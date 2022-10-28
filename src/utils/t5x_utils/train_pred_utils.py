from absl import logging

# Set Linen to add profiling information when constructing Modules.
# Must be set before flax imports.
# pylint:disable=g-import-not-at-top
import os
os.environ['FLAX_PROFILE'] = 'true'
from tqdm import tqdm
from flax import linen as nn
from flax import optim
from flax.metrics import tensorboard
from flax.training import checkpoints
from flax.training import common_utils
import jax
from jax import lax
from jax import random
from jax.interpreters.sharded_jit import sharded_jit
import jax.numpy as jnp
import ml_collections
import numpy as np
import t5
from utils.t5x import checkpoint_importer
from utils.t5x import input_pipeline
from utils.t5x import partitions
from utils.t5x import train_lib
from utils.t5x import models
from utils.t5x import decode
import tensorflow as tf
import time
import functools
from transformers import AutoModelWithLMHead
from utils.T5X_utils.models import set_hardware_bernoulli, TransformerConfig, Transformer
from utils.T5X_utils.load_pytorch_weights import load_weights_from_pytorch
import csv
from os.path import basename
from flax.core.frozen_dict import FrozenDict
import operator
from transformers import T5ForConditionalGeneration

def get_configs(config):
  """Get train, eval, and predict model configs.
  Args:
    config: The config dict for the experiment.
  Returns:
    A triple (train_config, eval_config, predict_config).
  """
  train_config = TransformerConfig(
      vocab_size=config.vocab_size,
      output_vocab_size=config.vocab_size,
      share_embeddings=config.share_embeddings,
      logits_via_embedding=config.logits_via_embedding,
      dtype=jnp.bfloat16 if config.use_bfloat16 else jnp.float32,
      emb_dim=config.emb_dim,
      num_heads=config.num_heads,
      num_layers=config.num_layers,
      qkv_dim=config.qkv_dim,
      mlp_dim=config.mlp_dim,
      mlp_activations=config.mlp_activations,
      position_embeddings='relative',
      relative_attention_num_buckets=config.relative_attention_num_buckets,
      relative_attention_max_distance=config.relative_attention_max_distance,
      max_len=max(config.max_input_length, config.max_target_length,
                  config.max_eval_input_length, config.max_eval_target_length),
      dropout_rate=config.dropout_rate,
      attention_dropout_rate=config.attention_dropout_rate,
      attention_fn=config.attention_fn,  # <---
      deterministic=False,
      decode=False,
      kernel_init=nn.initializers.xavier_uniform(),
      bias_init=nn.initializers.normal(stddev=1e-6))
  eval_config = train_config.replace(deterministic=True)  # pytype: disable=attribute-error
  predict_config = train_config.replace(  # pytype: disable=attribute-error
      deterministic=True,
      decode=True,
      max_decode_len=config.max_eval_target_length)

  return (train_config, eval_config, predict_config)



def get_initial_params(rng, config,
                       transformer_config,
                       optimizer_def):
  """Get the initial parameter tree."""
  if (config.load_pytorch_weights is not None):
    # Retrieve the state dict from the specified HuggingFace PyTorch model
    #torch_model = AutoModelWithLMHead.from_pretrained(config.load_pytorch_weights)
    torch_model = T5ForConditionalGeneration.from_pretrained(config.load_pytorch_weights)
    torch_model_params = torch_model.state_dict()
  # Create and initialize the transformer model
  input_shape = (config.batch_size, config.max_input_length)
  target_shape = (config.batch_size, config.max_target_length)
  initial_variables = Transformer(transformer_config).init(
      rng, jnp.ones(input_shape, jnp.float32),
      jnp.ones(target_shape, jnp.float32))
  if (config.load_pytorch_weights is not None):
    flax_weights = initial_variables['params'].unfreeze()
    loaded_weights = load_weights_from_pytorch(
        torch_model_params, flax_weights, config, debug=False)
    return optimizer_def.create(FrozenDict(loaded_weights))
  return optimizer_def.create(initial_variables['params'])




def get_optimizer(
    config,
    transformer_config,
    topology,
    broadcast,
    unbroadcast,
    init_rng,
    model_dir=None):
  """Get the optimizer according to the model and the topology
  specified in the configuration dictionary.
  Args:
    config: The config dict for the experiment
    transformer_config: The config for transformer init
    topology: The system topology to consider for sharding and parallelism
    broadcast: The broadcast function to use for tree device communication
    unbroadcast: The function to use for broadcast replicated axis removal
    init_rng: Initialization random seed
    model_dir: Optional; the model directory from which load checkpoints"""

  # Define the optimizer (with preinit params or loaded ones)
  # - First, we only abstractly initialize the optimizer and model parameters,
  #   since the parameters may not even fit in device memory!
  optimizer_def = optim.Adafactor(
      config.learning_rate, decay_rate=0.8, step_offset=config.step_offset)
  initialize_params_fn = functools.partial(
      get_initial_params,
      config=config,
      transformer_config=transformer_config,
      optimizer_def=optimizer_def)
  optimizer = jax.eval_shape(initialize_params_fn, init_rng)
  # Tuple-like pytree leaves for global_arg_shapes
  optimizer_shapes = jax.tree_map(lambda x: partitions.Spec(*x.shape),
                                  optimizer)
  optimizer_partitions = None
  per_host_optimizer_partitions = None

  # Handle (optional) multiple partitions in the optimizer
  # - Build parameter partition annotations for preserving partitions from train to eval
  if config.num_partitions > 1:
    optimizer_partitions = optimizer.restore_state(
        partitions.set_partitions(config.num_partitions,
                                  optimizer.state_dict()))
    per_host_optimizer_partitions = optimizer.restore_state(
        partitions.set_partitions(topology.per_host_num_partitions,
                                  optimizer.state_dict()))

  # Optionally restore current state a checkpoint file

  # a) Search last checkpoint in model_dir
  existing_checkpoint_found = False
  if config.restore_checkpoints:
    existing_checkpoint_found = train_lib.checkpoint_exists(model_dir)
    optimizer = checkpoints.restore_checkpoint(model_dir, optimizer)

  # b) Use a specified checkpoint path (downloaded from the Web)
  # Import a pretrained-T5 checkpoint only if we didn't import a local
  # "native" checkpoint (e.g. due to resuming a pre-empted finetuning run)
  if config.restore_t5_checkpoint and not existing_checkpoint_found:
    optimizer = checkpoint_importer.restore_from_t5_checkpoint(
        optimizer, config.restore_t5_checkpoint)
  
  # c) Only if we load a checkpoint (found or specified)...
  if config.restore_t5_checkpoint or existing_checkpoint_found:
    if config.num_partitions > 1:
      # Share params along the topology
      # Until checkpoint/restore is sharded, the restored checkpoint is global
      # and we need to slice each sharded parameter into the chunk containing
      # only the partitions that are present on this host.
      def per_host_chunk(x, spec):
        if spec is None or spec is x:  # unsharded or not a parameter
          return x
        if spec[0] == 1:
          dim_size = x.shape[1]
        elif spec[1] == 1:
          dim_size = x.shape[0]
        else:
          raise NotImplementedError()
        chunk_size = (
            dim_size * topology.per_host_num_partitions // config.num_partitions)
        lower = topology.per_replica_set_host_id * chunk_size
        upper = (topology.per_replica_set_host_id + 1) * chunk_size
        if spec[0] == 1:
          return x[:, lower:upper]
        else:
          return x[lower:upper]
      optimizer = jax.tree_multimap(per_host_chunk, optimizer,
                                    optimizer_partitions)
  else:
    # If pretraining and no checkpoint imported, we jit the (sharded-) init
    # function to minimize fragmentation. We use the same pmap(sharded_jit)
    # setup as the training step/loop to initialize everything "in-place" and
    # avoid communication or OOM.
    if config.num_partitions > 1:
      initialize_params_fn = sharded_jit(
          initialize_params_fn,
          in_parts=None,
          local_in_parts=None,
          out_parts=optimizer_partitions,
          local_out_parts=per_host_optimizer_partitions,
      )
      initialize_params_fn = jax.pmap(
          initialize_params_fn,
          'batch',
          in_axes=0,
          axis_size=topology.num_replicas,
          devices=topology.device_assignment)
      init_rng = broadcast(init_rng)
      optimizer = initialize_params_fn(init_rng)
      # We maintain the optimizer in unbroadcasted form (i.e. with no leading
      # replica axis). This is equivalent to the as-yet-nonexistent pmap kwarg
      # out_axes=None.
      optimizer = unbroadcast(optimizer)
    else:
      optimizer = jax.jit(initialize_params_fn)(init_rng)

  return optimizer, optimizer_shapes, \
    optimizer_partitions, per_host_optimizer_partitions


def decode_tokens(toks,
                  eos_id,
                  encoder,
                  max_id = 32000):
    """Decode tokens back to unicode."""
    del eos_id
    # TODO(levskaya): T5 doesn't seem to emit EOS tokens?  double check this
    # is the best decoding function or just switch to using tf_decode.
    # valid_toks = toks[:np.argmax(toks == eos_id) + 1].astype(np.int32)
    valid_toks = toks.astype(np.int32)
    valid_toks[valid_toks >= max_id] = 3
    return encoder.detokenize(valid_toks).numpy().decode('utf-8')


def predict_step(inputs,
                 params,
                 eos_id,
                 max_decode_len,
                 config,
                 beam_size = 4,
                 return_entire_beam = False):
  """Predict translation with fast decoding beam search on a batch."""
  # Prepare zeroed-out autoregressive cache.
  target_shape = (inputs.shape[0], max_decode_len) + inputs.shape[2:]
  cache = models.Transformer(config).init(
      jax.random.PRNGKey(0), jnp.ones(inputs.shape, config.dtype),
      jnp.ones(target_shape, config.dtype))['cache']
  # Prepare transformer fast-decoder call for beam search: for beam search, we
  # need to set up our decoder model to handle a batch size equal to
  # batch_size * beam_size, where each batch item's data is expanded in-place
  # rather than tiled.
  # i.e. if we denote each batch element subtensor as el[n]:
  # [el0, el1, el2] --> beamsize=2 --> [el0,el0,el1,el1,el2,el2]
  encoded_inputs = decode.flat_batch_beam_expand(
      models.Transformer(config).apply({'params': params},
                                       inputs,
                                       method=models.Transformer.encode),
      beam_size)
  raw_inputs = decode.flat_batch_beam_expand(inputs, beam_size)

  def tokens_ids_to_logits(flat_ids,
                           flat_cache):
    """Token slice to logits from decoder model."""
    # --> [batch * beam, vocab]
    flat_logits, new_vars = models.Transformer(config).apply(
        {
            'params': params,
            'cache': flat_cache
        },
        encoded_inputs,
        raw_inputs,  # only needed for input padding mask
        flat_ids,
        mutable=['cache'],
        method=models.Transformer.decode)
    new_flat_cache = new_vars['cache']
    return flat_logits, new_flat_cache

  # Using the above-defined single-step decoder function, run a
  # beam search over possible sequences given input encoding.
  # - beam_seqs [batch, beams, length + 1]
  # - beam_seqs_scores [batch, beams]
  beam_seqs, beam_seq_scores = decode.beam_search(        # <--
      inputs,
      cache,
      tokens_ids_to_logits,
      beam_size=beam_size,
      alpha=0.6,
      eos_id=eos_id,
      max_decode_len=max_decode_len)

  # Beam search output has beam dimension sorted in increasing order
  # of log-probability; "1:" filter drops first dummy 0 token.
  if return_entire_beam:
    # Return the highest scoring beam sequence
    return beam_seqs[:, :, 1:], beam_seq_scores[:, :]     # <--
  else:
    return beam_seqs[:, -1, 1:], beam_seq_scores[:, -1]   # <--



def get_p_pred_step(config,
                    predict_config,
                    topology,
                    optimizer_shapes,
                    optimizer_partitions,
                    per_host_optimizer_partitions,
                    eos_id=1):
  """Get the parallel prediction function according to the specified topology.
  Args:
    config: The config dict for the experiment
    predict_config: The config for prediction init
    topology: The system topology to consider for sharding and parallelism
    optimizer_shapes: ...
    optimizer_partitions: ...
    per_host_optimizer_partitions: ...
    eos_id: The identifier for eos token
  """

  # predict_step(inputs, params,
  #              eos_id, max_decode_len, config, beam_size=4) --> beam_seqs
  def p_pred_step(inputs, params):
    return predict_step(inputs, params, eos_id,
                        config.max_eval_target_length, predict_config,
                        config.beam_size,
                        return_entire_beam = True)
  
  if config.num_partitions > 1:
    p_pred_step = sharded_jit(
        p_pred_step,
        in_parts=(None, optimizer_partitions.target),
        local_in_parts=(None, per_host_optimizer_partitions.target),
        out_parts=None,
        local_out_parts=None)
  p_pred_step = jax.pmap(
      p_pred_step,
      axis_name='batch',
      in_axes=(0, None),
      global_arg_shapes=(None, optimizer_shapes.target),
      axis_size=topology.num_replicas,
      devices=topology.device_assignment)  # pytype: disable=wrong-arg-types

  return p_pred_step

def predict_output(pred_step_fn,
                   tokenized_batches,
                   optimizer,
                   topology,
                   padded_batch_size,
                   per_replica_set_eval_batch_size,
                   encoder,
                   tokenized_batches_field=None,
                   eos_id=1):
    """Apply a prediction function on the T5X model optimizer and
    the tokenized batches provided as input, returning (i) predicting tokens
    and (ii) detokenized predicted text.
    Args:
      pred_step_fn: Function to use for step prediction
      tokenized_batches: Batches for which make predictions
      tokenized_batches_field: Optional, if batches are structured,
        the name of the field to consider
      optimizer: T5X model optimizer to use for prediction
      topology: The system topology to consider for sharding and parallelism
      padded_batch_size: Batch size to consider for padding
      encoder: The tokenizer to use during decoding
      per_replica_set_eval_batch_size: Eval batch size for each host
      eos_id: The identifier for eos token
    """

    # Notes:
    # - all_predicted_seqs will come to contain the results of the prediction
    #   for the various input batches, possibly calculated in parallel on
    #   several replicates. The prediction of each tokenized text in each
    #   batch occurs with the application of the model and the use of
    #   beam search during the decoding phase.
    # - all_predicted_scores will come to contain the prediction score for
    #   each sequence obtained with beam search
    # - all_bs will come to contain the batch size of each input batch for
    #   which make a prediction, without padding (i.e., only effective inputs).
    #   E.g., considering a single batch with two tokenized texts to predict,
    #   it will be [2, 0, ... , 0]. It will be useful for padding removal.
    all_predicted_seqs, all_predicted_scores, all_bs = [], [], []

    for pred_batch in tqdm(tokenized_batches):

      # Get batch for which make prediction
      if (tokenized_batches_field is not None):
        pred_batch = pred_batch[tokenized_batches_field]
      
      # Handle final odd-sized batch by padding instead of dropping it.
      # Repeat the last item in the batch until it reaches a batch size
      # that is multiple of the desired one.
      # Shape notes:
      # BEFORE PADDING
      # - input_batch (
      #   pred_batch_size, i.e. number of sentences to predict in the batch,
      #   pred_batch_input_length, i.e. max encoding length for batch sentences)
      #   E.g., (2, 512)
      # AFTER PADDING (per_replica_set_eval_batch_size = 8)
      # - input_batch (8, 512)
      input_batch, unpadded_batch_size = train_lib.pad_batch_to_size(
          pred_batch, per_replica_set_eval_batch_size)
      all_bs.append(unpadded_batch_size)

      # Split batch dimensions for pmap
      # Shape notes:
      # - input_batch (1,
      #                per_replica_set_eval_batch_size,
      #                max_eval_input_length)
      #   E.g., (1, 8, 512)
      input_batch = jax.tree_map(
          lambda x: x.reshape(
              (topology.per_replica_set_num_replicas, -1) + x.shape[1:]),
          input_batch)
      
      # Run fast inference on batch applying beam search as decoding strategy
      # Shape notes:
      # - pred_seq (1,
      #             per_replica_set_eval_batch_size, i.e.
      #               input batch_size after padding
      #               or tokenized texts for which calculate a prediction
      #             beam_size, i.e, number of predictions to consider for each input,
      #             max_eval_target_length - 1, i.e. prediction length)
      #   E.g., (1, 8, 4, 61)
      # - pred_scores (1,
      #                per_replica_set_eval_batch_size,
      #                beam_size)
      #   E.g., (1, 8, 4)
      # - all_predicted_seqs (number of batches to predict, 1, 8, 4, 61)
      #   E.g., (1, 1, 8, 4, 61)
      # - all_predicted_scores (number of batches to predict, 1, 8, 4)
      #   E.g., (1, 1, 8, 4)
      pred_seq, pred_score = pred_step_fn(input_batch, optimizer.target)
      beam_size = pred_seq.shape[2]
      all_predicted_seqs.append(pred_seq)
      all_predicted_scores.append(pred_score)

    # Pad out the number of batches so each host has the same number
    # Notes:
    # - padded_batch_size specified in input
    #   E.g., 512
    # - len(all_predicted) = number of batches with beam_size texts to predict
    #     for each instance in each batch
    #   E.g., 1
    batch_shortfall = padded_batch_size - len(all_predicted_seqs)
    if batch_shortfall > 0:

      # To make sure the cross-host barriers work, we run the program the same
      # number of times on all hosts. The results of this call is ignored, and
      # the predictions are populated with zeros instead
      pred_step_fn(input_batch, optimizer.target)  # Dummy call

      # Extend combines two arrays into one.
      # Insert a new head-dimension made of empty batch-predictions (0s);
      # one for each missing batch
      # Notes:
      # - all_predicted_seqs (batch_shortfall,
      #                       len(all_predicted), i.e. number of predicted batches,
      #                       input batch size after padding,
      #                       beam_size,
      #                       prediction length)
      #   E.g., (511, 1, 8, 4, 61)
      # - all_predicted_scores
      #   E.g., (511, 1, 8, 4)
      all_predicted_seqs.extend([jnp.zeros_like(all_predicted_seqs[0])] *
                                batch_shortfall)
      all_predicted_scores.extend([jnp.zeros_like(all_predicted_scores[0])] *
                                  batch_shortfall)
      all_bs.extend([0] * batch_shortfall)
    
    # Concatenate the first two dimensions to reach the target padded_batch_size
    # Notes:
    # - all_predicted_seqs (padded_batch_size,
    #                       input batch size after padding,
    #                       beam_size,
    #                       prediction length)
    #   E.g., (512, 8, 4, 61)
    # - all_predicted_scores
    #   E.g., (512, 8, 4)
    all_predicted_seqs = jnp.concatenate(all_predicted_seqs)
    all_predicted_scores = jnp.concatenate(all_predicted_scores)
    all_bs = jnp.array(all_bs)

    # Collect all batches from across hosts and reverse sharding
    # It brings together the prediction results of the various hosts.
    # Build an array with a new leading dimension of size num_replica_sets,
    # carrying the data copied from all hosts
    # Notes:
    # - all_predicted_seqs (num_replica_sets,
    #                       padded_batch_size,
    #                       input batch size after padding,
    #                       beam_size,
    #                       prediction length)
    #   E.g., (1, 512, 8, 4, 61)
    # - all_predicted_scores
    #   E.g., (1, 512, 8, 4)
    all_predicted_seqs = train_lib.host_allgather(
        all_predicted_seqs,
        topology.num_replica_sets, topology.replica_set_id,
        topology.per_replica_set_host_id == 0)
    all_predicted_scores = train_lib.host_allgather(
        all_predicted_scores,
        topology.num_replica_sets, topology.replica_set_id,
        topology.per_replica_set_host_id == 0)
    seqlength = all_predicted_seqs.shape[-1] # E.g., 61

    # Calculates the total number of effective input texts (i.e., without
    # considering padding) putting together the batches assigned to the
    # various hosts
    # Notes:
    # - total_examples (padded_batch_size,)
    #   E.g., (2,), such as [2, 0, ..., 0]
    total_examples = np.sum(
        train_lib.host_allgather(all_bs, topology.num_replica_sets,
                                 topology.replica_set_id,
                                 topology.per_replica_set_host_id == 0))
    del all_bs

    # De-shard the collected predicted tokens and remove padding
    # Notes:
    # (i) change all_predicted dimensions order
    #     all_predicted_seqs = (input batch size after padding,
    #                           num_replica_sets,
    #                           padded_batch_size,
    #                           beam_size,
    #                           prediction length)
    #     E.g., (8, 1, 512, 4, 61)
    # (ii) reshape the result as sequence of predictions with length seqlength
    #      E.g., (16384, 61)
    # (iii) filter only valid predictions (the first ones); for each input text
    #       (total_examples) there are beam_size possible outputs
    #       E.g., (2*4, 61) = (8, 61)
    # Similarly, the output for all_predicted_scores will be of shape (8, 1)
    all_predicted_seqs = np.transpose(
        all_predicted_seqs, (1, 2, 0, 3, 4)).reshape(
            -1, seqlength)[:total_examples*beam_size]
    all_predicted_scores = np.transpose(
        all_predicted_scores, (1, 2, 0, 3)).reshape(
            -1, 1)[:total_examples*beam_size]
    
    # Concat score results
    # E.g., (8, 1) --> (8,)
    all_predicted_scores = functools.reduce(
        operator.iconcat, all_predicted_scores, [])

    # We now extract raw predictions a single host
    if jax.host_id() == 0:
      raw_predictions = []
      for tokens in all_predicted_seqs:
        raw_predictions.append(decode_tokens(tokens, eos_id, encoder))

    return all_predicted_seqs, raw_predictions, all_predicted_scores
