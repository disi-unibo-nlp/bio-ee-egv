# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This script pre-trains or fine-tunes a Transformer using the T5 data pipeline."""
from concurrent.futures import thread
import functools
import importlib
import os
import time
from typing import Any, Mapping, Sequence, Tuple
from tqdm import tqdm
from absl import logging
# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This script pre-trains or fine-tunes a Transformer using the T5 data pipeline."""
from concurrent.futures import thread
import functools
import importlib
import os
from typing import Any, Mapping, Sequence, Tuple
from flax.core.frozen_dict import FrozenDict
from absl import logging

# Set Linen to add profiling information when constructing Modules.
# Must be set before flax imports.
# pylint:disable=g-import-not-at-top
os.environ['FLAX_PROFILE'] = 'true'
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
from transformers import AutoModelWithLMHead
from utils.T5X_utils.models import set_hardware_bernoulli, TransformerConfig, Transformer
from utils.T5X_utils.load_pytorch_weights import load_weights_from_pytorch
from utils.T5X_utils.train_pred_utils import get_configs, get_initial_params, get_optimizer, decode_tokens, get_p_pred_step, predict_output
import csv
from os.path import basename


# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This script pre-trains or fine-tunes a Transformer using the T5 data pipeline."""


# Show logging info on console
logging.set_verbosity('info')

# log_file_name_path = "/content/metrics.csv"
# pylint:disable=g-long-lambda

CFG = None
PyTreeDef = type(jax.tree_structure(None))
ConfigDict = ml_collections.ConfigDict


def train(model_dir=None,
          data_dir=None,
          config=None):
  """
  Args:
    model_dir: Directory to store model data.
    data_dir: Tensorflow datasets directory.
    config: Training config file.
  """

  global CFG
  CFG = config

  #  ---------------------------
  # | 1. PRELIMINARY OPERATIONS |
  #  ---------------------------

  # Guarantee that the JAX bfloat16 extension is used rather than TF bfloat16.
  _ = np.array(jnp.array([1.0], dtype=jnp.bfloat16))

  # Use hardware RNG for bernoulli randoms in dropout mask creation.
  if CFG.hardware_rng:
    set_hardware_bernoulli()

  if 'module_import' in CFG and CFG.module_import:
    for module in CFG.module_import:
      importlib.import_module(module)

  if 'additional_task_cache_dirs' in CFG and CFG.additional_task_cache_dirs:
    t5.data.add_global_cache_dirs(CFG.additional_task_cache_dirs)

  # Define the topology for distributed computation based on config params
  # and check batch shapes
  num_partitions = CFG.num_partitions
  topology = train_lib.compute_multihost_topology(num_partitions)
  batch_size = CFG.batch_size
  eval_batch_size = CFG.eval_batch_size
  per_replica_set_eval_batch_size = eval_batch_size // topology.num_replica_sets
  if batch_size % topology.num_replicas:
    raise ValueError('Batch size must be divisible by the number of replicas.')

  steps_per_epoch = CFG.steps_per_epoch
  logging.info('steps per epoch: %d', steps_per_epoch)

  broadcast = functools.partial(
      train_lib.broadcast,
      num_replicas=topology.per_replica_set_num_replicas,
      num_partitions=topology.per_host_num_partitions,
      devices=topology.this_host_device_assignment)

  # Initialize TensorBoard summaries under model_dir/train(and eval)
  if jax.host_id() == 0:
    tf.io.gfile.makedirs(model_dir)
    tensorboard_folder = model_dir + "../model_tensorboard/"
    tf.io.gfile.makedirs(tensorboard_folder)
    train_summary_writer = tensorboard.SummaryWriter(
        os.path.join(tensorboard_folder, 'train'))
    eval_summary_writer = tensorboard.SummaryWriter(
        os.path.join(tensorboard_folder, 'eval'))
    gpu_summary_writer = tensorboard.SummaryWriter(
        os.path.join(tensorboard_folder, 'train'))
    time_summary_writer = tensorboard.SummaryWriter(
        os.path.join(tensorboard_folder, 'eval'))
  else:
    train_summary_writer = None
    eval_summary_writer = None
    gpu_summary_writer = None
    time_summary_writer = None

  if CFG.infeed:
    # Write summaries in background thread to avoid blocking on device sync
    # Infeed is currently synchronous, so do it in a background thread too
    infeed_pool = thread.ThreadPoolExecutor(jax.local_device_count(), 'infeed')

  # Obtain TRAIN and EVAL datasets (not TEST)
  # Eval cache contained the tokenized batches for each training task
  # to consider for evaluation purposes
  (train_ds, eval_ds), eval_cache = input_pipeline.get_datasets_and_cache(
      CFG, topology.num_replica_sets, topology.replica_set_id,
      topology.per_replica_set_host_id)

  # Retrieve Vocabulary and Tokenizer
  vocab = input_pipeline.get_vocabulary(CFG.mixture_or_task_name)
  encoder = vocab.tf_tokenizer
  eos_id = vocab.tokenizer.eos_id()

  #  ----------------------------
  # | 2. T5 MODEL INITIALIZATION |
  #  ----------------------------

  logging.info('Initializing model, optimizer, and step functions.')

  # 1) Divide the original configuration params in multiple configs by their use
  train_config, eval_config, predict_config = get_configs(CFG)

  # 2) Random seed
  rng = random.PRNGKey(CFG.random_seed)
  rng, init_rng = random.split(rng)

  # 3) Variables for parallelized training (according to the topology)
  # This is used for infeed conversion from feature dict <--> tuple
  train_keys = [
      'inputs', 'targets', 'inputs_position', 'targets_position',
      'inputs_segmentation', 'targets_segmentation'
  ]
  device_train_input_shape = tuple([
      (batch_size // topology.num_replicas,
       CFG.max_input_length if 'inputs' in k else CFG.max_target_length)
      for k in train_keys
  ])

  # 4) Define the learning rate function
  # Note: if schedule is "constant", warmup is not used
  learning_rate_fn = train_lib.create_learning_rate_scheduler(
      factors=CFG.schedule,
      base_learning_rate=CFG.learning_rate,
      warmup_steps=CFG.warmup_steps)

  # 5) Define the optimizer
  optimizer, optimizer_shapes, \
  optimizer_partitions, \
  per_host_optimizer_partitions = get_optimizer(
    config=CFG,
    transformer_config=eval_config,
    topology=topology,
    broadcast=broadcast,
    unbroadcast=train_lib.unbroadcast,
    init_rng=init_rng,
    model_dir=model_dir)

  #  -----------------------------------------------------------
  # | 3. FUNCTION DEFINITIONS FOR TRAINING                      |
  # | These functions will be called later                      |
  # |                                                           |
  # | Compile multidevice versions of train/eval/predict step   |
  # | and cache init fn.                                        |
  #  -----------------------------------------------------------

  # NOTE: the following A and B defintions refer to two alternative functions
  # with parallelized execution (i.e., pmap)

  # We can use either a single train-step for a host training loop:

  #  ---------------------
  # | 3.1.A) P-TRAIN STEP |
  #  ---------------------

  # train_step(optimizer, batch, prev_metrics, dropout_rng, **kwargs)
  #  --> new_optimizer, metrics, new_dropout_rng
  def p_train_step(optimizer, batch,
                   prev_metrics,
                   dropout_rng):
    return train_lib.train_step(
        optimizer,
        batch,
        prev_metrics,
        dropout_rng,
        config=train_config,
        learning_rate_fn=learning_rate_fn,
        num_microbatches=CFG.microbatches,
        label_smoothing=CFG.label_smoothing,
        z_loss=CFG.z_loss,
        use_bfloat16=CFG.use_bfloat16)

  if num_partitions > 1:
    p_train_step = sharded_jit(
        p_train_step,
        in_parts=(optimizer_partitions, None, None, None),
        local_in_parts=(per_host_optimizer_partitions, None, None, None),
        out_parts=(optimizer_partitions, None, None),
        local_out_parts=(per_host_optimizer_partitions, None, None))
  # TODO(levskaya): the in_axes spec below might be wrong, double-check.
  p_train_step = jax.pmap(
      p_train_step,
      axis_name='batch',
      in_axes=(None, 0, 0, 0),
      donate_argnums=(0,),
      global_arg_shapes=(optimizer_shapes, None, None, None),
      axis_size=topology.num_replicas,
      devices=topology.device_assignment)  # pytype: disable=wrong-arg-types

  #  ----------------------
  # | 3.1.B) P-TRAIN EPOCH |
  #  ----------------------
  # OR, we use an on-device loop that feeds the training step via infeed queue.

  def device_train_loop_cond(
      args
  ):
    """Stopping criterion for on-device loop."""
    _, _, _, _, step, epoch = args
    return step // steps_per_epoch == epoch

  def device_train_loop_body(
      args
  ):
    """On-device loop body."""
    optimizer, dropout_rngs, metrics, token, step, epoch = args
    # Ordering input data from infeed requires threading a symbolic token
    # through the computation.
    input_data, token = lax.infeed(
        token,
        shape=tuple(
            [jax.ShapedArray(s, jnp.int32) for s in device_train_input_shape]))
    # Rebuild input dict from infeed data tuple.
    batch = {k: v for k, v in zip(train_keys, input_data)}
    # Run the train_step function and return the loop state.
    optimizer, metrics, dropout_rngs = train_lib.train_step(
        optimizer,
        batch,
        metrics,
        dropout_rngs,
        train_config,
        learning_rate_fn,
        num_microbatches=CFG.microbatches,
        label_smoothing=CFG.label_smoothing,
        z_loss=CFG.z_loss)
    step += 1
    return optimizer, dropout_rngs, metrics, token, step, epoch

  def device_train_loop(optimizer, dropout_rngs,
                        metrics, step,
                        epoch):
    # Create symbolic token for threading infeed data.
    token = lax.create_token(step)
    # Run on-device loop.
    optimizer, dropout_rngs, metrics, _, step, _ = lax.while_loop(
        device_train_loop_cond, device_train_loop_body,
        (optimizer, dropout_rngs, metrics, token, step, epoch))
    return optimizer, dropout_rngs, metrics, step

  if num_partitions > 1:
    device_train_loop = sharded_jit(
        device_train_loop,
        in_parts=(optimizer_partitions, None, None, None, None),
        local_in_parts=(per_host_optimizer_partitions, None, None, None, None),
        out_parts=(optimizer_partitions, None, None, None),
        local_out_parts=(per_host_optimizer_partitions, None, None, None))
  p_train_epoch = jax.pmap(
      device_train_loop,
      axis_name='batch',
      in_axes=(None, 0, 0, None, None),
      donate_argnums=(0,),
      global_arg_shapes=(optimizer_shapes, None, None, None, None),
      axis_size=topology.num_replicas,
      devices=topology.device_assignment)  # pytype: disable=wrong-arg-types

  #  ----------------------------------
  # | 3.2) P-ALLREDUCE FOR METRIC DATA |
  #  ----------------------------------
  # Reduction psum for metric data.

  def p_allreduce_metrics(x):
    return lax.psum(x, axis_name='batch')

  if num_partitions > 1:
    p_allreduce_metrics = sharded_jit(
        p_allreduce_metrics,
        in_parts=None,
        local_in_parts=None,
        out_parts=None,
        local_out_parts=None,
        num_partitions=num_partitions,
        local_num_partitions=topology.per_host_num_partitions)
  p_allreduce_metrics = jax.pmap(
      p_allreduce_metrics,
      axis_name='batch',
      global_arg_shapes=None,
      axis_size=topology.num_replicas,
      devices=topology.device_assignment)

  #  ------------------
  # | 3.3) P-EVAL STEP |
  #  ------------------

  # Training evaluation computation.

  # eval_step(params, batch, config, label_smoothing=0.0) --> metrics
  def p_eval_step(params, batch):
    return train_lib.eval_step(
        params, batch, config=eval_config, label_smoothing=CFG.label_smoothing)

  if num_partitions > 1:
    p_eval_step = sharded_jit(
        p_eval_step,
        in_parts=(optimizer_partitions.target, None),
        local_in_parts=(per_host_optimizer_partitions.target, None),
        out_parts=None,
        local_out_parts=None)
  p_eval_step = jax.pmap(
      p_eval_step,
      axis_name='batch',
      in_axes=(None, 0),
      global_arg_shapes=(optimizer_shapes.target, None),
      axis_size=topology.num_replicas,
      devices=topology.device_assignment)  # pytype: disable=wrong-arg-types

  # start time
  start_time = time.time()

  #  ------------------
  # | 3.4) P-PRED STEP |
  #  ------------------

  # Fast autoregressive decoding loop.
  # For inference and model evaluation.

  p_pred_step = get_p_pred_step(
      CFG,
      predict_config,
      topology,
      optimizer_shapes,
      optimizer_partitions,
      per_host_optimizer_partitions,
      eos_id=1)


  #  --------------------
  # | 4. MAIN TRAIN LOOP |
  #  --------------------

  # We init the first set of dropout PRNG keys, but update it afterwards inside
  # the main pmap'd training update for performance.
  # There should be a unique dropout key for each replica represented on this
  # host, but the key should be the same for the same replica on other hosts.
  # Again, this is what the replica set abstraction is for.
  dropout_rngs = random.split(
      random.fold_in(rng, topology.replica_set_id),
      topology.per_replica_set_num_replicas)

  # Restore step from last checkpoint
  host_step = int(optimizer.state.step)

  empty_metrics = broadcast({
      'loss': 0.0,
      'accuracy': 0.0,
      'learning_rate': 0.0,
      'denominator': 0.0
  })

  if CFG.infeed:
    # Execute p_train_epoch if infeed is true, p_train_step within a loop otherwise
    logging.info('Precompiling training loop and moving optimizer to device.')
    optimizer, _, metrics, _ = p_train_epoch(optimizer, dropout_rngs,
                                             empty_metrics,
                                             jnp.array(0, dtype=jnp.int32), 1)
    optimizer = train_lib.unbroadcast(optimizer)
    metrics['loss'].block_until_ready()

  logging.info('Starting training loop.')

  # Init devices info
  local_devices = jax.local_devices()
  device_step = broadcast(host_step)

  # Init first epoch index
  # E.g. host_step = 1000, steps_per_epoch = 10, we are at epoch 100
  # Note: // (division with integer result, floor rounding)
  first_epoch = host_step // steps_per_epoch
  #first_epoch = 0

  # Define an iterator on TRAINING DATASET (batch-level)
  train_iter = train_ds.as_numpy_iterator()

  #  ---------------------
  # | 4.1. FOR EACH EPOCH |
  #  ---------------------

  for epoch in range(first_epoch, first_epoch + CFG.num_epochs):

    logging.info("Epoch %d", epoch)

    # Init (empty) metrics
    metrics = empty_metrics

    # NOTE: 'optimizer' is unbroadcast by construction at initialization or
    # when loading a checkpoint. It is maintained in 'unbroadcast' state to
    # enable the XLA cross-replica sharding optimization.  The broadcasting is
    # handled automatically by the pmap'd functions that use it.

    #  ----------------------------------
    # | 4.1.1 EVALUATION BEFORE TRAINING |
    #  ----------------------------------
    # - Eval Dataset is divided in dictionary with two keys: inputs and targets.
    #   Each item contains a batch (size: eval_batch_dim) of tokenized array
    # - An internal Transformer model is created by calling P-PRED STEP
    # - P-PRED STEP apply the model (calling "decode") on the input-side and
    #   compare the prediction (obtained using beam search) with the target
    # - This evaluation process is repeated for each batch obtained by the eval
    #   dataset: greater is the eval dataset's size, greater is the requested
    #   computation time

    # Gather all task evaluation metrics
    logging.info('Evaluating tasks.')
    if epoch == first_epoch + 1:
      train_lib.sync_devices()
    for task in eval_cache.tasks:
      logging.info('Evaluating task %s', task.name)
      max_host_batch_number = np.max(
          eval_cache.preprocessed_batch_sizes[task.name])
      pred_tokens, pred_texts, pred_scores = predict_output(
          pred_step_fn=p_pred_step,
          tokenized_batches=eval_cache.preprocessed_examples[task.name],
          tokenized_batches_field='inputs',
          optimizer=optimizer,
          topology=topology,
          padded_batch_size=max_host_batch_number,
          encoder=encoder,
          per_replica_set_eval_batch_size=per_replica_set_eval_batch_size,
          eos_id=eos_id)

      # We now run the post-processing and metric-fns on a single host
      if jax.host_id() == 0:
        assert eval_summary_writer

        # post-process predictions for metric fns
        # note:
        # - there are beam_size predictions for each eval example,
        #   len(eval_cache.examples[task.name]) = beam_size * len(pred_texts);
        #   we need to repeat each example item beam_size times
        repeated_task_examples = np.repeat(
            eval_cache.examples[task.name], CFG.beam_size)
        repeated_targets = [str(d['targets_plaintext'], 'utf-8') for d in repeated_task_examples]
        predictions = [
            task.postprocess_fn(p, example=ex)
            for p, ex in zip(
                pred_texts,
                repeated_task_examples)
        ]

        # calculate metric scores after making all predictions
        for metric_fn in task.metric_fns:
          scores = metric_fn(repeated_targets, predictions)
          for metric_name, metric_value in scores.items():
            tag = f'eval/{task.name}/{metric_name}'
            eval_summary_writer.scalar(tag, metric_value, host_step)
            logging.info('EVAL %s at step %d: %.3f', tag, host_step,
                         metric_value)

            #with open(log_file_name_path, 'a', newline='') as file:
            # writer = csv.writer(file, delimiter=';')
            # writer.writerow([time.time(), tag, metric_value, host_step])

          eval_summary_writer.flush()
          write_info_tensorboard(CFG.mixture_or_task_name,
                                 gpu_summary_writer,
                                 time_summary_writer,
                                 start_time,
                                 host_step)

        # Save text samples for tensorboard
        exemplars = ''
        N_EXAMPLES_TO_SHOW = 8
        for n in np.random.choice(
            np.arange(len(predictions)), N_EXAMPLES_TO_SHOW):
          input_txt = tf.compat.as_text(
              repeated_task_examples[n]['inputs_plaintext'])
          tgt_txt = tf.compat.as_text(
              repeated_task_examples[n]['targets_plaintext'])
          pred_txt = pred_texts[n]
          print(tgt_txt)
          print(pred_txt)
          exemplars += (f'{input_txt}\n\n'
                        f'target: {tgt_txt}\n\n'
                        f'prediction: {pred_txt}\n\n')
        eval_summary_writer.text(f'{task.name} samples', exemplars, host_step)
        eval_summary_writer.flush()

        write_info_tensorboard(CFG.mixture_or_task_name,
                               gpu_summary_writer,
                               time_summary_writer,
                               start_time,
                               host_step)
        #with open(log_file_name_path, 'a', newline='') as file:
        # writer = csv.writer(file, delimiter=';')
        # writer.writerow([time.time(), f'{task.name} samples', exemplars, host_step])

    # Take an Xprof trace after the first loop has compiled everything
    if epoch == first_epoch + 1:
      train_lib.sync_devices()

    #  ----------------
    # | 4.1.2 TRAINING |
    #  ----------------

    # For on-device loop, we launch the computation before feeding data.
    logging.info('BEGIN Train loop.')
    if CFG.infeed:
      optimizer, dropout_rngs, metrics, device_step = p_train_epoch(
          optimizer, dropout_rngs, metrics, train_lib.unbroadcast(device_step),
          epoch)
      optimizer = train_lib.unbroadcast(optimizer)

    #  -----------------------------------------
    # | 4.1.2.1 FOR EACH BATCH WITHIN THE EPOCH |
    #  -----------------------------------------

    # I.e., 1000 / 10 = 100, 1001 / 10 = 100,1 = 100...
    # 1010 / 10 = 101 (new epoch)
    #target_step = host_step + CFG.num_epochs * steps_per_epoch
    #while host_step <= target_step:
    while int(host_step // steps_per_epoch) == epoch:

      logging.info("Host Step: %d", host_step)

      # Get the next batch
      batch = next(train_iter)
      batch = jax.tree_map(
          lambda x: x.reshape(
              (topology.per_replica_set_num_replicas, -1) + x.shape[1:]), batch)
      print(batch)

      # Feed the on-device training loop
      # Note: p_train_epoch is used with infeed (see above),
      # and p_train step otherwise (in fact is located inside the loop)
      if CFG.infeed:
        for i, device in enumerate(local_devices):
          # When using infeed to provide data to the computation, we're on our
          # own for feeding the right values to the right devices. Each device
          # should get the minibatch corresponding to its replica, a slice of
          # the larger batch corresponding to the host's replica set.
          if device.platform == 'tpu':
            device_coords = (*device.coords, device.id % 2)
          else:
            device_coords = (device.host_id, i)
          per_replica_set_device_coords = tuple(
              dc % prsm
              for dc, prsm in zip(device_coords, topology.per_replica_set_mesh))
          per_replica_set_replica_coords = tuple(
              prsdc // prm for prsdc, prm in zip(per_replica_set_device_coords,
                                                 topology.per_replica_mesh))
          per_replica_set_replica_id = 0
          for prsm, prm, prsrc in zip(topology.per_replica_set_mesh,
                                      topology.per_replica_mesh,
                                      per_replica_set_replica_coords):
            per_replica_set_replica_id = (
                per_replica_set_replica_id * prsm // prm + prsrc)
          input_tuple = tuple(
              [batch[k][per_replica_set_replica_id] for k in train_keys])
          # Safety check: infeed does not check shape or types but requires
          # them to agree with on-device spec, otherwise the queue and program
          # stalls.
          tuple_shapes = jax.tree_map(jnp.shape, input_tuple)
          tuple_dtypes = jax.tree_map(lambda x: x.dtype, input_tuple)
          assert tuple_shapes == device_train_input_shape, (
              'infeed shape error %s != %s' %
              (tuple_shapes, device_train_input_shape))
          assert tuple(set(tuple_dtypes)) == (jnp.int32,), \
              ('infeed dtype error %s not all of type %s' % (
                  tuple_dtypes, jnp.int32))
          infeed_pool.submit(
              functools.partial(device.transfer_to_infeed, input_tuple))
      # Host training loop.
      else:
        optimizer, metrics, dropout_rngs = p_train_step(optimizer, batch,
                                                        metrics, dropout_rngs)
        optimizer = train_lib.unbroadcast(optimizer)
      host_step += 1
    logging.info('END Train loop.')

    # Maybe save a checkpoint on one host
    if (CFG.save_checkpoints and
        epoch % CFG.checkpoint_freq == CFG.checkpoint_freq - 1 and
        jax.host_id() == 0):
      checkpoints.save_checkpoint(model_dir, optimizer, host_step)

    # Gather training metrics
    metrics = p_allreduce_metrics(metrics)
    metrics = jax.tree_map(lambda x: jax.device_get(x[0]), metrics)
    denominator = metrics.pop('denominator')
    summary = jax.tree_map(lambda x: x / denominator, metrics)  # pylint: disable=cell-var-from-loop
    logging.info('train in step: %s, %s', host_step, summary)
    if jax.host_id() == 0:
      #with open(log_file_name_path, 'a', newline='') as file:
      #   writer = csv.writer(file, delimiter=';')
      #   writer.writerow([time.time(), key, val, host_step])
      assert train_summary_writer
      for key, val in summary.items():
        train_summary_writer.scalar(key, val, host_step)
      train_summary_writer.flush()
      write_info_tensorboard(CFG.mixture_or_task_name,
                             gpu_summary_writer,
                             time_summary_writer,
                             start_time,
                             host_step)

    # Gather training evaluation metrics
    logging.info('Gathering training evaluation metrics.')
    eval_metrics = []
    eval_iter = eval_ds.as_numpy_iterator()
    for _, eval_batch in zip(range(CFG.num_eval_steps), eval_iter):
      eval_batch = jax.tree_map(
          lambda x: x.reshape(
              (topology.per_replica_set_num_replicas, -1) + x.shape[1:]),
          eval_batch)
      metrics = p_eval_step(optimizer.target, eval_batch)
      eval_metrics.append(metrics)
    # average metrics across devices
    eval_metrics = p_allreduce_metrics(eval_metrics)
    eval_metrics = common_utils.get_metrics(eval_metrics)
    # average metrics across steps
    eval_metrics = jax.tree_map(np.sum, eval_metrics)
    eval_denominator = eval_metrics.pop('denominator')
    eval_summary = jax.tree_map(
        lambda x: x / eval_denominator,  # pylint: disable=cell-var-from-loop
        eval_metrics)
    logging.info('eval in step: %s, %s', host_step, eval_summary)
    if jax.host_id() == 0:
      #with open(log_file_name_path, 'a', newline='') as file:
      #   writer = csv.writer(file, delimiter=';')
      #   writer.writerow([time.time(), key, val, host_step])
      assert eval_summary_writer
      for key, val in eval_summary.items():
        eval_summary_writer.scalar(key, val, host_step)
      eval_summary_writer.flush()
      write_info_tensorboard(CFG.mixture_or_task_name,
                             gpu_summary_writer,
                             time_summary_writer,
                             start_time,
                             host_step)

  # Wait until computations are done before exiting
  logging.info('Finished.')
  train_lib.sync_devices()
  # Shut down the infeed threadpool.
  if CFG.infeed:
    infeed_pool.shutdown()
