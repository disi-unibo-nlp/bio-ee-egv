def load_weights_from_pytorch(pytorch_model,
                              flax_model,
                              config,
                              debug=False):
  """Load T5-base F32 model weights from HuggingFace PyTorch to Google Research FLAX.

  Args:
    pytorch_model: the pre-trained PyTorch T5 model with the weights to load
    flax_model: the Google Research FLAX model on which to transfer the weights
    config: the configuration file with model architecture info
    debug: boolean; true if you want to print architectural info for the models

  Return:
    The state_dict with updated target (state is left unchanged)
  """

  if (debug):
    print(pytorch_model)
    print(flax_model)

  num_hidden_layers = config.num_layers
  final_dense_layer = config.final_dense_layer
  
  # EMBEDDINGS
  # -----------------------
  # - In a complete Transformer architecture like that of T5,
  #   the encoder and decoder embeddings can be different or shared
  # - We consider shared embeddings, being the most widespread architectural choice
  # - In the case of shared embedding, T5X does not introduce redundant modules
  #   with equal weights for encoder and decoder, but PyTorch does.
  #   These layers (i.e., encoder.embed_tokens.weight and decoder.embed_tokens.weight)
  #   are consequently ignored.

  flax_model["shared_embedding"]["embedding"] = \
    pytorch_model["shared.weight"].numpy()

  # ENCODER
  # -----------------------

  flax_model["encoder"]["encoder_relative_posemb"]["rel_embedding"] = \
    pytorch_model["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"].T.numpy()

  for i in range(0, num_hidden_layers):

    flax_model["encoder"]["encoderblock_"+str(i)]["LayerNorm_0"]["scale"] = \
      pytorch_model["encoder.block."+str(i)+".layer.0.layer_norm.weight"].numpy()

    flax_model["encoder"]["encoderblock_"+str(i)]["SelfAttention_0"]["key"]["kernel"] = \
      pytorch_model["encoder.block."+str(i)+".layer.0.SelfAttention.k.weight"].T.numpy()
    flax_model["encoder"]["encoderblock_"+str(i)]["SelfAttention_0"]["query"]["kernel"] = \
      pytorch_model["encoder.block."+str(i)+".layer.0.SelfAttention.q.weight"].T.numpy()
    flax_model["encoder"]["encoderblock_"+str(i)]["SelfAttention_0"]["value"]["kernel"] = \
      pytorch_model["encoder.block."+str(i)+".layer.0.SelfAttention.v.weight"].T.numpy()
    flax_model["encoder"]["encoderblock_"+str(i)]["SelfAttention_0"]["out"]["kernel"] = \
      pytorch_model["encoder.block."+str(i)+".layer.0.SelfAttention.o.weight"].T.numpy()

    flax_model["encoder"]["encoderblock_"+str(i)]["MlpBlock_0"]["wo"]["kernel"] = \
      pytorch_model["encoder.block."+str(i)+".layer.1.DenseReluDense.wo.weight"].T.numpy()
    flax_model["encoder"]["encoderblock_"+str(i)]["MlpBlock_0"]["wi"]["kernel"] = \
      pytorch_model["encoder.block."+str(i)+".layer.1.DenseReluDense.wi.weight"].T.numpy()

    flax_model["encoder"]["encoderblock_"+str(i)]["LayerNorm_1"]["scale"] = \
      pytorch_model["encoder.block."+str(i)+".layer.1.layer_norm.weight"].numpy()

  flax_model["encoder"]["encoder_norm"]["scale"] = \
    pytorch_model["encoder.final_layer_norm.weight"].numpy()
  
  # DECODER
  # -----------------------

  flax_model["decoder"]["decoder_relative_posemb"]["rel_embedding"] = \
    pytorch_model["decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"].T.numpy()

  for i in range(0, num_hidden_layers):

    flax_model["decoder"]["encoderdecoderblock_"+str(i)]["LayerNorm_0"]["scale"] = \
      pytorch_model["decoder.block."+str(i)+".layer.0.layer_norm.weight"].numpy()
    
    flax_model["decoder"]["encoderdecoderblock_"+str(i)]["SelfAttention_0"]["key"]["kernel"] = \
      pytorch_model["decoder.block."+str(i)+".layer.0.SelfAttention.k.weight"].T.numpy()
    flax_model["decoder"]["encoderdecoderblock_"+str(i)]["SelfAttention_0"]["query"]["kernel"] = \
      pytorch_model["decoder.block."+str(i)+".layer.0.SelfAttention.q.weight"].T.numpy()
    flax_model["decoder"]["encoderdecoderblock_"+str(i)]["SelfAttention_0"]["value"]["kernel"] = \
      pytorch_model["decoder.block."+str(i)+".layer.0.SelfAttention.v.weight"].T.numpy()
    flax_model["decoder"]["encoderdecoderblock_"+str(i)]["SelfAttention_0"]["out"]["kernel"] = \
      pytorch_model["decoder.block."+str(i)+".layer.0.SelfAttention.o.weight"].T.numpy()

    flax_model["decoder"]["encoderdecoderblock_"+str(i)]["LayerNorm_1"]["scale"] = \
      pytorch_model["decoder.block."+str(i)+".layer.1.layer_norm.weight"].numpy()
    
    flax_model["decoder"]["encoderdecoderblock_"+str(i)]["MultiHeadDotProductAttention_0"]["key"]["kernel"] = \
      pytorch_model["decoder.block."+str(i)+".layer.1.EncDecAttention.k.weight"].T.numpy()
    flax_model["decoder"]["encoderdecoderblock_"+str(i)]["MultiHeadDotProductAttention_0"]["query"]["kernel"] = \
      pytorch_model["decoder.block."+str(i)+".layer.1.EncDecAttention.q.weight"].T.numpy()
    flax_model["decoder"]["encoderdecoderblock_"+str(i)]["MultiHeadDotProductAttention_0"]["value"]["kernel"] = \
      pytorch_model["decoder.block."+str(i)+".layer.1.EncDecAttention.v.weight"].T.numpy()
    flax_model["decoder"]["encoderdecoderblock_"+str(i)]["MultiHeadDotProductAttention_0"]["out"]["kernel"] = \
      pytorch_model["decoder.block."+str(i)+".layer.1.EncDecAttention.o.weight"].T.numpy()

    flax_model["decoder"]["encoderdecoderblock_"+str(i)]["LayerNorm_2"]["scale"] = \
      pytorch_model["decoder.block."+str(i)+".layer.2.layer_norm.weight"].numpy()
    
    flax_model["decoder"]["encoderdecoderblock_"+str(i)]["MlpBlock_0"]["wo"]["kernel"] = \
      pytorch_model["decoder.block."+str(i)+".layer.2.DenseReluDense.wo.weight"].T.numpy()
    flax_model["decoder"]["encoderdecoderblock_"+str(i)]["MlpBlock_0"]["wi"]["kernel"] = \
      pytorch_model["decoder.block."+str(i)+".layer.2.DenseReluDense.wi.weight"].T.numpy()

  flax_model["decoder"]["encoderdecoder_norm"]["scale"] = \
    pytorch_model["decoder.final_layer_norm.weight"].numpy()

  if (final_dense_layer):
    flax_model["decoder"]["logits_dense"]["kernel"] = \
      pytorch_model["lm_head.weight"].T.numpy()

  return flax_model
