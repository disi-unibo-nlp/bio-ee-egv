#!/bin/sh
import jax
import jax.numpy as jnp
from utils.linear_attention.fast_attention_module import make_fast_softmax_attention, make_fast_generalized_attention 


def get_performer_attentions(m=64, qkv_dim=None, num_heads=None):
  """Get the fast attentions introduced with Performer paper.
  
  Args:
    m: Optional; the low-rank dimensionality. If None, is derived by qkv_dim and num_heads.
    qkv_dim: Optional; the dimensionality of the query, key, value arrays. It is required only if m is not specified.
    num_heads: Optional; the number of attention heads. It is required only if m is not specified.
  
  Returns:
    - Softmax attention (i.e., the main one discussed in the paper)
    - ReLU attention (i.e., the one related to better results in the paper)"""
  if (m is None and (qkv_dim is not None and num_heads is not None)):
    # https://github.com/google-research/google-research/issues/465
    d = qkv_dim // num_heads
    m = int(d * jnp.log(d))
  return make_fast_softmax_attention(m, lax_scan_unroll=16), \
      make_fast_generalized_attention(m, kernel_fn=jax.nn.relu, lax_scan_unroll=16)
