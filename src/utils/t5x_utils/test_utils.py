import gin
import tensorflow as tf


def get_inputs_from_file(input_filename, ignore_comments=False):
  """Read data from file and strip new lines."""
  inputs = [line.rstrip() for line in tf.io.gfile.GFile(input_filename)]

  # Strip the last empty line.
  if not inputs[-1]:
    inputs.pop()

  if ignore_comments:
    inputs = [l for l in inputs if not l.startswith("#")]

  return inputs


def inputs_vocabulary(vocabulary):
  """Get the inputs vocabulary.
  Args:
    vocabulary: Vocabulary or (inputs_vocabulary, targets_vocabulary) tuple.
  Returns:
    a Vocabulary
  """
  if isinstance(vocabulary, tuple):
    vocabulary = vocabulary[0]
  return vocabulary


def encode_inputs(inputs,
                  vocabulary,
                  model_type,
                  batch_size,
                  sequence_length,
                  eos_id=1,
                  unscored_prefix=None):
  """Encode string inputs for inference/scoring.
  Args:
    inputs: list of strings
    vocabulary: a mtf.transformer.vocabulary.Vocabulary
    model_type: a string
    batch_size: an integer
    sequence_length: an integer (maximum decode length)
    eos_id: EOS id
    unscored_prefix: an optional list of strings
  Returns:
    all_input_ids: encoded inputs
  """
  n = len(inputs)
  all_input_ids = []
  for line_num, line in enumerate(inputs):
    ids = inputs_vocabulary(vocabulary).encode(line.strip())
    if unscored_prefix:
      prefix_str = unscored_prefix[line_num].strip()
      ids = [-i for i in inputs_vocabulary(vocabulary).encode(prefix_str)] + ids
    if model_type != "lm":
      # for text2self problems, the inputs represent a partial sequence
      # to be continued, and should not be terminated by EOS.
      # for sequence-to-sequence problems, the input needs to be EOS-terminated
      ids += [eos_id]
    if len(ids) > sequence_length:
      ids = ids[:sequence_length]
    else:
      ids.extend([0] * (sequence_length - len(ids)))
    all_input_ids.append(ids)
  # pad to make an integral number of batches
  all_input_ids.extend([all_input_ids[0]] * (-n % batch_size))
  all_input_ids = np.array(all_input_ids, dtype=np.int32)

  return all_input_ids


def decode_from_file(estimator,
                     vocabulary,
                     model_type,
                     batch_size,
                     sequence_length,
                     checkpoint_path=None,
                     input_filename=gin.REQUIRED,
                     output_filename=gin.REQUIRED,
                     eos_id=1,
                     repeats=1):
  """Decode from a text file and write to output_filename.
  Args:
    estimator: a TPUEstimator
    vocabulary: a mtf.transformer.vocabulary.Vocabulary
    model_type: a string
    batch_size: an integer
    sequence_length: an integer or a dict from feature-key to integer
      the (packed) sequence length, e.g. {"inputs": 512, "targets": 128}
    checkpoint_path: an optional string
    input_filename: a string
    output_filename: a string
    eos_id: EOS id
    repeats: an integer, the number of times to repeat each input.
  """
  inputs = get_inputs_from_file(input_filename)

  all_input_ids = encode_inputs(inputs, vocabulary, model_type, batch_size,
                                sequence_length["inputs"], eos_id=eos_id)
  def input_fn(params):
    del params
    dataset = tf.data.Dataset.from_tensor_slices({"inputs": all_input_ids})
    dataset = dataset.flat_map(
        lambda x: tf.data.Dataset.from_tensors(x).repeat(repeats))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

  checkpoint_step = get_step_from_checkpoint_path(checkpoint_path)
  decodes = decode(
      estimator, input_fn, vocabulary, checkpoint_path=checkpoint_path)
  # Remove any padded examples
  dataset_size = len(inputs) * repeats
  decodes = decodes[:dataset_size]
  output_filename = "{}-{}".format(output_filename, checkpoint_step)
  write_lines_to_file(decodes, output_filename)
