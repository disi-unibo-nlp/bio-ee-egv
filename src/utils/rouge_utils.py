from rouge_score import rouge_scorer
from rouge_score import scoring
from absl import logging
import numpy as np


def rouge_top_beam(targets, predictions, beam_size, score_keys=None):
  """Computes rouge score, considering only the best score within each beam.
  In other words, for each target, it keeps the rouge only for the most similar
  sentence among those produced with beam search.
  Args:
    targets: list of strings (with each item repeated beam_size times)
    predictions: list of strings (flat sequences of multi-output predictions)
    beam_size: the number of predictions for each original input
    score_keys: list of strings with the keys to compute
  Returns:
    dict with score_key: rouge score across all targets and predictions
  """
  
  if score_keys is None:
    score_keys = ["rouge1", "rouge2", "rougeLsum"]
  scorer = rouge_scorer.RougeScorer(score_keys)
  aggregator = scoring.BootstrapAggregator()

  def _prepare_summary(summary):
    # Make sure the summary is not bytes-type
    # Add newlines between sentences so that rougeLsum is computed correctly.
    summary = summary.replace(" . ", " .\n")
    return summary

  seq_idx = 0
  beam_max_score = None
  max_avg_fmeasure = None

  for prediction, target in zip(predictions, targets):

    target = _prepare_summary(target)
    prediction = _prepare_summary(prediction)

    # Calculate a Score object (containing precision, recall, and fmeasure)
    # for each specified ROUGE metric
    score = scorer.score(target=target, prediction=prediction)

    print(target)
    print(prediction)
    print(score)

    # For each beam, keep the Score object related to the prediction with
    # the best overall fmeasure (i.e., average between the various ROUGE
    # metrics)
    avg_fmeasure = np.average([score[key].fmeasure for key in score_keys])
    if (beam_max_score is None or
        avg_fmeasure > max_avg_fmeasure):
      beam_max_score = score
      max_avg_fmeasure = avg_fmeasure

    seq_idx += 1
    if (seq_idx % beam_size == 0):
      print("SELECTED")
      print(beam_max_score)
      print("---------------------")
      aggregator.add_scores(beam_max_score)
      beam_max_score = max_avg_fmeasure = None
  
  result = aggregator.aggregate() # Works with percentiles

  for key in score_keys:
    logging.info(
        """%s => recall = %.2f, 95%% confidence [%.2f, %.2f]
                 precision = %.2f, 95%% confidence [%.2f, %.2f]
                 fmeasure = %.2f, 95%% confidence [%.2f, %.2f]""",
        key,
        result[key].mid.recall*100,
        result[key].low.recall*100,
        result[key].high.recall*100,
        result[key].mid.precision*100,
        result[key].low.precision*100,
        result[key].high.precision*100,
        result[key].mid.fmeasure*100,
        result[key].low.fmeasure*100,
        result[key].high.fmeasure*100,
    )
  
  return {key: result[key].mid.fmeasure*100 for key in score_keys}
