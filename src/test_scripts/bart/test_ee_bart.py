from datasets import Dataset
import pandas as pd
from transformers import BartForConditionalGeneration, BartTokenizer
import gdown, torch, glob, zipfile

model_name = 'facebook/bart-base'
tokenizer = BartTokenizer.from_pretrained(model_name)
batch_size = 16

def load_model():
  checkpoints = glob.glob('../../../data/model_data/EE/bart/model_checkpoints/checkpoint-*')
  if len(checkpoints) > 0:
    model = BartForConditionalGeneration.from_pretrained(checkpoints[0])
    print("Loaded checkpoint from " + checkpoints[0])
  else:
    model = BartForConditionalGeneration.from_pretrained(model_name)
    print("Loaded HuggingFace model: " + model_name)
  return model

model = load_model().to("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_predictions(batch):
    """Generates the predictions on the test set with the fine-tuned model.
    Args:
        batch: The batch to process.
    Returns:
        The batch processed to obtain the predictions.
    """
    inputs = tokenizer(batch["event"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs.input_ids.cuda("cuda:0" if torch.cuda.is_available() else "cpu")
    attention_mask = inputs.attention_mask.cuda("cuda:0" if torch.cuda.is_available() else "cpu")

    outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=62, num_beams=1)

    batch["predicted_events"] = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return batch

with zipfile.ZipFile("../../../data/datasets/single_task_validation_sets/test_ee.zip", 'r') as zip_ref:
    zip_ref.extractall("../../../data/datasets/single_task_validation_sets/")

datasetsNames = ['gro-2013', 'mlee', 'cg-2013', 'pc-2013', 'ge-2013', 'genia-mk', 'ge-2011', 'id-2011', 'epi-2011', 'st-09']

for dataset in datasetsNames:
  test_data = Dataset.from_pandas(pd.read_csv("../../../data/datasets/single_task_validation_sets/test_EE_" + dataset + ".tsv", sep='\t'))
  results = test_data.map(generate_predictions, batched=True, batch_size=batch_size, remove_columns=["event"])
  pred_str = results["predicted_events"]
  label_str = results["target_events"]

  with open('preds_EE_' + dataset + '.txt', 'w+') as f:
    for pred in results["predicted_events"]:
        f.write("%s\n" % pred)

test_data = Dataset.from_pandas(pd.read_csv("../../../data/datasets/test_ee" + dataset + ".tsv", sep='\t'))
results = test_data.map(generate_predictions, batched=True, batch_size=batch_size, remove_columns=["event"])
pred_str = results["predicted_events"]
label_str = results["target_events"]

with open('preds_EE_BIOT2E' + '.txt', 'w+') as f:
  for pred in results["predicted_events"]:
      f.write("%s\n" % pred)
