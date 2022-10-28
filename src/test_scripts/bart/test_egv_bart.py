from datasets import Dataset
import pandas as pd
from transformers import BartForConditionalGeneration, BartTokenizer
import gdown, torch, glob

model_name = 'facebook/bart-base'
tokenizer = BartTokenizer.from_pretrained(model_name)
batch_size = 16

def load_model():
  checkpoints = glob.glob('../../../data/model_data/EGV/bart/model_checkpoints/checkpoint-*')
  if len(checkpoints) > 0:
    model = BartForConditionalGeneration.from_pretrained(checkpoints[0])
    print("Loaded checkpoint from " + checkpoints[0])
  else:
    model = BartForConditionalGeneration.from_pretrained(model_name)
    print("Loaded HuggingFace model: " + model_name)
  return model

model = load_model().to("cuda:0" if torch.cuda.is_available() else "cpu")

test_data = Dataset.from_pandas(pd.read_csv('test.tsv', sep='\t'))

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

    batch["predicted_event_mention"] = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return batch

results = test_data.map(generate_predictions, batched=True, batch_size=batch_size, remove_columns=["event"])

pred_str = results["predicted_event_mention"]
label_str = results["target_event_mention"]

with open('preds_EGV_BART.txt', 'w+') as f:
    for pred in results["predicted_event_mention"]:
        f.write("%s\n" % pred)

with open('labels_EGV_BART.txt', 'w+') as f:
    for label in results["target_event_mention"]:
        f.write("%s\n" % label)
