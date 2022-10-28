from datasets import load_metric

def calculate_rouge(pred_filename, label_filename, output_filename = "metrics.txt"):
    with open(pred_filename, 'r') as file:
        pred_str = file.readlines()
        pred_str = [line.rstrip() for line in pred_str]
    with open(label_filename, 'r') as file:
        label_str = file.readlines()
        label_str = [line.rstrip() for line in label_str]
   
    print("Examples")
    for i, label in enumerate(label_str[:5]):
            print("Target")
            print(label)
            print("Prediction")
            print(pred_str[i])
            print("End prediction\n\n")
    
    rouge = load_metric("rouge")
    rouge_types = ["rouge1", "rouge2", "rougeLsum"]
    rouge_outputs = rouge.compute(predictions=pred_str, references=label_str, rouge_types=rouge_types, use_agregator=True)

    rouge1 = rouge_outputs["rouge1"].mid
    rouge2 = rouge_outputs["rouge2"].mid
    rougeLsum = rouge_outputs["rougeLsum"].mid

    metrics = {
        "rouge1_precision": round(rouge1.precision, 4)*100,
        "rouge1_recall": round(rouge1.recall, 4)*100,
        "rouge1_fmeasure": round(rouge1.fmeasure, 4)*100,
        "rouge2_precision": round(rouge2.precision, 4)*100,
        "rouge2_recall": round(rouge2.recall, 4)*100,
        "rouge2_fmeasure": round(rouge2.fmeasure, 4)*100,
        "rougeLsum_precision": round(rougeLsum.precision, 4)*100,
        "rougeLsum_recall": round(rougeLsum.recall, 4)*100,
        "rougeLsum_fmeasure": round(rougeLsum.fmeasure, 4)*100,
    }
    print("Calculated metrics")
    print(metrics)
    f = open(output_filename, "w+")
    f.write(str(metrics))
    f.close()