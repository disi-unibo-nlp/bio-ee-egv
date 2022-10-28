import sys
sys.path.append("...")
from t5.data import dataset_providers
import os, glob, time, zipfile, functools
from t5.data import dataset_providers
from t5.data import preprocessors
from configs import ee_t5 as t5_base
from utils.rouge_utils import rouge_top_beam
from utils.t5x_utils.test import test
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)
fine_tuning_cfg = t5_base.get_config()
fine_tuning_cfg.beam_size = 1

with zipfile.ZipFile("../../data/datasets/single_task_validation_sets/test_ee.zip", 'r') as zip_ref:
    zip_ref.extractall("../../data/datasets/single_task_validation_sets/")

datasetsNames = ['gro-2013', 'mlee', 'cg-2013', 'pc-2013', 'ge-2013', 'genia-mk', 'ge-2011', 'id-2011', 'epi-2011', 'st-09']

start = time.time()
for dataset in datasetsNames:
    data_dir = "../../../data/datasets/single_task_validation_sets/"
    train_file = "test_EE_" + dataset + ".tsv"
    test_file = "test_EE_" + dataset + ".tsv"
    TaskRegistry = dataset_providers.TaskRegistry
    TaskRegistry.remove("event_extraction_task")

    TaskRegistry.add(
    "event_extraction_task",
    dataset_providers.TextLineTask,
    split_to_filepattern = {
        "train": os.path.join(data_dir, train_file),
        "validation": os.path.join(data_dir, test_file)
    },
    skip_header_lines = 1,
    text_preprocessor = preprocessors.preprocess_tsv,
    metric_fns=[functools.partial(
        rouge_top_beam, beam_size=fine_tuning_cfg.beam_size)]
    )
    print("Evaluating EE on " + dataset + " dataset.")
    test(task_name="event_extraction_task", model_dir="../../../data/model_data/ee/best_checkpoint/", config=fine_tuning_cfg, output_prediction_postfix=dataset, ee=True)
