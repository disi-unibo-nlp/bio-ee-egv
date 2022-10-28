import gdown
import zipfile

def download_file(id, filename):
    url = 'https://drive.google.com/uc?id=' + id
    gdown.download(url, filename, quiet=False)

def download_folder(id, output):
    gdown.download_folder('https://drive.google.com/drive/folders/' + id, output=output, quiet=False, remaining_ok=True)

def download_summarization_dataset():
    download_file('1JVNCzQyrNV0bOh4lpZpIS-qZiSrZcjuD', './datasets/summarization/train.tsv')
    download_file('1RToJ0ZSQXCO0Gh28eH2Ha89Ukd1Sn7-9', './datasets/summarization/validation.tsv')
    download_file('1SicrCWpUNhiyISYHxho_dpCBCBKQASZT', './datasets/summarization/test.tsv')

def download_ee_predictions():
    download_folder('1A5yTS_sA1aKaAFkCcrRLTSRSMrQHc0Dr', './model_data/ee/t5x/')

def download_model_checkpoints():
    download_folder('15vg-2tqHXlezz6N3cKWMeLVECUuZX2kr', './model_data/')
    download_folder('1eJKG5BOo5BQ2pEeSS3EmyhssqJJb7tr3', './model_data/')
    download_folder('1ntLlX81Hl1G0qmTwIwdrt0Hdmnx7eH7b', './model_data/')
    download_folder('1UAYBRO-pU5rjjt1wAfWfQZa00Q4wczm5', './model_data/')

download_summarization_dataset()
download_model_checkpoints()
download_ee_predictions()
