import sys
import os

# Import BART
HOME_REPO = "/home/lr/faza.thirafi/raid/repository-kenkyuu-models/"
sys.path.insert(0, HOME_REPO + "transformers/")
sys.path.insert(0, HOME_REPO + "transformers/src")

import datasets

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def batch_tokenize_preprocess(batch, tokenizer, max_source_length, max_target_length):
    source, target = batch["document"], batch["summary"]
    source_tokenized = tokenizer(
        source, padding="max_length", truncation=True, max_length=max_source_length
    )
    target_tokenized = tokenizer(
        target, padding="max_length", truncation=True, max_length=max_target_length
    )

    batch = {k: v for k, v in source_tokenized.items()}
    # Ignore padding in the loss
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in l]
        for l in target_tokenized["input_ids"]
    ]
    return batch

def preprocess_data(dataset):

    model_name = "sshleifer/distilbart-xsum-12-3" # TODO FT move as param
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # tokenization
    encoder_max_length = 256  # demo
    decoder_max_length = 64

    data = dataset.map(
        lambda batch: batch_tokenize_preprocess(
            batch, tokenizer, encoder_max_length, decoder_max_length
        ),
        batched=True,
        remove_columns=dataset.column_names,
    ) 

    return data

def load_bart_dataset(dataset_name='xsum', data_type='train'):
    data = []
    if dataset_name =='xsum':
        if data_type == 'train':
            data = datasets.load_dataset(dataset_name, name='english', split="train")
        elif data_type == 'test':
            data = datasets.load_dataset(dataset_name, name='english', split="test")
        elif data_type == 'valid':
            data = datasets.load_dataset(dataset_name, name='english', split="validation")

        dataset = data.map(remove_columns=["id"])

    return data


def load_bart_model(model_name="sshleifer/distilbart-xsum-12-3"):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model


def main():

    data = load_bart_dataset("xsum", "valid")
    data = preprocess_data(data)

    print(data[0])

if __name__ == "__main__":
    main()


