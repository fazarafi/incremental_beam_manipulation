import sys
import os

# Import BART modules
HOME_REPO = "/home/lr/faza.thirafi/raid/repository-kenkyuu-models/"
sys.path.insert(0, HOME_REPO + "transformers/")
sys.path.insert(0, HOME_REPO + "transformers/src")

import datasets

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

BART_ENCODER_MAX_LENGTH = 256
BART_DECODER_MAX_LENGTH = 64

BART_START_TOKEN = 0
BART_PAD_TOKEN = 1 #TODO bener?
BART_END_TOKEN = 2

# def list2samples(example):
#     documents = []
#     summaries = []
#     for sample in zip(example["articles"], example["highlight"]):
#         if len(sample[0]) > 0:
#             documents += sample[0]
#             summaries += sample[1]
#     return {"document": documents, "summary": summaries}

def cnndm_flatten(example):
    return {
        "document": example["article"],
        "summary": example["highlights"],
    }



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

    return dataset

def load_bart_dataset(dataset_name='xsum', data_type='train'):
    dataset = []
    if dataset_name == 'xsum':
        data = datasets.load_dataset(dataset_name, name='english', 
            split="validation" if data_type=="valid" else data_type)
        dataset = data.map(remove_columns=["id"])
    
    elif dataset_name == '"cnn_dailymail"':
        data = datasets.load_dataset(dataset_name, name='3.0.0', 
            split="validation" if data_type=="valid" else data_type)
        dataset = data.map(cnndm_flatten , remove_columns=["highlights","article", "id"])

    return preprocess_data(dataset)

def load_bart_model(model_name="sshleifer/distilbart-xsum-12-3"):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model

def convert_ids_to_text(tokenizer, ids):
    text = tokenizer.decode(ids, skip_special_tokens=True)

    return text

def beam_search_expand_single_bart(summ_model, summ_paths, beam_size, step, summ_data, input_params, **model_kwargs):

    updated_summ_paths, return_params, input_ids, model_kwargs, this_peer_finished = summ_model.beam_search_expand_single(
        input_params,
        step,
        summ_paths,
        input_params["input_ids"],
        input_params["beam_scorer"],
        logits_processor=input_params["logits_processor"],
        stopping_criteria=input_params["stopping_criteria"],
        pad_token_id=input_params["pad_token_id"],
        eos_token_id=input_params["eos_token_id"],
        output_scores=input_params["output_scores"],
        return_dict_in_generate=input_params["return_dict_in_generate"],
        synced_gpus=input_params["synced_gpus"],
        **model_kwargs,
    )

    if (this_peer_finished):
        print("SELESAII")

    return updated_summ_paths, return_params, input_ids, model_kwargs, this_peer_finished

def finalize_beam_search_expand_single_bart(summ_model, params):
    result = summ_model.finalize_beam_search_expand_single(
        params,
        params["beam_scorer"],
        params["input_ids"],
        params["beam_scores"],
        params["next_tokens"],
        params["next_indices"],
        params["pad_token_id"],
        params["eos_token_id"],
        params["stopping_criteria"],
        params["beam_indices"],
        params["return_dict_in_generate"],
        params["output_scores"],
    )

    return result

def main():

    data = load_bart_dataset("xsum", "valid")

    print(data[:3])

if __name__ == "__main__":
    main()


