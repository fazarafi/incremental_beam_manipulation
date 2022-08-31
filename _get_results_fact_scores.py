import datetime
import os
import sys
import time
import argparse

from e2e_metrics.metrics.pymteval import BLEUScore
from e2e_metrics.measure_scores import load_data
from utils import RESULTS_DIR, VALIDATION_NOT_TEST, DATASET_WEBNLG, \
    get_args_presumm, convert_id_to_text, SUMM_RESULTS_DIR


from PreSumm.src.models import data_loader, model_builder
from PreSumm.src.models.data_loader import load_dataset
# from PreSumm.src.models.model_builder import AbsSummarizer
# from PreSumm.src.models.predictor import build_predictor
from pytorch_transformers import BertTokenizer
import torch

from fact_scorer.fact_factcc.factcc_caller_model import FactccCaller
from fact_scorer.fact_summac.summac_caller import classify as summac_cls, evaluate_batch as summac_evaluate_batch
from rouge import Rouge

import pickle

# TODO FT import with sys
HOME_REPO = "/home/lr/faza.thirafi/raid/repository-kenkyuu-models/"
HOME_DATA = "/home/lr/faza.thirafi/raid_elmo/cache/"
sys.path.insert(0, HOME_REPO + "feqa/")
from feqa import FEQA


def average(lst):
    return sum(lst) / len(lst)

def load_data(pred_file, ref_file):
    return []

def test_summary_scores_calon_beneran(args, pred_file_name, scorer, pred_mode=None):

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir=args.temp_dir)

    pred_file = os.path.join(SUMM_RESULTS_DIR, pred_file_name)

    summaries_sys = []

    for paths in pred_file:
        summaries_sys.append(paths[1])

    mode = args.use_data
    
    summ_data = None
    if mode == 'test':
        summ_data = data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False), args.batch_size, args.device, shuffle=False, is_test=True)
    elif mode == 'valid':
        summ_data = data_loader.Dataloader(args, load_dataset(args, 'valid', shuffle=False), args.batch_size, args.device, shuffle=False, is_test=True)
    else:
        summ_data = data_loader.Dataloader(args, load_dataset(args, 'train', shuffle=False), args.batch_size, args.device, shuffle=False, is_test=True)

    summaries_ref = list()
    documents_ref = list()
    usage_data_for_train = 10000 # TODO FT revise properly
    counter = 0
    for batch in summ_data:
        if mode == 'train' and counter >= usage_data_for_train:
            break
        for src in batch.src:
            documents_ref.append(convert_id_to_text(tokenizer, src))
        for tgt in batch.tgt:
            summaries_ref.append(convert_id_to_text(tokenizer, src))
        counter += len(batch.src)
    
    final_scores = {}
    final_scores["scorer"] = scorer

    data_length = min(len(documents_ref, summaries_ref, summaries_sys))

    # equalify all dataset arrays
    documents_ref = documents_ref[:data_length]
    summaries_ref = summaries_ref[:data_length]
    summaries_sys = summaries_sys[:data_length]

    if (scorer=='factcc'):
        print('TEST WITH FactCC')
        factcc_scorer = FactccCaller()
        results = factcc_scorer.evaluate_batch(documents_ref, summaries_sys)
        final_scores["raw_scores"] = results

    elif (scorer=='summac'):
        print('TEST WITH SummaC')
        f1, scores = summac_evaluate_batch(documents_ref, summaries_sys)
        final_scores["raw_scores"] = scores
        final_scores["f1_score"] = f1

    elif (scorer=='feqa'):
        print('TEST WITH FEQA')
        model = FEQA(use_gpu=True)
        # scores = model.compute_score(documents_ref, summaries_ref, aggregate=False)
        # average = average(scores)
        # final_scores["average"] = average            
        agg_score = model.compute_score(documents_ref, summaries_sys, aggregate=True)
        final_scores["raw_scores"] = agg_score
        

    elif (scorer=='rouge'):
        print('TEST WITH ROUGE')
        rouge = Rouge()
        scores = rouge.get_scores(summaries_sys, summaries_ref, avg=True)
        final_scores["raw_scores"] = scores

    else:
        print('Test module is not recognized.')

    return final_scores

# TODO FT remove below if above works
def test_summary_scores(args, pred_file_name, scorer, mode, pred_mode=None):

    use_data_mode = args.use_data
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir=args.temp_dir)

    pred_file = os.path.join(SUMM_RESULTS_DIR, pred_file_name)

    final_scores = {}
    final_scores["scorer"] = scorer
    
    dummy_documents = [
                "The world's oldest person has died a \
                few weeks after celebrating her 117th birthday.  \
                Born on March 5, 1898, the greatgrandmother had lived through two world \
                wars, the invention of the television and the \
                first successful powered aeroplane.", 
                "The world's oldest person has died a \
                few weeks after celebrating her 117th birthday.  \
                Born on March 5, 1898, the greatgrandmother had lived through two world \
                wars, the invention of the television and the \
                first successful powered aeroplane."]
    dummy_ref_summaries = [
                "The world's oldest person died in 1898",
                "The world's oldest person died after her 117th birthday"]

    dummy_sys_summaries = [
                "Someone oldest person died",
                "The world's born after her 117th birthday"]
    documents_ref = dummy_documents
    summaries_ref = dummy_ref_summaries
    summaries_sys = dummy_sys_summaries

    if (scorer=='factcc'):
        print('TEST WITH FactCC')
        factcc_scorer = FactccCaller()
        results = factcc_scorer.evaluate_batch(documents_ref, summaries_sys)
        final_scores["raw_scores"] = results

    elif (scorer=='summac'):
        print('TEST WITH SummaC')
        f1, scores = summac_evaluate_batch(documents_ref, summaries_sys)
        final_scores["raw_scores"] = scores
        final_scores["f1_score"] = f1

    elif (scorer=='feqa'):
        print('TEST WITH FEQA')
        model = FEQA(use_gpu=True)
        agg_score = model.compute_score(documents_ref, summaries_sys, aggregate=True)
        final_scores["raw_scores"] = agg_score

    elif (scorer=='rouge'):
        print('TEST WITH ROUGE')
        rouge = Rouge()
        scores = rouge.get_scores(summaries_sys, summaries_ref, avg=True)
        final_scores["raw_scores"] = scores

    else:
        print('Test module is not recognized.')

    return final_scores




def test_res_fact_scores(pred_file_name, scorer, mode):
    pred_file = os.path.join(RESULTS_DIR, pred_file_name)
    if DATASET_WEBNLG:
        if VALIDATION_NOT_TEST:
            true_file = "WebNLG_Reader/data/webnlg/valid.txt"
        else:
            true_file = "WebNLG_Reader/data/webnlg/test.txt"
    else:
        if VALIDATION_NOT_TEST:
            true_file = "tgen/e2e-challenge/input/devel-conc.txt"
        else:
            true_file = "tgen/e2e-challenge/input/test-conc.txt"

    _, data_ref, data_sys = load_data(true_file, pred_file)

    bleu = BLEUScore()
    for sents_ref, sent_sys in zip(data_ref, data_sys):
        bleu.append(sent_sys, sents_ref)

    total_bleu_score = bleu.score()
    bleu_scores = []
    # for sents_ref, sent_sys in zip(data_ref, data_sys):
    #     bleu.reset()
    #     bleu.append(sent_sys, sents_ref)
    #     bleu_scores.append(bleu.score())
    # # return the computed scores
    # if total_bleu_score > 0.6:
    #     print(bleu_scores)
    #     pass

    return total_bleu_score

def test_res_official(pred_file_name):
    pred_file = os.path.join(RESULTS_DIR, pred_file_name)
    if DATASET_WEBNLG:
        if VALIDATION_NOT_TEST:
            true_file = "WebNLG_Reader/data/webnlg/valid.txt"
        else:
            true_file = "WebNLG_Reader/data/webnlg/test.txt"
    else:
        if VALIDATION_NOT_TEST:
            true_file = "tgen/e2e-challenge/input/devel-conc.txt"
        else:
            true_file = "tgen/e2e-challenge/input/test-conc.txt"

    _, data_ref, data_sys = load_data(true_file, pred_file)

    bleu = BLEUScore()
    for sents_ref, sent_sys in zip(data_ref, data_sys):
        bleu.append(sent_sys, sents_ref)

    total_bleu_score = bleu.score()
    bleu_scores = []
    # for sents_ref, sent_sys in zip(data_ref, data_sys):
    #     bleu.reset()
    #     bleu.append(sent_sys, sents_ref)
    #     bleu_scores.append(bleu.score())
    # # return the computed scores
    # if total_bleu_score > 0.6:
    #     print(bleu_scores)
    #     pass

    return total_bleu_score


def print_results(args, summ_scorer=None):
    if summ_scorer == None:
        summ_scorer = 'rouge'

    day_seconds = 24 * 60 * 60
    # print(sys.argv)
    filename_bs = []
    for filename in os.listdir(SUMM_RESULTS_DIR):
        if '*' in filename:
            continue
        splits = filename.split('-')
        try:
            beam_size = int(splits[-1].split('.')[0])
        except:
            beam_size = 0
        filter_name = '-'.join(splits[:-1])
        if (len(sys.argv) > 1 and sys.argv[1] == 'all') or os.path.getmtime(
                os.path.join(SUMM_RESULTS_DIR, filename)) > time.time() - day_seconds / 2:
            filename_bs.append((filter_name, filename, beam_size))

    for _, filename, bs in sorted(filename_bs, key=lambda x: (x[0], int(x[2]))):
        print(filename, bs, test_summary_scores(args, filename, summ_scorer, 'test'))


if __name__ == "__main__":
    # SUMM_RESULTS_DIR = 'output_files/from_gpu_2/out-text-dir-v3'
    parser = argparse.ArgumentParser()


    # PreSumm parser
    parser = get_args_presumm(parser)

    parser.add_argument('-c', default=None)
    args = parser.parse_args()


    # Setup GPU for PreSumm
    args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
    args.world_size = len(args.gpu_ranks)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    # init_logger(args.log_file) TODO add logger
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    device_id = 0 if device == "cuda" else -1
    
    print_results(args)
    
    
    result = test_summary_scores(args, '', 'factcc', 'test')
    print(result)
    
    result = test_summary_scores(args, '', 'summac', 'test')
    print(result)
    
    result = test_summary_scores(args, '', 'feqa', 'test')
    print(result)
    
    result = test_summary_scores(args, '', 'rouge', 'test')
    print(result)
    

    

    # scorer = FEQA(use_gpu=True)

    # documents = [
    #             "The world's oldest person has died a \
    #             few weeks after celebrating her 117th birthday.  \
    #             Born on March 5, 1898, the greatgrandmother had lived through two world \
    #             wars, the invention of the television and the \
    #             first successful powered aeroplane.", 
    #             "The world's oldest person has died a \
    #             few weeks after celebrating her 117th birthday.  \
    #             Born on March 5, 1898, the greatgrandmother had lived through two world \
    #             wars, the invention of the television and the \
    #             first successful powered aeroplane."]
    # summaries = [
    #             "The world's oldest person died in 1898",
    #             "The world's oldest person died after her 117th birthday"]
    # scores1 = scorer.compute_score(documents, summaries, aggregate=False)

    
    # scores2 = scorer.compute_score(documents, summaries, aggregate=True)

    # print("[DEBUG FT] non SCORE: " + str(scores1))
    # print("[DEBUG FT] Aggregate SCORE: " + str(scores2))
