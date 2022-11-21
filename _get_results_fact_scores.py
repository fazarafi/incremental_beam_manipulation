import datetime
import os
import sys
import time
import argparse

from utils import RESULTS_DIR, VALIDATION_NOT_TEST, DATASET_WEBNLG, \
    get_args_presumm, convert_id_to_text, SUMM_RESULTS_DIR, get_timestamp_file

from pytorch_transformers import BertTokenizer
import torch

from fact_scorer.fact_factcc.factcc_caller_model import FactccCaller
from fact_scorer.fact_summac.summac_caller import classify as summac_cls, evaluate_batch as summac_evaluate_batch
from rouge import Rouge
from fact_scorer.fact_coco.coco_caller import initialize_coco, evaluate_coco

import pickle

# TODO FT import with sys
HOME_REPO = "/home/lr/faza.thirafi/raid/repository-kenkyuu-models/"
HOME_DATA = "/home/lr/faza.thirafi/raid_elmo/cache/"
sys.path.insert(0, HOME_REPO + "feqa/")
from feqa import FEQA


def average(lst):
    return sum(lst) / len(lst)

def test_summary_scores_official(args, pred_file_name, scorers):

    pred_file = os.path.join(SUMM_RESULTS_DIR, pred_file_name)

    summaries_sys = []
    summaries_ref = []
    documents_ref = []

    with open(pred_file, "r", encoding="utf-8") as fin1:
        for line in fin1:
            summaries_sys.append(line)

    with open(pred_file + '.raw_src', "r", encoding="utf-8") as fin2:
        for line in fin2:
            summaries_ref.append(line)

    with open(pred_file + '.gold', "r", encoding="utf-8") as fin3:
        for line in fin3:
            documents_ref.append(line)

    
    final_results = []
    print("len(documents_ref), len(summaries_ref), len(summaries_sys):", len(documents_ref)," ", len(summaries_ref)," ", len(summaries_sys))
    data_length = min(len(documents_ref), len(summaries_ref), len(summaries_sys))

    # equalify all dataset arrays
    documents_ref = documents_ref[:data_length]
    summaries_ref = summaries_ref[:data_length]
    summaries_sys = summaries_sys[:data_length]

    if 'rouge' in scorers:
        print('TEST WITH ROUGE')
        final_scores = {}
        final_scores["scorer"] = 'rouge'
        rouge = Rouge()
        scores = rouge.get_scores(summaries_sys, summaries_ref, avg=True)
        final_scores["raw_scores"] = scores
        
        print('TEST WITH ROUGE: ', final_scores)
        final_results.append(final_scores)

    if 'factcc' in scorers:
        final_scores = {}
        print('TEST WITH FactCC')
        final_scores["scorer"] = 'factcc'
        factcc_scorer = FactccCaller()
        results = factcc_scorer.evaluate_batch(documents_ref, summaries_sys)
        final_scores["raw_scores"] = results

        print('TEST WITH FactCC: ', final_scores)
        final_results.append(final_scores)

    if 'coco' in scores:
        final_scores = {}
        print('TEST WITH CoCo')
        final_scores["scorer"] = 'coco'
        coco_params = initialize_coco()
        
        results, avg, f1 = evaluate_batch_coco(coco_params, documents_ref, summaries_sys)

        final_scores["raw_scores"] = results
        final_scores["average"] = avg
        final_scores["f1_score"] = f1_score
        
        print('TEST WITH CoCo: ', final_scores)
        final_results.append(final_scores)
    
    if 'summac' in scorers:
        print('TEST WITH SummaC')
        final_scores = {}
        final_scores["scorer"] = 'summac'
        f1, f1_score = summac_evaluate_batch(documents_ref, summaries_sys)
        final_scores["f1_score"] = f1_score
        # final_scores["raw_scores"] = f1
        
        print('TEST WITH SummaC: ', final_scores)
        final_results.append(final_scores)

    if 'feqa' in scorers:
        print('TEST WITH FEQA')
        final_scores = {}
        final_scores["scorer"] = 'feqa'
        model = FEQA(use_gpu=True)
        agg_score = model.compute_score(documents_ref, summaries_sys, aggregate=True)
        final_scores["raw_scores"] = agg_score
        
        print('TEST WITH FEQA: ', final_scores)
        final_results.append(final_scores)
    
    return final_results

def print_results(args, summ_scorers=None):
    if summ_scorers == None:
        summ_scorers = ['rouge']

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
        # print(filename, bs, test_summary_scores(args, filename, summ_scorers, 'test'))
        print(filename, bs, test_summary_scores_official(args, filename, summ_scorers))


if __name__ == "__main__":
    # SUMM_RESULTS_DIR = 'output_files/from_gpu_2/out-text-dir-v3'
    parser = argparse.ArgumentParser()

    # PreSumm parser
    parser = get_args_presumm(parser)

    parser.add_argument('-pred_file', default=None)
    parser.add_argument('-eval_scorers', default=None)
    args = parser.parse_args()


    # Setup GPU for PreSumm
    args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
    args.world_size = len(args.gpu_ranks)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    # init_logger(args.log_file) TODO add logger
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    device_id = 0 if device == "cuda" else -1
    
    # print_results(args)
    
    scorers = args.eval_scorers.split(',')

    pred_file_name = args.pred_file
    
    result = test_summary_scores_official(args, pred_file_name, scorers)    
    
    pred_file = os.path.join(SUMM_RESULTS_DIR, pred_file_name)
    
    print(result)
    print("END")

    with open(pred_file + '.test_result_cust_' + get_timestamp_file(), "w+", encoding='utf-8') as out_file_result:
        for res in result:
            out_file_result.write(str(res) + '\n')
