import argparse
import os
import random
import sys
import yaml
from pathlib import Path
from embedding_extractor import TokEmbeddingSeq2SeqExtractor, DAEmbeddingSeq2SeqExtractor
from _get_results_fact_scores import print_results, test_summary_scores_official
from _beam_search_fact import run_beam_search_with_rescorer
from _scorer_functions_fact import get_score_function, get_score_function_fact
from utils import get_training_variables, apply_absts, get_abstss_train, get_test_das, \
    get_true_sents, get_abstss_test, get_training_das_texts, SUMM_RESULTS_DIR, CONFIGS_MODEL_DIR, CONFIGS_DIR, postprocess, \
    get_multi_reference_training_variables, tgen_postprocess, get_args_presumm, get_timestamp_file, SUMM_START_TOK, SUMM_END_TOK, SUMM_PAD_TOK

import sys


# Import FT Project
from PreSumm.src.models import data_loader, model_builder
from PreSumm.src.models.data_loader import load_dataset
from PreSumm.src.models.model_builder import AbsSummarizer
from PreSumm.src.models.predictor import build_predictor
from pytorch_transformers import BertTokenizer
import torch
from time import time
import pickle
sys.path.insert(0, './PreSumm/src') # hacky

# Import BART
HOME_REPO = "/home/lr/faza.thirafi/raid/repository-kenkyuu-models/"
sys.path.insert(0, HOME_REPO + "transformers/")
from _bart_utils import load_bart_dataset, load_bart_model, get_bart_tokenizer

import logging
logger = logging.getLogger(__name__)

MAX_LEN = 150


def load_presumm(args, device):
    summ_data = [] 
    summary_embedder = []
    document_embedder = []
    summ_model = []
    len_summ_data = 0

    summ_data = data_loader.Dataloader(args, load_dataset(args, args.use_data, shuffle=False),
                                            args.batch_size, device,
                                            shuffle=False, is_test=False)

    print("PreSumm: Counting dataset length...")

    for batch in summ_data:
        len_summ_data += len(batch.src)
        document_embedder.append(batch.src)
        summary_embedder.append(batch.tgt)
    
    print('Loading checkpoint from %s' % args.test_from)
    checkpoint = torch.load(args.test_from, map_location=lambda storage, loc: storage)
    summarization_models = AbsSummarizer(args, device, checkpoint)
    summarization_models.eval()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir=args.temp_dir)
    symbols = {'BOS': tokenizer.vocab['[unused0]'], 'EOS': tokenizer.vocab['[unused1]'],
            'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused2]']}
    summ_model = build_predictor(args, tokenizer, symbols, summarization_models, logger)
    

    return summ_data, summary_embedder, document_embedder, summ_model, len_summ_data

def load_bart(args, cfg):
    summ_data = [] 
    summary_embedder = []
    document_embedder = []
    summ_model = []

    summ_data = load_bart_dataset(args)
    print("BART: Counting dataset length...")

    max_data = args.use_size if args.use_size > 0 else len(summ_data)
    print("MAX DATA: ", max_data)
    summ_data = summ_data[:max_data]

    len_summ_data = len(summ_data)
    
    document_embedder = summ_data["document"]
    summary_embedder = summ_data["summary"]
    
    tokenizer = get_bart_tokenizer(args)
    # tokenize all
    summary_embedder = tokenizer(
        summary_embedder,
        truncation=True,
        return_tensors="np",
    ).input_ids
    
    document_embedder = tokenizer(
        document_embedder,
        truncation=True,
        return_tensors="np",
    ).input_ids

    max_len_summ = max([len(x) for x in summary_embedder])
    max_len_docs = max([len(x) for x in document_embedder])
    
    lens = {}
    lens['max_len_docs'] = max_len_docs
    lens['max_len_summ'] = max_len_summ

    # document_embedder = summ_data["document"]
    # summary_embedder = summ_data["summary"]

    summ_model = load_bart_model(args)

    return summ_data, summary_embedder, document_embedder, summ_model, lens

def do_beam_search_fact(args, beam_size, cfg, models, das_test, da_embedder, text_embedder, true_vals, absts, summ_model, summ_data, len_summ_data, document_embedder, summary_embedder, lens=None):
    print("Beam size = {} ".format(beam_size))
    beam_save_path = cfg.get('beam_save_path', '')
    if beam_save_path:
        scorer_label = cfg['scorer']
        if cfg['scorer'] in ['fact_rouge', 'fact_mixed', 'fact_bart', 'coco_bart']:
            scorer_label = cfg['scorer'] + '-' + str(int(args.w1)) + '-' + str(int(args.w2))
        if cfg['scorer'] in ['fact_coco_bart', 'fact_coco_bart_rouge']:
            suffix = ''
            weights = args.multi_w.split(',')
            for wei in weights:
                suffix += ',' + str(int(wei))
            scorer_label = cfg['scorer'] + suffix

        beam_save_path = beam_save_path.format(args.use_dataset, args.pretrained_model, scorer_label, beam_size)

    parent = os.path.abspath(os.path.join(beam_save_path, os.pardir))
    if not os.path.exists(parent):
        os.makedirs(parent)

    # This is a horrible hack
    alpha = 0.65 if 'alpha' not in cfg else cfg['alpha'][beam_size]
    scorer_func = get_score_function_fact(args, cfg['scorer'], cfg, models, true_vals, beam_size, alpha, summary_embedder, document_embedder, lens=lens)
    max_pred_len = MAX_LEN

    non_greedy_score_func = None
    if "non_greedy_scorer" in cfg:
        non_greedy_score_func = get_score_function_fact(args, cfg['non_greedy_scorer'], cfg, models, true_vals, beam_size, alpha, lens=lens)
    if "greedy_complete_at" in cfg:
        greedy_complete = cfg["greedy_complete_at"]
    else:
        greedy_complete_rate = cfg.get("greedy_complete_rate", max_pred_len + 1)
        greedy_complete = [list(range(greedy_complete_rate, max_pred_len, greedy_complete_rate))]

    for gred_comp in greedy_complete:
        if gred_comp == ['random']:
            gred_comp = sorted(random.choices(list(range(3, 15)), k=random.randint(1, 4)))
        preds, srcs, tgts = run_beam_search_with_rescorer(args, scorer_func, models, das_test, beam_size,
                                              only_rerank_final=cfg['only_rerank_final'],
                                              save_final_beam_path=beam_save_path,
                                              greedy_complete=gred_comp,
                                              max_pred_len=max_pred_len,
                                              save_progress_path=cfg.get('save_progress_file', None),
                                              also_rerank_final=cfg.get('also_rerank_final', False),
                                              cfg=cfg,
                                              non_greedy_rescorer=non_greedy_score_func,
                                              length_norm_alpha=alpha if cfg.get('non_greedy_scorer', None) == 'length_normalised' else None,
                                              summ_scorer=scorer_func,
                                              summ_beam_search_model=summ_model,
                                              summ_data=summ_data,
                                              device=args.device,
                                              len_summ_data=len_summ_data)
        print("[DEBUG FT] preds before: ", len(preds))
        preds = [[x for x in pred if x not in [SUMM_START_TOK, SUMM_END_TOK, SUMM_PAD_TOK]] for pred in preds] # TODO FT evaluate this whether it's okay to use or not
        print("[DEBUG FT] preds AFTER: ", len(preds))
        if "res_save_format" in cfg:
            suffix_str = args.use_dataset + '_' + args.pretrained_model + '_' + str(beam_size)

            if cfg['scorer'] in ['fact_coco_bart', 'fact_coco_bart_rouge', 'fact_coco_bart_rouge']:
                suffix_weights = ''
                weights = args.multi_w.split(',')
                for wei in weights:
                    suffix_weights += ',' + str(int(wei))
                suffix_str += suffix_weights
            elif cfg['scorer'] in ['fact_rouge', 'fact_mixed', 'fact_bart', 'coco_bart']:
                suffix_str += '-' + str(int(args.w1)) + '-' + str(int(args.w2))

            save_filename = cfg["res_save_format"].format(suffix_str)
        # TODO FT store list of scorer as const
        elif 'trainable_reranker_config' in cfg and cfg['scorer'] in ['baseline','factcc', 'fact_mixed', 'summac', 'fact_rouge', 'rouge', 'bart_penalty', 'fact_bart', 'coco_bart', 'fact_coco_bart', 'fact_coco_bart_rouge']:
            fact_cfg = yaml.safe_load(open(cfg["trainable_reranker_config"], 'r+'))
            save_filename = "-{}-{}-{}-{}-{}-{}.txt".format(cfg["summary_dataset"], cfg['scorer'], fact_cfg["output_type"],
                                                        fact_cfg["logprob_preprocess_type"],
                                                        fact_cfg['beam_size'], beam_size)
            save_filename = cfg.get("save_prefix", "") + save_filename
        
        elif cfg['scorer'] in ['surrogate', 'greedy_decode_surrogate', 'surrogate_rev', 'surrogate_fact']:
            # Example surrogate-regression_reranker_relative-categorical_order_10_10.txt
            surrogate_cfg = yaml.safe_load(open(cfg["trainable_reranker_config"], 'r+'))
            save_filename = "all_test-{}-{}-{}-{}-{}-{}.txt".format(cfg["summary_dataset"], cfg['scorer'], surrogate_cfg["output_type"],
                                                        surrogate_cfg["logprob_preprocess_type"],
                                                        surrogate_cfg['beam_size'], beam_size)
            save_filename = cfg.get("save_prefix", "") + save_filename
        
        else:
            raise ValueError('Not saving files any where')
        save_filename_update = save_filename + "." + str(len(preds))
        save_path = os.path.join(SUMM_RESULTS_DIR, save_filename_update)

        cfg["re-lexicalise"] = False # TODO FT check if re-lexicalise not needed

        # if cfg.get("re-lexicalise", True):
        #     print("Applying abstract")
        #     post_abstr = apply_absts(absts, preds)
        # else:
        #     print("Abstract not applied")
        #     post_abstr = preds
        
        if not(args.use_data=='train' or len(preds) == 0):
            print("Saving to {}".format(save_path))
            parent = os.path.abspath(os.path.join(save_path, os.pardir))
            if not os.path.exists(parent):
                os.makedirs(parent)
            with open(save_path, "w+", encoding='utf-8') as out_file:
                for pa in preds:
                    out_file.write("".join(pa) + '\n')
                    
            with open(save_path + '.raw_src', "w+", encoding='utf-8') as out_file_src:
                for pa in srcs:
                    out_file_src.write(str(''.join(pa.splitlines())) + '\n')
            
            with open(save_path + '.gold', "w+", encoding='utf-8') as out_file_tgt:
                for pa in tgts:
                    out_file_tgt.write(str(pa.replace('\n', ' ')) + '\n')
                    
            
            
            scorers = ['factcc', 'rouge', 'coco', 'summac', 'rouge-d'] # TODO FT for first step, complete it later
            complete_scorers = ['factcc', 'rouge', 'summac', 'feqa', 'coco']
            test_result, _ = test_summary_scores_official(args, save_filename_update, scorers)
            print("Summary Score: ", test_result)

            with open(save_path + '.test_result', "w+", encoding='utf-8') as out_file_result:
                for res in test_result:
                    out_file_result.write(str(res) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = get_args_presumm(parser)
    args = parser.parse_args()

    # Setup GPU for PreSumm
    args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
    args.world_size = len(args.gpu_ranks)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    device_id = 0 if device == "cuda" else -1

    args.device = device
    cfg_path = args.c
    if cfg_path is None:
        filenames = os.listdir(CONFIGS_DIR)
        filepaths = [os.path.join(CONFIGS_DIR, filename) for filename in filenames]
        mod_times = [(os.path.getmtime(x), i) for i, x in enumerate(filepaths) if not os.path.isdir(x)]
        cfg_path = filepaths[max(mod_times)[1]]

    print("Using config from: {}".format(cfg_path))
    cfg = yaml.safe_load(open(cfg_path, "r", encoding='utf-8'))
    if "trainable_reranker_config" in cfg:
        cfg["train_reranker"] = yaml.safe_load(open(cfg["trainable_reranker_config"], "r", encoding='utf-8'))
    print("Config:")
    [print("\t{}: {}".format(k, v)) for k, v in cfg.items()]
    print("*******")

    len_summ_data = 0

    summ_data = []
    document_embedder = []
    summary_embedder = []
        
    if (args.pretrained_model == 'presumm'):
        summ_data, summary_embedder, document_embedder, summ_model, len_summ_data = load_presumm(args, device)
     
    elif (args.pretrained_model == 'bart'):
        summ_data, summary_embedder, document_embedder, summ_model, lens = load_bart(args, cfg)
        len_summ_data = len(summ_data)
        
    print("Total data: ", len_summ_data)
   
    if cfg.get("first_x", False):
        das_test = das_test[:cfg['first_x']]

    absts = get_abstss_test()
    for beam_size in cfg["beam_sizes"]:
        # summ_data = pickle.loads(pickle.dumps(summ_data_ori, -1))
        st = time()
        
        if (args.pretrained_model == 'presumm'):
            summ_data = data_loader.Dataloader(args, load_dataset(args, args.use_data, shuffle=False),
                                               args.batch_size, device, 
                                               shuffle=False, is_test=False)
        print("Dataset loading time: ", time() - st)
        do_beam_search_fact(args, beam_size, cfg, None, None, None, None, None, None, summ_model, summ_data, len_summ_data, document_embedder, summary_embedder, lens=lens)

    # print_results(args)
