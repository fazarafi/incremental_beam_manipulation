import argparse
import os
import pickle
import sys
from collections import Counter

import msgpack
import numpy as np
import yaml
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from utils import get_training_variables, START_TOK, PAD_TOK, END_TOK, get_multi_reference_training_variables, \
    get_final_beam, get_test_das, get_true_sents, TRAIN_BEAM_SAVE_FORMAT, TEST_BEAM_SAVE_FORMAT, RESULTS_DIR, \
    CONFIGS_MODEL_DIR, get_section_cutoffs, get_section_value, get_regression_vals, \
    get_args_presumm, SUMM_START_TOK, SUMM_END_TOK, SUMM_PAD_TOK, SUMM_CLS_TOK, convert_id_to_text, get_timestamp_file
from _base_models_fact import TGEN_Model, TrainableReranker, PairwiseReranker, SummaryFactTrainableReranker
from e2e_metrics.metrics.pymteval import BLEUScore
from embedding_extractor import TokEmbeddingSeq2SeqExtractor, DAEmbeddingSeq2SeqExtractor
from _beam_search_fact import run_beam_search_with_rescorer
from _scorer_functions_fact import get_score_function, get_score_function_fact

# Import FT Project
from PreSumm.src.models import data_loader, model_builder
from PreSumm.src.models.data_loader import load_dataset
from PreSumm.src.models.model_builder import AbsSummarizer
from PreSumm.src.models.predictor import build_predictor
from pytorch_transformers import BertTokenizer
import torch

from fact_scorer.fact_factcc.factcc_caller_model import FactccCaller
from fact_scorer.fact_summac.summac_caller import classify as summac_cls

import sys
sys.path.insert(0, './PreSumm/src') # hacky

import logging
logger = logging.getLogger(__name__)


def get_fact_scores(args, scorer, factcc_scorer, docs, summ):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir=args.temp_dir)
    
    # print(docs)
    # print(summ)
    
    # convert to factually scorable texts
    # if (docs) token_ids
    docs = convert_id_to_text(tokenizer, docs[0])
    # if (summ) token_ids
    summ = convert_id_to_text(tokenizer, summ)

    final_score = 0
    if scorer == 'factcc':
        final_score = factcc_scorer.classify(docs, summ)
    elif scorer == 'summac':
        final_score = summac_cls(docs, summ)
    elif scorer == 'fact_mixed':
        w1 = args.w1
        w2 = args.w2
        factcc_score = factcc_scorer.classify(docs, summ)
        summac_score = summac_cls(docs, summ)
        final_score = (w1*factcc_score + w2*summac_score)/(w1+w2)
    # TODO FT use weight

    return final_score


def get_embeddings_summary(tokenised_texts, symbols, length):
    pad_token = symbols[SUMM_PAD_TOK]
    
    # length = 0
    # if len(tokenised_texts[0]) == 1:
    #     length = max([len(x[0]) for x in tokenised_texts])    
    # else:
    #     print("SUMM DATA")
    #     print([len(x) for x in tokenised_texts])
    #     length = max([len(x) for x in tokenised_texts])
    print(length)

    
    embs = []
    for toks in tokenised_texts:
        tok_in_np = toks.cpu().tolist()
        
        # emb = tok_in_np

        if len(tok_in_np) == 1:
            emb = tok_in_np[0]
            # print(emb)
        else:
            emb = tok_in_np
        
        pad = [pad_token for _ in range(length - len(emb))]
        
        embs.append(pad + emb)
        
        
        
    result = [e[:length] for e in embs]
    # print("result.shape")
    # print(result.shape)
    return result


def get_scores_ordered_beam_fact(args, device, cfg, documents, summaries, beam_save_path=None, summ_train_data=None):
    print("Loading Training Data")
    beam_size = cfg["beam_size"]
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir=args.temp_dir)
    symbols = {'BOS': tokenizer.vocab['[unused0]'], 'EOS': tokenizer.vocab['[unused1]'],
               'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused2]']}
    
     
    train_texts, train_das = get_multi_reference_training_variables()
    if beam_save_path is None:
        beam_save_path = TRAIN_BEAM_SAVE_FORMAT.format(beam_size, cfg["fact_model_config"].split('.')[0].split('/')[-1])
    
    print("path exist? ", os.path.exists(beam_save_path))
    if not os.path.exists(beam_save_path):
        models = TGEN_Model(da_embedder, text_embedder, cfg["fact_model_config"])
        models.load_models()
        print("Creating test final beams")
        scorer = get_score_function('identity', cfg, models, None, beam_size)
        
        print('Loading checkpoint from %s' % args.test_from)
        checkpoint = torch.load(args.test_from, map_location=lambda storage, loc: storage)
        # opt = vars(checkpoint['opt'])
        summarization_models = AbsSummarizer(args, device, checkpoint)
        summarization_models.eval()

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir=args.temp_dir)
        symbols = {'BOS': tokenizer.vocab['[unused0]'], 'EOS': tokenizer.vocab['[unused1]'],
                'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused2]']}
        summ_predictor = build_predictor(args, tokenizer, symbols, summarization_models, logger)
    
        summarization_scorer = get_score_function_fact(args, cfg['scorer'], cfg, summ_predictor, None, beam_size)

        run_beam_search_with_rescorer(args, summarization_scorer, models, train_das, beam_size, cfg, only_rerank_final=True,
                                      save_final_beam_path=beam_save_path,
                                      summ_scorer=summarization_scorer, summ_beam_search_model=summ_predictor, summ_data=summ_train_data, device=device)
        
        # run_beam_search_with_rescorer(scorer, models, train_das, beam_size, cfg, only_rerank_final=True,
        #                               save_final_beam_path=beam_save_path)
    
    
    factcc_scorer = FactccCaller()
    
    fact_scores = []
    docs_seqs = []
    summ_seqs = []
    
    
    bleu = BLEUScore()

    scores = []
    final_beam = pickle.load(open(beam_save_path, "rb"))
    # print("ISII ")
    # print(final_beam)
    summ_seqs = []
    docs_seqs = []
    fact_scores = []
    log_probs = []
    with_ref_train_flag = cfg["with_refs_train"]
    num_ranks = cfg["num_ranks"]
    cut_offs = get_section_cutoffs(num_ranks)
    regression_vals = get_regression_vals(num_ranks, with_ref_train_flag)
    if cfg["output_type"] != 'pair':
        print("Cut off values:", cut_offs)
        print("Regression vals:", regression_vals)

    only_top = cfg.get("only_top", False)
    only_bottom = cfg.get("only_bottom", False)
    merge_middles = cfg["merge_middle_sections"]
    if only_top:
        print("Only using top value")
    if merge_middles and only_top:
            print("Ignoring only top since have merge_middle_sections set")
    
    # training_vals = list(zip(final_beam, train_texts, train_das))


    # Wrap summarization training set
    train_docs = []
    train_summ = []
    
    for batch in summ_train_data:
        train_docs.append(batch.src)
        train_summ.append(batch.tgt)

    print("LEN")    
    print(len(final_beam), " ", len(train_summ), " ", len( train_docs))

    training_vals = list(zip(final_beam, train_summ, train_docs))
    # TODO FT is remove testing setup
    training_vals = training_vals[:cfg.get("use_size", len(training_vals))]
    # training_vals = training_vals[:20]

    for beam, real_summs, docs in tqdm(training_vals):
        
        beam_scores = []
        if with_ref_train_flag:
            # I am not sure how to do log probs?
            summ_seqs.extend(real_summs)
            docs_seqs.extend([docs for _ in real_summs])
            fact_scores.extend([0 for _ in real_summs])

        for i, path in enumerate(beam):
            # print(i)
            summ = path[1] 
            # print("summ: ", summ)
            # print("docs: ", docs)
            # TODO FT check whether it's in the same data
            fact_score = get_fact_scores(args, cfg['scorer'], factcc_scorer, docs, summ)

            beam_scores.append((fact_score, summ, path))


        # for i,path in enumerate(beam):

        #     bleu.reset()
        #     hyp = [x for x in text_embedder.reverse_embedding(path[1]) if x not in [START_TOK, END_TOK, PAD_TOK]]
        #     bleu.append(hyp, [x for x in real_summs if x not in [START_TOK, END_TOK]])
        #     beam_scores.append((bleu.score(), hyp, path))

            
        #     # log_probs.append(i)
        # print("beam_scores: ", beam_scores)
        for i, (score, hyp, path) in enumerate(sorted(beam_scores, key=lambda x: x[0], reverse=True)):
            # TODO FT check, is it necessary for tokens?
            # summ_seqs.append([SUMM_START_TOK] + hyp + [SUMM_END_TOK])
            summ_seqs.append(hyp)
            docs_seqs.append(docs)
            
            if cfg["output_type"] in ['fact']:
                fact_scores.append(score)
            elif cfg["output_type"] in ['bleu', 'pair']:
                fact_scores.append(score)
            elif cfg["output_type"] == 'order_discrete':
                fact_scores.append(to_categorical([i], num_classes=beam_size))
            elif cfg["output_type"] in ['regression_ranker', 'regression_reranker_relative']:
                fact_scores.append(i / (beam_size - 1))
            elif cfg["output_type"] in ['regression_sections', 'binary_classif']:
                val = (i / (beam_size - 1))
                regression_val = get_section_value(val, cut_offs, regression_vals,
                                                   merge_middles, only_top, only_bottom)
                fact_scores.append(regression_val) # converts range from [0,1] to [-1,1] (which has mean of 0)
            else:
                raise ValueError("Unknown output type")

            log_probs.append([path[0]])

    len_summ = max([len(x[0]) for x in summaries])
    # TODO FT make embedder for better code
    summ_seqs = np.array(get_embeddings_summary(summ_seqs, symbols, len_summ))
    
    # TODO FT make sure, PAD in docs_seqs?
    len_docs = max([len(x[0]) for x in documents])
    docs_seqs = np.array(get_embeddings_summary(docs_seqs, symbols, len_docs))

    # print("docs_seqs")
    # print(docs_seqs[:3])


    if cfg["output_type"] in ['fact']:
        # print("SCORES: ", Counter(fact_scores))
        fact_scores = np.array(fact_scores).reshape((-1, 1))
    elif cfg["output_type"] in ['regression_ranker', 'bleu', 'regression_reranker_relative', 'pair',
                              'regression_sections', 'binary_classif']:
        # print("SCORES: ", Counter(fact_scores))
        fact_scores = np.array(fact_scores).reshape((-1, 1))
    elif cfg["output_type"] == 'order_discrete':
        fact_scores = np.array(fact_scores).reshape((-1, beam_size))
    
    # fact_scores is already in [-1,1]

    log_probs = np.array(log_probs) # TODO FT what is it for?

    return summ_seqs, docs_seqs, fact_scores, log_probs
    
    

# MAIN PROGRAM

# TO run: 
# python3 get_results.py -c new_configs/fact_setup_bm_traindata.yaml
# python3 _fact_train_beam_manipulator.py -c new_configs/model_configs/fact_bm_model.yaml -gpu_ranks 0 -visible_gpus 0

parser = argparse.ArgumentParser()


# PreSumm parser
parser = get_args_presumm(parser)

parser.add_argument('-c', default=None)
parser.add_argument('-should_skip_beam', default=False)
args = parser.parse_args()


# Setup GPU for PreSumm
args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
args.world_size = len(args.gpu_ranks)
os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

# init_logger(args.log_file) TODO add legger
device = "cpu" if args.visible_gpus == '-1' else "cuda"
device_id = 0 if device == "cuda" else -1


cfg_path = args.c
if cfg_path is None:
    filenames = os.listdir(CONFIGS_MODEL_DIR)
    filepaths = [os.path.join(CONFIGS_MODEL_DIR, filename) for filename in filenames]
    mod_times = [(os.path.getmtime(x), i) for i, x in enumerate(filepaths)]
    cfg_path = filepaths[max(mod_times)[1]]

print("Using config from: {}".format(cfg_path))
cfg = yaml.safe_load(open(cfg_path, "r"))
print("Config:")
[print("\t{}: {}".format(k,v)) for k,v in cfg.items()]
print("*******")
texts, das = get_multi_reference_training_variables()
da_embedder = DAEmbeddingSeq2SeqExtractor(das)

# This is a very lazy move
texts_flat, _ = get_training_variables()
text_embedder = TokEmbeddingSeq2SeqExtractor(texts_flat)



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir=args.temp_dir)
symbols = {'BOS': tokenizer.vocab['[unused0]'], 'EOS': tokenizer.vocab['[unused1]'],
            'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused2]']}

# TODO FT check if it's correct

summ_train_data = data_loader.Dataloader(args, load_dataset(args, args.use_data, shuffle=False), args.batch_size, device, shuffle=False, is_test=True)

summary_embedder = list()
document_embedder = list()

i=0
max_data = cfg['use_size'] if 'use_size' in cfg else -99
batch_list = []
for batch in summ_train_data:
    i+=1
    # TODO FT remove little data
    document_embedder.append(batch.src)
    summary_embedder.append(batch.tgt)
    batch_list.append(batch)
    if (i==max_data):
        break

print("LEN document_embedder, summary_embedder, batch_list")    
print(len(document_embedder), " ", len(summary_embedder), " ", len(batch_list))

# print("batch_list: ", batch_list)


if cfg['output_type'] == 'fact':
    reranker = SummaryFactTrainableReranker(summary_embedder, document_embedder, cfg_path, tokenizer=tokenizer)
elif cfg['output_type'] != 'pair':
    reranker = TrainableReranker(da_embedder, text_embedder, cfg_path)
else:
    reranker = PairwiseReranker(da_embedder, text_embedder, cfg_path)


if reranker.load_model():
    print("WARNING THE TRAINING START POINT IS AN ALREADY TRAINED MODEL")

if cfg["train"]:
    print("Training")
    summ_seqs, docs_seqs, scores, log_probs = get_scores_ordered_beam_fact(args, device, cfg, document_embedder, summary_embedder, 
                                                                    beam_save_path=cfg.get("beam_save_path", None), summ_train_data=batch_list)
    
    if type(scores[0]) ==  int:
        print("Score Distributions:")
        print(Counter([x for x in scores]))
    print("LP distributions")
    print("min", min(log_probs))
    print("max", max(log_probs))
    print("mean", sum(log_probs)/len(log_probs))

    if cfg['output_type'] == 'fact':
        reranker.train(summ_seqs, docs_seqs, scores, log_probs, cfg["epoch"], cfg["valid_size"],
                   cfg.get("min_training_passes", 5))
    elif cfg['output_type'] != 'pair':
        reranker.train(summ_seqs, docs_seqs, scores, log_probs, cfg["epoch"], cfg["valid_size"],
                   cfg.get("min_training_passes", 5))
    else:
        reranker.train(summ_seqs, docs_seqs, scores, log_probs, cfg["epoch"], cfg["valid_size"], cfg["num_ranks"],
                       cfg.get("only_bottom", False),
                       cfg.get("only_top", False),
                       cfg.get("min_training_passes", 5))

if cfg["show_reranker_post_training_stats"]:
    test_das = get_test_das()
    test_texts = get_true_sents()
    final_beam_path = TEST_BEAM_SAVE_FORMAT.format(10)


    # GET Summary model testing
    test_docs_data = []
    test_summ_data = []
    summ_test_data = data_loader.Dataloader(args, load_dataset(args, args.use_data, shuffle=False),
                                        args.batch_size, device, shuffle=False, is_test=True)

    for batch in summ_test_data:
        test_docs_data.append(batch.src)
        test_summ_data.append(batch.tgt)

    if not os.path.exists(final_beam_path):
        # print("Creating final beams file")
        # models = TGEN_Model(da_embedder, text_embedder, cfg['tgen_seq2seq_config'])
        # models.load_models()
        # scorer = get_score_function('identity', cfg, models, None, 10)


        print('Loading checkpoint from %s' % args.test_from)
        checkpoint = torch.load(args.test_from, map_location=lambda storage, loc: storage)
        # opt = vars(checkpoint['opt'])
        summarization_models = AbsSummarizer(args, device, checkpoint)
        summarization_models.eval()

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir=args.temp_dir)
        symbols = {'BOS': tokenizer.vocab['[unused0]'], 'EOS': tokenizer.vocab['[unused1]'],
                'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused2]']}
        summ_predictor = build_predictor(args, tokenizer, symbols, summarization_models, logger)
    
        print("Load")
        # fact scorer
        summarization_scorer = get_score_function_fact(args, 'factcc', cfg, summ_predictor, None, beam_size)


        run_beam_search_with_rescorer(args, scorer, models, test_das, 10, only_rerank_final=True,
                                      save_final_beam_path=final_beam_path,
                                      summ_scorer=summarization_scorer, summ_beam_search_model=summ_predictor, summ_data=summ_test_data)

    
    factcc_scorer = FactccCaller()
    # feqa_scorer = 

    
    bleu = BLEUScore()
    test_da_embs = da_embedder.get_embeddings(test_das)
    final_beam = pickle.load(open(final_beam_path, 'rb+'))
    all_reals = []
    all_preds = []
    
    # TODO FT reference for below loop
    # for da_emb, beam, true in zip(test_da_embs, final_beam, test_texts):
    #     real_scores = []
    #     lp_probs_beam = [x[0] for x in beam]
    #     for i, path in enumerate(beam):
    #         logp, text_emb, _ = path
    #         toks = text_embedder.reverse_embedding(text_emb)
    #         lp_rank = [sum([1 for x in lp_probs_beam if x > logp + 0.000001])]
    #         lp_rank_cat = to_categorical([lp_rank], num_classes=10)
    #         da_seqs = np.array([da_emb])
    #         text_seqs = np.array(text_embedder.get_embeddings([toks], pad_from_end=False))
    #         score = reranker.predict_bleu_score(text_seqs, da_seqs, lp_rank_cat)
    #         score = 9 - np.argmax(score[0])
    #         all_preds.append(score)
    #         pred = [x for x in toks if x not in [START_TOK, END_TOK, PAD_TOK]]
    #         bleu.reset()
    #         bleu.append(pred, true)
    #         real_scores.append((bleu.score(), i))
    #     sorted_reals = sorted(real_scores)
    #     all_reals.extend([i for _, i in sorted_reals])
    # print(confusion_matrix(all_reals, all_preds))

    # beam is the path 
    # path is generated summ

    for test_docs, beam, test_summ in zip(test_docs_data, final_beam, test_summ_data):
        real_scores = []
        lp_probs_beam = [x[0] for x in beam]
        for i, path in enumerate(beam):
            logp, summ_emb, _ = path
            docs_seqs = test_docs
            summ_seqs = summ_emb
            
            score = reranker.predict_fact_score(summ_seqs, docs_seqs, lp_rank_cat)
            score = 9 - np.argmax(score[0]) # TODO FT make sure
            
            all_preds.append(score)
            pred = [x for x in toks if x not in [SUMM_START_TOK, SUMM_END_TOK, SUMM_PAD_TOK]]

            # bleu.append(pred, test_summ)

            score = get_fact_scores(args, cfg['scorer'], factcc_scorer, docs_seqs, test_summ)

            real_scores.append((score), i)

        sorted_reals = sorted(real_scores)
        all_reals.extend([i for _, i in sorted_reals])


    print(confusion_matrix(all_reals, all_preds))

    beam_texts = [[text for text, _ in beam] for beam in final_beam]
    beam_tok_logprob = [[tp for _, tp in beam] for beam in final_beam]
    # test_text_embs = [text_embedder.get_embeddings(beam) for beam in beam_texts]
    mapping = []
    order_correct_surrogate = 0
    order_correct_seq2seq = 0
    
    # for texts, da_emb, tp_emb, true_texts in zip(beam_texts, test_da_embs, beam_tok_logprob, test_texts):
    #     summ_seqs = np.array(text_embedder.get_embeddings(texts, pad_from_end=False))
    #     docs_seqs = np.array([da_emb for _ in range(len(summ_seqs))])
    #     tp_seqs = np.array(tp_emb).reshape(-1, 1)
    #     preds = reranker.predict_bleu_score(summ_seqs, docs_seqs, tp_seqs)
    #     beam_scores = []
    #     for i, (pred, text, tp) in enumerate(zip(preds, texts, tp_seqs)):
    #         bleu.reset()
    #         bleu.append(text, true_texts)
    #         real = bleu.score()
    #         mapping.append((pred[0], real))
    #         beam_scores.append((real, pred[0], i, tp[0]))
    #     sorted_beam_scores = sorted(beam_scores, reverse=True)
    #     best = sorted_beam_scores[0][2]
    #     best_surrogate = sorted(beam_scores, key=lambda x: x[1])[0][2]
    #     best_seq2seq = sorted(beam_scores, key=lambda x: x[3], reverse=True)[0][2]
    #     if best == best_surrogate:
    #         order_correct_surrogate += 1
    #     if best == best_seq2seq:
    #         order_correct_seq2seq += 1
    # print(len(beam_texts), order_correct_surrogate, order_correct_seq2seq)

    for beam_summ, test_docs, tp_emb, true_summ in zip(beam_texts, test_docs_data, beam_tok_logprob, test_summ_data):
        summ_beam_hypo = beam_summ
        # TODO FT make sure data correctness
        docs_seqs = np.array([test_docs for _ in range(len(summ_beam_hypo))])
        
        tp_seqs = np.array(tp_emb).reshape(-1, 1)
        # TODO FT the pred calculation true_summ from test data
        preds = reranker.predict_fact_score(true_summ, docs_seqs, tp_seqs)
        beam_scores = []
        # for i, (pred, text, tp) in enumerate(zip(preds, texts, tp_seqs)):
        #     bleu.reset()
        #     bleu.append(text, true_summ)
        #     real = bleu.score()
        #     mapping.append((pred[0], real))
        #     beam_scores.append((real, pred[0], i, tp[0]))
        for i, (pred, summ_hypo, tp) in enumerate(zip(preds, beam_summ, tp_seqs)):
            real = get_fact_scores(args, cfg['scorer'], factcc_scorer, docs, summ_hypo)
            mapping.append((pred[0], real))
            beam_scores.append((real, pred[0], i, tp[0]))

        
        sorted_beam_scores = sorted(beam_scores,  key=lambda x: x[0], reverse=True)
        best = sorted_beam_scores[0][2]
        best_surrogate = sorted(beam_scores, key=lambda x: x[0])[0][2]
        best_seq2seq = sorted(beam_scores, key=lambda x: x[0], reverse=True)[0][2]
        if best == best_surrogate:
            order_correct_surrogate += 1
        if best == best_seq2seq:
            order_correct_seq2seq += 1
    print(len(beam_texts), order_correct_surrogate, order_correct_seq2seq)

    # print(mapping)
    preds = [x for x, _ in mapping]
    reals = [x for _, x in mapping]
    plt.scatter(reals, preds, alpha=0.05)
    plt.plot([0, 1], [0, 1], color='red')
    plt.xlabel("Real Score")
    plt.ylabel("Predicted")
    plt.show()
