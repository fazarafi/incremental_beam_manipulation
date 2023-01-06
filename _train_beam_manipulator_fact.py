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
    get_args_presumm, SUMM_START_TOK, SUMM_END_TOK, SUMM_PAD_TOK, SUMM_CLS_TOK, get_timestamp_file
from _base_models_fact import TrainableReranker, PairwiseReranker, SummaryFactTrainableReranker
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
from rouge import Rouge

from _bart_utils import load_bart_dataset, load_bart_model, get_bart_tokenizer, convert_ids_to_text, \
    BART_ENCODER_MAX_LENGTH, BART_PAD_TOKEN

import sys
sys.path.insert(0, './PreSumm/src') # hacky

import logging
logger = logging.getLogger(__name__)

from fact_scorer.fact_coco.coco_caller import initialize_coco, evaluate_coco

def convert_id_list_to_text(pretrained_model, tokenizer, token_ids):
    text = ""

    if type(token_ids) is str:
        return token_ids

    if (pretrained_model=='presumm'):
        # reshape first
        token_ids = token_ids.flatten()

        # Convert token_ids to text for factual consistency scoring
        text = tokenizer.convert_ids_to_tokens([int(n) for n in token_ids])
        text = ' '.join(text).replace(' ##','')
        text = text.replace('[unused0]', '').replace('[unused3]', '').replace('[PAD]', '').replace('[unused1]', '').replace(r' +', ' ').replace(' [unused2] ', '<q>').replace('[unused2]', '').strip()
    elif (pretrained_model=='bart'):
        text = convert_ids_to_text(tokenizer, token_ids)
    
    return text


def get_fact_scores(args, scorer, tokenizer, scorers, docs, summ_hypo, summ_tgt):
    
    pretrained_model = args.pretrained_model
    # convert to factually scorable texts
    # if (docs) token_ids
    docs = convert_id_list_to_text(pretrained_model, tokenizer, docs)
    summ_hypo = convert_id_list_to_text(pretrained_model, tokenizer, summ_hypo)

    summ_tgt = convert_id_list_to_text(pretrained_model, tokenizer, summ_tgt)

    final_score = 0
    if scorer == 'factcc':
        final_score = scorers['factcc'].classify(docs, summ_hypo)
    elif scorer == 'summac':
        final_score = summac_cls(docs, summ_hypo)
    elif scorer == 'fact_mixed':
        w1 = args.w1
        w2 = args.w2
        factcc_score = scorers['factcc'].classify(docs, summ_hypo)
        coco_score = evaluate_coco(scorers['coco'], docs, summ_hypo)
        final_score = (w1*factcc_score + w2*coco_score)/(w1+w2)
    elif scorer == 'fact_rouge':
        factcc_score = scorers['factcc'].classify(docs, summ_hypo)

        if (summ_tgt != None) and (summ_tgt != "") and (summ_tgt != "."):
            rouge_scores = scorers['rouge'].get_scores(summ_hypo, summ_tgt, avg=True)
            rouge_score = rouge_scores['rouge-1']['f']
            w1 = args.w1
            w2 = args.w2
            final_score = (w1 * factcc_score + w2 * rouge_score)/(w1 + w2)
        else:
            print("summ_tgt: ", summ_tgt)
            final_score = factcc_score
    elif scorer == 'bart_penalty':
        final_score = 0 # TODO take from paths

        
    # elif scorer == 'fact_mixed':
    #     w1 = args.w1
    #     w2 = args.w2
    #     factcc_score = scorers['factcc'].classify(docs, summ_hypo)
    #     summac_score = summac_cls(docs, summ_hypo)
    #     final_score = (w1*factcc_score + w2*summac_score)/(w1+w2)
    
    return final_score

def load_presumm(args):
    summ_data = [] 
    summary_embedder = []
    document_embedder = []
    summ_model = []
    len_summ_data = 0

    summ_data = data_loader.Dataloader(args, load_dataset(args, args.use_data, shuffle=False),
                                            args.batch_size, args.device,
                                            shuffle=False, is_test=False)

    print("PreSumm: Counting dataset length...")

    for batch in summ_data:
        len_summ_data += len(batch.src)
        document_embedder.append(batch.src)
        summary_embedder.append(batch.tgt)
    
    print('Loading checkpoint from %s' % args.test_from)
    checkpoint = torch.load(args.test_from, map_location=lambda storage, loc: storage)
    summarization_models = AbsSummarizer(args, args.device, checkpoint)
    summarization_models.eval()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir=args.temp_dir)
    symbols = {'BOS': tokenizer.vocab['[unused0]'], 'EOS': tokenizer.vocab['[unused1]'],
            'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused2]']}
    summ_model = build_predictor(args, tokenizer, symbols, summarization_models, logger)
    

    return summ_data, summary_embedder, document_embedder, summ_model, len_summ_data

def load_bart(args):
    summ_data = [] 
    summary_embedder = []
    document_embedder = []
    summ_model = []

    summ_data = load_bart_dataset(args)
    print("BART: Counting dataset length...")

    len_summ_data = len(summ_data)
    document_embedder = summ_data["document"]
    summary_embedder = summ_data["summary"]

    summ_model = load_bart_model(args)

    return summ_data, summary_embedder, document_embedder, summ_model


def get_embeddings_summary(tokenised_texts, pad_token, length):
    
    embs = []
    for toks in tokenised_texts:
        tok_in_np = toks
        if (type(toks)==torch.Tensor):
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

def init_scorers():
    scorers = {}
    coco_params = initialize_coco()
    fact_scorer = FactccCaller()
    rouge_scorer = Rouge()

    return {
        'coco': coco_params,
        'factcc': fact_scorer,
        'rouge': rouge_scorer
    }

def get_tokenizer(args):
    if (args.pretrained_model=='presumm'):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir=args.temp_dir)
        symbols = {'BOS': tokenizer.vocab['[unused0]'], 'EOS': tokenizer.vocab['[unused1]'],
                'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused2]']}

    if (args.pretrained_model=='bart'):
        tokenizer = get_bart_tokenizer(args)
    
    return tokenizer

    

def get_scores_ordered_beam_fact(args, device, cfg, documents, summaries, beam_save_path=None, summ_train_data=None):
    print("Loading Training Data")
    beam_size = cfg["beam_size"]
    
    tokenizer = get_tokenizer(args)
    if beam_save_path is None:
        beam_save_path = TRAIN_BEAM_SAVE_FORMAT.format(beam_size, cfg["fact_model_config"].split('.')[0].split('/')[-1])
    
    print("path exist? ", os.path.exists(beam_save_path))
    if not os.path.exists(beam_save_path):
        summ_data = []
        summary_embedder = []
        document_embedder = []

        if (args.pretrained_model == 'presumm'):
            symbols = {'BOS': tokenizer.vocab['[unused0]'], 'EOS': tokenizer.vocab['[unused1]'],
               'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused2]']}
    
            summ_data, summary_embedder, document_embedder, summ_model, len_summ_data = load_presumm(args)
        
        elif (args.pretrained_model == 'bart'):
            summ_data, summary_embedder, document_embedder, summ_model = load_bart(args)
            len_summ_data = len(summ_data)
        
        alpha = 0.65
        summarization_scorer = get_score_function_fact(args, cfg['scorer'], cfg, None, None, beam_size, alpha, summary_embedder, document_embedder)

        run_beam_search_with_rescorer(args, summarization_scorer, None, None, beam_size, cfg, only_rerank_final=True,
                                      save_final_beam_path=beam_save_path,
                                      summ_scorer=summarization_scorer, summ_beam_search_model=summ_model, 
                                      summ_data=summ_data, device=device, len_summ_data=len_summ_data)
        
    
    scorers = init_scorers()

    fact_scores = []
    docs_seqs = []
    summ_seqs = []
    
    scores = []
    final_beam = pickle.load(open(beam_save_path, "rb"))
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
    
    if (args.pretrained_model == 'presumm'):
        for batch in summ_train_data:
            train_docs.append(batch.src)
            train_summ.append(batch.tgt)
    elif (args.pretrained_model == 'bart'):
        train_docs = tokenizer(
            summ_train_data["document"],
            truncation=True,
            return_tensors="np",
        ).input_ids
        
        train_summ = tokenizer(
            summ_train_data["summary"],
            truncation=True,
            return_tensors="np",
        ).input_ids
        

    print("LEN")    
    print(len(final_beam), " ", len(train_summ), " ", len( train_docs))

    training_vals = list(zip(final_beam, train_summ, train_docs))
    # TODO FT is remove testing setup
    training_vals = training_vals[:cfg.get("use_size", len(training_vals))]
    

    if not os.path.exists(cfg["reranker_loc"]):
        print("Create working folder: ", cfg["reranker_loc"])
        os.mkdir(cfg["reranker_loc"])

    beam_scores_path = os.path.join(cfg["reranker_loc"], "beam_scores." + str(cfg["use_size"]) + ".pickle")
    summ_seqs_path = os.path.join(cfg["reranker_loc"], "summ_seqs." + str(cfg["use_size"]) + ".pickle")
    docs_seqs_path = os.path.join(cfg["reranker_loc"], "docs_seqs." + str(cfg["use_size"]) + ".pickle")
    fact_scores_path = os.path.join(cfg["reranker_loc"], "fact_scores." + str(cfg["use_size"]) + ".pickle")
    log_probs_path = os.path.join(cfg["reranker_loc"], "log_probs." + str(cfg["use_size"]) + ".pickle")
    
    if (os.path.exists(beam_scores_path)):
        print("Loading saved beam_scores, fact_scores, etc.")
        beam_scores = pickle.load(open(beam_scores_path, "rb"))
        summ_seqs = pickle.load(open(summ_seqs_path, "rb"))
        docs_seqs = pickle.load(open(docs_seqs_path, "rb"))
        fact_scores = pickle.load(open(fact_scores_path, "rb"))
        log_probs = pickle.load(open(log_probs_path, "rb"))
    else:
        print("Creating beam_scores_path: ", beam_scores_path)
        # Load all scores
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
                # TODO FT check whether it's in the same data
                fact_score = get_fact_scores(args, cfg['scorer'], tokenizer, scorers, docs, summ, real_summs)

                beam_scores.append((fact_score, summ, path))


            # for i,path in enumerate(beam):

            #     bleu.reset()
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

        pickle.dump(beam_scores, open(beam_scores_path, "wb+"))
        pickle.dump(summ_seqs, open(summ_seqs_path, "wb+"))
        pickle.dump(docs_seqs, open(docs_seqs_path, "wb+"))
        pickle.dump(fact_scores, open(fact_scores_path, "wb+"))
        pickle.dump(log_probs, open(log_probs_path, "wb+"))

        

    if (args.pretrained_model=='bart'):
        len_summ = max([len(x) for x in train_summ])
        print("len_summ: ", len_summ)
        summ_seqs = np.array(get_embeddings_summary(summ_seqs, BART_PAD_TOKEN, len_summ))
        
        len_docs = max([len(x) for x in train_docs])
        print("len_docs: ", len_docs)
        docs_seqs = np.array(get_embeddings_summary(docs_seqs, BART_PAD_TOKEN, len_docs))


    elif (args.pretrained_model=='presumm'):
        len_summ = max([len(x[0]) for x in summaries])
        summ_seqs = np.array(get_embeddings_summary(summ_seqs, symbols[SUMM_PAD_TOK], len_summ))
        
        # TODO FT make sure, PAD in docs_seqs?
        len_docs = max([len(x[0]) for x in documents])
        docs_seqs = np.array(get_embeddings_summary(docs_seqs, symbols[SUMM_PAD_TOK], len_docs))

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
parser = get_args_presumm(parser)
args = parser.parse_args()



# Setup GPU for PreSumm
args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
args.world_size = len(args.gpu_ranks)
os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

# init_logger(args.log_file) TODO add legger
device = "cpu" if args.visible_gpus == '-1' else "cuda"
device_id = 0 if device == "cuda" else -1
args.device = device

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



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir=args.temp_dir)
symbols = {'BOS': tokenizer.vocab['[unused0]'], 'EOS': tokenizer.vocab['[unused1]'],
            'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused2]']}

# TODO FT check if it's correct

batch_list = []
summary_embedder = list()
document_embedder = list()

max_data = cfg['use_size'] if 'use_size' in cfg else -99
if (args.pretrained_model == 'presumm'):
    summ_data, summary_embedder, document_embedder, summ_model, len_summ_data = load_presumm(args)
    
    i=0
    for batch in summ_data:
        i+=1
        # TODO FT remove little data
        document_embedder.append(batch.src)
        summary_embedder.append(batch.tgt)
        batch_list.append(batch)
        if (i==max_data):
            break
    
elif (args.pretrained_model == 'bart'):
    summ_data, summary_embedder, document_embedder, summ_model = load_bart(args)
    len_summ_data = len(summ_data)

    tokenizer = get_bart_tokenizer(args)

    summary_embedder = summary_embedder[:max_data]
    document_embedder = document_embedder[:max_data]

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
    
    batch_list = summ_data[:max_data]    
    


        
# print("LEN document_embedder, summary_embedder, batch_list")    
# print(len(document_embedder), " ", len(summary_embedder), " ", len(batch_list))


if cfg['output_type'] == 'fact':
    reranker = SummaryFactTrainableReranker(summary_embedder, document_embedder, cfg_path, tokenizer=tokenizer, pretrained_model=args.pretrained_model)
# elif cfg['output_type'] != 'pair':
#     reranker = TrainableReranker(da_embedder, text_embedder, cfg_path)
# else:
#     reranker = PairwiseReranker(da_embedder, text_embedder, cfg_path)


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

# LOW PRIORITY
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
        
        summ_data = []
        summary_embedder = []
        document_embedder = []

        if (args.pretrained_model == 'presumm'):
            summ_data, summary_embedder, document_embedder, summ_model, len_summ_data = load_presumm(args)
        
        elif (args.pretrained_model == 'bart'):
            summ_data, summary_embedder, document_embedder, summ_model = load_bart(args)
            len_summ_data = len(summ_data)
        alpha = 0.65

        summarization_scorer = get_score_function_fact(args, cfg['scorer'], cfg, None, None, beam_size, alpha, summary_embedder, document_embedder)

        run_beam_search_with_rescorer(args, scorer, models, test_das, 10, only_rerank_final=True,
                                      save_final_beam_path=final_beam_path,
                                      summ_scorer=summarization_scorer, summ_beam_search_model=summ_predictor, summ_data=summ_test_data)

    
    scorers = init_scorers()

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

            score = get_fact_scores(args, cfg['scorer'], tokenizer, scorers, docs_seqs, test_summ)

            real_scores.append((score), i)

        sorted_reals = sorted(real_scores)
        all_reals.extend([i for _, i in sorted_reals])


    print(confusion_matrix(all_reals, all_preds))

    beam_texts = [[text for text, _ in beam] for beam in final_beam]
    beam_tok_logprob = [[tp for _, tp in beam] for beam in final_beam]
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
            real = get_fact_scores(args, cfg['scorer'], tokenizer, scorers, docs, summ_hypo)
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


