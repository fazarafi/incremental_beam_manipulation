import random
import numpy as np
from keras.utils import to_categorical

from utils import START_TOK, END_TOK, PAD_TOK, get_features, SUMM_PAD_TOK

from _base_models_fact import TGEN_Reranker, TrainableReranker, SummaryFactTrainableReranker

from fact_scorer.fact_factcc.factcc_caller_model import FactccCaller
from fact_scorer.fact_summac.summac_caller import classify as summac_cls

from fact_scorer.fact_coco.coco_caller import initialize_coco, evaluate_coco
from rouge import Rouge

from pytorch_transformers import BertTokenizer
import torch

from time import time 

from _bart_utils import get_bart_tokenizer, convert_ids_to_text, BART_PAD_TOKEN

def get_regressor_score_func(regressor, text_embedder, w2v):
    def func(path, logprob, da_emb, da_i, beam_size, docs, tgt=None):
        features = get_features(path, text_embedder, w2v, logprob)
        regressor_score = regressor.predict(features.reshape(1, -1))[0][0]
        return regressor_score

    return func

def get_tgen_rerank_score_func(tgen_reranker):
    def func(path, logprob, da_emb, da_i, enc_outs):
        text_emb = path[1]
        pads = [tgen_reranker.text_embedder.tok_to_embed[PAD_TOK]] * \
               (tgen_reranker.text_embedder.length - len(text_emb))
        text_emb = pads + text_emb
        text_emb = text_emb[:tgen_reranker.text_embedder.length]
        reranker_score = tgen_reranker.get_pred_hamming_dist(text_emb, da_emb)
        return path[0] - 100 * reranker_score

    return func

def get_identity_score_func():
    def func(path, logprob, da_emb, da_i, beam_size, docs, tgt=None):
        return path[0]

    return func


power_cache = {}


#This cache appears not to be speeding things up
def get_power(num, power):
    key = (num, power)
    if key in power_cache:
        return power_cache[key]
    else:
        val = pow(num, power)
        power_cache[key] = val
        return val


def get_length_normalised_score_func(alpha):
    def func(path, logprob, da_emb, da_i, beam_size, docs, tgt=None):
        return path[0] / get_power(len(path[1]), alpha)

    return func

# def get_greedy_decode_score_func(models, final_scorer, max_length_out, save_scores=None):
#     def func(path, logprob, da_emb, da_i, enc_outs):
#         path = models.naive_complete_greedy(path, enc_outs, max_length_out - len(path[1]))
#         score = final_scorer(path, logprob, da_emb, da_i, enc_outs)
#         if save_scores is not None:
#             if type(save_scores) is dict:
#                 raise NotImplementedError("This bit need to be rewritten")
#         return score
#
#     return func

def get_oracle_score_func(bleu, true_vals, text_embedder, reverse):
    def func(path, logprob, da_emb, da_i, beam_size, docs, tgt=None):
        true = true_vals[da_i]
        toks = text_embedder.reverse_embedding(path[1])
        pred = [x for x in toks if x not in [START_TOK, END_TOK, PAD_TOK]]
        bleu.reset()
        # pred -> text_embedder
        # ture -> true_vals, gold
        bleu.append(pred, true)
        if reverse:
            return 1 - bleu.score()
        return bleu.score()

    return func

def get_random_score_func():
    def func(path, logprob, da_emb, da_i, beam_size, docs, tgt=None):
        return random.random()

    return func


def get_learned_score_func(trainable_reranker, select_max=False, reverse_order=False):
    def func(path, logprob, da_emb, da_i, beam_size, docs, tgt=None):
        text_emb = path[1]
        pads = [trainable_reranker.text_embedder.tok_to_embed[PAD_TOK]] * \
               (trainable_reranker.text_embedder.length - len(text_emb))
        if trainable_reranker.logprob_preprocess_type == 'categorical_order':
            logprob_rank = logprob * trainable_reranker.beam_size // beam_size
            logprob_val = to_categorical([logprob_rank], num_classes=trainable_reranker.beam_size)
        else:
            logprob_val = [path[0]]

        text_seqs = [pads + text_emb]
        text_seqs = [text_seqs[0][:trainable_reranker.text_embedder.length]]
        pred = trainable_reranker.predict_bleu_score(
            np.array(text_seqs),
            np.array([da_emb]),
            np.array(logprob_val))

        if trainable_reranker.output_type in ["regression_ranker", "regression_reranker_relative"]:
            return 1 - pred[0][0]
        elif trainable_reranker.output_type in ["regression_sections"]:
            if reverse_order:
                return -pred[0][0], path[0]
            return pred[0][0], path[0]
        elif trainable_reranker.output_type in ["binary_classif"]:
            pred = 1 if pred[0][0] > 0.5 else 0
            return 1 - pred, path[0]

        if select_max:
            max_pred = np.argmax(pred[0])
            return 10 - max_pred, pred[0][0]
        elif reverse_order:
            return -pred[0][0]
        else:
            return pred[0][0]

    return func

def get_score_function(scorer, cfg, models, true_vals, beam_size, alpha=0.65):
    da_embedder = models.da_embedder
    text_embedder = models.text_embedder
    print("Using Scorer: {}".format(scorer))
    if scorer == "TGEN":
        tgen_reranker = TGEN_Reranker(da_embedder, text_embedder, cfg['tgen_reranker_config'])
        tgen_reranker.load_model()
        return get_tgen_rerank_score_func(tgen_reranker)
    elif scorer == 'identity':
        return get_identity_score_func()
    elif scorer in ['surrogate', "surrogate_rev"]:
        learned = TrainableReranker(da_embedder, text_embedder, cfg['trainable_reranker_config'])
        learned.load_model()
        select_max = cfg.get("order_by_max_class", False)
        reverse_order = scorer == 'surrogate_rev'
        return get_learned_score_func(learned, select_max, reverse_order)
    elif scorer == 'random':
        return get_random_score_func()
    elif scorer == 'length_normalised':
        return get_length_normalised_score_func(alpha)
    else:
        raise ValueError("Unknown Scorer {}".format(cfg['scorer']))

def get_learned_fact_score_func(trainable_reranker, select_max=False, reverse_order=False, pad_symbol=None, len_summ=None, len_docs=None):
    def func(path, logprob, da_emb, da_i, beam_size, docs, tgt=None):
        summ_emb = path[1]
        pads = [pad_symbol] * \
               (len_summ - len(summ_emb))
        if trainable_reranker.logprob_preprocess_type == 'categorical_order':
            logprob_rank = logprob * trainable_reranker.beam_size // beam_size
            logprob_val = to_categorical([logprob_rank], num_classes=trainable_reranker.beam_size)
        else:
            logprob_val = [path[0]]

        if (type(summ_emb)==torch.Tensor):
            summ_emb = summ_emb.cpu().tolist()
        
        summ_seqs = [pads + summ_emb]
        summ_seqs = [summ_seqs[0][:len_summ]]

        # print("DEB ", docs)
        if (type(docs) == torch.Tensor):
            if (len(docs.shape)>1):
                docs_emb = docs[0].cpu().tolist()    
            else:
                docs_emb = docs.cpu().tolist()
        else:
            docs_emb = docs
        
        # print("DEBUG 185 ", docs_emb)
        docs_pads = [pad_symbol] * \
               (len_docs - len(docs_emb))
        docs_seqs = [docs_pads + docs_emb]
        docs_seqs = [docs_seqs[0][:len_docs]]

        # print("docs_seqs: ", docs_seqs)
        # print("summ_seqs: ", summ_seqs)
        
        pred = trainable_reranker.predict_fact_score(
            np.array(summ_seqs),
            np.array(docs_seqs),
            np.array(logprob_val))

        # print("PRED: ", pred)

        if trainable_reranker.output_type in ["regression_ranker", "regression_reranker_relative"]:
            return 1 - pred[0][0]
        elif trainable_reranker.output_type in ["regression_sections"]:
            if reverse_order:
                return -pred[0][0], path[0]
            return pred[0][0], path[0]
        elif trainable_reranker.output_type in ["binary_classif"]:
            pred = 1 if pred[0][0] > 0.5 else 0
            return 1 - pred, path[0]

        if select_max:
            max_pred = np.argmax(pred[0])
            return 10 - max_pred, pred[0][0]
        elif reverse_order:
            return -pred[0][0]
        else:
            return pred[0][0]

    return func


def convert_id_to_text(pretrained_model, tokenizer, token_ids):
    text = ""    
    if type(token_ids) is str:
        return token_ids

    # handle more gracefully
    if (type(token_ids)==tuple):
        token_ids = token_ids[1]
    
    if (type(token_ids)==torch.Tensor):
        token_ids = token_ids.flatten()

    if (pretrained_model=='presumm'):
        # Convert token_ids to text for factual consistency scoring
        text = tokenizer.convert_ids_to_tokens([int(n) for n in token_ids])
        text = ' '.join(text).replace(' ##','')
        text = text.replace('[unused0]', '').replace('[unused3]', '').replace('[PAD]', '').replace('[unused1]', '').replace(r' +', ' ').replace(' [unused2] ', '<q>').replace('[unused2]', '').strip()
    elif (pretrained_model=='bart'):
        text = convert_ids_to_text(tokenizer, token_ids)
    # print("text ", text)
    return text

def get_bart_score(path):
    cur_len = len(path[1])
    score = path[2] if path[2] is not None else 0 
    adjusted_score = score / cur_len
    # print("[DEBUG] len: ", cur_len, ", score: ", score, ", adjusted: ", adjusted_score)
    return adjusted_score

def get_bart_score_function():
    def func(path, logprob, da_emb, da_i, beam_size, docs, tgt=None):
        score = get_bart_score(path)
        # print("BARTT score path: ", path)
        return score
    return func

def get_bart_fact_score_function(pretrained_model, factcc_scorer, tokenizer, w1, w2): # TODO FT use array of w instead of parameters
    def func(path, logprob, da_emb, da_i, beam_size, docs, tgt=None):    
        
        docs = convert_id_to_text(pretrained_model, tokenizer, docs)
        summ_hypo = convert_id_to_text(pretrained_model, tokenizer, path[1])
        
        factcc_score = factcc_scorer.classify(docs, summ_hypo)
        bart_score = get_bart_score(path)
        
        w_1 = w2
        w_2 = w1

        score = (w_1 * factcc_score + w_2 * bart_score)/(w_1 + w_2) 
        # print("SKOR Fact MIXED: ", str(score), "|  w1:", w_1," w2:", w_2)

        return score

    return func

def get_rouge_score_function(pretrained_model, scorer, tokenizer):
    def func(path, logprob, da_emb, da_i, beam_size, docs, tgt=None):
        summ_hypo = convert_id_to_text(pretrained_model, tokenizer, path[1])
        tgt_len = len(path[1]) # give the same length to the current hypothesis
        summ_tgt = convert_id_to_text(pretrained_model, tokenizer, tgt[0:tgt_len])
        scores = scorer.get_scores(summ_hypo, summ_tgt, avg=True)
        score = scores['rouge-2']['f']
        
        return score
    return func

def get_factcc_score_function(pretrained_model, scorer, tokenizer):
    def func(path, logprob, da_emb, da_i, beam_size, docs, tgt=None):
        docs = convert_id_to_text(pretrained_model, tokenizer, docs)
        summ_hypo = convert_id_to_text(pretrained_model, tokenizer, path[1])

        # print("docs ", docs)
        # print("summ_hypo ", summ_hypo)

        score = scorer.classify(docs, summ_hypo)
        
        return score
        
    return func

def get_summac_score_function(pretrained_model, tokenizer):
    def func(path, logprob, da_emb, da_i, beam_size, docs, tgt=None):    
        docs = convert_id_to_text(pretrained_model, tokenizer, docs)
        summ_hypo = convert_id_to_text(pretrained_model, tokenizer, path[1])

        start = time()
        score = summac_cls(docs, summ_hypo)
        # print("SKOR SummaC: ", str(score))
        # print("summac scoring: ", time()-start)
        return score

    return func

# DEPRECATED
# def get_mixed_fact_score_function(pretrained_model, fact_scorer, tokenizer, w1, w2): # TODO FT use array of w instead of parameters
#     def func(path, logprob, da_emb, da_i, beam_size, docs, tgt=None):    
#         docs = convert_id_to_text(pretrained_model, tokenizer, docs)
#         summ_hypo = convert_id_to_text(pretrained_model, tokenizer, path[1])
        
#         start = time()
#         summac_score = summac_cls(docs, summ_hypo)
#         # print("summac scoring: ", time()-start)

#         factcc_score = fact_scorer.classify(docs, summ_hypo)
        
#         w_1 = w2
#         w_2 = w1

#         score = (w_1 * factcc_score + w_2 * summac_score)/(w_1 + w_2)  # TODO FT need to train weight
#         # print("SKOR Fact MIXED: ", str(score))

#         return score

#     return func

def get_mixed_fact_score_2_function(pretrained_model, coco_params, factcc_scorer, tokenizer, w1, w2): # TODO FT use array of w instead of parameters
    def func(path, logprob, da_emb, da_i, beam_size, docs, tgt=None):    
        
        docs = convert_id_to_text(pretrained_model, tokenizer, docs)
        summ_hypo = convert_id_to_text(pretrained_model, tokenizer, path[1])
        
        factcc_score = factcc_scorer.classify(docs, summ_hypo)
        coco_score = evaluate_coco(coco_params, docs, summ_hypo)
        
        # print("factcc_score: ", factcc_score)
        # print("coco_score: ", coco_score)
        
        w_1 = w2
        w_2 = w1

        score = (w_1 * factcc_score + w_2 * coco_score)/(w_1 + w_2) 
        # print("SKOR Fact MIXED: ", str(score), "|  w1:", w_1," w2:", w_2)

        return score

    return func

def get_mixed_coco_bart_score_function(pretrained_model, coco_params, tokenizer, w1, w2): # TODO FT use array of w instead of parameters
    def func(path, logprob, da_emb, da_i, beam_size, docs, tgt=None):    
        
        docs = convert_id_to_text(pretrained_model, tokenizer, docs)
        summ_hypo = convert_id_to_text(pretrained_model, tokenizer, path[1])
        
        bart_score = get_bart_score(path)
        coco_score = evaluate_coco(coco_params, docs, summ_hypo)
        
        # print("bart_score: ", bart_score)
        # print("coco_score: ", coco_score)
        
        w_1 = w2
        w_2 = w1

        score = (w_1 * coco_score + w_2 * bart_score)/(w_1 + w_2) 
        # print("SKOR Fact MIXED: ", str(score), "|  w1:", w_1," w2:", w_2)

        return score

    return func

def get_factcc_coco_bart_score_function(pretrained_model, coco_params, factcc_scorer, tokenizer, multi_w): # TODO FT use array of w instead of parameters
    def func(path, logprob, da_emb, da_i, beam_size, docs, tgt=None):    
        
        docs = convert_id_to_text(pretrained_model, tokenizer, docs)
        summ_hypo = convert_id_to_text(pretrained_model, tokenizer, path[1])
        
        factcc_score = factcc_scorer.classify(docs, summ_hypo)
        coco_score = evaluate_coco(coco_params, docs, summ_hypo)
        bart_score = get_bart_score(path)
        
        # print("factcc_score: ", factcc_score)
        # print("coco_score: ", coco_score)
        
        weights = multi_w.split(',')
        w_1 = int(weights[0])
        w_2 = int(weights[1])
        w_3 = int(weights[2])

        # TODO FT add w_3
        score = (w_1 * normalize_score(factcc_score, -1, 1) + w_2 * normalize_score(coco_score, -1, 1) + w_3 * normalize_score(bart_score,0 ,1))/(w_1 + w_2 + w_3) 

        # print("SKOR ALL MIXED: ", str(score), "|  w1:", w_1," w2:", w_2)

        return score

    return func

def get_factcc_coco_bart_rouge_score_function(pretrained_model, coco_params, factcc_scorer, rouge_scorer, tokenizer, multi_w): # TODO FT use array of w instead of parameters
    def func(path, logprob, da_emb, da_i, beam_size, docs, tgt=None):    
        
        docs = convert_id_to_text(pretrained_model, tokenizer, docs)
        summ_hypo = convert_id_to_text(pretrained_model, tokenizer, path[1])
        
        factcc_score = factcc_scorer.classify(docs, summ_hypo)
        coco_score = evaluate_coco(coco_params, docs, summ_hypo)
        bart_score = get_bart_score(path)

        tgt_len = len(path[1]) # give the same length to the current hypothesis
        summ_tgt = convert_id_to_text(pretrained_model, tokenizer, tgt[0:tgt_len])
        
        rouge_scores = rouge_scorer.get_scores(summ_hypo, summ_tgt, avg=True)
        rouge_score = rouge_scores['rouge-2']['f']
        
        
        # print("factcc_score: ", factcc_score)
        # print("coco_score: ", coco_score)
        
        weights = multi_w.split(',')
        w_1 = int(weights[0])
        w_2 = int(weights[1])
        w_3 = int(weights[2])
        w_4 = int(weights[3])

        # TODO FT add w_3
        score = (w_1 * normalize_score(factcc_score, -1, 1) + w_2 * normalize_score(coco_score, -1, 1) + w_3 * normalize_score(bart_score, 0, 1) + w_4 * normalize_score(rouge_score, 0, 1))/(w_1 + w_2 + w_3 + w_4) 

        # print("SKOR ALL MIXED: ", str(score), "|  w1:", w_1," w2:", w_2)

        return score

    return func

def get_rouge_fact_score_function(pretrained_model, fact_scorer, rouge_scorer, tokenizer, w1, w2): # TODO FT use array of w instead of parameters
    def func(path, logprob, da_emb, da_i, beam_size, docs, tgt=None):    
        docs = convert_id_to_text(pretrained_model, tokenizer, docs)
        summ_hypo = convert_id_to_text(pretrained_model, tokenizer, path[1])
        
        tgt_len = len(path[1]) # give the same length to the current hypothesis
        summ_tgt = convert_id_to_text(pretrained_model, tokenizer, tgt[0:tgt_len])
        
        start = time()
        
        factcc_score = fact_scorer.classify(docs, summ_hypo)

        # print("summ_hypo ", summ_hypo)
        # print("summ_tgt ", summ_tgt)
        rouge_scores = rouge_scorer.get_scores(summ_hypo, summ_tgt, avg=True)
        rouge_score = rouge_scores['rouge-2']['f']
        
        w_1 = w2
        w_2 = w1

        score = (w_1 * normalize_score(factcc_score, -1, 1) + w_2 * normalize_score(rouge_score, 0, 1))/(w_1 + w_2)  # TODO FT need to train weight
        # print("SKOR Fact MIXED: ", str(score))

        return score

    return func

# Function to get base scores: ROUGE and LP
def get_baseline(pretrained_model, tokenizer, tgt, summ_hypo, rouge_scorer, path):
    score = 0

    tgt_len = len(path[1]) # give the same length to the current hypothesis
    summ_tgt = convert_id_to_text(pretrained_model, tokenizer, tgt[0:tgt_len])
    
    rouge_scores = rouge_scorer.get_scores(summ_hypo, summ_tgt, avg=True)
    rouge_score = rouge_scores['rouge-2']['f'] # or AVG of all R-1, R-2, R-3 scores?
    bart_score = get_bart_score(path)

    if (bart_score==0):
        bart_score = rouge_score # TODO mana yg oke?

    score = rouge_score + bart_score / 2

    return score

def get_score_baseline_fact(pretrained_model, rouge_scorer, tokenizer, w1=0, w2=0):
    def func(path, logprob, da_emb, da_i, beam_size, docs, tgt=None):   
        summ_hypo = convert_id_to_text(pretrained_model, tokenizer, path[1])
        score = get_baseline(pretrained_model, tokenizer, tgt, summ_hypo, rouge_scorer, path)
        # print("score baseline: ", score)
        return score

    return func

def get_score_completed_function_fact(pretrained_model, coco_params, factcc_scorer, tokenizer, multi_w):
    def func(path, logprob, da_emb, da_i, beam_size, docs, tgt=None):    
        score = 0
        docs = convert_id_to_text(pretrained_model, tokenizer, docs)
        summ_hypo = convert_id_to_text(pretrained_model, tokenizer, path[1])
        
        tgt_len = len(path[1]) # give the same length to the current hypothesis
        summ_tgt = convert_id_to_text(pretrained_model, tokenizer, tgt[0:tgt_len])
        

        return score
    return func


def normalize_score(score, min, max):
    return (score - min) / (max - min)


def get_score_function_fact(args, scorer, cfg, summ_data, true_summ, beam_size, alpha=0.65, summary_embedder=None, document_embedder=None, lens=None):
    print("Using Scorer: {}".format(scorer))

    pretrained_model = args.pretrained_model

    symbols = {}
    tokenizer = None

    if (pretrained_model=='presumm'):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir=args.temp_dir)
        symbols = {'BOS': tokenizer.vocab['[unused0]'], 'EOS': tokenizer.vocab['[unused1]'],
                'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused2]']}

    if (pretrained_model=='bart'):
        tokenizer = get_bart_tokenizer(args)

    # convert docs and hypo to text
    if scorer == "factcc":
        factcc = FactccCaller()
        return get_factcc_score_function(pretrained_model, factcc, tokenizer)
    elif scorer == "summac":
        return get_summac_score_function(pretrained_model, tokenizer)
    elif scorer == "fact_mixed":
        factcc = FactccCaller()
        coco_params = initialize_coco()
        return get_mixed_fact_score_2_function(pretrained_model, coco_params, factcc, tokenizer, args.w1, args.w2)
    elif scorer == "rouge":
        rouge_model = Rouge()
        return get_rouge_score_function(pretrained_model, rouge_model, tokenizer)
    elif scorer == "fact_rouge":
        factcc = FactccCaller()
        rouge_model = Rouge()
        return get_rouge_fact_score_function(pretrained_model, factcc, rouge_model, tokenizer, args.w1, args.w2)
    elif scorer == "bart_penalty":
        return get_bart_score_function()
    elif scorer == "fact_bart":
        factcc = FactccCaller()
        return get_bart_fact_score_function(pretrained_model, factcc, tokenizer, args.w1, args.w2)
    elif scorer == 'coco_bart':
        coco_params = initialize_coco()
        return get_mixed_coco_bart_score_function(pretrained_model, coco_params,tokenizer, args.w1, args.w2)
    elif scorer == 'fact_coco_bart':
        factcc = FactccCaller()
        coco_params = initialize_coco()
        return get_factcc_coco_bart_score_function(pretrained_model, coco_params, factcc, tokenizer, args.multi_w)
    elif scorer == 'fact_coco_bart_rouge':
        factcc = FactccCaller()
        coco_params = initialize_coco()
        rouge_model = Rouge()
        return get_factcc_coco_bart_rouge_score_function(pretrained_model, coco_params, factcc, rouge_model, tokenizer, args.multi_w)
    elif scorer == "weighted_fact":
        print("TODO")
    elif scorer == "baseline":
        rouge_model = Rouge()
        return get_score_baseline_fact(pretrained_model, rouge_model, tokenizer)
        print("TODO")
    elif scorer == "surrogate_fact":
        # TODO FT use bart tokenizer below
        learned = SummaryFactTrainableReranker(summary_embedder, document_embedder, cfg['trainable_reranker_config'], tokenizer=tokenizer, pretrained_model=args.pretrained_model)
        learned.load_model()
        select_max = cfg.get("order_by_max_class", False)
        reverse_order = scorer == 'surrogate_rev'
        
        if (pretrained_model == 'presumm'):
            len_summ = max([len(x[0]) for x in summary_embedder])
            len_docs = max([len(x[0]) for x in document_embedder])
            pad_symbol = symbols[SUMM_PAD_TOK]

        elif (pretrained_model == 'bart'):
            # len_summ = lens['max_len_summ']
            # len_docs = lens['max_len_docs']
            len_summ = max([len(x) for x in summary_embedder])
            len_docs = max([len(x) for x in document_embedder])
            pad_symbol = BART_PAD_TOKEN

        print("len_summ ", len_summ)
        print("len_docs ", len_docs)
        return get_learned_fact_score_func(learned, select_max, reverse_order, pad_symbol, len_summ, len_docs)
    else:
        raise ValueError("Unknown Scorer {}".format(cfg['scorer']))
