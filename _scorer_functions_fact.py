import random
import numpy as np
from keras.utils import to_categorical

from utils import START_TOK, END_TOK, PAD_TOK, get_features, SUMM_PAD_TOK

from _base_models_fact import TGEN_Reranker, TrainableReranker, SummaryFactTrainableReranker

from fact_scorer.fact_factcc.factcc_caller_model import FactccCaller
from fact_scorer.fact_summac.summac_caller import classify as summac_cls
from rouge import Rouge

from pytorch_transformers import BertTokenizer

from time import time 

from _bart_utils import get_bart_tokenizer, convert_ids_to_text, BART_XSUM_MODEL

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

def get_learned_fact_score_func(trainable_reranker, select_max=False, reverse_order=False, summ_symbols=None, len_summ=None, len_docs=None):
    def func(path, logprob, da_emb, da_i, beam_size, docs, tgt=None):
        summ_emb = path[1]
        pads = [summ_symbols[SUMM_PAD_TOK]] * \
               (len_summ - len(summ_emb))
        if trainable_reranker.logprob_preprocess_type == 'categorical_order':
            logprob_rank = logprob * trainable_reranker.beam_size // beam_size
            logprob_val = to_categorical([logprob_rank], num_classes=trainable_reranker.beam_size)
        else:
            logprob_val = [path[0]]

        summ_seqs = [pads + summ_emb.cpu().tolist()]
        summ_seqs = [summ_seqs[0][:len_summ]]

        docs_emb = docs[0].cpu().tolist()
        docs_pads = [summ_symbols[SUMM_PAD_TOK]] * \
               (len_docs - len(docs_emb))
        docs_seqs = [docs_pads + docs_emb]
        docs_seqs = [docs_seqs[0][:len_docs]]

        pred = trainable_reranker.predict_fact_score(
            np.array(summ_seqs),
            np.array(docs_seqs),
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


def convert_id_to_text(pretrained_model, tokenizer, token_ids):
    text = ""
    if (pretrained_model=='presumm'):
        # Convert token_ids to text for factual consistency scoring
        text = tokenizer.convert_ids_to_tokens([int(n) for n in token_ids])
        text = ' '.join(text).replace(' ##','')
        text = text.replace('[unused0]', '').replace('[unused3]', '').replace('[PAD]', '').replace('[unused1]', '').replace(r' +', ' ').replace(' [unused2] ', '<q>').replace('[unused2]', '').strip()
    elif (pretrained_model=='bart'):
        text = convert_ids_to_text(tokenizer, token_ids)
    
    return text


def get_rouge_score_function(pretrained_model, scorer, tokenizer):
    def func(path, logprob, da_emb, da_i, beam_size, docs, tgt=None):
        summ_hypo = convert_id_to_text(pretrained_model, tokenizer, path[1])
        summ_tgt = convert_id_to_text(pretrained_model, tokenizer, tgt)
        scores = scorer.get_scores(summ_hypo, tgt, avg=True)
        score = scores['rouge-l']['f']
        
        return score
    return func

def get_factcc_score_function(pretrained_model, scorer, tokenizer):
    def func(path, logprob, da_emb, da_i, beam_size, docs, tgt=None):
        docs = convert_id_to_text(pretrained_model, tokenizer, docs[0])
        summ_hypo = convert_id_to_text(pretrained_model, tokenizer, path[1])
        score = scorer.classify(docs, summ_hypo)
        
        return score
        
    return func

def get_summac_score_function(pretrained_model, tokenizer):
    def func(path, logprob, da_emb, da_i, beam_size, docs, tgt=None):    
        docs = convert_id_to_text(pretrained_model, tokenizer, docs[0])
        summ_hypo = convert_id_to_text(pretrained_model, tokenizer, path[1])

        start = time()
        score = summac_cls(docs, summ_hypo)
        # print("SKOR SummaC: ", str(score))
        # print("summac scoring: ", time()-start)
        return score

    return func

def get_mixed_fact_score_function(pretrained_model, fact_scorer, tokenizer, w1, w2): # TODO FT use array of w instead of parameters
    def func(path, logprob, da_emb, da_i, beam_size, docs, tgt=None):    
        docs = convert_id_to_text(pretrained_model, tokenizer, docs[0])
        summ_hypo = convert_id_to_text(pretrained_model, tokenizer, path[1])
        
        start = time()
        summac_score = summac_cls(docs, summ_hypo)
        # print("summac scoring: ", time()-start)

        factcc_score = fact_scorer.classify(docs, summ_hypo)
        
        w_1 = w1
        w_2 = w2

        score = (w_1 * factcc_score + w_2 * summac_score)/(w_1 + w_2)  # TODO FT need to train weight
        # print("SKOR Fact MIXED: ", str(score))

        return score

    return func

def get_rouge_fact_score_function(pretrained_model, fact_scorer, rouge_scorer, tokenizer, w1, w2): # TODO FT use array of w instead of parameters
    def func(path, logprob, da_emb, da_i, beam_size, docs, tgt=None):    
        docs = convert_id_to_text(pretrained_model, tokenizer, docs[0])
        summ_hypo = convert_id_to_text(pretrained_model, tokenizer, path[1])
        summ_tgt = convert_id_to_text(pretrained_model, tokenizer, tgt)
        
        start = time()
        
        factcc_score = fact_scorer.classify(docs, summ_hypo)

        rouge_scores = rouge_scorer.get_scores(summ_hypo, summ_tgt, avg=True)
        rouge_score = rouge_scores['rouge-l']['f']
        
        w_1 = w1
        w_2 = w2

        score = (w_1 * factcc_score + w_2 * rouge_score)/(w_1 + w_2)  # TODO FT need to train weight
        # print("SKOR Fact MIXED: ", str(score))

        return score

    return func

def get_score_function_fact(args, scorer, cfg, summ_data, true_summ, beam_size, alpha=0.65, summary_embedder=None, document_embedder=None):
    print("Using Scorer: {}".format(scorer))

    pretrained_model = args.pretrained_model

    symbols = {}
    tokenizer = None

    if (pretrained_model=='presumm'):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir=args.temp_dir)
        symbols = {'BOS': tokenizer.vocab['[unused0]'], 'EOS': tokenizer.vocab['[unused1]'],
                'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused2]']}

    if (pretrained_model=='bart'):
        tokenizer = get_bart_tokenizer(BART_XSUM_MODEL)

    # convert docs and hypo to text
    if scorer == "factcc":
        factcc = FactccCaller()
        return get_factcc_score_function(pretrained_model, factcc, tokenizer)
    elif scorer == "summac":
        return get_summac_score_function(pretrained_model, tokenizer)
    elif scorer == "fact_mixed":
        factcc = FactccCaller()
        return get_mixed_fact_score_function(pretrained_model, factcc, tokenizer, args.w1, args.w2)
    elif scorer == "rouge":
        rouge = Rouge()
        return get_rouge_score_function(pretrained_model, rouge, tokenizer)
    elif scorer == "fact_rouge":
        factcc = FactccCaller()
        rouge = Rouge()
        return get_rouge_fact_score_function(pretrained_model, factcc, rouge, tokenizer, args.w1, args.w2)
    elif scorer == "weighted_fact":
        print("TODO")
    elif scorer == "surrogate_fact":
        # TODO FT use bart tokenizer below
        learned = SummaryFactTrainableReranker(summary_embedder, document_embedder, cfg['trainable_reranker_config'], tokenizer=tokenizer)
        learned.load_model()
        select_max = cfg.get("order_by_max_class", False)
        reverse_order = scorer == 'surrogate_rev'

        len_summ = max([len(x[0]) for x in summary_embedder])
        len_docs = max([len(x[0]) for x in document_embedder])
        return get_learned_fact_score_func(learned, select_max, reverse_order, symbols, len_summ, len_docs)
    else:
        raise ValueError("Unknown Scorer {}".format(cfg['scorer']))
