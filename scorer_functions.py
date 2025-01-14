import random
import numpy as np
from keras.utils import to_categorical

from base_models import TGEN_Reranker, TrainableReranker
from e2e_metrics.metrics.pymteval import BLEUScore
from utils import START_TOK, END_TOK, PAD_TOK, get_features

from fact_scorer.fact_factcc.factcc_caller_model import FactccCaller
from fact_scorer.fact_summac.summac_caller import classify as summac_cls
from pytorch_transformers import BertTokenizer



def get_regressor_score_func(regressor, text_embedder, w2v):
    def func(path, logprob, da_emb, da_i, beam_size):
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
    def func(path, logprob, da_emb, da_i, beam_size):
        print("1 PATH: ",str(len(path)))
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
    def func(path, logprob, da_emb, da_i, beam_size):
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
    def func(path, logprob, da_emb, da_i, beam_size):
        true = true_vals[da_i]
        toks = text_embedder.reverse_embedding(path[1])
        pred = [x for x in toks if x not in [START_TOK, END_TOK, PAD_TOK]]
        bleu.reset()
        bleu.append(pred, true)
        if reverse:
            return 1 - bleu.score()
        return bleu.score()

    return func


def get_random_score_func():
    def func(path, logprob, da_emb, da_i, beam_size):
        return random.random()

    return func


def get_learned_score_func(trainable_reranker, select_max=False, reverse_order=False):
    def func(path, logprob, da_emb, da_i, beam_size):
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
    elif scorer in ['oracle', 'rev_oracle']:
        bleu_scorer = BLEUScore()
        return get_oracle_score_func(bleu_scorer, true_vals, text_embedder, reverse=(scorer == 'rev_oracle'))
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