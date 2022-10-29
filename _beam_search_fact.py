import pickle
from collections import defaultdict, Counter
from itertools import product
from math import log

from gensim.models import Word2Vec
import random
import sys
import os

# from importlib import reload
# reload(sys)
# sys.setdefaultencoding('utf-8')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from time import time
import numpy as np
import yaml
import tensorflow as tf
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from tqdm import tqdm

from embedding_extractor import TokEmbeddingSeq2SeqExtractor, DAEmbeddingSeq2SeqExtractor
from _scorer_functions_fact import get_identity_score_func
from utils import get_texts_training, RERANK, get_training_das_texts, safe_get_w2v, apply_absts, PAD_TOK, END_TOK, \
    START_TOK, get_section_cutoffs, get_section_value, get_regression_vals, get_timestamp_file

import torch
from pytorch_transformers import BertTokenizer

# Import BART
HOME_REPO = "/home/lr/faza.thirafi/raid/repository-kenkyuu-models/"
sys.path.insert(0, HOME_REPO + "transformers/")
from _bart_utils import load_bart_dataset, load_bart_model, convert_ids_to_text, BART_ENCODER_MAX_LENGTH, BART_START_TOKEN, \
    beam_search_expand_single_bart, finalize_beam_search_expand_single_bart, get_bart_tokenizer, \
    BART_XSUM_MODEL

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import random

MAX_LEN = 150

SUMM_END_TOKENS = ['EOS', 2]
 
def score_beams_fact(rescorer, beam, da_emb, i, docs, summ_tgt):
    path_scores = []
    logprobs = [x[0] for x in beam]
    for path in beam:
        # lp_pos = sum([1 for lp in logprobs if lp > path[0] + 0.000001])
        lp_pos = 0

        hyp_score = rescorer(path, lp_pos, da_emb, i, len(beam), docs, summ_tgt)
        path_scores.append((hyp_score, path))
    return path_scores

recorded_sections = []


# Ignore flags is a horrible hack to get non-greedy-rescorers not to use sections/pairwise flags of greedy
# TODO pass in the value of the flags and then can controll these at a level where can distinguish between
# greedy and non greedy
def order_beam_acording_to_rescorer(rescorer, beam, da_emb, i, cfg, out_beam=None, ignore_flags=False, summ_data=None, summ_tgt=None):
    
    # this only works if rescorer is the one used in cfg
    global recorded_sections
    if "train_reranker" in cfg:
        sections_flag = cfg["train_reranker"]["output_type"] in ['regression_sections'] and not ignore_flags
        pairwise_flag = cfg["train_reranker"]["output_type"] in ['pair'] and not ignore_flags
    else:
        sections_flag = False
        pairwise_flag = False

    if sections_flag:
        num_ranks = cfg["train_reranker"]["num_ranks"]
        cut_offs = get_section_cutoffs(num_ranks)
        regression_vals = get_regression_vals(num_ranks, cfg["train_reranker"]["with_refs_train"])
        if cfg["train_reranker"]["with_refs_train"]:
            NotImplementedError()

        scored_finished_beams = score_beams_fact(rescorer, beam, da_emb, i, docs, summ_tgt)

        mms = cfg["train_reranker"]["merge_middle_sections"]
        ot = cfg["train_reranker"]["only_top"]
        ob = cfg["train_reranker"]["only_bottom"]
        av = sum([x for (x, _), _ in scored_finished_beams]) / len(scored_finished_beams)
        sections = [get_section_value(x - av + 0.5, cut_offs, regression_vals, mms, ot, ob) for (x, _), _ in
                    scored_finished_beams]

        recorded_sections.extend(sections)
        path_scores = [((1 - x, y[1]), z) for x, (y, z) in zip(sections, scored_finished_beams)]
    elif pairwise_flag:
        print("TODO FT removed pairwise")
    else:
        path_scores = score_beams_fact(rescorer, beam, da_emb, i, summ_data, summ_tgt)

    order = sorted(enumerate(path_scores), reverse=True, key=lambda x: x[1][0])

    # if i == 0 and sections_flag:
        # print("Path scores:", [x for _, (x, _) in order])
        # print([x for x, _ in scored_finished_beams])
    if out_beam is not None:
        beam = out_beam
    result = [beam[i] for i, _ in order]
    return result


def order_beam_after_greedy_complete(rescorer, beam, da_emb, i, enc_outs, seq2seq, max_pred_len, cfg, length_norm_alpha=None, 
    summ_data=None, summ_model=None, summ_enc_outs=None, beam_size=None, summ_tgt=None):
    finished_beam = beam.copy()
    toks_pred_so_far = max([len(x[1]) for x in beam])
    # print("max_pred_len: ", max_pred_len)
    # print("toks_pred_so_far: ", toks_pred_so_far)
    # print("finished beam: \n", finished_beam)
    for step in range(max_pred_len - toks_pred_so_far):
        finished_beam = summ_model.beam_search_expand(finished_beam, 1, step, summ_data, summ_enc_outs)
        
        if all([p[1][-1] in SUMM_END_TOKENS for p in finished_beam]):
            break
    result = order_beam_acording_to_rescorer(rescorer, finished_beam, da_emb, i, cfg, beam, summ_data=summ_data, summ_tgt=summ_tgt)
    return result

def order_beam_after_greedy_complete_bart(rescorer, beam, i, max_pred_len, cfg, length_norm_alpha=None, 
    summ_data=None, summ_model=None, summ_enc_outs=None, beam_size=None, summ_tgt=None, bart_params=None, model_kwargs=None):
    finished_beam = beam.copy()
    toks_pred_so_far = max([len(x[1]) for x in beam])
    
    for step in range(max_pred_len - toks_pred_so_far):
        
        # finished_beam = summ_model.beam_search_expand_single(finished_beam, 1, step, summ_data, summ_enc_outs)
        # TODO FT 
        new_paths, return_params, input_ids, model_kwargs = \
            beam_search_expand_single_bart(summ_model, finished_beam, beam_size, summ_data, bart_params, **model_kwargs)

        if all([p[1][-1] in SUMM_END_TOKENS for p in finished_beam]):
            break
    result = order_beam_acording_to_rescorer(rescorer, finished_beam, da_emb, i, cfg, beam, summ_data=summ_data, summ_tgt=summ_tgt)

    return result

def _run_beam_search_with_rescorer(args, i, da_emb, paths, enc_outs, beam_size, max_pred_len, seq2seq, cfg,
                                   rescorer=None, greedy_complete=[],
                                   save_progress_file=None, non_greedy_rescorer=None, length_norm_alpha=None, 
                                   summ_model=None, summ_data=None, summ_enc_outs=None, summ_paths=None, summ_tgt=None):
    for step in range(max_pred_len):
        # print("STEP KE- ", str(step))
        # expand
        # if type(summ_paths)=='list':
        #     summ_paths = torch.stack(summ_paths)

        # print("step MAIN: ",step)
        # print("PATHS: ", summ_paths)
        new_paths = summ_model.beam_search_expand(summ_paths, beam_size, step, summ_data, summ_enc_outs)

        # prune
        # randomize whether we use scoring or not
        if (random.randint(0,9) % 2):
            if step in greedy_complete and rescorer is not None:
                summ_paths = order_beam_after_greedy_complete(rescorer, new_paths, da_emb, i, enc_outs, seq2seq, max_pred_len,
                                                        cfg, length_norm_alpha, 
                                                        summ_data=summ_data, summ_model=summ_model, summ_enc_outs=summ_enc_outs, beam_size=beam_size)
            elif step not in greedy_complete and non_greedy_rescorer is not None and cfg['non_greedy_scorer'] != 'identity':
                summ_paths = order_beam_acording_to_rescorer(non_greedy_rescorer, new_paths, da_emb, i, cfg, ignore_flags=True, summ_data=summ_data, summ_tgt=summ_tgt)
            elif not greedy_complete and rescorer is not None and cfg['scorer'] != 'identity':
                summ_paths = order_beam_acording_to_rescorer(rescorer, new_paths, da_emb, i, cfg, summ_data=summ_data, summ_tgt=summ_tgt)
            else:
                summ_paths = sorted(new_paths, reverse=True, key=lambda x: x[1][0])
        else:
            # summ_paths = new_paths
            summ_paths = sorted(new_paths, reverse=True, key=lambda x: x[1][0])
            # print("Skip path scoring")
        
        summ_paths = summ_paths[:beam_size]
        # print("NEW PATH: ", new_paths)
        # print("SAVE? ",save_progress_file)
        if save_progress_file:
            # print("SAVE!")
            save_progress_file.write("Step: {}\n".format(step))
            for path in summ_paths:
                # toks = [x for x in seq2seq.text_embedder.reverse_embedding(path[1]) if x != PAD_TOK]
                toks = summ_model.convert_id_to_text(path[1])
                # print("save: ",str(toks))
                # save_progress_file.write(" ".join(toks.encode('utf-8').decode('utf-8')) + '\n')
                # print(type(toks))
                save_progress_file.write(" ".join(str(toks.encode('utf-8', 'ignore'))) + '\n')
            save_progress_file.write("\n")

        if all([p[1][-1] in SUMM_END_TOKENS for p in summ_paths]):
            break

    return summ_paths

# TODO FT merge procedure later
def _run_beam_search_with_rescorer_bart(args, i, beam_size, max_pred_len, cfg,
                                        rescorer=None, greedy_complete=[],
                                        save_progress_file=None, non_greedy_rescorer=None, length_norm_alpha=None, 
                                        summ_model=None, summ_data=None, summ_paths=None, summ_tgt=None,
                                        bart_params=None, bart_tokenizer=None, **model_kwargs):
    params = bart_params.copy()
    for step in range(max_pred_len):
        # new_paths = summ_model.beam_search_expand_single(summ_paths, beam_size, step, summ_data, summ_enc_outs)
        new_paths, return_params, input_ids, model_kwargs = \
            beam_search_expand_single_bart(summ_model, summ_paths, beam_size, summ_data, params, **model_kwargs)
        
        params = return_params

        print("input_ids: ", input_ids)
        # prune
        # randomize whether we use scoring or not
        if (random.randint(0,9) % 2):
            if step in greedy_complete and rescorer is not None:
                summ_paths = order_beam_after_greedy_complete_bart(rescorer, new_paths, i, max_pred_len, cfg, length_norm_alpha, 
                                                        summ_data=summ_data, summ_model=summ_model, summ_enc_outs=summ_enc_outs, beam_size=beam_size,
                                                        bart_params=params, **model_kwargs)
            elif step not in greedy_complete and non_greedy_rescorer is not None and cfg['non_greedy_scorer'] != 'identity':
                summ_paths = order_beam_acording_to_rescorer(non_greedy_rescorer, new_paths, None, i, cfg, ignore_flags=True, summ_data=summ_data, summ_tgt=summ_tgt)
            elif not greedy_complete and rescorer is not None and cfg['scorer'] != 'identity':
                summ_paths = order_beam_acording_to_rescorer(rescorer, new_paths, None, i, cfg, summ_data=summ_data, summ_tgt=summ_tgt)
            else:
                summ_paths = sorted(new_paths, reverse=True, key=lambda x: x[1][0])
        else:
            summ_paths = sorted(new_paths, reverse=True, key=lambda x: x[1][0])
        
        summ_paths = summ_paths[:beam_size]
        
        if save_progress_file:
            # print("SAVE!")
            save_progress_file.write("Step: {}\n".format(step))
            for path in summ_paths:
                # toks = [x for x in seq2seq.text_embedder.reverse_embedding(path[1]) if x != PAD_TOK]
                toks = convert_ids_to_text(bart_tokenizer, path[1])
                # print("save: ",str(toks))
                # save_progress_file.write(" ".join(toks.encode('utf-8').decode('utf-8')) + '\n')
                # print(type(toks))
                save_progress_file.write(" ".join(str(toks.encode('utf-8', 'ignore'))) + '\n')
            save_progress_file.write("\n")

        if all([p[1][-1] in SUMM_END_TOKENS for p in summ_paths]):
            break

    return summ_paths

def run_beam_search_with_rescorer(args, scorer, beam_search_model, das, beam_size, cfg, only_rerank_final=False,
                                  save_final_beam_path='', greedy_complete=[], max_pred_len=MAX_LEN, save_progress_path=None,
                                  also_rerank_final=False, non_greedy_rescorer=None, length_norm_alpha=None,
                                  summ_scorer=None, summ_beam_search_model=None, summ_data=None, device='cpu', len_summ_data=None):
    global recorded_sections
    recorded_sections = []
    save_final_beam_path_toggle = False
    if save_progress_path is not None:
        # save_progress_file = open(save_progress_path.format(beam_size), 'w+')
        save_progress_file = open(save_progress_path.format(args.use_dataset, args.pretrained_model, cfg["scorer"], beam_size), 'w+')
    else:
        save_progress_file = None

    pred_results = []
    src_data = []
    tgt_data = []
    
    should_skip_beam = args.should_skip_beam
    
    should_load_beams = save_final_beam_path and os.path.exists(save_final_beam_path) and only_rerank_final
    should_save_beams = save_final_beam_path and not should_load_beams
    
    load_final_beams = []
    final_beams = []
    if should_save_beams and os.path.exists(save_final_beam_path):
        print("Loading partial beams from", save_final_beam_path)
        final_beams = pickle.load(open(save_final_beam_path, "rb"))
        print("Loaded {} from saved final beams".format(len(final_beams)))

    if should_load_beams:
        print("Loading beams from", save_final_beam_path)
        load_final_beams = pickle.load((open(save_final_beam_path, "rb")))

    start = time()
    print("Start generating")
    
    print("save_final_beam_path: ", save_final_beam_path)
    print("Final beam: ",len(final_beams))
    
    counted = False
    i = 0

    batch_skipped = 0
    remaining = 0
    batch_size = 0

    bart_tokenizer = get_bart_tokenizer()
    
    if (args.pretrained_model=='presumm'):
        with torch.no_grad():
            for batch in summ_data:
                if not(counted):
                    batch_size = len(batch.src)

                    batch_skipped = int(len(final_beams)/batch_size)
                    remaining = len(final_beams) % batch_size
                    counted = True

                if should_skip_beam and i <= batch_skipped:
                    i += 1 # TODO FT revamp this hack
                    continue
                
                j = 0
                for src, segs, mask_src, tgt in zip(batch.src, batch.segs, batch.mask_src, batch.tgt):
                    if (should_skip_beam and i == batch_skipped+1 and j < remaining):
                        print("SKIP j: ", (i * batch_size) + j)
                    else:
                        print("Process summ_data: ke-", (i * batch_size) + j)

                        src = src.view(1, -1)
                        segs = segs.view(1, -1)
                        mask_src = mask_src.view(1, -1)
                        
                        if save_progress_file:
                            save_progress_file.write("Test {}\n".format(i))

                        summ_enc_outs = summ_beam_search_model.encode_batch(src, segs, mask_src)
                        
                        # inf_enc_out = beam_search_model.encoder_model.predict(np.array([da_emb]))
                        # enc_outs = inf_enc_out[0]
                        # enc_last_state = inf_enc_out[1:]
                        # paths = [(log(1.0), text_embedder.start_emb, enc_last_state)]
                        
                        summ_enc_out = summ_enc_outs[0]
                        summ_enc_last_state = summ_enc_outs[1:]
                        summ_paths = [(
                            log(1.0),
                            torch.tensor(
                                [summ_beam_search_model.start_token],
                                dtype=torch.long,
                                device=device
                            ),
                            summ_enc_last_state
                        )]
                            
                        if should_load_beams:
                            paths = load_final_beams[(i * batch_size) + j]
                        else:
                            paths = _run_beam_search_with_rescorer(
                                args, 
                                i=i,
                                da_emb=None,
                                paths=None,
                                enc_outs=None,
                                beam_size=beam_size,
                                max_pred_len=max_pred_len,
                                seq2seq=beam_search_model,
                                rescorer=scorer if not only_rerank_final else None,
                                greedy_complete=greedy_complete,
                                save_progress_file=save_progress_file,
                                cfg=cfg,
                                non_greedy_rescorer=non_greedy_rescorer,
                                length_norm_alpha=length_norm_alpha,
                                summ_model=summ_beam_search_model,
                                summ_data=src,
                                summ_enc_outs=summ_enc_outs,
                                summ_paths=summ_paths,
                                summ_tgt=tgt
                            )

                        final_beams.append(paths)

                        if only_rerank_final or also_rerank_final:
                            paths = order_beam_acording_to_rescorer(scorer, paths, None, i, cfg, summ_data=src, summ_tgt=tgt)
                            
                        # A hack to handle the what we need right now - this should be updated
                        elif non_greedy_rescorer:
                            # TODO FT PARTIAL
                            paths = order_beam_acording_to_rescorer(non_greedy_rescorer, paths, None, i, cfg, ignore_flags=True, summ_data=src, summ_tgt=tgt)

                        best_path = paths[0]
                        
                        pred_toks = summ_beam_search_model.convert_id_to_text(best_path[1])

                        src_toks = summ_beam_search_model.convert_id_to_text(src[0])
                        tgt_toks = summ_beam_search_model.convert_id_to_text(tgt)
                
                        # print("BEST")
                        # print(pred_toks)
                        pred_results.append(pred_toks)
                        src_data.append(src_toks)
                        tgt_data.append(tgt_toks)

                        save_final_beam_path_toggle = not save_final_beam_path_toggle

                        # TODO fix hack on len, investigate why
                        if should_save_beams and (i % 100 == 0 or len(final_beams) == len_summ_data or len(final_beams) == len_summ_data-1 or len(final_beams) == len_summ_data-2):
                            print("Saving final beam states at ", save_final_beam_path)
                            if save_final_beam_path_toggle:
                                toggledPath = save_final_beam_path
                            else:
                                splits = save_final_beam_path.split('.')
                                toggledPath = splits[0] + '.' + splits[1]
                            pickle.dump(final_beams, open(toggledPath, "wb+"))

                    j += 1
                i += 1
    elif (args.pretrained_model=='bart'):
        with torch.no_grad():
            model_name = "sshleifer/distilbart-xsum-12-3" # TODO FT store it somewhere
            tokenizer = get_bart_tokenizer(model_name)

            i = len(final_beams)
            
            # TODO FT work on multiprocessing here
            for src, tgt in zip(summ_data["document"][len(final_beams):], summ_data["summary"][len(final_beams):]):
                i += 1 
                # src = batch["document"]
                # tgt = batch["summary"]
                # print("src: ", src)
                inputs = tokenizer(
                    [src],
                    padding="max_length",
                    truncation=True,
                    max_length=BART_ENCODER_MAX_LENGTH,
                    return_tensors="pt",
                )
                input_ids = inputs.input_ids.to(summ_beam_search_model.device)
                # print(len(input_ids), "  ","input_ids ", input_ids)
                attention_mask = inputs.attention_mask.to(summ_beam_search_model.device)

                params, model_kwargs = summ_beam_search_model.generate_custom_beam(
                    input_ids, attention_mask=attention_mask
                )

                # summ_enc_out = summ_enc_outs[0]
                # summ_enc_last_state = summ_enc_outs[1:]
                summ_paths = [(
                    log(1.0),
                    torch.tensor(
                        [BART_START_TOKEN],
                        dtype=torch.long,
                        device=device
                    )
                )]
                    
                if should_load_beams:
                    paths = load_final_beams[(i * batch_size) + j]
                else:
                    paths = _run_beam_search_with_rescorer_bart(
                        args, 
                        i=i,
                        beam_size=beam_size,
                        max_pred_len=max_pred_len,
                        rescorer=scorer if not only_rerank_final else None,
                        greedy_complete=greedy_complete,
                        save_progress_file=save_progress_file,
                        cfg=cfg,
                        non_greedy_rescorer=non_greedy_rescorer,
                        length_norm_alpha=length_norm_alpha,
                        summ_model=summ_beam_search_model,
                        summ_data=src,
                        summ_paths=summ_paths,
                        summ_tgt=tgt,
                        bart_params=params,
                        bart_tokenizer=bart_tokenizer,
                        **model_kwargs
                    )

                final_beams.append(paths)

                if only_rerank_final or also_rerank_final:
                    paths = order_beam_acording_to_rescorer(scorer, paths, None, i, cfg, summ_data=src, summ_tgt=tgt)
                    
                # A hack to handle the what we need right now - this should be updated
                elif non_greedy_rescorer:
                    # TODO FT PARTIAL
                    paths = order_beam_acording_to_rescorer(non_greedy_rescorer, paths, None, i, cfg, ignore_flags=True, summ_data=src, summ_tgt=tgt)

                best_path = paths[0]
                
                pred_toks = convert_ids_to_text(bart_tokenizer, best_path[1])

                src_toks = convert_ids_to_text(bart_tokenizer, src[0])
                tgt_toks = convert_ids_to_text(bart_tokenizer, tgt)

                pred_results.append(pred_toks)
                src_data.append(src_toks)
                tgt_data.append(tgt_toks)

                save_final_beam_path_toggle = not save_final_beam_path_toggle

                if should_save_beams and (i % 100 == 0 or len(final_beams) == len_summ_data or len(final_beams) == len_summ_data-1 or len(final_beams) == len_summ_data-2):
                    print("Saving final beam states at ", save_final_beam_path)
                    if save_final_beam_path_toggle:
                        toggledPath = save_final_beam_path
                    else:
                        splits = save_final_beam_path.split('.')
                        toggledPath = splits[0] + '.' + splits[1]
                    pickle.dump(final_beams, open(toggledPath, "wb+"))

    
    print("*** Time to generate text =", time() - start)


    if recorded_sections:
        print("SECTIONS:", Counter(recorded_sections))

    return pred_results, src_data, tgt_data
