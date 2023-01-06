import os
import re
import sys
from collections import defaultdict
from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False

import h5py
import nltk
# from keras.engine.saving import load_weights_from_hdf5_group
from tensorflow.python.keras import saving

import numpy as np
import json

sys.path.append(os.path.join(os.getcwd(), 'tgen'))
from enum import Enum
# from tgen.futil import read_das
# from tgen.futil import "smart_load_absts"
from regex import Regex, UNICODE, IGNORECASE

import time
import calendar

from pytorch_transformers import BertTokenizer

# Import BART
HOME_REPO = "/home/lr/faza.thirafi/raid/repository-kenkyuu-models/"
sys.path.insert(0, HOME_REPO + "transformers/")
from src.transformers.models.auto.tokenization_auto import AutoTokenizer

START_TOK = '<S>'
END_TOK = '<E>'
PAD_TOK = '<>'
RESULTS_DIR = 'output_files/out-text-dir-v3'
SUMM_RESULTS_DIR = 'output_files/out-text-dir-summ'
CONFIGS_DIR = 'new_configs'
CONFIGS_MODEL_DIR = 'new_configs/model_configs'
TRAIN_BEAM_SAVE_FORMAT = 'output_files/saved_beams/train_vanilla_{}_{}.pickle'
TEST_BEAM_SAVE_FORMAT = 'output_files/saved_beams/vanilla_{}.pickle'
VALIDATION_NOT_TEST= True
DATASET_WEBNLG=False


SUMM_START_TOK = 'BOS'
SUMM_END_TOK = 'EOS'
SUMM_PAD_TOK = 'PAD'
SUMM_CLS_TOK = 'CLS'


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# TODO remove
model_flags = ['hidden_size', 'ff_size', 'heads', 'emb_size', 'enc_layers', 'enc_hidden_size', 'enc_ff_size',
               'dec_layers', 'dec_hidden_size', 'dec_ff_size', 'encoder', 'ff_actv', 'use_interval']


class fakeDAI:
    def __init__(self, triple):
        self.slot, self.da_type, self.value = triple

    def __lt__(self, other):
        return (self.slot, self.da_type, self.value) < (other.slot, other.da_type, other.value)


def get_das_texts_from_webnlg(filepath):
    json_data = json.load(open(filepath, 'r'));
    das = []
    tokss = []
    for item in json_data:
        ent2ner = {v: k for k, v in item['ner2ent'].items()}
        lexicalized_triples = [[ent2ner.get(t, t) for t in trip] for trip in item['triples']]

        da = [fakeDAI(trip) for trip in lexicalized_triples]
        toks = [START_TOK] + item['target'].split(' ') + [END_TOK]
        das.append(da)
        tokss.append(toks)
    return das, tokss


def normalise(s):
    s = s.lower()
    words = s.split(" ")
    pos = nltk.pos_tag(words)
    result_words = []
    for word, tag in pos:
        if tag == 'NNS':
            if word == "children":
                result_words.append("child")
                result_words.append("-s")
                continue
            if not word.endswith("s"):
                print(word)
            result_words.append(word[:-1])
            result_words.append('-s')
        else:
            result_words.append(word)
    result = " ".join(result_words)
    return result


class RERANK(Enum):
    RANDOM = 0
    VANILLA = 1
    TGEN = 2
    ORACLE = 3
    REVERSE_ORACLE = 4

    def __str__(self):
        names = ["random", "vanilla", "tgen", "oracle", "reverse_oracle"]
        return names[self.value]


def get_true_sents():
    if DATASET_WEBNLG:
        if VALIDATION_NOT_TEST:
            return get_das_texts_from_webnlg("WebNLG_Reader/data/webnlg/valid.json")[1]
        else:
            return get_das_texts_from_webnlg("WebNLG_Reader/data/webnlg/test.json")[1]


    if VALIDATION_NOT_TEST:
        true_file = "tgen/e2e-challenge/input/devel-text.txt"
    else:
        true_file = "tgen/e2e-challenge/input/test-text.txt"

    true_sentences = []
    with open(true_file, "r", encoding='utf-8') as true:
        current_set = []
        for line in true:
            if len(line) > 1:
                current_set.append(line.strip('\n').split(" "))
            else:
                true_sentences.append(current_set)
                current_set = []
    return true_sentences


def count_lines(filepath):
    return sum([1 for _ in open(filepath, "r").readlines()])


def get_texts_training():
    true_file_path = "tgen/e2e-challenge/input/train-text.txt"
    with open(true_file_path, "r", encoding='utf-8') as fp:
        return [x.strip("\n").split(" ") for x in fp.readlines()]


def apply_absts(absts, texts):
    results = []
    pattern = re.compile("X-[a-z]+")
    for abst, text in zip(absts, texts):
        text_res = []
        # print(text)
        for tok in text:
            if pattern.match(tok):
                slot = tok[2:]
                for a in abst:
                    if a.slot == slot:
                        text_res.append(a.value)
                        break
            else:
                text_res.append(tok)
        # assert(len(text) == len(text_res))
        results.append(text_res)
    return results


def get_training_das_texts():
    if DATASET_WEBNLG:
        return get_das_texts_from_webnlg('WebNLG_Reader/data/webnlg/train.json')
    # das = read_das("tgen/e2e-challenge/input/train-das.txt")
    # texts = [[START_TOK] + x + [END_TOK] for x in get_texts_training()]
    # return das, texts


def safe_get_w2v(w2v, tok):
    unimp_toks = [PAD_TOK]
    tok = END_TOK if tok in unimp_toks else tok
    return w2v[tok]


def remove_strange_toks(tok):
    unimp_toks = ['<VOID>', '<UNK>', '<-s>']
    return '<STOP>' if tok in unimp_toks else tok


def get_hamming_distance(xs, ys):
    return sum([1 for x, y in zip(xs, ys) if x != y])


def get_training_variables():
    if DATASET_WEBNLG:
        das,texts = get_das_texts_from_webnlg('WebNLG_Reader/data/webnlg/train.json')
        return texts,das
    # das = read_das("tgen/e2e-challenge/input/train-das.txt")
    # texts = [[START_TOK] + x + [END_TOK] for x in get_texts_training()]
    # return texts, das


def get_multi_reference_training_variables():
    texts, das = get_training_variables()

    da_text_map = defaultdict(list)
    for da, text in zip(das, texts):
        da_text_map[tuple(da)].append(text)
    das_mr = []
    texts_mr = []
    for da, text in da_text_map.items():
        # print("[FT DEBUG]")
        # print(text[0][0].encode('utf-8'))
        das_mr.append(da)
        texts_mr.append(text)
    return texts_mr, das_mr


def get_test_das():
    if DATASET_WEBNLG:
        if VALIDATION_NOT_TEST:
            return get_das_texts_from_webnlg("WebNLG_Reader/data/webnlg/valid.json")[0]
        else:
            return get_das_texts_from_webnlg("WebNLG_Reader/data/webnlg/test.json")[0]

    if VALIDATION_NOT_TEST:
        das_file = "tgen/e2e-challenge/input/devel-das.txt"
    else:
        das_file = "tgen/e2e-challenge/input/test-das.txt"

    # das = read_das(das_file)
    return ""


def get_abstss_train():
    return "smart_load_absts('tgen/e2e-challenge/input/train-abst.txt')"


def get_abstss_test():
    if VALIDATION_NOT_TEST:
        absts_file = 'tgen/e2e-challenge/input/devel-abst.txt'
    else:
        absts_file = 'tgen/e2e-challenge/input/test-abst.txt'

    return "smart_load_absts(absts_file)"


def get_final_beam(beam_size, train=False):
    if train:
        path_format = "output_files/saved_beams/train_vanilla_{}.txt"
    else:
        path_format = "output_files/saved_beams/vanilla_{}.txt"
    path = path_format.format(beam_size)
    output = []
    current = []
    for line in open(path, "r+"):
        if line == '\n':
            output.append(current)
            current = []
            continue

        toks = line.strip('\n').split(" ")
        logprob = float(toks.pop())
        current.append((toks, logprob))
    return output


def load_model_from_gpu(model, filepath):
    f = h5py.File(filepath, mode='r')
    # load_weights_from_hdf5_group(f['model_weights'], model.layers)

    saving.hdf5_format.load_weights_from_hdf5_group_by_name(f['model_weights'], model.layers)
    saving.hdf5_format.load_weights_from_hdf5_group(f['model_weights'], model.layers)


def get_features(path, text_embedder, w2v, tok_prob):
    h = path[2][0][0]
    c = path[2][1][0]
    pred_words = [text_embedder.embed_to_tok[x] for x in path[1]]

    return np.concatenate((h, c,
                           safe_get_w2v(w2v, pred_words[-1]), safe_get_w2v(w2v, pred_words[-2]),
                           [tok_prob, path[0], len(pred_words)]))


def postprocess(text):
    text = re.sub(r'([a-zA-Z]) - ([a-zA-Z])', r'\1-\2', text)
    text = re.sub(r'£ ([0-9])', r'£\1', text)
    text = re.sub(r'([a-zA-Z]) *\' *(s|m|d|ll|re|ve|t)', r"\1'\2", text)
    text = re.sub(r'([a-zA-Z]) *n\'t', r"\1n't", text)
    text = re.sub(r' \' ([a-zA-Z ]+) \' ', r" '\1' ", text)
    return text


def tgen_postprocess(text):
    currency_or_init_punct = Regex(r' ([\p{Sc}\(\[\{\¿\¡]+) ', flags=UNICODE)
    noprespace_punct = Regex(r' ([\,\.\?\!\:\;\\\%\}\]\)]+) ', flags=UNICODE)
    contract = Regex(r" (\p{Alpha}+) ' (ll|ve|re|[dsmt])(?= )", flags=UNICODE | IGNORECASE)
    dash_fixes = Regex(r" (\p{Alpha}+|£ [0-9]+) - (priced|star|friendly|(?:£ )?[0-9]+) ",
                       flags=UNICODE | IGNORECASE)
    dash_fixes2 = Regex(r" (non) - ([\p{Alpha}-]+) ", flags=UNICODE | IGNORECASE)

    text = ' ' + text + ' '
    text = dash_fixes.sub(r' \1-\2 ', text)
    text = dash_fixes2.sub(r' \1-\2 ', text)
    text = currency_or_init_punct.sub(r' \1', text)
    text = noprespace_punct.sub(r'\1 ', text)
    text = contract.sub(r" \1'\2", text)
    text = text.strip()
    # capitalize
    if not text:
        return ''
    text = text[0].upper() + text[1:]
    return text


def get_regression_vals(num_ranks, with_train_refs):
    if with_train_refs:
        return [i / num_ranks for i in range(1, num_ranks + 1)]
    else:
        return [i / (num_ranks - 1) for i in range(num_ranks)]


def get_section_cutoffs(num_ranks):
    return [i / num_ranks for i in range(1, num_ranks)]


def get_section_value(val, cut_offs, regression_vals, merge_middle=False, only_top=False, only_bottom=False):
    def group_sections(x):
        if merge_middle:
            return 1 if x > 0.999 else (0 if x < 0.001 else 0.5)
        elif only_bottom:
            return 1 if x > 0.999 else 0
        elif only_top:
            return 0 if x < 0.001 else 1
        else:
            return x

    for i, co in enumerate(cut_offs):
        if val <= co:
            return group_sections(regression_vals[i])
    return group_sections(1)


def get_args_presumm(parser):
    parser.add_argument("-task", default='abs', type=str, choices=['ext', 'abs'])
    parser.add_argument("-encoder", default='bert', type=str, choices=['bert', 'baseline'])
    parser.add_argument("-mode", default='train', type=str, choices=['train', 'validate', 'test'])
    parser.add_argument("-bert_data_path", default='PreSumm/bert_data/bert_data_dummy/xsum') # TODO change
    parser.add_argument("-model_path", default='PreSumm/models/')
    parser.add_argument("-use_data", default='train', type=str, choices=['train', 'valid', 'test'])

    parser.add_argument("-result_path", default='PreSumm/results/xsum')
    parser.add_argument("-temp_dir", default='PreSumm/temp')

    parser.add_argument("-test_batch_size", default=16, type=int)

    parser.add_argument("-max_pos", default=512, type=int)
    parser.add_argument("-use_interval", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-large", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-load_from_extractive", default='', type=str)

    parser.add_argument("-sep_optim", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-lr_bert", default=2e-3, type=float)
    parser.add_argument("-lr_dec", default=2e-3, type=float)
    parser.add_argument("-use_bert_emb", type=str2bool, nargs='?',const=True,default=False)

    parser.add_argument("-share_emb", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-finetune_bert", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-dec_dropout", default=0.2, type=float)
    parser.add_argument("-dec_layers", default=6, type=int)
    parser.add_argument("-dec_hidden_size", default=768, type=int)
    parser.add_argument("-dec_heads", default=8, type=int)
    parser.add_argument("-dec_ff_size", default=2048, type=int)
    parser.add_argument("-enc_hidden_size", default=512, type=int)
    parser.add_argument("-enc_ff_size", default=512, type=int)
    parser.add_argument("-enc_dropout", default=0.2, type=float)
    parser.add_argument("-enc_layers", default=6, type=int)

    parser.add_argument("-label_smoothing", default=0.1, type=float)
    parser.add_argument("-generator_shard_size", default=32, type=int)
    parser.add_argument("-alpha",  default=0.6, type=float)
    parser.add_argument("-beam_size", default=5, type=int)
    parser.add_argument("-min_length", default=15, type=int)
    parser.add_argument("-max_length", default=150, type=int)
    parser.add_argument("-max_tgt_len", default=140, type=int)

    parser.add_argument("-param_init", default=0, type=float)
    parser.add_argument("-param_init_glorot", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-optim", default='adam', type=str)
    parser.add_argument("-lr", default=1, type=float)
    parser.add_argument("-beta1", default= 0.9, type=float)
    parser.add_argument("-beta2", default=0.999, type=float)
    parser.add_argument("-warmup_steps", default=8000, type=int)
    parser.add_argument("-warmup_steps_bert", default=8000, type=int)
    parser.add_argument("-warmup_steps_dec", default=8000, type=int)
    parser.add_argument("-max_grad_norm", default=0, type=float)

    parser.add_argument("-save_checkpoint_steps", default=5, type=int)
    parser.add_argument("-accum_count", default=1, type=int)
    parser.add_argument("-report_every", default=1, type=int)
    parser.add_argument("-train_steps", default=1000, type=int)
    parser.add_argument("-recall_eval", type=str2bool, nargs='?',const=True,default=False)

    parser.add_argument('-visible_gpus', default='-1', type=str)
    parser.add_argument('-gpu_ranks', default='-1', type=str)
    parser.add_argument('-log_file', default='PreSumm/logs/xsum.log')
    parser.add_argument('-seed', default=666, type=int)
    
    parser.add_argument("-test_all", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-test_from", default='PreSumm/models/model_step_30000.pt') # TODO FT: for xsum, cnndm change?
    parser.add_argument("-test_start_from", default=-1, type=int)

    parser.add_argument("-train_from", default='')
    parser.add_argument("-report_rouge", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)

    parser.add_argument("-lr_fact", default=0.01, type=float)
    parser.add_argument("-w1", default=1.00, type=float)
    parser.add_argument("-w2", default=1.00, type=float)
    parser.add_argument("-train_weight", default=False, type=float)
    
    parser.add_argument("-device", default='', type=str)

    parser.add_argument('-c', default=None)
    parser.add_argument('-should_skip_beam', default=False)
    parser.add_argument('-use_dataset', default='xsum', type=str, choices=['xsum', 'cnndm'])
    parser.add_argument('-pretrained_model', default='presumm', type=str, choices=['presumm', 'bart'])
    parser.add_argument("-batch_size", default=1, type=int)
    
    parser.add_argument("-use_size", default=-99, type=int)
    parser.add_argument("-skip_save_beam", default=False, type=bool)
    
    return parser


def get_timestamp_file():
    # gmt stores current gmtime
    gmt = time.gmtime()
    ts = calendar.timegm(gmt)
    
    return str(ts)
        
def convert_id_to_text(tokenizer, token_ids):
    text = tokenizer.convert_ids_to_tokens([int(n) for n in token_ids])
    # Convert token_ids to text for factual consistency scoring
    text = ' '.join(text).replace(' ##','')
    text = text.replace('[unused0]', '').replace('[unused3]', '').replace('[PAD]', '').replace('[unused1]', '').replace(r' +', ' ').replace(' [unused2] ', '<q>').replace('[unused2]', '').strip()
    
    return text

def convert_id_to_text_bart(token_ids):
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-xsum-12-3")
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return text


# def get_embedding_symbols(token):
#     symbols = {'BOS': tokenizer.vocab['[unused0]'], 'EOS': tokenizer.vocab['[unused1]'],
#                'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused2]']}
#     return symbols[token]
            