import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

from base_model import TGEN_Reranker
from embedding_extractor import TokEmbeddingSeq2SeqExtractor, DAEmbeddingSeq2SeqExtractor
from utils import get_training_variables, get_hamming_distance

cfg = yaml.load(open("configs/tgen_reranker_train.yaml", "r"))
texts, das, = get_training_variables()
text_embedder = TokEmbeddingSeq2SeqExtractor(texts)
da_embedder = DAEmbeddingSeq2SeqExtractor(das)
train_text = np.array(text_embedder.get_embeddings(texts, pad_from_end=False) + [text_embedder.empty_embedding])
das_inclusions = np.array([da_embedder.get_inclusion(da) for da in das] + [da_embedder.empty_inclusion])
reranker = TGEN_Reranker(da_embedder, text_embedder, cfg)
if os.path.exists(cfg["reranker_loc"]) and cfg["load_reranker"]:
    reranker.load_model()
else:
    valid_size = cfg['valid_size']
    reranker.train(das_inclusions[:-valid_size], train_text[:-valid_size], cfg["reranker_epoch"],
                   das_inclusions[-valid_size:], train_text[-valid_size:], cfg["min_passes"])
if cfg["plot_reranker_stats"]:
    preds = reranker.predict(train_text)
    ham_dists = [get_hamming_distance(x, y) for x, y in zip(preds, das_inclusions)]
    filter_hams = [x for x in ham_dists if x != 0]
    plt.hist(filter_hams)
    plt.show()
