import sys
HOME_REPO = "/home/lr/faza.thirafi/raid/repository-kenkyuu-models/"
sys.path.insert(0, HOME_REPO + "factual_coco/")

from run_coco import initialize_model, evaluate

def initialize_coco():
    return initialize_model()

def evaluate_coco(coco_params, document, summary):
    return evaluate(coco_params, document, summary)

def evaluate_batch_coco(coco_params, documents, summaries):
    results = []
    sum_coco_scores = 0
    count0 = 0
    count1 = 0
    for docs, summ in zip(documents, summaries):
        coco_score = evaluate(coco_params, docs, summ)
        results.append(coco_score)
        sum_coco_scores += coco_score
        if (coco_score>=0):
            count1 += 1
        else:
            count0 += 1

    precision = 1 # TODO FT make sure it's correct
    recall = float(count1/(count0 + count1))
    f1 =  2 * float(precision * recall /(precision + recall))

    avg = sum_coco_scores/len(documents)

    return results, avg, f1