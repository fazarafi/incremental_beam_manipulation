import sys


HOME_REPO = "/home/lr/faza.thirafi/raid/repository-kenkyuu-models/"
sys.path.insert(0, HOME_REPO + "factuality-datasets/")


from dae_evaluate_generated_outputs import evaluate_summary, initialize_model

def calculate_f1(li):
    count1 = li.count(1)
    count0 = li.count(0)
    precision = 1 # TODO FT make sure it's correct
    recall = float(count1/(count0 + count1))
    f1 =  2 * float(precision * recall /(precision + recall))
    
    return f1 

def initialize_dae(dataset_name):
    return initialize_model(dataset_name)

def evaluate_dae(dae_params, document, summary):

    article_data = document
    score = evaluate_summary(article_data, summary, dae_params['tokenizer'], dae_params['model'], dae_params['nlp'], dae_params['args'])

    return score

def evaluate_batch_dae(dae_params, documents, summaries):
    score_list = []
    
    for docs, summ in zip(documents, summaries):
        dae_score = evaluate_dae(dae_params, docs, summ)
        score_list.append(dae_score)

    f1 =  calculate_f1(score_list)
    
    return f1, score_list