from fact_scorer.fact_summac.model_summac import SummaCZS
from time import time

def classify(document, summary):
    model = SummaCZS(granularity="sentence", model_name="vitc", device="cuda")

    # INPUT EXAMPLE
    # document = """Scientists are studying Mars to learn about the Red Planet and find landing sites for future missions. One possible site, known as Arcadia Planitia, is covered instrange sinuous features. The shapes could be signs that the area is actually made of glaciers, which are large masses of slow-moving ice. Arcadia Planitia is in Mars' northern lowlands."""
    # summary = "There are strange shape patterns on Arcadia Planitia. The shapes could indicate the area might be made of glaciers. This makes Arcadia Planitia ideal for future missions."

    score = model.score([document], [summary])
    return score["scores"][0]

def evaluate_batch(documents, summaries):
    model = SummaCZS(granularity="sentence", model_name="vitc", device="cuda")

    count0 = 0
    count1 = 0
    
    scores = model.score(documents, summaries)

    for score in scores["scores"]:
        if (score>=0):
            count1 += 1
        else:
            count0 += 1
    precision = 1 # TODO FT make sure it's correct
    recall = float(count1/(count0 + count1))
    f1 =  2 * float(precision * recall /(precision + recall))

    return scores["scores"], f1

