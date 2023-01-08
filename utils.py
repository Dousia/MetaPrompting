import json
import os

import torch
import datetime
from collections import defaultdict
import numpy as np
from tqdm import tqdm, trange


def tprint(s):
    '''
        print datetime and s
        @params:
            s (str): the string to be printed
    '''
    print('{}: {}'.format(
        datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S'), s),
          flush=True)


def exact_match(predictions: np.ndarray, actuals: np.ndarray, question_ids: np.ndarray):
    """Compute the exact match (EM) for a sequence of predictions and actual labels"""
    unique_questions = set(question_ids)

    q_actuals = list(zip(question_ids, actuals))
    q_predictions = list(zip(question_ids, predictions))

    actuals_per_question = defaultdict(list)
    predictions_per_question = defaultdict(list)

    for qid, val in q_actuals:
        actuals_per_question[qid].append(val)
    for qid, val in q_predictions:
        predictions_per_question[qid].append(val)

    em = 0
    for qid in unique_questions:
        if actuals_per_question[qid] == predictions_per_question[qid]:
            em += 1
    em /= len(unique_questions)

    return em


