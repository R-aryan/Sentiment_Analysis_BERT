from Training import config
import torch
import flask
import time
from flask import Flask
from flask import request
from model import BERTBaseUncased
import functools
import torch.nn as nn
import joblib

app = Flask(__name__)


MODEL = None
DEVICE = "cpu"
PREDICTION_DICT = dict()
memory = joblib.Memory("../input/", verbose=0)


def predict_from_cache(sentence):
    if sentence in PREDICTION_DICT:
        return PREDICTION_DICT[sentence]
    
    else:
        result= sentence_prediction(sentence)
        PREDICTION_DICT[sentence]=result
        return result



@memory.cache
def sentence_prediction(sentence):
    tokenizer=config.TOKENIZER
    max_len= config.MAX_LEN
