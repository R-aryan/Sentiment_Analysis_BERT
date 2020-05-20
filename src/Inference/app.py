import config
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