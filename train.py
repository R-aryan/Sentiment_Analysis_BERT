import config
import dataset
import engine
import torch
import pandas as pd
import torch.nn as nn
import numpy as np

from model import BERTBaseUncased
from sklearn import model_selection
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup



def run():
    #reading the dataseet
    dfx= pd.read_csv(config.TRAINING_FILE).fillna("none")

    dfx.sentiment= dfx.sentiment.apply(
        lambda x:1 if x=="positive" else 0
    )

    

    #splitting into training and validation set
    df_train,df_valid= model_selection.train_test_split(
        dfx,
        test_size=0.1,
        random_state=42,
        stratify= dfx.sentiment.values
    )

    df_train= df_train.reset_index(drop=True)
    df_valid= df_valid.reset_index(drop=True)