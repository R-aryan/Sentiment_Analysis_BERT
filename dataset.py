import torch
import config


class BERTDataset():

    def __init__(self,review,target):
        self.review=review
        self.target=target
        self.tokenizer=config.TOKENIZER
        self.max_len=config.MAX_LEN


    def __len__(self):
        return len(self.review)

    
    def __getitem__(self,item):
        review= str(self.review[item])
