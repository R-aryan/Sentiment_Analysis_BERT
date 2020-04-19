import torch.nn as nn
import transformers
import config

class BERTBaseUncased(nn.Module):

    ##defining the init function
    def __init__(self):

        super(BERTBaseUncased,self).__init__()
        self.bert=transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.bert_drop=nn.Dropout(0.3)
        self.out=nn.Linear(768,1)





