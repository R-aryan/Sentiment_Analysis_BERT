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

        review= ".".join(review.split())

        inputs= self.tokenizer.encode_plus(
            review,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True
        )

        ids= inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
