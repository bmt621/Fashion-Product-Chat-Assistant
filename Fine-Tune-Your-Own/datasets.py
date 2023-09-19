import re
import torch
from torch.utils.data import DataLoader

class DataSet():
    def __init__(self,df,tokenizer,max_length:int=200):
        self.data_x = self.clean_text(df['text'])
        self.labels = torch.tensor(df['label'],dtype=torch.long)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self,idx):

        tokens = self.tokenizer(self.data_x[idx],max_length=self.max_length, padding='max_length',truncation=True,add_special_tokens=True)

        ids = tokens['input_ids']
        mask = tokens['attention_mask']
        
        items = {'input_ids':torch.tensor(ids,dtype=torch.long)}
        items['label'] = self.labels[idx]

        return (items)

    def __len__(self):
        return len(self.labels)

    def clean_text(self,texts):
        txts = []
        for text in texts:
            text = text.strip().lower()
            text = text.replace('&nbsp;',' ')
            text = re.sub(r'<br(\s\/)?>',' ',text)
            text = re.sub(r' +',' ',text)
            text = re.sub(r'[^A-Za-z0-9]+',' ',text)
            txts.append(text)

        return txts