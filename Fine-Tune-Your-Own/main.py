from trainer import EmbeddingModel
from datasets import *
from configs import *
from transformers import AutoTokenizer
from transformers import get_scheduler
import torch
import torch.nn as nn

def main():
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    config = TransformerEncoderConfigs()
    encod_config = TransformerEncoderConfigs()
    encod_config.vocab_size = tokenizer.vocab_size
    encod_config.sinusoid = True
    encod_config.n_encoder_layer = 3
    encod_config.use_flash_attn = False
    encod_config.norm_first = False

    model = EmbeddingModel(encod_config, 6)
    train_loader = None # pass train loader
    train_df = None # pass valid loader

    # this implementation will be used to pretrained an encoder models before further finetuning for sentence-embeddings


    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4,betas=(0.9, 0.98), eps=1e-9)
    num_epoch = 30
    num_training_steps = num_epoch *len(train_loader)
    lr_scheduler = get_scheduler(name='linear',num_warmup_steps=0.0,optimizer=optimizer,num_training_steps=num_training_steps)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_weight = torch.from_numpy((1 - (train_df['label'].value_counts(normalize=True).sort_index()).values)).float()
    loss_fn = nn.CrossEntropyLoss(weight=class_weight.to(device))

