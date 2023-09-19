from embedding_transformers import *
from torch.utils.data import DataLoader
from transformers import get_scheduler
from tqdm.auto import tqdm
from timeit import default_timer as timer
import numpy as np


class EmbeddingModel(nn.Module):
    def __init__(self,encoder_configs, n_outs,dropout: float = 0.1):
        super(EmbeddingModel,self).__init__()
        self.model = TransformerEncoder(encoder_configs)
        self.ffl = nn.Linear(encoder_configs.embed_dim, n_outs)
        self.dropout = nn.Dropout(dropout)
    
    
        #self.apply(self._init_weights_)

    def _init_weights_(self, module):

        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.xavier_normal_(module.weight)

    def forward(self, src,pad_id):
        device = src.device
        src_mask = self.make_src_pad(src,pad_id).to(device)

        hidden_state = self.model(src,src_mask)
        mean_pooled = self.mean_pooling(hidden_state=hidden_state,attention_mask=src_mask)
        out = self.dropout(self.ffl(mean_pooled))

        return out
        
    def make_src_pad(self,src,pad_idx):
        src_padding = (src == pad_idx).float()
        
        return src_padding
    
    
    def mean_pooling(self,hidden_state,attention_mask):

        padding_expanded = (attention_mask.unsqueeze(-1)==0).expand(hidden_state.size()).float()
        return torch.sum(hidden_state*padding_expanded,1)/torch.clamp(padding_expanded.sum(1),min=1e-9)
    


class Trainer:
    def __init__(self,model: torch.nn.Module,
                 pad_id: int,
                 train_data: DataLoader, 
                 optimizer: torch.optim.Optimizer,
                 schedular: get_scheduler,
                 save_every: int,
                 checkpoint: dict,
                 checkpoint_dir:str,
                 loss_fn: None,
                 gpu_id: torch.device,
                 )->None:
        
        self.model = model
        self.train_data = train_data
        self.optimizer = optimizer
        self.gpu_id = gpu_id
        self.save_every = save_every
        self.loss_fn = loss_fn
        self.pad_id = pad_id
        self.checkpoint = checkpoint
        self.checkpoint_dir = checkpoint_dir
        self.scheduler = schedular
    
        
    def _run_batch(self,batch):

        self.optimizer.zero_grad()
        input_id = batch['input_ids'].to(self.gpu_id)
        label = batch['label'].to(self.gpu_id)

        output = self.model(input_id,self.pad_id)
        loss = self.loss_fn(output,label)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def _run_epoch(self,epoch):
        losses = []
        for batch in self.train_data:
            loss = self._run_batch(batch)
            losses.append(loss)
        
        return np.mean(losses)
        

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            start_time=timer()
            train_loss = self._run_epoch(epoch)
            end_time = timer()
            print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
            
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


    def _save_checkpoint(self,epoch):
        torch.save(self.checkpoint,self.checkpoint_dir)
        print("Saved checkpoint {} at Epoch {}".format(self.checkpoint_dir,epoch))
