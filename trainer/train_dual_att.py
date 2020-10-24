import pickle 
import argparse
import json
import os
import time
import re
from collections import defaultdict
import csv
import gzip
import math

import torch 
import torch.nn as nn
from torch import LongTensor, FloatTensor
import numpy as np
from gensim.models import KeyedVectors

from models.dual_att.dual_att import DualAtt
from experiment import Experiment
from utils import get_mask
from preprocess.divide_and_create_example_sent import clean_str

class Args(object):
    pass

def parse_args(config):
    args = Args()
    with open(config, 'r') as f:
        config = json.load(f)
    for name, val in config.items():
        setattr(args, name, val)

    return args

def load_pretrained_embeddings(vocab, word2vec, emb_size):
    """
    NOTE:
        tensorflow version.
    Args:
        vocab: a Vocab object
        word2vec: dictionry, (str, np.ndarry with type of np.float32)

    Return:
        pre_embeddings: torch.FloatTensor
    """
    pre_embeddings = np.random.uniform(-1.0, 1.0, size=[len(vocab), emb_size]).astype(np.float32)
    for word in vocab._token2id:
        if word in word2vec:
            pre_embeddings[vocab._token2id[word]] = word2vec[word]
    return torch.FloatTensor(pre_embeddings)

class AvgMeters(object):
    def __init__(self):
        self.count = 0
        self.total = 0. 
        self._val = 0.
    
    def update(self, val, count=1):
        self.total += val
        self.count += count

    def reset(self):
        self.count = 0
        self.total = 0. 
        self._val = 0.

    @property
    def val(self):
        return self.total / self.count

class EarlyStop(Exception):
    pass

class DualAttExperiment(Experiment):
    def __init__(self, args, dataloaders):
        super(DualAttExperiment, self).__init__(args, dataloaders)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # dataloader
        self.train_dataloader = dataloaders["train"]
        self.valid_dataloader = dataloaders["valid"] if dataloaders["valid"] is not None else None

        # stats
        self.train_stats = defaultdict(list)
        self.valid_stats = defaultdict(list)
        self._best_rmse = 1e3
        self.patience = 0

        # create output path
        self.setup()
        self.build_model() # self.model
        self.build_optimizer() #self.optimizer
        self.build_scheduler() #self.scheduler
        self.build_loss_func() #self.loss_func

        # print
        self.print_args()
        self.print_model_stats()

    def build_scheduler(self):
        pass

    def build_model(self):
        # dirty implementation
        if self.args.use_pretrain:
            data_prefix = "/raid/hanszeng/Recommender/NARRE/data/"
            pretrain_path = "GoogleNews-vectors-negative300.bin"
            pretrain_path = data_prefix + pretrain_path

            
            wv_from_bin = KeyedVectors.load_word2vec_format(pretrain_path, binary=True)

            word2vec = {}
            for word, vec in zip(wv_from_bin.vocab, wv_from_bin.vectors):
                word2vec[word] = vec
            
            
            _dataset = self.train_dataloader.dataset
            word_pretrained = load_pretrained_embeddings(_dataset.word_vocab, word2vec, self.args.embedding_dim)
        else:
            _dataset  = self.train_dataloader.dataset
            word_pretrained=None

        self.model = DualAtt(vocab_size=len(_dataset.word_vocab), 
                doc_len=_dataset.doc_len, l_window_size=self.args.l_window_size, l_out_size=self.args.l_out_size, 
                g_out_size=self.args.g_out_size, emb_size=self.args.emb_size,
                 hidden_size_1=self.args.hidden_size_1, hidden_size_2=self.args.hidden_size_2, dropout=self.args.dropout, 
                 pretrained_embeddings=word_pretrained)

        if self.args.parallel:
            self.model = torch.nn.DataParallel(self.model)
            self.print_write_to_log("the model is parallel training.")
        self.model.to(self.device)

    def build_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        if self.args.verbose:
            self.print_write_to_log(re.sub(r"\n", "", self.optimizer.__repr__()))
        
    def build_loss_func(self):
        self.loss_func = nn.MSELoss()


    def train_one_epoch(self, current_epoch):
        avg_loss = AvgMeters()
        square_error = 0.
        accum_count = 0
        start_time = time.time()

        self.model.train()
        for i, (u_docs, i_docs, ratings) in enumerate(self.train_dataloader):
            if i == 0 and current_epoch == 0:
                print("u_docs", u_docs.shape, "i_docs", i_docs.shape)
            u_docs = u_docs.to(self.device)
            i_docs = i_docs.to(self.device)
            ratings = ratings.to(self.device)

            self.optimizer.zero_grad()
            y_pred = self.model(u_docs, i_docs)
            loss = self.loss_func(y_pred, ratings)
            loss.backward()

            gnorm = nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()

            # val 
            avg_loss.update(loss.mean().item())
            square_error += loss.mean().item() * ratings.size(0)
            accum_count += ratings.size(0)

            # log
            if (i+1) % self.args.log_idx == 0 and self.args.log:
                elpased_time = (time.time() - start_time) / self.args.log_idx
                rmse = math.sqrt(square_error / accum_count)

                log_text = "epoch: {}/{}, step: {}/{}, loss: {:.3f}, rmse: {:.3f}, lr: {}, gnorm: {:3f}, time: {:.3f}".format(
                    current_epoch, self.args.epochs,  (i+1), len(self.train_dataloader), avg_loss.val, rmse, 
                    self.optimizer.param_groups[0]["lr"], gnorm, elpased_time
                )
                self.print_write_to_log(log_text)

                avg_loss.reset()
                square_error = 0. 
                accum_count = 0
                start_time = time.time()

    def valid_one_epoch(self):
        square_error = 0.
        accum_count = 0
        avg_loss = AvgMeters()

        self.model.eval()
        for i, (u_docs, i_docs, ratings) in enumerate(self.valid_dataloader):
            u_docs = u_docs.to(self.device)
            i_docs = i_docs.to(self.device)
            ratings = ratings.to(self.device)

            with torch.no_grad():
                y_pred = self.model(u_docs, i_docs)
                loss = self.loss_func(y_pred, ratings)

            square_error += loss.mean().item() * ratings.size(0)
            accum_count += ratings.size(0)
            avg_loss.update(loss.mean().item())

        rmse = math.sqrt(square_error / accum_count)
        if rmse < self.best_rmse:
            self.best_rmse =  rmse 
            self.save("best_model.pt")
            self.patience = 0
        else:
            self.patience += 1

        log_text =  "valid loss: {:.3f}, valid rmse: {:.3f}, best rmse: {:.3f}".format(avg_loss.val, rmse, self.best_rmse)
        self.print_write_to_log(log_text)

        # ealry stop
        if self.patience >= self.args.patience:
            # write stats 
            if self.args.stats:
                self.write_stats("train")
                self.write_stats("valid")

            raise EarlyStop("early stop")

    @property
    def best_rmse(self):
        return self._best_rmse
    
    @best_rmse.setter
    def best_rmse(self, val):
        self._best_rmse = val

    def train(self):
        print("start training ...")
        for epoch in range(self.args.epochs):
            self.train_one_epoch(epoch)
            self.valid_one_epoch()

class DualAttDataset(torch.utils.data.Dataset):
    def __init__(self, args, set_name):
        super(DualAttDataset, self).__init__()

        self.args = args
        self.set_name = set_name
        param_path = os.path.join(self.args.data_dir, "meta.pkl")
        with open(param_path, "rb") as f:
            para = pickle.load(f)

        self.user_num = para['user_num']
        self.item_num = para['item_num']
        self.indexlizer = para['indexlizer']
        self.u_docs = para['user_docs']
        self.i_docs = para['item_docs']
        self.doc_len = para["doc_len"]
        self.word_vocab = self.indexlizer._vocab

        example_path = os.path.join(self.args.data_dir, f"{set_name}_exmaples.pkl")
        with open(example_path, "rb") as f:
            self.examples = pickle.load(f)


    def __getitem__(self, i):
        # for each review(u_docs or i_docs) [...] 
        # NOTE: not padding 
        u_id, i_id, rating, u_doc, i_doc = self.examples[i]

        return rating, u_doc, i_doc

    def __len__(self):
        return len(self.examples)

    def collate_fn(self, batch):
        ratings, u_docs, i_docs = zip(*batch)
        
        ratings = FloatTensor(ratings)
        u_docs = LongTensor(u_docs)
        i_docs= LongTensor(i_docs)

        return u_docs, i_docs, ratings

if __name__ == "__main__":
    config_file = "./models/dual_att/default_dual_att.json"
    args = parse_args(config_file)
    train_dataset = DualAttDataset(args, "train")
    valid_dataset = DualAttDataset(args, "valid")

    train_dataloder = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn, num_workers=8)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=valid_dataset.collate_fn, num_workers=8)

    dataloaders = {"train": train_dataloder, "valid": valid_dataloader, "test": None}
    experiment = DualAttExperiment(args, dataloaders)
    experiment.train()