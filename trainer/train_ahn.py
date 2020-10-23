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
from tensorboardX import SummaryWriter

from experiment import Experiment
from gensim.models import KeyedVectors
from utils import get_mask, get_seq_lengths_from_mask
#from ahn import LSTMForUserItemPredictionHIRCOAA as AHN
from models.ahn.ahn_model import AHN
from preprocess.divide_and_create_sent import clean_str

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
    for word in vocab.token2id:
        if word in word2vec:
            pre_embeddings[vocab.token2id[word]] = word2vec[word]
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

# self.args.lr
# self.args.verbose
class AhnExperiment(Experiment):
    def __init__(self, args, dataloaders):
        super(AhnExperiment, self).__init__(args, dataloaders)

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
        self.global_step = 0
        self.print_args()
        self.print_model_stats()
        if self.args.tensorboard:
            self.writer = SummaryWriter(log_dir=self.out_dir)

    def build_scheduler(self):
        pass

    def build_model(self) -> None:
        # dirty implementation
        
        """
        data_prefix = "/raid/hanszeng/Recommender/NARRE/data/"
        pretrain_path = "GoogleNews-vectors-negative300.bin"
        pretrain_path = data_prefix + pretrain_path

        
        wv_from_bin = KeyedVectors.load_word2vec_format(pretrain_path, binary=True)

        word2vec = {}
        for word, vec in zip(wv_from_bin.vocab, wv_from_bin.vectors):
            word2vec[word] = vec
        """
        _dataset = self.train_dataloader.dataset
        #user_pretrained = load_pretrained_embeddings(_dataset.user_word_vocab, word2vec, 300)
        #item_pretrained = load_pretrained_embeddings(_dataset.item_word_vocab, word2vec, 300)

        self.model = AHN(self.args.embedding_dim, self.args.hidden_dim, self.args.k_factor, 
                        user_size=_dataset.user_num, item_size=_dataset.item_num, 
                        word_vocab_size=len(_dataset.word_vocab), 
                        pretrained_word_embeddings=None,
                        rnn_dropout=self.args.rnn_dropout, dropout=self.args.dropout,
                        item_review_num=_dataset.rv_num)
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

    @staticmethod
    def get_sent_mask(batch_reviews):
        """
        NOTE: only deal with special condition where,
        batch_reviews: [bz, rn, sn, wn]
        sent_mask: [bz, rn, sn]
        """ 
        batch_sents = batch_reviews.sum(dim=-1) #[bz, rn, sn]
        sent_mask = get_mask(batch_sents)
        return sent_mask

    @staticmethod
    def get_sent_lengths(batch_reviews):
        """
        NOTE: only deal with special condition where,
        batch_reviews: [bz, rn, sn, wn]
        sent_lengths: [bz, rn, sn]
        """ 
        sent_lengths = torch.ones(size=list(batch_reviews.size()), dtype=torch.int64)
        sent_lengths[batch_reviews == 0] = 0
        sent_lengths = sent_lengths.sum(dim=-1) 

        return sent_lengths

    @staticmethod
    def get_review_mask(batch_reviews):
        """
        NOTE: only deal with special condition where,
        batch_reviews: [bz, rn, sn, wn]
        review mask: [bz, rn]
        """
        bz, rn = batch_reviews.size(0), batch_reviews.size(1)
        review_mask = torch.ones(size=(bz, rn), dtype=torch.bool)
        batch_reviews = batch_reviews.sum(dim=-1).sum(dim=-1) #[bz, rn]
        review_mask[batch_reviews == 0] = False

        return review_mask


    def train_one_epoch(self, current_epoch):
        avg_loss = AvgMeters()
        square_error = 0.
        accum_count = 0
        start_time = time.time()

        self.model.train()
        for i, (u_text, i_text, u_id, i_id, _, _, label) in enumerate(self.train_dataloader):
            self.global_step += 1
            # form mask and lengths 
            u_sent_mask = self.get_sent_mask(u_text).to(self.device)
            i_sent_mask = self.get_sent_mask(i_text).to(self.device)
            u_sent_lengths = self.get_sent_lengths(u_text).to(self.device)
            i_sent_lengths = self.get_sent_lengths(i_text).to(self.device)
            u_review_mask = self.get_review_mask(u_text).to(self.device)
            i_review_mask = self.get_review_mask(i_text).to(self.device)


            # to devicde
            u_text = u_text.to(self.device)
            i_text = i_text.to(self.device)
            u_id = u_id.to(self.device)
            i_id = i_id.to(self.device)
            
            label = label.to(self.device)

            self.optimizer.zero_grad()
            y_pred, us_weights, is_weights, ur_weights, ir_weights \
                    = self.model(u_text, i_text, u_sent_mask, i_sent_mask, u_sent_lengths, i_sent_lengths,
                                u_review_mask, i_review_mask, u_id, i_id)
            loss = self.loss_func(y_pred, label)
            loss.backward()

            gnorm = nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()

            # val 
            avg_loss.update(loss.mean().item())
            square_error += loss.mean().item() * label.size(0)
            accum_count += label.size(0)

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

            # tensorboard 
            if (i+1) % self.args.tensorboard_idx == 0 and self.args.tensorboard:
                self.writer.add_histogram("user sentence attention weights", us_weights.clone().cpu().data.numpy(), global_step=self.global_step)
                self.writer.add_histogram("item sentence attention weights", is_weights.clone().cpu().data.numpy(), global_step=self.global_step)
                self.writer.add_histogram("user review attention weights", ur_weights.clone().cpu().data.numpy(), global_step=self.global_step)
                self.writer.add_histogram("item review attention weights", ir_weights.clone().cpu().data.numpy(), global_step=self.global_step)

    def valid_one_epoch(self):
        square_error = 0.
        accum_count = 0
        avg_loss = AvgMeters()

        self.model.eval()
        for i, (u_text, i_text, u_id, i_id, _, _, label) in enumerate(self.valid_dataloader):
            # form mask and lengths 
            u_sent_mask = self.get_sent_mask(u_text).to(self.device)
            i_sent_mask = self.get_sent_mask(i_text).to(self.device)
            u_sent_lengths = self.get_sent_lengths(u_text).to(self.device)
            i_sent_lengths = self.get_sent_lengths(i_text).to(self.device)
            u_review_mask = self.get_review_mask(u_text).to(self.device)
            i_review_mask = self.get_review_mask(i_text).to(self.device)


            # to devicde
            u_text = u_text.to(self.device)
            i_text = i_text.to(self.device)
            u_id = u_id.to(self.device)
            i_id = i_id.to(self.device)

            label = label.to(self.device)

            with torch.no_grad():
                y_pred, _, _, _, _= self.model(u_text, i_text, u_sent_mask, i_sent_mask, u_sent_lengths, i_sent_lengths,
                                u_review_mask, i_review_mask, u_id, i_id)
                #y_pred = self.model(u_id, i_id)
                loss = self.loss_func(y_pred, label)

            square_error += loss.mean().item() * label.size(0)
            accum_count += label.size(0)
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

class AhnDataset(torch.utils.data.Dataset):
    def __init__(self, args, set_name):
        super(AhnDataset, self).__init__()

        self.args = args
        self.set_name = set_name
        param_path = os.path.join(self.args.data_dir, "meta.pkl")
        with open(param_path, "rb") as f:
            para = pickle.load(f)

        self.user_num = para['user_num']
        self.item_num = para['item_num']
        self.indexlizer = para['indexlizer']
        self.rv_num = para["rv_num"]
        self.sent_num = para["sent_num"]
        self.word_num = para["word_num"]
        self.u_text = para['user_reviews']
        self.i_text = para['item_reviews']
        self.u_rids = para["user_rids"]
        self.i_rids = para["item_rids"]
        self.word_vocab = self.indexlizer._vocab

        example_path = os.path.join(self.args.data_dir, f"{set_name}_exmaples.pkl")
        with open(example_path, "rb") as f:
            self.examples = pickle.load(f)

    def __getitem__(self, i):
        # for each review(u_text or i_text) [...] 
        # NOTE: not padding 
        if self.set_name == "train":
            u_id, i_id, rating, u_revs, i_revs, u_rids, i_rids, _= self.examples[i]

            return u_id, i_id, rating, u_revs, i_revs, u_rids, i_rids

        else:
            u_id, i_id, rating, u_revs, i_revs, u_rids, i_rids = self.examples[i]
            return u_id, i_id, rating, u_revs, i_revs, u_rids, i_rids
        
    def __len__(self):
        return len(self.examples)

    @staticmethod
    def truncate_tokens(tokens, max_seq_len):
        if len(tokens) > max_seq_len:
            tokens = tokens[:max_seq_len]
        return tokens

    @staticmethod
    def get_rev_mask(inputs):
        """
        If rv_len are all 0, then corresponding position in rv_num should be 0
        Args:
            inputs: [bz, rv_num, rv_len]
        """
        bz, rv_num, _ = list(inputs.size())

        masks = torch.ones(size=(bz, rv_num)).int()
        inputs = inputs.sum(dim=-1) #[bz, rv_num]
        masks[inputs==0] = 0 

        return masks.bool()

    def collate_fn(self, batch):
        # u_revs: [bz, ur_num, us_num, uw_num] #but variable length, 
        u_ids, i_ids, ratings, u_revs, i_revs, u_rids, i_rids = zip(*batch)

        bz = len(ratings)
        rv_num = self.rv_num 
        word_num = self.word_num
        sent_num = self.sent_num

        tensor_u_revs = torch.zeros(size=(bz, rv_num, sent_num, word_num)).long()
        tensor_i_revs = torch.zeros(size=(bz, rv_num, sent_num, word_num)).long()
        tensor_u_rids = torch.zeros(size=(bz, rv_num)).long()
        tensor_i_rids = torch.zeros(size=(bz, rv_num)).long() 

        # form tensor 
        for b_idx, reviews in enumerate(u_revs):
            for i, review in enumerate(reviews):
                for j, sent in enumerate(review):
                    sent_len = len(sent)
                    sent = LongTensor(sent)
                    tensor_u_revs[b_idx, i, j, :sent_len] = sent
        for b_idx, reviews in enumerate(i_revs):
            for i, review in enumerate(reviews):
                for j, sent in enumerate(review):
                    sent_len = len(sent)
                    sent = LongTensor(sent)
                    tensor_i_revs[b_idx, i, j, :sent_len] = sent
        for b_idx, u_rid in enumerate(u_rids):
            len_u_rid = len(u_rid)
            tensor_u_rids[b_idx, :len_u_rid] = LongTensor(u_rid)
        for b_idx, i_rid in enumerate(i_rids):
            len_i_rid = len(i_rid)
            tensor_i_rids[b_idx, :len_i_rid] = LongTensor(i_rid)
        
        u_ids = LongTensor(u_ids)
        i_ids = LongTensor(i_ids)
        ratings = FloatTensor(ratings)
        
        return tensor_u_revs, tensor_i_revs, u_ids, i_ids, tensor_u_rids, tensor_i_rids, ratings
     
if __name__ == "__main__":
    """
    args = Args()
    setattr(args, "data_dir", "./data/Toys_and_Games_5/randomly/kim_clean/sep_sentence/")
    valid_dataset = NarreSentDataset(args, "valid")
    valid_dataloder = torch.utils.data.DataLoader(valid_dataset, batch_size=4, shuffle=False, collate_fn=valid_dataset.collate_fn)
    batch_data = list(valid_dataloder)[19]

    u_text, i_text, u_id, i_id, _, _, label = batch_data

    u_sent_mask = NarreExperiment.get_sent_mask(u_text)
    i_sent_mask = NarreExperiment.get_sent_mask(i_text)
    u_sent_lengths = NarreExperiment.get_sent_lengths(u_text)
    i_sent_lengths = NarreExperiment.get_sent_lengths(i_text)
    u_review_mask = NarreExperiment.get_review_mask(u_text)
    i_review_mask = NarreExperiment.get_review_mask(i_text)

    print("u_text")
    print(u_text)
    print("u sent_mask")
    print(u_sent_mask)
    print("u_sent_lengths")
    print(u_sent_lengths)
    print("u_review_mask")
    print(u_review_mask)
    """ 

    config_file = "./models/ahn/default_ahn.json"
    args = parse_args(config_file)
    train_dataset = AhnDataset(args, "train")
    valid_dataset = AhnDataset(args, "valid")

    train_dataloder = torch.utils.data.DataLoader(train_dataset, batch_size=50, shuffle=True, collate_fn=train_dataset.collate_fn, num_workers=8)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=50, shuffle=False, collate_fn=valid_dataset.collate_fn, num_workers=8)
    #train_dataset.print_info()
    #valid_dataset.print_info()

    dataloaders = {"train": train_dataloder, "valid": valid_dataloader, "test": None}
    experiment = AhnExperiment(args, dataloaders)
    experiment.train()
