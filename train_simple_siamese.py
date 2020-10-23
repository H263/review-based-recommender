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
import random

import torch 
import torch.nn as nn
from torch import LongTensor, FloatTensor
import numpy as np
from gensim.models import KeyedVectors

from experiment import Experiment
from utils import get_mask
from data.divide_and_create_examples import clean_str

class MultipleOptimizer(object):
    def __init__(self, *op):
        self.optimizers = op 
        self.param_groups = self.optimizers[-1].param_groups
    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()
    def step(self):
        for op in self.optimizers:
            op.step()

    def state_dict(self):
        list_of_state_dict = []
        for op in self.optimizers:
            list_of_state_dict.append(op.state_dict())
        return list_of_state_dict

class MultipleScheduler(object):
    def __init__(self, Scheduler, *ops, **kwargs):
        self._optimizers = ops
        self._schedulers =  [Scheduler(optim, **kwargs) for optim in self._optimizers]
    
    def step(self, val):
        for sl in self._schedulers:
            sl.step(val)

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

class NarreExperiment(Experiment):
    def __init__(self, args, dataloaders):
        super(NarreExperiment, self).__init__(args, dataloaders)

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
        if self.args.sparse:
            self.scheduler = MultipleScheduler(torch.optim.lr_scheduler.ReduceLROnPlateau, self.sparse_optim, self.dense_optim,
                                            mode="min", factor=0.5, patience=0)
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=0)

    def parse_kernel_sizes(self, str_kernel_sizes):
        kernel_sizes = [int(x) for x in str_kernel_sizes.split(",")]
        print("kernel sizes are: ", kernel_sizes)
        return kernel_sizes
        
    def build_model(self):
        # import different model 
        from models.siamese.simple_siamese import SimpleSiamese
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

        self.model = SimpleSiamese(embedding_dim=self.args.embedding_dim, 
                             latent_dim=self.args.latent_dim, vocab_size=len(_dataset.word_vocab), 
                             user_size=_dataset.user_num, item_size=_dataset.item_num,  
                             pretrained_embeddings=word_pretrained, freeze_embeddings=self.args.freeze_embeddings,
                             dropout=self.args.dropout, word_dropout=self.args.word_dropout, review_dropout=self.args.review_dropout,
                             use_ui_bias=self.args.use_ui_bias,
                             latent_transform=self.args.latent_transform)
        if self.args.parallel:
            self.model = torch.nn.DataParallel(self.model)
            self.print_write_to_log("the model is parallel training.")
        self.model.to(self.device)

    def build_optimizer(self):
        def get_sparse_and_dense_parameters(model):
            sparse_params = []
            dense_params = []
            for name, params in model.named_parameters():
                if name == "word_embedding.embedding.weight":
                    sparse_params.append(params)
                else:
                    dense_params.append(params)
            print(f"len of params, sparse params, dense params: {len(model.state_dict())}, {len(sparse_params)}, {len(dense_params)}")
            return sparse_params, dense_params

        if self.args.sparse:
            sparse_params, dense_params = get_sparse_and_dense_parameters(self.model)

            self.sparse_optim = torch.optim.SparseAdam(sparse_params, lr=self.args.lr)
            self.dense_optim = torch.optim.Adam(dense_params, lr=self.args.lr)
        
            self.optimizer = MultipleOptimizer(self.sparse_optim, self.dense_optim)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        if self.args.verbose:
            self.print_write_to_log(re.sub(r"\n", "", self.optimizer.__repr__()))
        
    def build_loss_func(self):
        self.loss_func = nn.MSELoss()
        self.bce_loss_func = nn.BCEWithLogitsLoss()



    def train_one_epoch(self, current_epoch):
        avg_mse_loss = AvgMeters()
        square_error = 0.
        accum_count = 0
        start_time = time.time()

        self.model.train()
        for i,  ((u_ids, i_ids, ratings), (u_revs, i_revs, u_rev_word_masks, i_rev_word_masks, u_rev_masks, i_rev_masks), (u_rids, i_rids), \
                (_)) in enumerate(self.train_dataloader):
            if i == 0 and current_epoch == 0:
                print("u_text", u_revs.shape, "i_text", i_revs.shape, "reuid", u_rids.shape, "reiid", i_rids.shape)
            u_ids = u_ids.to(self.device)
            i_ids = i_ids.to(self.device)
            ratings = ratings.to(self.device)
            u_revs = u_revs.to(self.device)
            i_revs = i_revs.to(self.device)
            u_rev_word_masks = u_rev_word_masks.to(self.device)
            i_rev_word_masks = i_rev_word_masks.to(self.device)
            u_rev_masks = u_rev_masks.to(self.device)
            i_rev_masks = i_rev_masks.to(self.device)
            self.optimizer.zero_grad()
            y_pred, _, _ = self.model(u_revs, i_revs, u_rev_word_masks, i_rev_word_masks, u_rev_masks, i_rev_masks, 
                                u_ids, i_ids)

            loss = self.loss_func(y_pred, ratings)
            #loss_2 = self.bce_loss_func(gt_logits, ui_labels) + self.bce_loss_func(ng_logits, neg_ui_labels)
            #loss = loss_1 + loss_2
            loss.backward()

            gnorm = nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()

            # val 
            avg_mse_loss.update(loss.mean().item())
            square_error += loss.mean().item() * ratings.size(0)
            accum_count += ratings.size(0)

            #avg_cls_loss.update(loss_2.mean().item())

            # log
            if (i+1) % self.args.log_idx == 0 and self.args.log:
                elpased_time = (time.time() - start_time) / self.args.log_idx
                rmse = math.sqrt(square_error / accum_count)

                """
                log_text = "epoch: {}/{}, step: {}/{}, loss mse: {:.3f}, loss cls: {:.3f}, rmse: {:.3f}, lr: {}, gnorm: {:3f}, time: {:.3f}".format(
                    current_epoch, self.args.epochs,  (i+1), len(self.train_dataloader), avg_mse_loss.val, avg_cls_loss.val, rmse, 
                    self.optimizer.param_groups[0]["lr"], gnorm, elpased_time
                )
                """
                log_text = "epoch: {}/{}, step: {}/{}, loss mse: {:.3f}, rmse: {:.3f}, lr: {}, gnorm: {:3f}, time: {:.3f}".format(
                    current_epoch, self.args.epochs,  (i+1), len(self.train_dataloader), avg_mse_loss.val, rmse, 
                    self.optimizer.param_groups[0]["lr"], gnorm, elpased_time
                )
                self.print_write_to_log(log_text)

                avg_mse_loss.reset()
                #avg_cls_loss.reset()
                square_error = 0. 
                accum_count = 0
                start_time = time.time()

    def valid_one_epoch(self):
        square_error = 0.
        accum_count = 0
        avg_loss = AvgMeters()

        self.model.eval()
        for i,  ((u_ids, i_ids, ratings), (u_revs, i_revs, u_rev_word_masks, i_rev_word_masks, u_rev_masks, i_rev_masks), \
                 (u_rids, i_rids)) in enumerate(self.valid_dataloader):
            u_ids = u_ids.to(self.device)
            i_ids = i_ids.to(self.device)
            ratings = ratings.to(self.device)
            u_revs = u_revs.to(self.device)
            i_revs = i_revs.to(self.device)
            u_rev_word_masks = u_rev_word_masks.to(self.device)
            i_rev_word_masks = i_rev_word_masks.to(self.device)
            u_rev_masks = u_rev_masks.to(self.device)
            i_rev_masks = i_rev_masks.to(self.device)

            with torch.no_grad():
                y_pred, _, _ = self.model(u_revs, i_revs, u_rev_word_masks, i_rev_word_masks, u_rev_masks, i_rev_masks, 
                                u_ids, i_ids)
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
        
        if self.args.use_scheduler:
            self.scheduler.step(rmse)

    @property
    def best_rmse(self):
        return self._best_rmse
    
    @best_rmse.setter
    def best_rmse(self, val):
        self._best_rmse = val

    def train(self):
        print("start training ...")
        for epoch in range(self.args.epochs):
            self.valid_one_epoch()
            self.train_one_epoch(epoch)

class SiameseDatasetNotBalance(torch.utils.data.Dataset):
    def __init__(self, args, set_name, u_rv_num):
        super(NarreDatasetSameUIReviewNum, self).__init__()

        self.args = args
        self.set_name = set_name
        param_path = os.path.join(self.args.data_dir, "meta.pkl")
        with open(param_path, "rb") as f:
            para = pickle.load(f)

        self.user_num = para['user_num']
        self.item_num = para['item_num']
        self.indexlizer = para['indexlizer']
        self.rv_num = para["rv_num"]
        self.rv_len = para["rv_len"]
        self.u_text = para['user_reviews']
        self.i_text = para['item_reviews']
        self.u_rids = para["user_rids"]
        self.i_rids = para["item_rids"]
        self.word_vocab = self.indexlizer._vocab
        
        self.u_rv_num = u_rv_num
        self.i_rv_num = self.rv_num
        self.sample_train_review = self.args.sample_train_review
        assert self.u_rv_num <= self.rv_num

        example_path = os.path.join(self.args.data_dir, f"{set_name}_exmaples.pkl")
        with open(example_path, "rb") as f:
            self.examples = pickle.load(f)

    def uniform_sample_reviews(self, revs, rv_num):
        non_zero_indicies = np.nonzero(np.sum(revs, axis=1))[0]
        np.random.shuffle(non_zero_indicies)

        new_revs = []
        for i, idx in enumerate(non_zero_indicies):
            if i < rv_num:
                new_revs.append(revs[idx])
        
        if len(new_revs) < rv_num:
            new_revs += [[0] * self.rv_len]

        return new_revs
        

    def __getitem__(self, i):
        # for each review(u_text or i_text) [...] 
        # NOTE: not padding 
        if self.set_name == "train":
            u_id, i_id, rating, u_revs, i_revs, u_rids, i_rids, _, _, ui_rev = self.examples[i]
            print("org: ", u_revs[:4])
            if self.sample_train_review:
                u_revs = self.uniform_sample_reviews(u_revs, self.u_rv_num)
            print("after: ", u_revs[:4])
            
            
            neg_idx = random.randint(0, len(self.examples)-1) 
            while self.examples[neg_idx][1] == i_id:
                neg_idx = random.randint(0, len(self.examples)-1)
            neg_ui_rev = self.examples[neg_idx][-1]


            ui_label = 1. 
            neg_ui_label = 0.

            return u_id, i_id, rating, u_revs, i_revs, u_rids, i_rids, ui_rev, neg_ui_rev, ui_label, neg_ui_label      

        else:
            u_id, i_id, rating, u_revs, i_revs, u_rids, i_rids, _, _ = self.examples[i]
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

    def train_collate_fn(self, batch):
        u_ids, i_ids, ratings, u_revs, i_revs, u_rids, i_rids, ui_revs, neg_ui_revs, ui_labels, neg_ui_labels = zip(*batch)
        u_ids = LongTensor(u_ids)
        i_ids = LongTensor(i_ids)
        ratings = FloatTensor(ratings)
        u_revs = LongTensor(u_revs)
        i_revs = LongTensor(i_revs)
        u_rids = LongTensor(u_rids)
        i_rids = LongTensor(i_rids)
        ui_revs = LongTensor(ui_revs) 
        neg_ui_revs = LongTensor(neg_ui_revs)
        ui_labels = FloatTensor(ui_labels)
        neg_ui_labels = FloatTensor(neg_ui_labels)

        u_rev_word_masks = get_mask(u_revs)
        i_rev_word_masks = get_mask(i_revs)
        ui_word_masks = get_mask(ui_revs)
        neg_ui_word_masks = get_mask(neg_ui_revs)
        u_rev_masks = self.get_rev_mask(u_revs)
        i_rev_masks = self.get_rev_mask(i_revs)

        return (u_ids, i_ids, ratings), (u_revs, i_revs, u_rev_word_masks, i_rev_word_masks, u_rev_masks, i_rev_masks), (u_rids, i_rids), \
                (ui_revs, neg_ui_revs, ui_word_masks, neg_ui_word_masks,  ui_labels, neg_ui_labels)

    def test_collate_fn(self, batch):
        u_ids, i_ids, ratings, u_revs, i_revs, u_rids, i_rids = zip(*batch)
        
        u_ids = LongTensor(u_ids)
        i_ids = LongTensor(i_ids)
        ratings = FloatTensor(ratings)
        u_revs = LongTensor(u_revs)
        i_revs = LongTensor(i_revs)
        u_rids = LongTensor(u_rids)
        i_rids = LongTensor(i_rids)

        u_rev_word_masks = get_mask(u_revs)
        i_rev_word_masks = get_mask(i_revs)
        u_rev_masks = self.get_rev_mask(u_revs)
        i_rev_masks = self.get_rev_mask(i_revs)

        return (u_ids, i_ids, ratings), (u_revs, i_revs, u_rev_word_masks, i_rev_word_masks, u_rev_masks, i_rev_masks), (u_rids, i_rids)
     


class NarreDatasetSameUIReviewNum(torch.utils.data.Dataset):
    def __init__(self, args, set_name):
        super(NarreDatasetSameUIReviewNum, self).__init__()

        self.args = args
        self.set_name = set_name
        param_path = os.path.join(self.args.data_dir, "meta.pkl")
        with open(param_path, "rb") as f:
            para = pickle.load(f)

        self.user_num = para['user_num']
        self.item_num = para['item_num']
        self.indexlizer = para['indexlizer']
        self.rv_num = para["rv_num"]
        self.rv_len = para["rv_len"]
        self.u_text = para['user_reviews']
        self.i_text = para['item_reviews']
        self.u_rids = para["user_rids"]
        self.i_rids = para["item_rids"]
        self.word_vocab = self.indexlizer._vocab

        self.sample_train_review = self.args.sample_train_review
        self.u_rv_num = self.args.u_rv_num
        self.i_rv_num = self.args.i_rv_num

        example_path = os.path.join(self.args.data_dir, f"{set_name}_exmaples.pkl")
        with open(example_path, "rb") as f:
            self.examples = pickle.load(f)

    def uniform_sample_reviews(self, revs, rv_num):
        non_zero_indicies = np.nonzero(np.sum(revs, axis=1))[0]
        np.random.shuffle(non_zero_indicies)

        new_revs = []
        for i, idx in enumerate(non_zero_indicies):
            if i < rv_num:
                new_revs.append(revs[idx])
        
        if len(new_revs) < rv_num:
            new_revs += [[0] * self.rv_len] * (rv_num - len(new_revs))

        return new_revs

    def __getitem__(self, i):
        # for each review(u_text or i_text) [...] 
        # NOTE: not padding 
        if self.set_name == "train":
            u_id, i_id, rating, u_revs, i_revs, u_rids, i_rids, _, _, ui_rev = self.examples[i]
            #print("org: ", u_revs[:4])
            if self.sample_train_review:
                u_revs = self.uniform_sample_reviews(u_revs, self.u_rv_num)
                i_revs = self.uniform_sample_reviews(i_revs, self.i_rv_num)
            #print("after: ", u_revs[:4])
            
            
            neg_idx = random.randint(0, len(self.examples)-1) 
            while self.examples[neg_idx][1] == i_id:
                neg_idx = random.randint(0, len(self.examples)-1)
            neg_ui_rev = self.examples[neg_idx][-1]


            ui_label = 1. 
            neg_ui_label = 0.

            return u_id, i_id, rating, u_revs, i_revs, u_rids, i_rids, ui_rev, neg_ui_rev, ui_label, neg_ui_label      

        else:
            u_id, i_id, rating, u_revs, i_revs, u_rids, i_rids, _, _ = self.examples[i]
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

    def train_collate_fn(self, batch):
        u_ids, i_ids, ratings, u_revs, i_revs, u_rids, i_rids, ui_revs, neg_ui_revs, ui_labels, neg_ui_labels = zip(*batch)
        u_ids = LongTensor(u_ids)
        i_ids = LongTensor(i_ids)
        ratings = FloatTensor(ratings)
        u_revs = LongTensor(u_revs)
        i_revs = LongTensor(i_revs)
        u_rids = LongTensor(u_rids)
        i_rids = LongTensor(i_rids)
        ui_revs = LongTensor(ui_revs) 
        neg_ui_revs = LongTensor(neg_ui_revs)
        ui_labels = FloatTensor(ui_labels)
        neg_ui_labels = FloatTensor(neg_ui_labels)

        u_rev_word_masks = get_mask(u_revs)
        i_rev_word_masks = get_mask(i_revs)
        ui_word_masks = get_mask(ui_revs)
        neg_ui_word_masks = get_mask(neg_ui_revs)
        u_rev_masks = self.get_rev_mask(u_revs)
        i_rev_masks = self.get_rev_mask(i_revs)

        return (u_ids, i_ids, ratings), (u_revs, i_revs, u_rev_word_masks, i_rev_word_masks, u_rev_masks, i_rev_masks), (u_rids, i_rids), \
                (ui_revs, neg_ui_revs, ui_word_masks, neg_ui_word_masks,  ui_labels, neg_ui_labels)

    def test_collate_fn(self, batch):
        u_ids, i_ids, ratings, u_revs, i_revs, u_rids, i_rids = zip(*batch)
        
        u_ids = LongTensor(u_ids)
        i_ids = LongTensor(i_ids)
        ratings = FloatTensor(ratings)
        u_revs = LongTensor(u_revs)
        i_revs = LongTensor(i_revs)
        u_rids = LongTensor(u_rids)
        i_rids = LongTensor(i_rids)

        u_rev_word_masks = get_mask(u_revs)
        i_rev_word_masks = get_mask(i_revs)
        u_rev_masks = self.get_rev_mask(u_revs)
        i_rev_masks = self.get_rev_mask(i_revs)

        return (u_ids, i_ids, ratings), (u_revs, i_revs, u_rev_word_masks, i_rev_word_masks, u_rev_masks, i_rev_masks), (u_rids, i_rids)
     

if __name__ == "__main__":
    config_file = "./models/siamese/defalut_simple_train.json"
    args = parse_args(config_file)
    train_dataset = NarreDatasetSameUIReviewNum(args, "train")
    valid_dataset = NarreDatasetSameUIReviewNum(args, "test")

    train_dataloder = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.train_collate_fn, num_workers=4)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=valid_dataset.test_collate_fn, num_workers=4)

    dataloaders = {"train": train_dataloder, "valid": valid_dataloader, "test": None}
    experiment = NarreExperiment(args, dataloaders)
    experiment.train()