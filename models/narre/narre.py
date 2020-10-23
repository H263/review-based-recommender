import torch
from torch import LongTensor, FloatTensor
import torch.nn as nn
import torch.nn.functional as F

from .layers import NgramFeat
from .utils import masked_tensor

class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pretrained_embeddings=None, padding_idx=0, freeze_embeddings=False):
        super(WordEmbedding, self).__init__()

        self.freeze_embeddings = freeze_embeddings

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.embedding.weight.requires_grad = not self.freeze_embeddings
        if pretrained_embeddings is not None:
            self.embedding.load_state_dict({"weight": torch.tensor(pretrained_embeddings)})
        else:
            print("[Warning] not use pretrained embeddings ...")

    def forward(self, inputs):
        out = self.embedding(inputs)     
        return out

class LinearAttention(nn.Module):
    def __init__(self, vocab_size, feat_size, hidden_dim, dropout, padding_idx = 0):
        super(LinearAttention, self).__init__()

        self.W_rv = nn.Parameter(torch.empty(feat_size, hidden_dim).uniform_(-0.1, 0.1))
        self.W_id = nn.Parameter(torch.empty(hidden_dim, hidden_dim).uniform_(-0.1,0.1))
        self.h = nn.Parameter(torch.empty(hidden_dim, 1).uniform_(-0.1, 0.1))
        self.b_1 = nn.Parameter(torch.empty(hidden_dim).fill_(0.1))
        self.b_2 = nn.Parameter(torch.empty(1).fill_(0.1))

        self.ebd_vals = nn.Embedding(vocab_size, hidden_dim, padding_idx=padding_idx)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, feat, other_id):
        """
        Args:
            feat: [bz, dnum, hidden_dim]
            other_id: [bz, dnum]

        Returns:
            out: [bz, hidden_dim]
            att_scores: [bz, dnum]
        """
        bz = feat.shape[0]
        dnum = feat.shape[1]

        other_ebd = self.ebd_vals(other_id) #[bz, dnum, hiddem_dim]

        att_logits = F.relu(feat @ self.W_rv + other_ebd @ self.W_id + self.b_1) @ self.h + self.b_2


        att_scores = att_logits.exp() / (att_logits.exp().sum(dim=1, keepdim=True) + 1e-8)

        out = torch.sum(torch.mul(att_scores, feat), dim=1) #[bz, hidden_dim]

        out = self.dropout(out)

        return out, att_scores

class LastFeat(nn.Module):
    def __init__(self, vocab_size, feat_size, latent_dim, padding_idx):
        super(LastFeat, self).__init__()

        self.W = nn.Parameter(torch.Tensor(feat_size, latent_dim))
        self.b = nn.Parameter(torch.Tensor(latent_dim))

        self.ebd = nn.Embedding(vocab_size, latent_dim, padding_idx=padding_idx)

        self.reset_parameters()

    def reset_parameters(self):
        bound = 0.1
        nn.init.uniform_(self.W, -bound, bound)
        nn.init.constant_(self.b, bound)
        nn.init.uniform_(self.ebd.weight, -bound, bound)


    def forward(self, text_feat, my_id):
        """
        Args:
            text_feat: [bz, feat_size]
            my_id: [bz]
        """

        out_feat = text_feat @ self.W + self.b + self.ebd(my_id)

        return out_feat

class FM(nn.Module):
    def __init__(self, user_size, item_size, latent_dim, dropout, user_padding_idx, item_padding_idx):
        super(FM, self).__init__()

        self.dropout = nn.Dropout(dropout)

        self.h = nn.Parameter(torch.Tensor(latent_dim, 1))

        self.user_bias = nn.Embedding(user_size, 1, padding_idx=user_padding_idx)
        self.item_bias = nn.Embedding(item_size, 1, padding_idx=item_padding_idx)
        self.g_bias = nn.Parameter(torch.Tensor(1))

        self.reset_parameters()

    def reset_parameters(self):
        bound = 0.1
        nn.init.uniform_(self.h, -bound, bound)
        nn.init.uniform_(self.user_bias.weight, -bound, bound)
        nn.init.uniform_(self.item_bias.weight, -bound, bound)
        nn.init.constant_(self.g_bias, bound)


    def forward(self, u_feat, i_feat, u_id, i_id):
        """
        Args:
            u_feat: [bz, latent_dim]
            i_feat: ...
            u_id: [bz]
            i_id: [bz]

        Returns:
            pred: [bz]
        """
        fm = torch.mul(u_feat, i_feat)
        fm = F.relu(fm)
        fm = self.dropout(fm) #[bz, latent_dim]

        u_bias = self.user_bias(u_id)
        i_bias = self.item_bias(i_id)

        pred = fm @ self.h + u_bias + i_bias + self.g_bias

        return pred

class NARRE(nn.Module):
    def __init__(self, user_size, item_size, vocab_size,
                kernel_sizes, hidden_dim, embedding_dim, att_dim, latent_dim,
                max_doc_num, max_doc_len, dropout, word_padding_idx,
                user_padding_idx, item_padding_idx, pretrained_embeddings, arch):
        super(NARRE, self).__init__()

        self.embedding_dim = embedding_dim
        self.hiddem_dim = hidden_dim
        self.doc_num = max_doc_num 
        self.doc_len = max_doc_len

        self.word_embeddings =  WordEmbedding(vocab_size, embedding_dim, pretrained_embeddings=pretrained_embeddings)

        self.ngram = NgramFeat(kernel_sizes, embedding_dim, hidden_dim,
                                   max_doc_len, arch=arch)

        self.user_att = LinearAttention(item_size, hidden_dim, att_dim, dropout, padding_idx=item_padding_idx)
        self.item_att = LinearAttention(user_size, hidden_dim, att_dim, dropout, padding_idx=user_padding_idx)

        self.user_feat = LastFeat(user_size, hidden_dim, latent_dim, padding_idx=user_padding_idx)
        self.item_feat = LastFeat(item_size, hidden_dim, latent_dim, padding_idx=item_padding_idx)

        self.fm = FM(user_size, item_size, latent_dim, dropout, user_padding_idx=user_padding_idx,
                    item_padding_idx=item_padding_idx)

    def forward(self, u_text, i_text, u_text_masks, i_text_masks, u_id, i_id, reuid, reiid):
        u_text = self.word_embeddings(u_text)
        i_text = self.word_embeddings(i_text)

        # get each doc feature 
        u_text = u_text.view(-1, self.doc_len, self.embedding_dim)
        i_text = i_text.view(-1, self.doc_len, self.embedding_dim) 
        u_text_masks = u_text_masks.view(-1, self.doc_len)
        i_text_masks = i_text_masks.view(-1, self.doc_len)

        u_feat = self.ngram(u_text, u_text_masks)
        i_feat = self.ngram(i_text, i_text_masks)

        u_feat = u_feat.view(-1, self.doc_num, self.hiddem_dim)
        i_feat = i_feat.view(-1, self.doc_num, self.hiddem_dim)
        u_text_masks = u_text_masks.view(-1, self.doc_num)
        i_text_masks = i_text_masks.view(-1, self.doc_num)


        u_feat, u_att_scores = self.user_att(u_feat, reuid)
        i_feat, i_att_scores = self.item_att(i_feat, reiid)

        u_feat = self.user_feat(u_feat, u_id)
        i_feat = self.item_feat(i_feat, i_id)

        pred = self.fm(u_feat, i_feat, u_id, i_id)

        return pred.view(-1), u_att_scores, i_att_scores


