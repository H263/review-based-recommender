import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from .layers import WordEmbedding, MaskedAvgPooling1d, AddictiveAttention, LastFeat, FM, VariationalDropout, FMWithoutUIBias, NodeDropout
from .utils import get_rev_mask

class SimpleSiamese(nn.Module):
    def __init__(self, embedding_dim, latent_dim,
                vocab_size, user_size, item_size,
                pretrained_embeddings, freeze_embeddings, 
                dropout, word_dropout, review_dropout, use_ui_bias, latent_transform):
        super().__init__() 
        self.use_ui_bias = use_ui_bias
        self.embedding_dim = embedding_dim
        self.latent_transform = latent_transform

        self.word_embedding = WordEmbedding(vocab_size, embedding_dim, pretrained_embeddings=pretrained_embeddings,
                                            freeze_embeddings=freeze_embeddings, padding_idx=0)      
        self.var_dropout = VariationalDropout(p=word_dropout)
        self.review_dropout = NodeDropout(p=review_dropout)
        self.masked_pooling_1d = MaskedAvgPooling1d()

        if self.latent_transform:
            self.latent_transform_layer = nn.Sequential(nn.Linear(embedding_dim, latent_dim),
                                                    nn.Tanh())

        self.user_last_feat_layer = LastFeat(user_size, latent_dim if self.latent_transform else embedding_dim, latent_dim, padding_idx=0)
        self.item_last_feat_layer = LastFeat(item_size, latent_dim if self.latent_transform else embedding_dim, latent_dim, padding_idx=0)

        self.review_att_layer = AddictiveAttention(latent_dim if self.latent_transform else embedding_dim, latent_dim)

        if self.use_ui_bias:
            self.fm = FM(user_size, item_size, latent_dim, dropout, user_padding_idx=0, item_padding_idx=0)
        else:
            self.fm = FMWithoutUIBias(user_size, item_size, latent_dim, dropout, user_padding_idx=0, item_padding_idx=0)

    def forward(self, u_revs, i_revs, u_rev_word_masks, i_rev_word_masks, u_rev_masks, i_rev_masks, u_ids, i_ids):
        """
        Args:
            u_revs: [bz, rv_num, rv_len]
            i_revs:
            u_rev_word_masks: [bz, rv_num, rv_len]
            i_rev_word_masks
            u_rev_masks: [bz, rv_num]
            i_rev_masks: [bz, rv_num]
            u_ids: [bz]
            i_ids: [bz]
        
        Returns:
            out_logits: [bz]
            u_rev_scores: [bz, rv_num]
            i_rev_scores: [bz, ]
        """
        bz, u_rv_num, rv_len = list(u_revs.size())
        bz, i_rv_num, rv_len = list(i_revs.size())

        # each review representation
        u_revs = self.var_dropout(self.word_embedding(u_revs).view(bz*u_rv_num, rv_len, self.embedding_dim)).transpose(1,2)
        i_revs = self.var_dropout(self.word_embedding(i_revs).view(bz*i_rv_num, rv_len, self.embedding_dim)).transpose(1,2)

        # avg pooling 
        u_revs = self.masked_pooling_1d(u_revs, u_rev_word_masks.view(bz*u_rv_num, rv_len)).view(bz, u_rv_num, self.embedding_dim)
        i_revs = self.masked_pooling_1d(i_revs, i_rev_word_masks.view(bz*i_rv_num, rv_len)).view(bz, i_rv_num, self.embedding_dim)

        if self.latent_transform:
            u_revs = self.latent_transform_layer(u_revs)
            i_revs = self.latent_transform_layer(i_revs)

        # review dropout 
        u_revs = self.review_dropout(u_revs)
        i_revs = self.review_dropout(i_revs)

        # user/item representation 
        u_rev_feat, _ = self.review_att_layer(u_revs, u_rev_masks)
        i_rev_feat, _ = self.review_att_layer(i_revs, i_rev_masks)

        # user/item combine representation 
        u_feat = self.user_last_feat_layer(u_rev_feat, u_ids)
        i_feat = self.item_last_feat_layer(i_rev_feat, i_ids) #[bz, hdim]

        # fm
        if self.use_ui_bias:
            out_logits = self.fm(u_feat, i_feat, u_ids, i_ids)
        else:
            out_logits = self.fm(u_feat, i_feat)

        return out_logits.view(bz), None, None