import math

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from .utils import masked_tensor, masked_softmax, attention_weighted_sum, get_mask, masked_colwise_mean

"""
Not use bias when computing WordScore Layer
"""

class VariationalDropout(torch.nn.Dropout):
    """
    Apply the dropout technique in Gal and Ghahramani, [Dropout as a Bayesian Approximation:
    Representing Model Uncertainty in Deep Learning](https://arxiv.org/abs/1506.02142) to a
    3D tensor.
    This module accepts a 3D tensor of shape `(batch_size, num_timesteps, embedding_dim)`
    and samples a single dropout mask of shape `(batch_size, embedding_dim)` and applies
    it to every time step.
    """

    def forward(self, input_tensor):

        """
        Apply dropout to input tensor.
        # Parameters
        input_tensor : `torch.FloatTensor`
            A tensor of shape `(batch_size, num_timesteps, embedding_dim)`
        # Returns
        output : `torch.FloatTensor`
            A tensor of shape `(batch_size, num_timesteps, embedding_dim)` with dropout applied.
        """
        ones = input_tensor.data.new_ones(input_tensor.shape[0], input_tensor.shape[-1])
        dropout_mask = torch.nn.functional.dropout(ones, self.p, self.training, inplace=False)
        if self.inplace:
            input_tensor *= dropout_mask.unsqueeze(1)
            return None
        else:
            return dropout_mask.unsqueeze(1) * input_tensor
            
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

class MaskedAvgPooling1d(nn.Module):
    def __init__(self):
        super().__init__() 
    
    def forward(self, inputs, input_masks):
        """
        Args:
            inputs: [bz, hdim, seq_len]
            input_masks: [bz, seq_len]

        Returns: 
            outputs: [bz, hdim, 1]
        """
        assert input_masks.dim() == 2
        float_masks = input_masks.unsqueeze(1).float() #[bz, 1, seq_len]
        input_lengths = float_masks.sum(dim=2, keepdim=True) + 1e-8  #[bz, 1, 1]
        #print("inputs: ", inputs)
        #print("input_lengths: ", input_lengths)
        sum_inputs = (inputs * float_masks).sum(dim=2, keepdim=True) #[bz, hdim, 1]
        #print(sum_inputs)
        return sum_inputs / input_lengths

class HierPooling(nn.Module):
    """
    Shout out to masiwei 

    First Avg Pooling with given kernels, then Max Pooling on top of the new generated features
    Support: 
        - 
    TODO:
        - Add Highway Layer after avg_pool, before max_pool ?
    """
    def __init__(self, in_features, out_features, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size

        if in_features != out_features:
            self.proj_layer = nn.Linear(in_features, out_features)
        else:
            self.proj_layer = None
    
    def forward(self, inputs):
        """
        NOTE: use N x C x L format
        Args:
            inputs: [bz, in_features, seq_len]
        
        Returns:
            outputs: [bz, out_features]
        """
        inputs = F.avg_pool1d(inputs, self.kernel_size, stride=1)
        seq_len = inputs.size(2)
        outputs = F.max_pool1d(inputs, seq_len).squeeze(2)
        assert outputs.dim() == 2

        if self.proj_layer is not None:
            outputs = self.proj_layer(outputs)

        return outputs

class MyConv1d(nn.Module):
    """
    Support:
        multiple kernel sizes
    """
    def __init__(self, kernel_sizes, in_features, out_features):
        super().__init__()

        if type(kernel_sizes) is str:
            # allow kernel_sizes look like "3,4,5"
            kernel_sizes = [int(x) for x in kernel_sizes.split(",")]

        assert out_features % len(kernel_sizes) == 0
        assert all([kz % 2 == 1 for kz in kernel_sizes])
        

        self.out_features_per_kz = out_features // len(kernel_sizes)
        self.list_of_conv1d = nn.ModuleList([
            nn.Conv1d(in_features, self.out_features_per_kz, kz, padding=(kz-1)//2) for kz in kernel_sizes
        ])
    def forward(self, inputs):
        """
        NOTE: is N x C x L format !!!
        Args:
            inputs: [bz, in_features, seq_len]
        Returns:
            outpus: [bz, out_features, seq_len]
        """
        list_of_outpus = []
        for conv1d_layer in self.list_of_conv1d:
            sub_outputs = conv1d_layer(inputs) #[bz, sub_feat, seq_len]
            list_of_outpus.append(sub_outputs)
        outputs = torch.cat(list_of_outpus, dim=1)

        return outputs

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
        nn.init.constant_(self.b, 0.)
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
        nn.init.constant_(self.g_bias, 0.)


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


class AddictiveAttention(nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        super().__init__()

        self.proj_layer = nn.Sequential(nn.Linear(hidden_dim, latent_dim),
                                        nn.Tanh())
        self.inner_product = nn.Linear(latent_dim, 1, bias=False)
    
    def forward(self, inputs, input_masks):
        """
        Args:
            inputs: [bz, seq_len, hdim]
            input_masks: [bz, seq_len]
        
        Returns:
            Outputs: [bz, hdim]
        """
        bz, seq_len, hdim = list(inputs.size())

        att_logtis = self.inner_product(self.proj_layer(inputs)) #[bz, seq_len, 1]
        input_masks = input_masks.unsqueeze(2) #[bz, seq_len, 1]
        att_scores = F.softmax(torch.masked_fill(att_logtis, ~input_masks, -1e8), dim=1) #[bz, seq_len, 1]

        outptus = torch.sum(att_scores * inputs, dim=1)

        return outptus, att_scores


class SeqEncoder(nn.Module):
    """
    Support:
        - Conv1d
        - AvgPooling with kernel size
    """
    def __init__(self, kernel_sizes, in_features, out_features, arch_encoder):
        super().__init__() 

        self.kernel_sizes = kernel_sizes
        self.in_features = in_features
        self.out_features = out_features 
        self.arch_encoder = arch_encoder

        if self.arch_encoder == "CNN":
            self.seq_encoder =  nn.Sequential(MyConv1d(kernel_sizes, in_features, out_features),
                                                nn.ReLU())
        elif self.arch_encoder =="AvgPooling":
            assert len(kernel_sizes) == 1
            kernel_size = kernel_sizes[0]

            self.seq_encoder = nn.Sequential(nn.AvgPool1d(kernel_size, stride=1),
                                            nn.ReLU())
        else:
            raise ValueError(f"{self.arch_encoder} is not predefined")

    def forward(self, inputs, input_masks):
        inputs = masked_tensor(inputs, input_masks)
        inputs = inputs.transpose(1,2) #[bz, feat, seq_len]
        outputs = self.seq_encoder(inputs)
        outputs = outputs.transpose(1,2) #[bz, seq_len,feat]
        return outputs.contiguous()

class TanhNgramFeat(nn.Module):
    def __init__(self, kernel_sizes, in_feature, out_feature_per_kernel, seq_len, mode="MAX_AVG"):
        super().__init__()
        assert all([ type(x)==int for x in kernel_sizes])
        self.out_feature_per_kernel = out_feature_per_kernel
        self.seq_len = seq_len
        self.mode = mode 

        self.ngrams = nn.ModuleList([nn.Sequential(nn.Conv1d(in_feature, out_feature_per_kernel, kz), nn.Tanh()) 
                        for kz in kernel_sizes])
        self.kernel_sizes = kernel_sizes
        
        if "AVG" in self.mode:
            self.masked_avgpool_1d = MaskedAvgPooling1d()
            print("use avg pooling")
        if "MAX" in self.mode:
            print('use maxpooling')
    
    def forward(self, inputs, input_masks):
        """
        Args:
            inputs: [bz, seq_len, in_feat]
            input_masks: [bz, seq_len]

        Returns:
            out_ngrams: [bz, out_feat]
            out_seqs: [bz, seq_len, out_feat]
        """
        bz, seq_len, _ = list(inputs.size())
        inputs = masked_tensor(inputs, input_masks)
        inputs = inputs.transpose(1,2)
        list_of_ngram = [ng_layer(inputs) for ng_layer in self.ngrams] # list of [bz, out_feat, var_seq_lens]
        #print("list of ngram: ", list_of_ngram)
        #print(list_of_ngram[0].shape)
        out_ngrams = []

        if "MAX" in self.mode:
            #print("use maxpooling")
            list_max_ngrams = [F.max_pool1d(ngram, self.seq_len-kz+1) for ngram, kz in zip(list_of_ngram, self.kernel_sizes)] # list of [bz, out_feat, 1]
            out_ngrams += list_max_ngrams
        if "AVG" in self.mode:
            #print("use avgpooling")
            list_avg_masks = [input_masks[:, :self.seq_len-kz+1]  for kz in self.kernel_sizes]
            list_avg_ngrams = [self.masked_avgpool_1d(ngram, mask) for ngram, mask in  zip(list_of_ngram, list_avg_masks)] # list of [bz, out_feat, 1]
            out_ngrams += list_avg_ngrams

        out_ngrams = torch.cat(out_ngrams, dim=1).view(bz, -1)

        return out_ngrams


class MaxAvgPooling1d(nn.Module):
    def __init__(self):
        super().__init__()

        self.masked_avgpool_1d = MaskedAvgPooling1d()
    
    def forward(self, inputs, masks):
        """
        Args:
            inputs: [bz, seq_len, hidden_dim]
            masks: [bz, seq_len]

        Returns:
            outputs: [bz, 2*hidden_dim]
        """
        bz, seq_len, hdim = list(inputs.size())

        inputs = inputs.transpose(1,2)

        avg_feat = self.masked_avgpool_1d(inputs, masks) #[bz, hidden_dim, 1]
        max_feat = self.max_pool1d(inputs, seq_len) #[bz, hidden_dim, 1]

        return torch.cat([avg_feat, max_feat], dim=1).view(bz, 2*hdim)

class NgramFeat(nn.Module):
    def __init__(self, kernel_sizes, in_features, out_features, seq_len, dropout=0., arch="CNN"):
        super().__init__()

        self.arch = arch
        if arch == "CNN":
            print("use CNN archiecture for Ngram.")
            self.feature_layer = nn.Sequential(MyConv1d(kernel_sizes, in_features, out_features),
                                                nn.ReLU(),
                                                nn.MaxPool1d(seq_len))
        elif arch == "HierPooling":
            print("use HierPooling arch for Ngram.")
            assert len(kernel_sizes) == 1
            self.feature_layer = nn.Sequential(HierPooling(in_features, out_features, kernel_sizes[0]),
                                                nn.ReLU())
        else:
            raise ValueError(f"{arch} is not predefined.")
        
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None 

    def forward(self, inputs, input_masks):
        """
        Args:
            inputs: [bz, seq_len, in_features]
            input_masks: [bz, seq_len]
        Returns:
            ouptuts: [bz, out_features]
        """
        inputs = masked_tensor(inputs, input_masks)
        inputs = inputs.transpose(1,2) #[bz, feat, seq_len]
        outputs = self.feature_layer(inputs)
        outputs = outputs.contiguous()
        
        return outputs

# ========= Interaction Modules ==============
class CosineInteraction(nn.Module):
    def __init__(self):
        super().__init__() 
    
    def forward(self, input1, input2):
        """
        input1: [*, seq_len_a, dim]
        input2: [*, seq_len_b, dim]
        """
        device = input1.device

        input1_norm = torch.norm(input1, p=2, dim=-1, keepdim=True)
        input2_norm = torch.norm(input2, p=2, dim=-1, keepdim=True) #[*, seq_len_b, 1]

        _y = torch.bmm(input1, input2.transpose(1,2)) # [*, seq_len_a, seq_len_b]
        _y_norm = torch.bmm(input1_norm, input2_norm.transpose(1,2)) #[*, seq_len_a, seq_len_b]
        epilson = torch.tensor(1e-6).to(device)

        return _y / torch.max(_y_norm, epilson)
        

class TensorInteraction(nn.Module):
    def __init__(self, in_feat, k_factor, bias=False):
        """
        implement the A_i = X^T W_i Y,  i = 1, ..., k_factor
        X: [bz, seq_len_a, dim]
        Y: [bz, seq_len_b, dim]
        W_i: [dim, dim] 

        And get A = element-wise-max(A_1, ..., A_k) -->: [bz, seq_len_a, seq_len_b]
        """
        super(TensorInteraction, self).__init__()

        self.weight = nn.Parameter(torch.Tenensor(k_factor, in_feat, in_feat))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(k_factor))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.weight.size(1))
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input1, input2):
        """
        Args:
            input_1: [bz, seq_len_a, dim]
            input_2: [bz, seq_len_b, dim]
        Returns:
            out_feat: [bz, seq_len_a, seq_len_b]
        """
        weight = self.weight
        bias = self.bias 
        out_feat = []
        
        weight_slices = torch.chunk(weight, weight.size(0), dim=0)
        input1 = input1.unsqueeze(-2) #[bz, *, 1, in1_feat]
        input2 = input2.unsqueeze(-2) #[bz, *, 1, in2_feat]
        
        for i, W in enumerate(weight_slices):
            _y = torch.matmul(input1, W) #[bz, seq_len_a, dim]
            _y = torch.bmm(_y, input2.transpose(1,2))  #[bz, seq_len_a, seq_len_b]
            if bias is not None:
                _y = _y + bias[i]
            out_feat.append(_y) # list of [bz, seq_len_a, seq_len_b]
        
        out_feat.cat(dim=3).max(dim=3)

        return out_feat

class BiLinearInteraction(nn.Module):
    """
    implement the A = X^T W Y, 
    X: [bz, seq_len_a, dim]
    Y: [bz, seq_len_b, dim]
    W: [dim, dim] 
    And get A: [bz, seq_len_a, seq_len_b]
    """
    def __init__(self, feat_dim, bias=False):
        super(BiLinearInteraction, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(feat_dim, feat_dim))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        bound = 1. / math.sqrt(self.weight.size(0))
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        
    def forward(self, input1, input2):
        _y = torch.matmul(input1, self.weight) #[bz, seq_len_a, dim]
        _y = torch.bmm(_y, input2.transpose(1,2)) #[bz, seq_len_a, seq_len_b]
        if self.bias is not None:
            _y += self.bias
        return _y

class DotInteraction(nn.Module):
    """
    implement the A = X^T Y 
    """
    def __init__(self, feat_dim, scale=False):
        super(DotInteraction, self).__init__()

        if scale:
            self.scale = 1. / math.sqrt(feat_dim)
        else:
            self.register_buffer("scale", None)

    def forward(self, input1, input2):
        _y = torch.bmm(input1, input2.transpose(1,2)) 
        if self.scale is not None:
            _y = _y * self.scale 
        return _y

class CoAttention(nn.Module):
    def __init__(self, in_feature, out_feature, interaction_type="DOT", feature_type="FC", pooling="MEAN", **kwargs):
        """
        Args:
            interaction_type: support `DOT`, `SCALEDDOT`, `BILINEAR` `TENSOR`
            feature_type: support `IDENTITY`, `FC`
            pooling: support `MATRIX`, `MAX, `MEAN`
        """
        super(CoAttention, self).__init__()
        
        self.pooling = pooling
        
        if interaction_type == "DOT":
            self.interaction = DotInteraction(out_feature, scale=False)
        elif interaction_type == "SCALEDDOT":
            self.interaction = DotInteraction(out_feature, scale=True)
        elif interaction_type == "BILINEAR":
            self.interaction = BiLinearInteraction(out_feature, bias=False)
        elif interaction_type == "TENSOR":
            k_factor = kwargs["k_factor"]
            if k_factor <= 1:
                raise ValueError("TENSOR interaction should have > 1 k_factor")
            self.interaction = TensorInteraction(out_feature, k_factor=k_factor, bias=False)
        else:
            raise ValueError("interaction_type {} is not predefined".format(interaction_type))

        if feature_type == "IDENTITY":
            self.transform_feature = nn.Identity()
        elif feature_type == "FC":
            self.transform_feature = nn.Sequential(nn.Linear(in_feature, out_feature),
                                                    nn.ReLU())
            nn.init.xavier_normal_(self.transform_feature[0].weight, gain=nn.init.calculate_gain("relu"))


    def forward(self, seq_a, seq_b, mask_a, mask_b):
        """
        Args:
            seq_a: [bz, doc_num, doc_length, dim] --> [bz, 1, doc_numxdoc_length, dim] --> [bz, doc_num, doc_numxdoc_length, dim]
                                                --> [bzxdoc_num, doc_numxdoc_length, dim]
            seq_b: [bz, doc_num, doc_length, dim]

            mask_a: [bz, doc_num, doc_length] --> [bz, 1, doc_numxdoc_length] --> [bz, doc_num, doc_numxdoc_length]
                                        --> [bzxdoc_num, doc_numxdoc_length]

            seq_a: [bzxdoc_num, doc_length, dim]
            expand_seq_b: [bzxdoc_num, doc_num, doc_length, dim]

            seq_b: [bzxdoc_num, doc_length, dim]
            expand_seq_a = [bzxdoc_num, doc_num, doc_length, dim]

        Returns:
            a_out: [bz*rv_num_a, dim]
            b_out: [bz*rv_num_b, dim]
            atob_weights: [bz*rv_num_a, rv_len_a]
            btoa_weights: [bz*rv_num_b, rv_len_b]
        """
        assert (seq_a.shape == seq_b.shape) and len(seq_a.size()) == 4
        bz, rv_num, rv_len, hdim = list(seq_a.size())
        
        seq_a = self.transform_feature(seq_a)
        seq_b = self.transform_feature(seq_b)

        expand_seq_a = seq_a.unsqueeze(1).view(bz, 1, rv_num*rv_len, hdim).repeat(1,rv_num, 1, 1).view(bz*rv_num, rv_num*rv_len, hdim)
        expand_seq_b = seq_b.unsqueeze(1).view(bz, 1, rv_num*rv_len, hdim).repeat(1,rv_num, 1, 1).view(bz*rv_num, rv_num*rv_len, hdim)
        #expand_mask_a = mask_a.unsqueeze(1).view(bz, 1, rv_num*rv_len).repeat(1, rv_num, 1).view(bz*rv_num, rv_num*rv_len)
        #expand_mask_b = mask_b.unsqueeze(1).view(bz, 1, rv_num*rv_len).repeat(1, rv_num, 1).view(bz*rv_num, rv_num*rv_len)
        seq_a = seq_a.view(bz*rv_num, rv_len, hdim)
        seq_b = seq_b.view(bz*rv_num, rv_len, hdim)
        mask_a = mask_a.view(bz*rv_num, rv_len)
        mask_b = mask_b.view(bz*rv_num, rv_len)

        atob_affinity = self.interaction(seq_a, expand_seq_b) #[*, rv_len_a, rv_num*rv_len_b]
        btoa_affinity = self.interaction(seq_b, expand_seq_a) #[*, rv_len_b, rv_num*rv_len_a]
        

        if self.pooling == "MAX":
            atob_scores = atob_affinity.max(dim=-1) #[*, rv_len_a]
            btoa_scores = btoa_affinity.max(dim=-1) #[*, rv_len_b]
        elif self.pooling == "MEAN":
            atob_scores = atob_affinity.mean(dim=-1)
            btoa_scores = btoa_affinity.mean(dim=-1)
        else:
            raise ValueError(f"pooling mode: {self.pooling} is not predefined")

        atob_weights = masked_softmax(atob_scores, mask_a)
        btoa_weights = masked_softmax(btoa_scores, mask_b)

        a_out = attention_weighted_sum(atob_weights, seq_a)
        b_out = attention_weighted_sum(btoa_weights, seq_b)

        
        return a_out, b_out, atob_weights, btoa_weights

class RelScore(nn.Module):
    def __init__(self, in_feature, latent_dim, vocab_size, dropout, id_as_gate=False):
        super().__init__() 
        self.id_as_gate = id_as_gate
        
        self.proj_layer = nn.Sequential(nn.Dropout(p=dropout),
                                    nn.Linear(in_feature, latent_dim))
        self.ebd_layer = nn.Embedding(vocab_size, latent_dim, padding_idx=0)
        
        if self.id_as_gate:
            self.score_layer = nn.Linear(latent_dim, 1)
        else: 
            self.score_layer = nn.Sequential(nn.Tanh(),
                                        nn.Linear(2*latent_dim, 1))
    
    def forward(self, inputs, ids):
        """
        Args:
            inputs: [bz, rv_num, in_feature] or [bz, 1, in_feature]
            ids: [bz]
        
        Returns:
            score: [bz, rv_num, 1]
        """
        bz, rv_num, in_feature = list(inputs.size())

        latent_feat = self.proj_layer(inputs) #[bz, rv_num, latent_dim]
        id_feat = self.ebd_layer(ids).unsqueeze(1) #[bz, 1, latent_dim]

        if self.id_as_gate:
            in_feat = F.tanh(latent_feat) * F.sigmoid(id_feat)
        else:
            in_feat = torch.cat([latent_feat, id_feat.repeat(1,rv_num,1)], dim=-1) #[bz, rv_num, 2*latent_dim]
        
        return self.score_layer(in_feat)


class CombineGlobalRevFeat(nn.Module):
    def __init__(self):
        # NOTE: test !!!
        super().__init__()
    
    def forward(self, rev_feats, rev_masks, global_feat, rev_logits):
        """
        Args:
            rev_feats: [bz, rv_num, hdim]
            rev_masks: [bz, rv_num]
            global_feat: [bz, hdim]
            rev_logits: [bz, rnum]

        Return:
            combine_feat: [bz, hdim]
        """
        bz, rv_num, hdim = rev_feats.size(0), rev_feats.size(1), rev_feats.size(2)
        _device = rev_feats.device
        global_masks = torch.ones(size=(bz, 1)).bool().to(_device)
        global_logits = torch.zeros(size=(bz, 1)).float().to(_device)

        rev_masks = torch.cat([rev_masks, global_masks], dim=-1) #[bz, rv_num+1]
        rev_logits = torch.cat([rev_logits, global_logits], dim=-1) #[bz, rv_num+1]
        rev_scores = F.softmax(torch.masked_fill(rev_logits, ~rev_masks, -1e8), dim=-1).view(bz, rv_num+1, 1) 

        global_feat = global_feat.view(bz, 1, hdim)
        combine_feat = torch.cat([rev_feats, global_feat], dim=1) #[bz, rv_num+1, hdim]

        combine_feat = torch.sum(combine_feat * rev_scores, dim=1) #[bz, hdim]

        return combine_feat, rev_logits

class SingleRelLogit(nn.Module):
    def __init__(self, hidden_dim, rv_len, cosine_interaction, word_score_layer, rel_score_layer):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.cosine_interaction = cosine_interaction
        self.word_score_layer = word_score_layer
        self.rel_score_layer = rel_score_layer

    def forward(self, ui_seq, seq_b, ui_seq_mask, mask_b):
        """
        Args:
            ui_seq: [bz, rv_length, dim]
            seq_b: [bz, rv_num, rv_length, dim]
            ui_mask: [bz, rv_length]
            mask_b: [bz, rv_num, rv_length]
        """
        bz, rv_num, rv_len, hdim = list(seq_b.size())

        seq_b = seq_b.view(bz, rv_num*rv_len, -1) #[bz, rv_numxrv_len, hdim]
        mask_b = mask_b.view(bz, rv_num*rv_len).unsqueeze(1) #[bz, 1, rv_num*rv_len]

        cosine_affinity = self.cosine_interaction(ui_seq, seq_b) #[bz, rv_len, rv_numxrv_len]
        mean_feature = masked_colwise_mean(cosine_affinity, mask_b) #[bz, rv_len, 1]
        max_feature, _ = cosine_affinity.max(dim=-1, keepdim=True) #[bz, rv_len, 1]
        out_feat = torch.cat([mean_feature, max_feature], dim=-1) #[bz, rv_len, 2]
        word_score = self.word_score_layer(ui_seq, ui_seq_mask).view(bz, rv_len, 1) #[bz, rv_len, 1]
        
        out_feat = (out_feat * word_score).view(bz, rv_len*2)
        rel_logit = self.rel_score_layer(out_feat) #[bz, 1]

        return rel_logit

class SingleRelLogitWithId(nn.Module):
    def __init__(self, hidden_dim, rv_len, cosine_interaction, word_score_layer, rel_score_layer):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.cosine_interaction = cosine_interaction
        self.word_score_layer = word_score_layer
        self.rel_score_layer = rel_score_layer

    def forward(self, ui_seq, seq_b, ui_seq_mask, mask_b, b_id):
        """
        Args:
            ui_seq: [bz, rv_length, dim]
            seq_b: [bz, rv_num, rv_length, dim]
            ui_mask: [bz, rv_length]
            mask_b: [bz, rv_num, rv_length]
            b_id: [bz]
        """
        bz, rv_num, rv_len, hdim = list(seq_b.size())

        seq_b = seq_b.view(bz, rv_num*rv_len, -1) #[bz, rv_numxrv_len, hdim]
        mask_b = mask_b.view(bz, rv_num*rv_len).unsqueeze(1) #[bz, 1, rv_num*rv_len]

        cosine_affinity = self.cosine_interaction(ui_seq, seq_b) #[bz, rv_len, rv_numxrv_len]
        mean_feature = masked_colwise_mean(cosine_affinity, mask_b) #[bz, rv_len, 1]
        max_feature, _ = cosine_affinity.max(dim=-1, keepdim=True) #[bz, rv_len, 1]
        out_feat = torch.cat([mean_feature, max_feature], dim=-1) #[bz, rv_len, 2]
        word_score = self.word_score_layer(ui_seq, ui_seq_mask).view(bz, rv_len, 1) #[bz, rv_len, 1]
        
        out_feat = (out_feat * word_score).view(bz, 1, rv_len*2)
        rel_logit = self.rel_score_layer(out_feat, b_id) #[bz, 1]

        return rel_logit

class UserCoRelLogitWithRepWithId(nn.Module):
    def __init__(self, hidden_dim, rv_len, cosine_interaction, word_score_layer, rel_score_layer):
        super().__init__() 

        self.hidden_dim = hidden_dim

        self.cosine_interaction = cosine_interaction
        self.word_score_layer = word_score_layer
        self.rel_score_layer = rel_score_layer

    def forward(self, seq_a, seq_b, mask_a, mask_b, b_id):
        """
        Args:
            seq_a: [bz, rv_num, rv_length, dim]
            seq_b: [bz, rv_num, rv_length, dim]
            mask_a: [bz, rv_num, rv_length]
            mask_b: [bz, rv_num, rv_length]
            b_id: [bz]

        Returns:
           ab_logits: [bz, rv_num]
           rel_seq_a: [bz, rv_num, hdim]
        """
           
        bz, rv_num, rv_len, hdim = list(seq_a.size())
        assert hdim == self.hidden_dim

        expand_seq_b = seq_b.unsqueeze(1).view(bz, 1, rv_num*rv_len, hdim).repeat(1,rv_num, 1, 1).view(bz*rv_num, rv_num*rv_len, hdim)
        expand_mask_b = mask_b.unsqueeze(1).view(bz, 1, rv_num*rv_len).repeat(1, rv_num, 1).view(bz*rv_num, 1, rv_num*rv_len)
        seq_a = seq_a.view(bz*rv_num, rv_len, hdim)
        mask_a = mask_a.view(bz*rv_num, rv_len)
        
        atob_cos_affinity = self.cosine_interaction(seq_a, expand_seq_b) #[*, rv_len, rv_num*rv_len]
        ab_mean_feat = masked_colwise_mean(atob_cos_affinity, expand_mask_b) #[*, rv_len, 1]
        ab_max_feat, _ = atob_cos_affinity.max(dim=-1, keepdim=True) #[*, rv_len, 1]
        ab_feat = torch.cat([ab_mean_feat, ab_max_feat], dim=-1).view(bz*rv_num, rv_len, 2)
        ab_word_score = self.word_score_layer(seq_a, mask_a).view(bz*rv_num, rv_len, 1)
        ab_feat = (ab_feat * ab_word_score).view(bz, rv_num, rv_len*2)
        ab_logits = self.rel_score_layer(ab_feat, b_id).view(bz, rv_num) #[bz*rv_num, 1]

        rel_seq_a = attention_weighted_sum(ab_word_score, seq_a).view(bz, rv_num, hdim)
        
        return ab_logits, rel_seq_a


class UserCoRelLogitWithRep(nn.Module):
    def __init__(self, hidden_dim, rv_len, cosine_interaction, word_score_layer, rel_score_layer):
        super().__init__() 

        self.hidden_dim = hidden_dim

        self.cosine_interaction = cosine_interaction
        self.word_score_layer = word_score_layer
        self.rel_score_layer = rel_score_layer

    def forward(self, seq_a, seq_b, mask_a, mask_b):
        """
        Args:
            seq_a: [bz, rv_num, rv_length, dim]
            seq_b: [bz, rv_num, rv_length, dim]
            mask_a: [bz, rv_num, rv_length]
            mask_b: [bz, rv_num, rv_length]

        Returns:
           ab_logits: [bz, rv_num]
           rel_seq_a: [bz, rv_num, hdim]
        """
           
        bz, rv_num, rv_len, hdim = list(seq_a.size())
        assert hdim == self.hidden_dim

        expand_seq_b = seq_b.unsqueeze(1).view(bz, 1, rv_num*rv_len, hdim).repeat(1,rv_num, 1, 1).view(bz*rv_num, rv_num*rv_len, hdim)
        expand_mask_b = mask_b.unsqueeze(1).view(bz, 1, rv_num*rv_len).repeat(1, rv_num, 1).view(bz*rv_num, 1, rv_num*rv_len)
        seq_a = seq_a.view(bz*rv_num, rv_len, hdim)
        mask_a = mask_a.view(bz*rv_num, rv_len)
        
        atob_cos_affinity = self.cosine_interaction(seq_a, expand_seq_b) #[*, rv_len, rv_num*rv_len]
        ab_mean_feat = masked_colwise_mean(atob_cos_affinity, expand_mask_b) #[*, rv_len, 1]
        ab_max_feat, _ = atob_cos_affinity.max(dim=-1, keepdim=True) #[*, rv_len, 1]
        ab_feat = torch.cat([ab_mean_feat, ab_max_feat], dim=-1).view(bz*rv_num, rv_len, 2)
        ab_word_score = self.word_score_layer(seq_a, mask_a).view(bz*rv_num, rv_len, 1)
        ab_feat = (ab_feat * ab_word_score).view(bz*rv_num, rv_len*2)
        ab_logits = self.rel_score_layer(ab_feat).view(bz, rv_num) #[bz*rv_num, 1]

        rel_seq_a = attention_weighted_sum(ab_word_score, seq_a).view(bz, rv_num, hdim)
        
        return ab_logits, rel_seq_a

class UserCoRelLogit(nn.Module):
    def __init__(self, hidden_dim, rv_len, cosine_interaction, word_score_layer, rel_score_layer):
        super().__init__() 

        self.hidden_dim = hidden_dim

        self.cosine_interaction = cosine_interaction
        self.word_score_layer = word_score_layer
        self.rel_score_layer = rel_score_layer

    def forward(self, seq_a, seq_b, mask_a, mask_b):
        """
        Args:
            seq_a: [bz, rv_num, rv_length, dim]
            seq_b: [bz, rv_num, rv_length, dim]
            mask_a: [bz, rv_num, rv_length]
            mask_b: [bz, rv_num, rv_length]
            ui_seq: [bz, rv_length, dim]
            ui_mask: [bz, rv_length]
            neg_ui_seq: [bz, rv_length, dim]
            neg_ui_mask: [bz, rv_length, dim]

        Returns:
           ab_logits: [bz, rv_num]
        """
        bz, rv_num, rv_len, hdim = list(seq_a.size())
        assert hdim == self.hidden_dim

        expand_seq_b = seq_b.unsqueeze(1).view(bz, 1, rv_num*rv_len, hdim).repeat(1,rv_num, 1, 1).view(bz*rv_num, rv_num*rv_len, hdim)
        expand_mask_b = mask_b.unsqueeze(1).view(bz, 1, rv_num*rv_len).repeat(1, rv_num, 1).view(bz*rv_num, 1, rv_num*rv_len)
        seq_a = seq_a.view(bz*rv_num, rv_len, hdim)
        mask_a = mask_a.view(bz*rv_num, rv_len)
        
        atob_cos_affinity = self.cosine_interaction(seq_a, expand_seq_b) #[*, rv_len, rv_num*rv_len]
        ab_mean_feat = masked_colwise_mean(atob_cos_affinity, expand_mask_b) #[*, rv_len, 1]
        ab_max_feat, _ = atob_cos_affinity.max(dim=-1, keepdim=True) #[*, rv_len, 1]
        ab_feat = torch.cat([ab_mean_feat, ab_max_feat], dim=-1).view(bz*rv_num, rv_len, 2)
        ab_word_score = self.word_score_layer(seq_a, mask_a).view(bz*rv_num, rv_len, 1)
        ab_feat = (ab_feat * ab_word_score).view(bz*rv_num, rv_len*2)
        ab_logits = self.rel_score_layer(ab_feat).view(bz, rv_num) # [bz, rv_num]
        
        return ab_logits

class UserCoRel(nn.Module):
    def __init__(self, in_feature, out_feature, feature_type="IDENTITY"):
        super().__init__()

        if feature_type == "IDENTITY":
            self.transform_feature = nn.Identity()
        elif feature_type == "FC":
            self.transform_feature = nn.Sequential(nn.Linear(in_feature, out_feature),
                                                    nn.ReLU())
            nn.init.xavier_normal_(self.transform_feature[0].weight, gain=nn.init.calculate_gain("relu"))
        
        self.cosine_interaction = CosineInteraction()
        self.word_score_layer = WordScore(out_feature)

    def forward(self, seq_a, seq_b, mask_a, mask_b):
        """
        Args:
            seq_a: [bz, rv_num, rv_length, dim]
            seq_b: [bz, rv_num, rv_length, dim]

            mask_a: [bz, rv_num, rv_length]
            mask_b: [bz, rv_num, rv_length]

            exapnd_seq_b: [bzxrv_num, rv_numxrv_length, dim]
            expand_mask_b: [bzxrv_num, rv_numxrv_length]

        Returns:
            out_feature: [bz, rv_num, rv_len*2]
            word_score: [bz, rv_num, rv_len]
        """
        seq_a = masked_tensor(seq_a, mask_a)
        seq_b = masked_tensor(seq_b, mask_b)
        assert (seq_a.shape == seq_b.shape) and len(seq_a.size()) == 4
        bz, rv_num, rv_len, hdim = list(seq_a.size())
        
        seq_a = self.transform_feature(seq_a)
        seq_b = self.transform_feature(seq_b)

        # [bz*rv_num_a, rv_len_a*rv_num_a, hdim]
        expand_seq_b = seq_b.unsqueeze(1).view(bz, 1, rv_num*rv_len, hdim).repeat(1,rv_num, 1, 1).view(bz*rv_num, rv_num*rv_len, hdim)
        expand_mask_b = mask_b.unsqueeze(1).view(bz, 1, rv_num*rv_len).repeat(1, rv_num, 1).view(bz*rv_num, rv_num*rv_len)
        seq_a = seq_a.view(bz*rv_num, rv_len, hdim)
        mask_a = mask_a.view(bz*rv_num, rv_len)

        atob_cos_affinity = self.cosine_interaction(seq_a, expand_seq_b) #[*, rv_len, rv_len*rv_num]
        expand_mask_b = expand_mask_b.unsqueeze(1) #[*, 1, rv_len*rv_num]

        mean_feature = masked_colwise_mean(atob_cos_affinity, expand_mask_b) #[*, rv_len, 1]
        max_feature, _ = atob_cos_affinity.max(dim=-1, keepdim=True) #[*, rv_len, 1]

        out_feature = torch.cat([mean_feature, max_feature], dim=-1).view(bz*rv_num, rv_len, 2)
        word_score = self.word_score_layer(seq_a, mask_a).view(bz*rv_num, rv_len, 1)

        out_feature = out_feature * word_score 
        out_feature = out_feature.view(bz, rv_num, rv_len*2) 

        return out_feature, word_score.view(bz, rv_num, rv_len)


class CoAlign(nn.Module):
    def __init__(self, in_feature, out_feature, interaction_type="SCALEDDOT", feature_type="IDENTITY", **kwargs):
        """
        Args:
            interaction_type: support `DOT`, `SCALEDDOT`, `BILINEAR` `TENSOR`
            feature_type: support `IDENTITY`, `FC`
            pooling: support `MATRIX`, `MAX, `MEAN`
        """
        super().__init__()
        
        if interaction_type == "DOT":
            self.interaction = DotInteraction(out_feature, scale=False)
        elif interaction_type == "SCALEDDOT":
            self.interaction = DotInteraction(out_feature, scale=True)
        elif interaction_type == "BILINEAR":
            self.interaction = BiLinearInteraction(out_feature, bias=False)
        elif interaction_type == "TENSOR":
            k_factor = kwargs["k_factor"]
            if k_factor <= 1:
                raise ValueError("TENSOR interaction should have > 1 k_factor")
            self.interaction = TensorInteraction(out_feature, k_factor=k_factor, bias=False)
        else:
            raise ValueError("interaction_type {} is not predefined".format(interaction_type))

        if feature_type == "IDENTITY":
            self.transform_feature = nn.Identity()
        elif feature_type == "FC":
            self.transform_feature = nn.Sequential(nn.Linear(in_feature, out_feature),
                                                    nn.ReLU())
            nn.init.xavier_normal_(self.transform_feature[0].weight, gain=nn.init.calculate_gain("relu"))
        

    def forward(self, seq_a, seq_b, mask_a, mask_b):
        """
        Args:
            seq_a: [bz, doc_num, doc_length, dim] --> [bz, 1, doc_numxdoc_length, dim] --> [bz, doc_num, doc_numxdoc_length, dim]
                                                --> [bzxdoc_num, doc_numxdoc_length, dim]
            seq_b: [bz, doc_num, doc_length, dim]

            mask_a: [bz, doc_num, doc_length] --> [bz, 1, doc_numxdoc_length] --> [bz, doc_num, doc_numxdoc_length]
                                        --> [bzxdoc_num, doc_numxdoc_length]

            seq_a: [bzxdoc_num, doc_length, dim]
            expand_seq_b: [bzxdoc_num, doc_numxdoc_length, dim]

            seq_b: [bzxdoc_num, doc_length, dim]
            expand_seq_a = [bzxdoc_num, doc_numxdoc_length, dim]

        Returns:
            align_a: [bzxrv_num_a, rv_len_a, dim]
            align_b: [bzxrv_num_b, rv_len_b, dim]
        """
        seq_a = masked_tensor(seq_a, mask_a)
        seq_b = masked_tensor(seq_b, mask_b)
        assert (seq_a.shape == seq_b.shape) and len(seq_a.size()) == 4
        bz, rv_num, rv_len, hdim = list(seq_a.size())
        
        seq_a = self.transform_feature(seq_a)
        seq_b = self.transform_feature(seq_b)

        expand_seq_a = seq_a.unsqueeze(1).view(bz, 1, rv_num*rv_len, hdim).repeat(1,rv_num, 1, 1).view(bz*rv_num, rv_num*rv_len, hdim)
        expand_seq_b = seq_b.unsqueeze(1).view(bz, 1, rv_num*rv_len, hdim).repeat(1,rv_num, 1, 1).view(bz*rv_num, rv_num*rv_len, hdim)
        # [*, rv_num*rv_len_b, hdim]
        expand_mask_a = mask_a.unsqueeze(1).view(bz, 1, rv_num*rv_len).repeat(1, rv_num, 1).view(bz*rv_num, rv_num*rv_len)
        expand_mask_b = mask_b.unsqueeze(1).view(bz, 1, rv_num*rv_len).repeat(1, rv_num, 1).view(bz*rv_num, rv_num*rv_len)
        seq_a = seq_a.view(bz*rv_num, rv_len, hdim)
        seq_b = seq_b.view(bz*rv_num, rv_len, hdim) #[*, rv_len_a, hdim]
        mask_a = mask_a.view(bz*rv_num, rv_len)
        mask_b = mask_b.view(bz*rv_num, rv_len)
        
        atob_affinity = self.interaction(seq_a, expand_seq_b) #[*, rv_len_a, rv_num*rv_len_b]
        btoa_affinity = self.interaction(seq_b, expand_seq_a) #[*, rv_len_b, rv_num*rv_len_a]

        expand_mask_b = expand_mask_b.unsqueeze(1)
        atob_weights = masked_softmax(atob_affinity, expand_mask_b) #[*, rv_len_a, rv_num*rv_len_b]
        expand_mask_a = expand_mask_a.unsqueeze(1)
        btoa_weights = masked_softmax(btoa_affinity, expand_mask_a) #[*, rv_len_b, rv_num*rv_len_a]

        align_a = torch.bmm(atob_weights, expand_seq_b) #[*, rv_len_a, hdim]
        align_b = torch.bmm(btoa_weights, expand_seq_a) #[*, rv_len_b, hdim]

        # NOTE: align_a not be masked 
        align_a = masked_tensor(align_a, mask_a)
        align_b = masked_tensor(align_b, mask_b)
        
        return align_a, align_b, atob_weights, btoa_weights

class AlignEnhance(nn.Module):
    """
    NOTE: default architecture
    """
    def __init__(self):
        super().__init__()

    def forward(self, seq_a, seq_b, align_a, align_b, mask_a, mask_b):
        """
        NOTE: use "MUL", "SUB", "CONCAT"
        seq_a: [bz, seq_a, in_feature]
        seq_b: [bz, seq_b, in_feature]
        mask_a: [bz, seq_a]
        mask_b: [bz, seq_b]
        """
        seq_a = masked_tensor(seq_a, mask_a)
        align_a = masked_tensor(align_a, mask_a)
        seq_b = masked_tensor(seq_b, mask_b)
        align_b = masked_tensor(align_b, mask_b)

        mul_a = torch.mul(seq_a, align_a)
        mul_b = torch.mul(seq_b, align_b)

        sub_a = torch.abs(torch.sub(seq_a, align_a))
        sub_b = torch.abs(torch.sub(seq_b, align_b))

        cat_a = torch.cat([seq_a, align_a], dim=-1)
        cat_b = torch.cat([seq_b, align_b], dim=-1)

        return (mul_a, sub_a, cat_a), (mul_b, sub_b, cat_b)

class AlignEnhanceFM(nn.Module):
    def __init__(self, in_features, fm_k_factor):
        super().__init__()
        self.fm_mul = FactorizationMachine(in_features, fm_k_factor)
        self.fm_sub = FactorizationMachine(in_features, fm_k_factor)
        self.fm_cat = FactorizationMachine(2*in_features, fm_k_factor)

    def forward(self, seq_a, seq_b, align_a, align_b, mask_a, mask_b):
        """
        NOTE: use "MUL", "SUB", "CONCAT"
        Args:
            seq_a: [bz, seq_a, in_feature]
            seq_b: [bz, seq_b, in_feature]
            mask_a: [bz, seq_a]
            mask_b: [bz, seq_b]
        Returns:
            mul_seq_a: [bz, seq_a, 1]
            sub_seq_a: [bz, seq_a, 1]
            cat_seq_a: [bz, seq_a, 1]
            --> enhance_seq_a: [bz, seq_a, 3]

            mul_seq_b: [bz, seq_b, 1]
            sub_seq_b: [bz, seq_b, 1]
            cat_seq_b: [bz, seq_b, 1]
            --> enhance_seq_b: [bz, seq_b, 3]
        """
        seq_a = masked_tensor(seq_a, mask_a)
        align_a = masked_tensor(align_a, mask_a)
        seq_b = masked_tensor(seq_b, mask_b)
        align_b = masked_tensor(align_b, mask_b)

        mul_a = seq_a * align_a
        mul_b = seq_b * align_b
        en_mul_a = self.fm_mul(mul_a)
        en_mul_b = self.fm_mul(mul_b)

        sub_a = torch.abs(seq_a - align_a)
        sub_b = torch.abs(seq_b - align_b)
        en_sub_a = self.fm_sub(sub_a)
        en_sub_b = self.fm_sub(sub_b)

        cat_a = torch.cat([seq_a, align_a], dim=-1)
        cat_b = torch.cat([seq_b, align_b], dim=-1)
        en_cat_a = self.fm_cat(cat_a)
        en_cat_b = self.fm_cat(cat_b)

        en_seq_a = torch.cat([en_mul_a, en_sub_a, en_cat_a], dim=-1)
        en_seq_b = torch.cat([en_mul_b, en_sub_b, en_cat_b], dim=-1)

        return en_seq_a, en_seq_b

# ============= Aggregate Module ==================
class EnhanceAggregateFM(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3):
        super().__init__()
        """
        self.aggre = nn.Sequential(HierPooling(in_features, out_features, kernel_size),
                                    nn.LayerNorm(out_features) if use_layer_norm else nn.Identity(),
                                    nn.ReLU())
        """
        self.conv = nn.Sequential(MyConv1d([kernel_size], in_features, out_features),
                                nn.ReLU())
        
    def forward(self, seq_a, seq_b, enh_seq_a, enh_seq_b, mask_a, mask_b):
        """
        Args:
            seq_a: [bz*rv_num_a, rv_len_a, hdim]
            seq_b: [bz*rv_num_b, rv_len_b, hdim]
            enh_seq_a: [bz*rv_num_a, rv_len_a, 3]
            enh_seq_b: [bz*rv_num_b, rv_len_b, 3]
            mask_a: [bz*rv_num_a, rv_len_a]
            mask_b: [bz*rv_num_b, rv_len_b]
        
        Returns:
            u_feat: [bz*rv_num_a, hdim]
            i_feat: [bz*rv_num_a, hdim]
        """
        # mask 
        seq_a = masked_tensor(seq_a, mask_a)
        enh_seq_a = masked_tensor(enh_seq_a, mask_a)
        seq_b = masked_tensor(seq_b, mask_b)
        enh_seq_b = masked_tensor(enh_seq_b, mask_b)

        # enhance
        enh_seq_a = torch.cat([seq_a, enh_seq_a], dim=-1).transpose(1,2)
        enh_seq_b = torch.cat([seq_b, enh_seq_b], dim=-1).transpose(1,2)

        # conv
        feat_seq_a = self.conv(enh_seq_a)
        feat_seq_b = self.conv(enh_seq_b)

        # max pooling
        assert feat_seq_a.size(-1) == feat_seq_b.size(-1)
        feat_a = F.max_pool1d(feat_seq_a, kernel_size=feat_seq_a.size(-1)) #
        feat_b = F.max_pool1d(feat_seq_b, kernel_size=feat_seq_b.size(-1))

        return feat_a, feat_b

class EnhanceAggregate(nn.Module):
    """
    NOTE: default architecture
    """
    def __init__(self, in_features, out_features, kernel_size=3):
        super().__init__()

        self.conv = nn.Sequential(MyConv1d([kernel_size], in_features, out_features),
                                nn.ReLU())
        
    def forward(self, mul_seq_a, sub_seq_a, cat_seq_a, mul_seq_b, sub_seq_b, cat_seq_b):
        seq_a = torch.cat([mul_seq_a, sub_seq_a, cat_seq_a], dim=-1).transpose(1,2)
        seq_b = torch.cat([mul_seq_b, sub_seq_b, cat_seq_b], dim=-1).transpose(1,2)
        
        # conv 
        feat_seq_a = self.conv(seq_a)
        feat_seq_b = self.conv(seq_b) #[bz*rv_num, hdim, seq_len]

        # max pooling 
        feat_a = F.max_pool1d(feat_seq_a, kernel_size=feat_seq_a.size(-1))
        feat_b = F.max_pool1d(feat_seq_a, kernel_size=feat_seq_b.size(-1))

        return feat_a.contiguous(), feat_b.contiguous()

class EnhanceAggregate(nn.Module):
    """
    NOTE: default architecture
    """
    def __init__(self, in_features, out_features, kernel_size=3):
        super().__init__()

        self.conv = nn.Sequential(MyConv1d([kernel_size], in_features, out_features),
                                nn.ReLU())
        
    def forward(self, mul_seq_a, sub_seq_a, cat_seq_a, mul_seq_b, sub_seq_b, cat_seq_b):
        seq_a = torch.cat([mul_seq_a, sub_seq_a, cat_seq_a], dim=-1).transpose(1,2)
        seq_b = torch.cat([mul_seq_b, sub_seq_b, cat_seq_b], dim=-1).transpose(1,2)
        
        # conv 
        feat_seq_a = self.conv(seq_a)
        feat_seq_b = self.conv(seq_b) #[bz*rv_num, hdim, seq_len]

        # max pooling 
        feat_a = F.max_pool1d(feat_seq_a, kernel_size=feat_seq_a.size(-1))
        feat_b = F.max_pool1d(feat_seq_a, kernel_size=feat_seq_b.size(-1))

        return feat_a.contiguous(), feat_b.contiguous()

class EnhanceAggregateWithMaxAvgPooling(nn.Module):
    """
    NOTE: default architecture
    """
    def __init__(self, in_features, out_features, kernel_size=3):
        super().__init__()

        self.conv = nn.Sequential(MyConv1d([kernel_size], in_features, out_features),
                                nn.ReLU())
        self.masked_avgpool_1d = MaskedAvgPooling1d()
        
    def forward(self, mul_seq_a, sub_seq_a, cat_seq_a, mul_seq_b, sub_seq_b, cat_seq_b, mask_a, mask_b):
        """
        mul_seq_a: [bz, seq_len, hdim]
        mask_a: [bz, seq_len]
        """
        bz, seq_len, hdim = list(mul_seq_a.size())
        seq_a = torch.cat([mul_seq_a, sub_seq_a, cat_seq_a], dim=-1).transpose(1,2)
        seq_b = torch.cat([mul_seq_b, sub_seq_b, cat_seq_b], dim=-1).transpose(1,2)
        
        # conv 
        feat_seq_a = self.conv(seq_a)
        feat_seq_b = self.conv(seq_b) #[bz*rv_num, hdim, seq_len]

        # max pooling 
        max_feat_a = F.max_pool1d(feat_seq_a, kernel_size=feat_seq_a.size(-1))
        max_feat_b = F.max_pool1d(feat_seq_a, kernel_size=feat_seq_b.size(-1))
        avg_feat_a = self.masked_avgpool_1d(feat_seq_a, mask_a)
        avg_feat_b = self.masked_avgpool_1d(feat_seq_b, mask_b)
        #print(mul_seq_a.shape, feat_seq_a.shape, max_feat_a.shape)

        feat_a = torch.cat([max_feat_a, avg_feat_a], dim=1).view(bz, 2*hdim)
        feat_b = torch.cat([max_feat_b, avg_feat_b], dim=1).view(bz, 2*hdim)

        return feat_a, feat_b
    
class NgramFeat_old(nn.Module):
    def __init__(self, vocab_size, kernel_size, filter_num, embedding_dim,
                max_doc_num, max_doc_len, padding_idx, pretrained_embeddings=None):
        super(NgramFeat, self).__init__()
        self.kernel_size = kernel_size

        self.filter_num = filter_num
        self.embeddings = WordEmbedding(vocab_size, embedding_dim, pretrained_embeddings=pretrained_embeddings, padding_idx=padding_idx)
        self.conv_1 = nn.Sequential(
                            nn.Conv1d(embedding_dim, filter_num, kernel_size=kernel_size),
                            nn.ReLU(),
                            nn.MaxPool1d(max_doc_len-kernel_size+1))
    def forward(self, text):
        """
        Args:
            text: [bz, max_doc_num, max_doc_len]

        Return:
            text: [bz, max_doc_num, filter_num]
        """
        text_ebd = self.embeddings(text) #[bz, dnum, dlen, ebd_dim]
        bz = text_ebd.size(0)
        dnum = text_ebd.size(1)
        dlen = text_ebd.size(2)
        ebd_dim = text_ebd.size(3)

        text_ebd = text_ebd.view(-1, dlen, ebd_dim).permute(0,2,1).contiguous() #[bz*dnum, ebd_dim, dlen]

        text_feat = self.conv_1(text_ebd).squeeze(-1) #[bz*dnum, filter_num, 1]

        text_feat = text_feat.view(bz, dnum, self.filter_num) #[bz, dnum, filter_num]

        return text_feat

# ========== Others ==============
class ReviewRelScore(nn.Module):
    def __init__(self, in_feature, dropout):
        super().__init__()

        self.trans_layer = nn.Sequential(nn.Dropout(p=dropout),
                                        nn.Linear(in_feature,1),
                                        nn.Tanh()) # NOTE: migth relu
    def forward(self, inputs, masks):
        """
        Args:
            inputs: [bz, rv_num, in_feature]  
            masks: [bz, rv_num]
        
        Returns:
            out_scores: [bz, rv_num]
        """
        bz, rv_num, _ = list(inputs.size())

        out_logits = self.trans_layer(inputs).view(bz, rv_num)
        out_scores = F.softmax(torch.masked_fill(out_logits, ~masks, -1e8), dim=-1)

        return out_scores

class WordScore(nn.Module):
    def __init__(self, feature):
        super().__init__()

        self.inner_product_layer = nn.Linear(feature, 1, bias=False)
        #self.inner_product_layer = nn.Sequential(nn.Linear(feature, 1))
    def forward(self, inputs, masks):
        """
        Args:
            inputs: [bz, seq_len, dim]
            masks: [bz, seq_len]

        Returns:
            out_scores: [bz, seq_len]
        """
        bz, seq_len, dim = list(inputs.size())
        inputs = self.inner_product_layer(inputs).view(bz, seq_len) #[bz, seq_len]
        assert inputs.dim() == masks.dim()

        masked_inputs = torch.masked_fill(inputs, ~masks, -1e8)
        out_scores = F.softmax(masked_inputs, dim=-1)

        return out_scores

class InputSelection(nn.Module):
    """
    inputs_1: reliable feature sources, like user/item features
    inputs_2: not that reliable, like user/item features from reviews.
    """
    def __init__(self, in_features_1, in_features_2, out_features):
        super().__init__() 
        self.gate = nn.Sequential(nn.Linear(in_features_1 + in_features_2, out_features),
                                    nn.Sigmoid())
        self.trans_layer = nn.Sequential(nn.Linear(in_features_1+in_features_2, out_features),
                                            nn.ReLU())
    
    def forward(self, inputs_1, inputs_2):
        """
        NOTE: inputs_1 is our more reliable feature source 
        """
        cat_inputs = torch.cat([inputs_1, inputs_2], dim=-1)
        gate = self.gate(cat_inputs)
        trans_outputs = self.trans_layer(cat_inputs)
        outputs = trans_outputs * gate + (1-gate) * inputs_1
        
        return outputs

        
class FactorizationMachine(nn.Module):
    def __init__(self, in_feat, latent_factor):
        super().__init__() 
        self.in_feat = in_feat

        self.Ww = nn.Parameter(torch.randn(in_feat, 1))
        self.Wv = nn.Parameter(torch.randn(in_feat, latent_factor))
        self.bias = nn.Parameter(torch.randn(1))

        self.reset_parameters()


    def reset_parameters(self):
        bound = 1. / math.sqrt(self.in_feat)
        nn.init.uniform_(self.Ww, -bound, bound)
        nn.init.uniform_(self.Wv, -bound, bound)
        nn.init.zeros_(self.bias)

    def forward(self, tensor):
        """
        Args:
            tensor: FloatTensor with shape of [bz, *, in_feat]
        
        Returns:
            out_tensor: FloatTensor with shape of [bz, *, 1]
        """
        dims_except_last, in_feat = list(tensor.size())[:-1], list(tensor.size())[-1]
        tensor = tensor.view(-1, in_feat) #[*, in_feat]

        linear_term = torch.matmul(tensor, self.Ww) 

        trans_tensor = torch.matmul(tensor, self.Wv)
        quadratic_term_1 = trans_tensor * trans_tensor #[*, latent_factor]
        quadratic_term_2 = torch.matmul(tensor * tensor, self.Wv * self.Wv) #[*, latent_factor]
        quadratic_term = (quadratic_term_1 - quadratic_term_2).sum(-1, keepdim=True)
        quadratic_term = 0.5 * quadratic_term 

        out_tensor = linear_term + quadratic_term + self.bias 
        
        new_dims = dims_except_last + [1]
        out_tensor = out_tensor.view(*new_dims) 

        return out_tensor


if __name__ == "__main__":
    """
    ins = torch.randn(4, 150,20)
    hier_pool_layer = HierPooling(150, 50, 5)
    print(hier_pool_layer(ins).shape)
    #print(hier_pool_layer(ins))
    my_conv1d_layer = MyConv1d([3,5,7], 300, 150)
    inps = torch.randn(8, 300, 20)
    outs = my_conv1d_layer(inps)
    print(outs.shape)

    ebd_layer = nn.Embedding(1000, 100)
    co_attention_layer = CoAttention(100, 100)
    seq_a = torch.randint(1, 1000, size=(2, 3, 20))
    seq_b = torch.randint(1, 1000, size=(2, 3, 20))
    seq_a[0, 0, 15:] = 0
    seq_b[0, 0, 14:] = 0
    mask_a = get_mask(seq_a)
    mask_b = get_mask(seq_b)
    seq_a = ebd_layer(seq_a)
    seq_b = ebd_layer(seq_b)

    out_a, out_b, att_col, att_row = co_attention_layer(seq_a, seq_b, mask_a, mask_b)
    print("out_a", out_a.shape, "out_b", out_b.shape)
    print("att_col", att_col.shape, att_col[0])
    print("att_row", att_row.shape, att_row[0])

    """
    """
    seq_encoder_layer = SeqEncoder([3], 100, 150, "CNN")
    out_seq = seq_encoder_layer(seq_a, mask_a)
    print("out_seq", out_seq.shape, out_seq[0, :, :8], out_seq.sum(dim=-1))

    co_align_layer = CoAlign(100, 100)
    out_a, out_b, atob_weights, btoa_weights = co_align_layer(seq_a, seq_b, mask_a, mask_b)
    print("CoAlign: ")
    print("out_a", out_a.shape, "out_b", out_b.shape)
    print("out_a", out_a.sum(dim=-1))
    print("out_b", out_b.sum(dim=-1))
    #print("att_weights", att_weights.shape, att_weights[0])
    
    seq_a = seq_a.view(-1, 20, 100)
    seq_b = seq_b.view(-1, 20, 100)
    mask_a = mask_a.view(-1, 20)
    mask_b = mask_b.view(-1, 20)
    enhance_layer = AlignEnhanceFM(100, 32)
    print(seq_a.shape, out_a.shape, mask_a.shape)
    x1, y1 = enhance_layer(seq_a, seq_b, out_a, out_b, mask_a, mask_b)
    print("a", x1.shape)
    print("b", y1.shape)
    print(x1.sum(dim=-1))
    print(y1.sum(dim=-1))
    
    aggre_layer = EnhanceAggregateFM()
    x2, y2 = aggre_layer(seq_a, seq_b, x1, y1, mask_a, mask_b)
    print("a", x2.shape)
    print("b", y2.shape)
    print(x2.sum(dim=-1))
    print(y2.sum(dim=-1))
    """
    a = torch.tensor([-1/2., 1/2., 1/3.])
    b = torch.tensor([1/3., 1/2., 1/2.])  
    c = torch.tensor([1,1.,1.]) 
    d = torch.tensor([0., 0., 0.])
    cosine_co = CosineInteraction()
    x = torch.cat([a,b,c,d], dim=0).view(1, 4, 3)
    y = torch.cat([a,c,b,d], dim=0).view(1,4,3)
    print(x) 
    print(y)
    cosine_matrix = cosine_co(x,y) #[1, 4, 4]
    from utils import masked_colwise_mean
    y_indices = torch.tensor([1,1,1,0]).view(1,4)
    y_mask = torch.ones(size=list(y_indices.size()),dtype=torch.bool)
    y_mask[y_indices==0] = False
    y_mask = y_mask.view(1,1,4) 
    cm_cmean = masked_colwise_mean(cosine_matrix, y_mask)

    print("-------")
    print(cosine_matrix)
    print(y_mask)
    print(cm_cmean)

    print(" ------- test WordScore Layer ------- ")

    fdim = 5
    wm_idx = 100
    bz = 4
    ws_layer = WordScore(fdim)
    eb_layer = nn.Embedding(wm_idx, fdim)

    in_words = torch.randint(low=1, high=wm_idx, size=(4, 6))
    in_masks = torch.tril(in_words).bool()
    in_words[in_masks == False ] = 0
    print(in_words)
    print(in_masks)

    in_words = eb_layer(in_words)
    print(ws_layer(in_words, in_masks))


    print(" ---- test CombineFeature layer -----")
    bz, rv_num, hdim = 5, 10, 8
    _rev_feat = torch.randn(bz, rv_num, hdim)
    _rev_logits = torch.randn(bz, rv_num)
    _rev_masks = torch.tril(torch.ones(bz, rv_num)).bool()
    _glb_feat = torch.randn(5, 8)
    
    comb_layer = CombineGlobalRevFeat()
    _combine_feat, _rev_scores = comb_layer(_rev_feat, _rev_masks, _glb_feat, _rev_logits)
    print("final feature: ", _combine_feat)
    print("rev scores: ", _rev_scores)