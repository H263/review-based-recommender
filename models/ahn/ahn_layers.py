import math

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

from utils import masked_softmax, attention_weighted_sum
# ======== Modules for Utils ==============
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

# ======= Modules for Input Encoder (Embedding) ===============
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

class CharEmbedding(nn.Module):
    """
    v1: easiest version
    """
    def __init__(self, vocab_size, embedding_dim, kernel_size, dropout=0., padding_idx=0, freeze_embeddings=False):
        super(CharEmbedding, self).__init__()

        self.kernel_size = kernel_size
        self.embedding_dim = embedding_dim

        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.embeddings.weight.requires_grad = not freeze_embeddings

        self.encoder = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=kernel_size, padding=(kernel_size-1)//2)
                                    

    def forward(self, inputs):
        """
        inputs: [bz, seq_len, char_max]
        """
        bz, seq_len, char_max = list(inputs.size())
        inputs = self.embeddings(inputs).view(bz*seq_len, char_max, -1)
        if self.dropout is not None:
            inputs = self.dropout(inputs)
        inputs = inputs.transpose(1,2) #[bz*seq_len, out_feat, char_max]
        outputs = self.encoder(inputs)
        outputs = F.max_pool1d(outputs, char_max).squeeze(2) #[bz*seq_len, out_feat]
        outputs = outputs.view(bz, seq_len, self.embedding_dim)

        return outputs

class FeatEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, dropout=0., padding_idx=0, freeze_embeddings=False):
        super(FeatEmbedding, self).__init__()
        self.freeze_embeddings = freeze_embeddings
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.embeddings.weight.requires_grad = not self.freeze_embeddings

    def forward(self, inputs):
        """
        Args:
            inputs: [bz, seq_len]
        """

        outputs = self.embeddings(inputs) #[bz, seq_len, embedding_dim]

        return outputs

class CombineEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pretrained_embeddings=None, freeze_word_embeddings=True, use_char=False, 
            use_pos=False, use_local_feat=False, char_dim=None, pos_dim=None, local_feat_dim=None, 
            char_size=None, pos_size=None, local_feat_size=None, dropout=0., padding_idx=0, char_kernel_size=3):
        super(CombineEmbedding, self).__init__()
        self.use_char = use_char
        self.use_pos = use_pos
        self.use_local_feat = use_local_feat

        self.word_embedddings = WordEmbedding(vocab_size, embedding_dim, pretrained_embeddings=pretrained_embeddings,
                        freeze_embeddings=freeze_word_embeddings)

        if use_char:
            self.char_embeddings = CharEmbedding(char_size, char_dim, char_kernel_size)
        if use_pos:
            self.pos_embeddings = FeatEmbedding(pos_size, pos_dim)
        if use_local_feat:
            self.local_feat_embeddings = FeatEmbedding(local_feat_size, local_feat_dim)

        # NOTE: apply dropout after concatenation
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
    
    def forward(self, word_inputs, char_inputs=None, pos_inputs=None, local_feat_inputs=None):
        word_inputs = self.word_embedddings(word_inputs)
        outputs = word_inputs

        if self.use_char:
            char_inputs = self.char_embeddings(char_inputs)
            outputs = torch.cat([outputs, char_inputs], dim=-1)
        if self.use_pos:
            pos_inputs = self.pos_embeddings(pos_inputs)
            outputs = torch.cat([outputs, pos_inputs], dim=-1)
        if self.use_local_feat:
            local_feat_inputs = self.local_feat_embeddings(local_feat_inputs)
            outputs = torch.cat([outputs, local_feat_inputs], dim=-1)

        if self.dropout is not None:
            outputs = self.dropout(outputs)

        return outputs

class HighwayLayer(nn.Module):
    def __init__(self, in_feat, out_feat, activation=nn.ReLU, dropout=0.):
        super().__init__()

        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

        self.trans_layer = nn.Sequential(nn.Linear(in_feat, out_feat),
                                        activation())
        self.gate_layer = nn.Sequential(nn.Linear(in_feat, out_feat),
                                        nn.Sigmoid())

        self.projection = True if in_feat != out_feat else False
        if self.projection:
            self.proj_layer = nn.Linear(in_feat, out_feat, bias=False)

    def forward(self, tensor):
        trans_tensor = self.trans_layer(tensor)
        gate_tensor = self.gate_layer(tensor)

        if self.projection:
            out_tensor = trans_tensor * gate_tensor + (1-gate_tensor) * self.proj_layer(tensor)
        else:
            out_tensor = trans_tensor * gate_tensor + (1-gate_tensor) * tensor 

        if self.dropout is not None:
            out_tensor = self.dropout(out_tensor)

        return out_tensor

class HighWayEncoder(nn.Module):
    def __init__(self, in_feat, out_feat, num_layers, activation=nn.ReLU, dropout=0.):
        """
        NOTE: Apply `dropout` on each output of highlayer
        """
        super(HighWayEncoder, self).__init__()
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(HighwayLayer(in_feat, out_feat, activation=activation, dropout=dropout))
            else:
                layers.append(HighwayLayer(out_feat, out_feat, activation=activation, dropout=dropout))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, inputs):
        """
        NOTE: Should I apply mask?
        """
        outputs = self.layers(inputs)
        return outputs

class Seq2SeqEncoder(nn.Module):
    """
    RNN taking variable length padded sequences of vectors as input and
    encoding them into padded sequences of vectors of the same length.
    This module is useful to handle batches of padded sequences of vectors
    that have different lengths and that need to be passed through a RNN.
    The sequences are sorted in descending order of their lengths, packed,
    passed through the RNN, and the resulting sequences are then padded and
    permuted back to the original order of the input sequences.
    """

    def __init__(self,
                 rnn_type,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 bias=True,
                 dropout=0.0,
                 bidirectional=False):
        """
        Args:
            rnn_type: The type of RNN to use as encoder in the module.
                Must be a class inheriting from torch.nn.RNNBase
                (such as torch.nn.LSTM for example).
            input_size: The number of expected features in the input of the
                module.
            hidden_size: The number of features in the hidden state of the RNN
                used as encoder by the module.
            num_layers: The number of recurrent layers in the encoder of the
                module. Defaults to 1.
            bias: If False, the encoder does not use bias weights b_ih and
                b_hh. Defaults to True.
            dropout: If non-zero, introduces a dropout layer on the outputs
                of each layer of the encoder except the last one, with dropout
                probability equal to 'dropout'. Defaults to 0.0.
            bidirectional: If True, the encoder of the module is bidirectional.
                Defaults to False.
        """
        assert issubclass(rnn_type, nn.RNNBase), \
            "rnn_type must be a class inheriting from torch.nn.RNNBase"

        super(Seq2SeqEncoder, self).__init__()

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.bidirectional = bidirectional

        if dropout:
            self.dropout = VariationalDropout(p=dropout)
        else:
            self.dropout = None

        self._encoder = rnn_type(input_size,
                                 hidden_size,
                                 num_layers=num_layers,
                                 bias=bias,
                                 batch_first=True,
                                 bidirectional=bidirectional)

    def forward(self, sequences_batch, sequences_lengths):
        """
        Args:
            sequences_batch: A batch of variable length sequences of vectors.
                The batch is assumed to be of size
                (batch, sequence, vector_dim).
            sequences_lengths: A 1D tensor containing the sizes of the
                sequences in the input batch.
        Returns:
            reordered_outputs: The outputs (hidden states) of the encoder for
                the sequences in the input batch, in the same order.
        """
        """
        sorted_batch, sorted_lengths, _, restoration_idx = \
            sort_by_seq_lens(sequences_batch, sequences_lengths)
        packed_batch = nn.utils.rnn.pack_padded_sequence(sorted_batch,
                                                         sorted_lengths,
                                                         batch_first=True)

        outputs, _ = self._encoder(packed_batch, None)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs,
                                                      batch_first=True)
        reordered_outputs = outputs.index_select(0, restoration_idx)
        """
        """
        There are some sequences_lengths have 0 length, use the solution of 
        https://github.com/pytorch/pytorch/issues/4582
        """
        if self.dropout:
            sequences_batch = self.dropout(sequences_batch)

        seq_lengths_clamped = torch.clamp(sequences_lengths, min=1, max=1000) 
        packed_batch = nn.utils.rnn.pack_padded_sequence(sequences_batch, seq_lengths_clamped, batch_first=True, enforce_sorted=False)
        outputs, _ = self._encoder(packed_batch)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs,
                                                      batch_first=True) #[bz, seq_len, hidden_dim]
        outputs[sequences_lengths == 0]  = 0. 

        return outputs


# ============== Modules for Feature Enhancements ===============
class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, dropout=0., padding_idx=0):
        super(Embedding, self).__init__()
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1. / math.sqrt(self.embedding_dim)
        nn.init.uniform_(self.embedding.weight, -bound, bound)

    def forward(self, inputs):
        out = self.embedding(inputs)

        if self.dropout is not None:
            out = self.dropout(out)
        
        return out
        
class DistanceBias(nn.Module):
    """
    Return a 2-d tensor with the values of distance bias to be applied on the
    intra-attention matrix with the size of time_steps
    """
    def __init__(self, max_steps=6):
        super().__init__()
        self.max_steps = max_steps

        vocab_size = 2 * max_steps + 1 
        self.embeddings = nn.Embedding(vocab_size, 1)
        nn.init.zeros_(self.embeddings.weight)
    
    def generate_distance_matrix(self, time_steps):
        max_steps = self.max_steps

        r_mat = torch.arange(time_steps).repeat(time_steps, 1) #[time_steps, time_steps]
        distance_mat = r_mat - r_mat.transpose(0,1)
        distance_mat = distance_mat.clamp(-max_steps, max_steps)
        distance_mat = distance_mat + max_steps

        return distance_mat

    def forward(self, time_steps):
        distance_matrix = self.generate_distance_matrix(time_steps)  #[time_steps, time_steps]
        distance_bias = self.embeddings(distance_matrix).squeeze() #[time_steps, time_steps]
        return distance_bias

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
        nn.init.zeros_(self.bias, )

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

class EnhancedFeature(nn.Module):
    def __init__(self, in_feature, compress="FM", k_factor=50, mode="MUL_MIN_CAT"):
        """
        """
        super(EnhancedFeature, self).__init__() 

        self.in_feature = in_feature

        if compress == "FM":
            self.compress_mul = FactorizationMachine(in_feature, k_factor) if "MUL" in mode else None
            self.compress_min = FactorizationMachine(in_feature, k_factor) if "MIN" in mode else None 
            self.compress_cat = FactorizationMachine(2*in_feature, k_factor) if "CAT" in mode else None

            if self.compress_mul == None:
                print("[Warning]: not use mul mode")
            if self.compress_min == None:
                print("[Warning]: not use min mode")
            if self.compress_cat == None:
                print("[Warning]: not use cat mode")
    
    def forward(self, input_a, input_b, align_a, align_b, mask_a=None, mask_b=None):
        """
        Args:
            input_*, align_*: [bz, seq_len_*, in_feature]
            mask_a: [bz, seq_len_a]
            mask_b: [bz, seq_len_b]

        Returns:
            features_a: [bz, seq_len_a, 3]
            features_b: [bz, seq_len_b, 3]
        """
        if mask_a != None and mask_b != None:
            mask_a = mask_a.unsqueeze(2)
            mask_b = mask_b.unsqueeze(2)
            input_a = input_a.masked_fill(~mask_a, 0.)
            align_a = align_a.masked_fill(~mask_a, 0.)
            input_b = input_b.masked_fill(~mask_b, 0.)
            align_b = align_b.masked_fill(~mask_b, 0.)
        else:
            print("[Warning]: not use mask in EnhancedFeature module.")

        features_a = [] 
        features_b = []

        if self.compress_mul is not None:
            feat_a = self.compress_mul(input_a * align_a)
            feat_b = self.compress_mul(input_b * align_b)
            features_a.append(feat_a)
            features_b.append(feat_b)

        if self.compress_min is not None:
            feat_a = self.compress_min(input_a - align_a)
            feat_b = self.compress_min(input_b - align_b)
            features_a.append(feat_a)
            features_b.append(feat_b)

        if self.compress_cat is not None:
            feat_a = self.compress_cat(torch.cat([input_a, align_a], dim=-1)) 
            feat_b = self.compress_cat(torch.cat([input_b, align_b], dim=-1))
            features_a.append(feat_a)
            features_b.append(feat_b)

        features_a = torch.cat(features_a, dim=-1)
        features_b = torch.cat(features_b, dim=-1)

        return features_a, features_b     

# ============== Modules for Aggregators ==================
class GatedAttention(nn.Module):
    """
    from https://arxiv.org/pdf/1802.04712.pdf. equation (9)
    level 2 
    """
    def __init__(self, in_features, out_features, dropout=0.0, batch_contains_review=False, review_num=None):
        super(GatedAttention, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.batch_contains_review = batch_contains_review
        self.review_num = review_num 
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

        self.trans_layer = nn.Sequential(nn.Linear(in_features, out_features, bias=False),
                                        nn.Tanh())
        self.gate_layer = nn.Sequential(nn.Linear(in_features, out_features, bias=False),
                                    nn.Sigmoid())
        self.proj_layer = nn.Linear(out_features, 1, bias=False)

    def forward(self, inputs, input_masks):
        """
        Args:
            inputs: [bz, seq_len, in_features]
            input_masks: [bz, seq_len]

        Returns:
            outputs: [bz, hidden_dim]
            att_weights: [bz, seq_len]

            if batch_contains_review:
                outputs: [bz, review_num, hidden_num]
                att_weights: [bz, review_num, seq_len]
                review_unfolded_att_weights: [bz, review_num * seq_len]
                ```where seq_len = sent_num```
        """
        att_scores = self.proj_layer(self.trans_layer(inputs) * self.gate_layer(inputs)).squeeze(-1) #[bz, seq_len]
        att_weights = F.softmax(att_scores.masked_fill(~input_masks, -1e8), dim=-1) #[bz, seq_len]
        assert att_weights.dim() == 2 
        outputs = torch.sum(att_weights.unsqueeze(2) * inputs, dim=1)

        if self.dropout is not None:
            outputs = self.dropout(outputs)

        if self.batch_contains_review:
            seq_len = att_weights.size(-1)
            sizes = [-1, self.review_num*seq_len]
            review_unfolded_att_scores = att_scores.view(*sizes)
            review_unfolded_input_masks = input_masks.view(*sizes) #[bz, review_num*seq_len] seq_len == sentence_num
            review_unfolded_att_weights = F.softmax(review_unfolded_att_scores.masked_fill(~review_unfolded_input_masks, -1e8), dim=-1) 
            
            batch_size = inputs.size(0) // self.review_num # NOTE: dirty implementation.
            outputs = outputs.view(batch_size, self.review_num, self.out_features)
            att_weights = att_weights.view(batch_size, self.review_num, seq_len)

            return outputs, att_weights, review_unfolded_att_weights
        else:
            return outputs, att_weights

# ============== Modules for Feature Transformation =================
class WeightedFeature(nn.Module):
    def __init__(self, in_features):
        super().__init__() 
    def forward(self, inputs, input_weights):
        """
        Args:
            inputs: [bz, seq_len, hidden_dim]
            input_weights: [bz, seq_len] 
            where seq_len = review_num X sentence_len

        Returns:
            outputs: [bz, seq_len, hidden_dim]
        """
        outputs = inputs * input_weights.unsqueeze(-1)
        return outputs

# ============== Modules for Alignment ================
class UnbalancedCoAttentionAggregatorReview(nn.Module):
    def __init__(self, in_features, out_features, interaction_type="BILINEAR"):
        super(UnbalancedCoAttentionAggregatorReview, self).__init__()
        self.in_features = in_features
        self.out_features = out_features 

        self.item_aggregator = GatedAttention(in_features, out_features)
        self.bilinear = BiLinearInteraction(out_features, bias=False)
    
    def forward(self, user_review_inputs, item_review_inputs, user_review_masks, item_review_masks):
        """
        Args:
            user_review_inputs: [bz, ur_num, in_features]
            item_review_inputs: [bz, ir_num, in_features]
            user_review_masks: [bz, ur_num]
            item_review_masks: [bz, ir_num]
        """
        # aggregate item_review_inputs
        item_outputs, item_review_weights = self.item_aggregator(item_review_inputs, item_review_masks)

        # interaction score
        review_similarity_scores = self.bilinear(user_review_inputs, item_review_inputs)
        user_review_scores, _ = torch.max(review_similarity_scores, dim=2) #[bz, ur_num]
        user_review_weights = masked_softmax(user_review_scores, user_review_masks)
        user_outputs = attention_weighted_sum(user_review_weights, user_review_inputs)

        return user_outputs, item_outputs, user_review_weights, item_review_weights

class UnbalancedCoAttentionAggregator(nn.Module):
    def __init__(self, in_features, out_features, item_review_num, interaction_type="BILINEAR"):
        """
        user_sent_inputs: [bz, ur_num, us_num, in_features]
        item_sent_inputs: [bz, ir_num, is_num, in_features]

        item_sent_inputs -- (GatedAttention) --> item_review_outputs[bz, ir_num, out_features]
        user_sent_inputs: -- (Alignment) -- (Aggregation) --> user_review_ouputs [bz, ur_num, out_features]
        """
        super(UnbalancedCoAttentionAggregator, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.interaction_type = interaction_type

        self.item_aggregator = GatedAttention(in_features, out_features, batch_contains_review=True, review_num=item_review_num)
        self.bilinear = BiLinearInteraction(in_features, bias=False)

    def forward(self, user_sent_inputs, item_sent_inputs, user_sent_masks, item_sent_masks, debug=False):
        """
        Args:
            user_sent_inputs: [bz, ur_num, us_num, in_features]
            item_sent_inputs: [bz, ir_num, is_num, in_features]
            user_sent_masks: [bz, ur_num, us_num]
            item_sent_masks: [bz, ir_num, is_num]

        returns:
            user_review_outputs: [bz, ur_num, out_features]
            item_review_outputs: [bz, ir_num, out_features]
        """
        # aggregate item_sent_inputs
        bz, ir_num, is_num, in_features = list(item_sent_inputs.size())
        item_sent_inputs = item_sent_inputs.view(bz*ir_num, is_num, in_features)
        item_sent_masks = item_sent_masks.view(bz*ir_num, is_num)
        item_review_outputs, item_sent_weights, item_all_sent_weights = self.item_aggregator(item_sent_inputs, item_sent_masks)

        # interaction score
        bz, ur_num, us_num, in_features = list(user_sent_inputs.size())
        item_all_sent_inputs = item_sent_inputs.view(bz, ir_num*is_num, in_features)
        item_all_sent_inputs = item_all_sent_inputs * item_all_sent_weights.unsqueeze(-1)

        chunks_of_user_sent_inputs = torch.chunk(user_sent_inputs, dim=1, chunks=ur_num)
        chunks_of_user_sent_masks = torch.chunk(user_sent_masks, dim=1, chunks=ur_num)
        user_review_outputs = []
        user_sent_weights = []
        for user_sent_inputs_pr, user_sent_masks_pr in zip(chunks_of_user_sent_inputs, chunks_of_user_sent_masks):
            user_sent_inputs_pr = user_sent_inputs_pr.squeeze(1) #[bz, us_num, in_features]
            user_sent_masks_pr = user_sent_masks_pr.squeeze(1) #[bz, us_num]
            
            ui_similarity_score_pr =  self.bilinear(user_sent_inputs_pr, item_all_sent_inputs) #[bz, us_num, ir_num*is_num]
            user_sent_scores_pr, _ = torch.max(ui_similarity_score_pr, dim=2) #[bz, us_num]
            user_sent_weights_pr = masked_softmax(user_sent_scores_pr, user_sent_masks_pr)
            
            user_review_output_pr = attention_weighted_sum(user_sent_weights_pr, user_sent_inputs_pr)
            user_review_output_pr = user_review_output_pr.unsqueeze(1) # NOTE: dirty implementation, [bz, 1, out_features]
            user_sent_weights_pr = user_sent_weights_pr.unsqueeze(1) # [bz, 1, us_num]
            user_review_outputs.append(user_review_output_pr) # list of [bz, 1, in_features]
            user_sent_weights.append(user_sent_weights_pr)
        #print("user review outputs element", user_review_outputs[0].shape)
        user_review_outputs = torch.cat(user_review_outputs, dim=1)
        user_sent_weights = torch.cat(user_sent_weights, dim=1)

        if debug:
            print("UnbalancedCoAttentionAggregator: ")
            print("user_review_outputs", user_review_outputs)
            print("item_review_outputs", item_review_outputs)
            print("user_sent_weights", user_sent_weights)
            print("item_sent_weights", item_sent_weights)
            print("item_all_sent_weights", item_all_sent_weights)

        return user_review_outputs, item_review_outputs, user_sent_weights, item_sent_weights, item_all_sent_weights
        

class ProjectionLayer(nn.Module):
    def __init__(self, in_feat, out_feat, dropout=0., num_layers=1, use_mode="FC", activation=nn.ReLU, initializer=None):
        chains = []
        for i in range(num_layers):
            if use_mode == "FC":
                if dropout:
                    chains.append(nn.Dropout(p=dropout))
                chains.append(nn.Linear(in_feat if i==0 else out_feat, out_feat))
                if activation:
                    chains.append(activation())
            elif use_mode == "HIGH":
                chains.append(HighwayLayer(in_feat if i == 0 else out_feat, out_feat, activation=nn.ReLU, dropout=dropout))
            else:
                raise ValueError("use_mode not defined.")

        self.chains = nn.Sequential(*chains)

    def forward(self, tensor):
        return self.chains(tensor)

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

        self.weight = nn.Parameter(torch.Tensor(out_feat, in_feat, in_feat))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feat))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.weight.size(1))
        nn.init.uniform_(self.weight, -bound, bound)
        if bias is not None:
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
    def __init__(self, in_feature, out_feature, interaction_type="DOT", feature_type="FC", pooling="MATRIX", dist_bias=0, k_factor=1):
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

        if dist_bias > 0:
            self.dist_bias = DistanceBias(max_steps=dist_bias)
        else:
            self.dist_bias = None 

    def forward(self, seq_a, seq_b, mask_a, mask_b):
        """
        Args:
            seq_a: [bz, seq_a, dim]
            seq_b: [bz, seq_b, dim]
            mask_a: [bz, seq_a]
            mask_b: [bz, seq_b]
        
        Returns:
            align_a: [bz, seq_a, dim]
            align_b: [bz, seq_b, dim]
            simlarity_matrix: [bz, seq_a, seq_b]
            _a: [bz, seq_a, seq_b] or [bz, seq_a]
            _b: [bz, seq_b, seq_a] or [bz, seq_b]
        """
        seq_a = self.transform_feature(seq_a)
        seq_b = self.transform_feature(seq_b)
        similarity_matrix = self.interaction(seq_a, seq_b)
        _similarity_matrix = similarity_matrix.transpose(1,2)

        if self.pooling == "MATRIX":
            mask_b = mask_b.unsqueeze(1)
            atob_soft_matrix = F.softmax(similarity_matrix.masked_fill(~mask_b, -1e8), dim=-1) #[bz, seq_a, seq_b]
            align_a = torch.bmm(atob_soft_matrix, seq_b)

            mask_a = mask_a.unsqueeze(1)
            btoa_soft_matrix = F.softmax(_similarity_matrix.masked_fill(~mask_a, -1e8), dim=-1) #[bz, seq_b, seq_a]
            align_b = torch.bmm(btoa_soft_matrix, seq_a)

            _a = atob_soft_matrix
            _b = btoa_soft_matrix
        else:
            if self.pooling == "MAX":
                att_row = similarity_matrix.max(dim=1) #[bz, seq_b]
                att_col = similarity_matrix.max(dim=2) #[bz, seq_a]
            elif self.pooling == "MEAN":
                att_row = similarity_matrix.mean(dim=1) #[bz, seq_b]
                att_col = similarity_matrix.mean(dim=2) #[bz, seq_a]
            
            att_col = F.softmax(att_col, dim=-1)
            att_row = F.softmax(att_row, dim=-1)
            _a = att_col
            _b = att_row 

            att_col = att_col.unsqueeze(2)
            att_row = att_row.unsqueeze(2)

            align_a = att_col * seq_a #[bz, seq_a, dim]
            align_b = att_row * seq_b #[bz, seq_b, dim]
        
        return align_a, align_b, similarity_matrix, _a, _b 


# ============ Modules for pooling features ===============
class MaxPooling(nn.Module):
    def __init__(self):
        super(MaxPooling, self).__init__()

    def forward(self, inputs):
        """
        inputs: [bz, seq_len, dim]
        """
        assert inputs.dim() == 3 
        
        return inputs.max(dim=1)[0]

class AvgPooling(nn.Module):
    def __init__(self):
        super(AvgPooling, self).__init__() 
    
    def forward(self, inputs):
        assert inputs.dim() == 3 
        
        return inputs.mean(dim=1)[0] 

class SequentialPooling(nn.Module):
    def __init__(self, pool_mode="MAX_AVG"):
        super(SequentialPooling, self).__init__()

        self.max_pool = MaxPooling() if "MAX" in pool_mode else None 
        self.avg_pool = MaxPooling() if "AVG" in pool_mode else None 

        if self.max_pool == None:
            print("[Warning] not use max pooling")
        if self.avg_pool == None:
            print("[Warning] not use avg pooling")
    
    def forward(self, inputs):
        features = []

        f = self.max_pool(inputs)
        features.append(f)
        
        f = self.avg_pool(inputs)
        features.append(f)

        features = torch.cat(features, dim=-1)

        return features
        
# ============ Others ===============
def glorot(shape):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    init = (2 * init_range) * torch.rand(shape[0], shape[1]) - init_range
    return init


class TorchFM(nn.Module):
    def __init__(self, n=None, k=None):
        super(TorchFM, self).__init__()
        # Initially we fill V with random values sampled from Gaussian distribution
        # NB: use nn.Parameter to compute gradients
        self.V = nn.Parameter(glorot([n, k]), requires_grad=True)
        self.lin = nn.Linear(n, 1)

    def forward(self, x):
        out_1 = torch.matmul(x, self.V).pow(2).sum(1, keepdim=True)  # S_1^2
        out_2 = torch.matmul(x.pow(2), self.V.pow(2)).sum(1, keepdim=True)  # S_2

        out_inter = 0.5 * (out_1 - out_2)
        out_lin = self.lin(x)
        out = out_inter + out_lin
        return out

if __name__ == "__main__":
    in_tensor = torch.FloatTensor([[0.4, .3, .3], [.6, 1., .4]])
    in_tensor = in_tensor.unsqueeze(0)
    FM = FactorizationMachine(3, 2)
    out_tensor = FM(in_tensor)
    print(out_tensor, out_tensor.shape)

    get_dist_bias = DistanceBias()
    dist_bias = get_dist_bias(15)
    print(dist_bias)