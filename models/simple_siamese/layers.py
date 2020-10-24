import torch.nn as nn 
import torch 
import torch.nn.functional as  F

from .utils import masked_tensor

class NodeDropout(torch.nn.Dropout):
    def forward(self, input_tensor):
        """
        Args:
            input_tensor: [bz, seq_len, embedding_dim]
        
        Returns:
            output: [bz, seq_len, embedding_dim]
        """
        ones = input_tensor.data.new_ones(input_tensor.shape[0], input_tensor.shape[1])
        dropout_mask = torch.nn.functional.dropout(ones, self.p, self.training, inplace=False)
        if self.inplace:
            input_tensor *= dropout_mask.unsqueeze(2)
            return None
        else:
            return dropout_mask.unsqueeze(2) * input_tensor

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
    def __init__(self, vocab_size, embedding_dim, pretrained_embeddings=None, padding_idx=0, freeze_embeddings=False, sparse=False):
        super(WordEmbedding, self).__init__()

        self.freeze_embeddings = freeze_embeddings

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx, sparse=sparse)
        self.embedding.weight.requires_grad = not self.freeze_embeddings
        if pretrained_embeddings is not None:
            self.embedding.load_state_dict({"weight": torch.tensor(pretrained_embeddings)})
        else:
            print("[Warning] not use pretrained embeddings ...")

    def forward(self, inputs):
        out = self.embedding(inputs)     
        return out

class PairWiseAggre(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inputs):
        """
        Args:
            inputs: [bz, rv_num, hdim]
        
        Returns:
            output: [bz, hdim]
        """
        bz, rv_num, hdim = list(inputs.size())

        inputs_sum = inputs.sum(dim=1)
        term_1 = inputs_sum * inputs_sum #[bz, hdim]
        
        term_2 = (inputs * inputs).sum(dim=1)

        return (term_1 - term_2) / (rv_num * 2.)

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
        if "ATT" in self.mode:
            self.att_layer = AddictiveAttention(out_feature_per_kernel, out_feature_per_kernel)
            print('use attention as pooling.')

    def forward(self, inputs, input_masks):
        """
        Args:
            inputs: [bz, seq_len, in_feat]
            input_masks: [bz, seq_len]

        Returns:
            out_ngrams: [bz, out_feat]
        """
        bz, _, _ = list(inputs.size())
        inputs = masked_tensor(inputs, input_masks)
        inputs = inputs.transpose(1,2)
        list_of_ngram = [ng_layer(inputs) for ng_layer in self.ngrams] # list of [bz, out_feat, var_seq_lens]
        #print("list of ngram: ", list_of_ngram)
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
        if "ATT" in self.mode:
            # transpose list of ngrams 
            list_of_ngram = [x.transpose(1,2) for x in list_of_ngram]
            list_att_masks = [input_masks[:, :self.seq_len-kz+1]  for kz in self.kernel_sizes]
            list_att_ngrams = [self.att_layer(ngram, mask)[0] for ngram, mask in  zip(list_of_ngram, list_att_masks)] # list of [bz, out_feat, 1]
            # transpose att ngrams
            list_att_ngrams = [x.unsqueeze(2) for x in list_att_ngrams]
            out_ngrams += list_att_ngrams

        out_ngrams = torch.cat(out_ngrams, dim=1).view(bz, -1)

        return out_ngrams

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
        assert input_masks.dim() == 2

        att_logtis = self.inner_product(self.proj_layer(inputs)) #[bz, seq_len, 1]
        input_masks = input_masks.unsqueeze(2) #[bz, seq_len, 1]
        att_scores = F.softmax(torch.masked_fill(att_logtis, ~input_masks, -1e8), dim=1) #[bz, seq_len, 1]

        outptus = torch.sum(att_scores * inputs, dim=1)

        return outptus, att_scores


class RepByRatMask(nn.Module):
    def __init__(self, hidden_dim, latent_dim, num_type_of_rating=5):
        super().__init__()
        self.num_type_of_rating = num_type_of_rating

        self.r_addict_attentions = nn.ModuleList(
            [AddictiveAttention(hidden_dim, latent_dim) for _ in range(num_type_of_rating)]
        )
    def forward(self, inputs, list_of_mask):
        """
        Args:
            inputs: [bz, seq_len, hdim]
            list_of_mask: list of [bz, seq_len]

        Returns:
            outputs: [bz, num_type_of_rating, hdim]
        """
        bz, seq_len, hdim = list(inputs.size())
        outputs = []
        for mask in list_of_mask:
            mask = mask.unsqueeze(2)
            outputs.append(torch.masked_fill(inputs, ~mask, 0.)) # list of [bz, seq_len, hdim]
        
        #for i, out in enumerate(outputs):
            #print(f"{i+1}, out: {out[:2]}")
        for i, (out, add_atten_layer, mask) in enumerate(zip(outputs, self.r_addict_attentions, list_of_mask)):
            #tmp = F.max_pool1d(out.transpose(1,2), kernel_size=seq_len).view(bz, hdim) # [bz, ]
            tmp = add_atten_layer(out, mask)[0].unsqueeze(1) # list of [bz, 1, hdim] with length of `num_type_of_rating`
            outputs[i] = tmp
        
        outputs = torch.cat(outputs, dim=1)

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

class FMWithoutUIBias(nn.Module):
    def __init__(self, user_size, item_size, latent_dim, dropout, user_padding_idx, item_padding_idx):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        self.h = nn.Parameter(torch.Tensor(latent_dim, 1))
        self.g_bias = nn.Parameter(torch.Tensor(1))

        self.reset_parameters()

    def reset_parameters(self):
        bound = 0.1
        nn.init.uniform_(self.h, -bound, bound)
        nn.init.constant_(self.g_bias, 4.0)


    def forward(self, u_feat, i_feat):
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

        pred = fm @ self.h + self.g_bias

        return pred

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
        nn.init.constant_(self.g_bias, 4.0)


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

if __name__ == "__main__":
    bz = 2
    inputs = torch.randn(bz, 10, 3)
    masks = torch.tril(torch.ones(bz,10)).bool()
    masks[0][0]= False
    print("masks", masks)

    ngram_feat_layer = TanhNgramFeat([2], 3, 3, 10)
    print("ngram feature", ngram_feat_layer(inputs, masks))

    print("------- test addictive attention layer ------")
    aa_layer = AddictiveAttention(3, 2)
    outs, scores = aa_layer(inputs, masks)
    print("outs: ", outs)
    print("scores: ", scores)

    print("----- test pairwise aggre --------")
    pw_aggre = PairWiseAggre()
    a = torch.randn(8, 11, 3)
    print("pairwise aggre: ", pw_aggre(a))

    print("------ test RepByRatMask --------")
    masks = []
    ratings = torch.randint(2, 6, size=(4,6))
    ratings[0][5:] = 0
    ratings[-1][:] = 0
    for scale in [1,2,3,4,5]:
        mask = torch.zeros_like(ratings)      
        mask_idx = torch.where(ratings == scale)
        mask[mask_idx] = 1
        masks.append(mask.bool())
    rep_by_mask_layer = RepByRatMask(3, 3, num_type_of_rating=5)
    
    inputs = torch.randn(4,6,3)
    print(f"ratings: {ratings} \n inputs: {inputs}")
    outputs = rep_by_mask_layer(inputs, masks)
    print(outputs, outputs.shape)
