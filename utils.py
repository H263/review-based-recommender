import torch
import torch.nn.functional as F 

def masked_softmax(input_scores, input_masks):
    """
    Args:
        inputs: [bz, seq_len]
        input_scores: [bz, seq_len]

    Returns:
        output_weights: [bz, seq_len]
    """
    return F.softmax(torch.masked_fill(input_scores, ~input_masks, -1e8), dim=-1)

def attention_weighted_sum(input_weights, inputs):
    """
    Args:
        input_weights: [bz, seq_len] or [bz, seq_len, 1]
        inputs: [bz, seq_len, hidden_dim]
    
    Returns:
        outputs: [bz, hidden_dim]
    """
    if input_weights.dim() == 2:
        input_weights = input_weights.unsqueeze(-1)
    
    outputs = torch.sum(input_weights * inputs, dim=1)
    return outputs

def get_mask(tensor, padding_idx=0):
    """
    Get a mask to `tensor`.
    Args:
        tensor: LongTensor with shape of [bz, seq_len]

    Returns:
        mask: BoolTensor with shape of [bz, seq_len]
    """
    mask = torch.ones(size=list(tensor.size()), dtype=torch.bool)
    mask[tensor == padding_idx] = False 

    return mask 

def get_seq_lengths_from_mask(mask_tensor):
    """
    NOTE: Not generalize, just deal with a special condition where
    mask_tensor: BoolTensor with shape of [bz, review_num, sent_num, word_num]
    length_tensor: LongTensor with shape of [bz, review_num, sent_num]
    """
    int_tensor = mask_tensor.int()
    length_tensor = int_tensor.sum(dim=-1)
   
    return length_tensor


if __name__ == "__main__":
    x = torch.BoolTensor([[[1,1,0,0],[1,0,0,0], [1,1,1,0]],
                            [[1,1,1,1], [1,0,0,0], [1,1,0,0]]])
    y = get_seq_lengths_from_mask(x)
    print("bool tensor")
    print(x)
    print("length tensor")
    print(y)

    x = torch.LongTensor([[[7,8,2,0],[1,4,5,0], [2,3,4,5]],
                            [[3,3,2,1], [1,0,0,0], [1,1,0,0]]])
    y = get_mask(x)

    print("tensor")
    print(x, x.shape)
    print("corresponding mask")
    print(y)
