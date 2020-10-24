import torch.nn as nn 
import torch 
import torch.nn.functional as  F




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

class LocalAttention(nn.Module):
    def __init__(self, doc_len, window_size, out_size, emb_size=100):
        super(LocalAttention, self).__init__()
        self.window_size = window_size
        self.doc_len = doc_len
        self.out_size = out_size
        self.emb_size = emb_size
        self.padding_size = (self.window_size - 1) // 2

        self.attn = nn.Sequential(
                        nn.Conv1d(emb_size, 1, kernel_size=window_size, padding=self.padding_size),
                        nn.Sigmoid())
        self.conv = nn.Sequential(
                        nn.Conv1d(emb_size, out_size, kernel_size=1),
                        nn.Tanh(),
                        nn.MaxPool1d(doc_len))


    def forward(self, x):
        """
        Args:
            x: torch.Tensor with shape of [bz, doc_len, emb_size]
        """
        x = x.permute(0,2,1).contiguous()
        score = self.attn(x)
        out = torch.mul(score, x) #[bz, emb_size, doc_len]
        out = self.conv(out) #[bz, out_size, 1]

        return out

class GlobalAttention(nn.Module):
    def __init__(self, doc_len, out_size, emb_size=100):
        """
        Note: we hard encode the window size [2, 3, 4]
        """
        super(GlobalAttention, self).__init__()
        self.doc_len = doc_len
        self.out_size = out_size
        self.emb_size = emb_size

        self.attn = nn.Sequential(
                        nn.Conv1d(emb_size, 1, kernel_size=doc_len),
                        nn.Sigmoid())
        self.conv1 = nn.Sequential(
                        nn.Conv1d(emb_size, out_size, kernel_size=2),
                        nn.Tanh(),
                        nn.MaxPool1d(doc_len-1))
        self.conv2 = nn.Sequential(
                        nn.Conv1d(emb_size, out_size, kernel_size=3),
                        nn.Tanh(),
                        nn.MaxPool1d(doc_len-2))
        self.conv3 = nn.Sequential(
                        nn.Conv1d(emb_size, out_size, kernel_size=4),
                        nn.Tanh(),
                        nn.MaxPool1d(doc_len-3))

    def forward(self, x):
        x = x.permute(0,2,1).contiguous()
        score = self.attn(x)
        out = torch.mul(score, x) #[bz, emb_size, doc_len]
        out_1 = self.conv1(out)
        out_2 = self.conv2(out)
        out_3 = self.conv3(out) #[bz, out_size, 1]

        return (out_1, out_2, out_3)

