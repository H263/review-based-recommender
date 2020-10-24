import torch
import torch.nn as nn
import numpy as np
from nltk import word_tokenize
import pandas as pd
import time
from .layers import WordEmbedding, LocalAttention, GlobalAttention
# ============ Helper function ===============
def to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    else:
        print("cuda is not available")
    return x


# ============= Create Model ====================
""" Create Model """
class DualAtt(nn.Module):
    def __init__(self, vocab_size, doc_len, l_window_size=5, l_out_size=200, g_out_size=100, emb_size=100,
                 hidden_size_1=500, hidden_size_2=50, dropout=0.5, pretrained_embeddings=None):
        super(DualAtt, self).__init__()
        self.fc_input = l_out_size + 3*g_out_size

        self.word_embeddings =  WordEmbedding(vocab_size, emb_size, pretrained_embeddings=pretrained_embeddings)
        self.u_local_atten = LocalAttention(doc_len, l_window_size, l_out_size, emb_size)
        self.u_global_atten = GlobalAttention(doc_len, g_out_size, emb_size)
        self.i_local_atten = LocalAttention(doc_len, l_window_size, l_out_size, emb_size)
        self.i_global_atten = GlobalAttention(doc_len, g_out_size, emb_size)

        self.fc = nn.Sequential(
                    nn.Linear(self.fc_input, hidden_size_1),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size_1, hidden_size_2),)
        
    def forward(self, u_docs, i_docs):
        """
        Args: 
            u_docs: [bz, doc_len]
            i_docs: [bz, doc_len]
        Returns:
            ratings: [bz]
        """
        user_input = self.word_embeddings(u_docs)
        item_input = self.word_embeddings(i_docs)
        u_local_out = self.u_local_atten(user_input)
        u_global_out_1, u_global_out_2, u_global_out_3 = self.u_global_atten(user_input)
        u_feat = torch.cat((u_local_out,u_global_out_1, u_global_out_2, u_global_out_3), 1)
        u_feat = u_feat.view(u_feat.size(0), -1) #[bz, feat_size]
        u_feat = self.fc(u_feat) # [bz, hidden_size_2]

        i_local_out = self.i_local_atten(item_input)
        i_global_out_1, i_global_out_2, i_global_out_3 = self.i_global_atten(item_input)
        i_feat = torch.cat((i_local_out, i_global_out_1, i_global_out_2, i_global_out_3), 1)
        i_feat = i_feat.view(i_feat.size(0), -1)
        i_feat = self.fc(i_feat)

        ratings = torch.sum(torch.mul(u_feat, i_feat), 1)

        return ratings.view(-1)

if __name__ == "__main__":
    # ====== Hyperparameters =======
    doc_len = 10000
    epoch_num = 40
    batch_size = 64
    learning_rate = 1e-4

    # ====== Data path =============
    #prefix = "/raid/hanszeng/datasets/temp/"
    prefix = "/raid/hanszeng/datasets/"
    train_csv_path = prefix + "yelp-recsys-2013/data/train.csv"
    test_csv_path = prefix + "yelp-recsys-2013/data/test.csv"
    glove_path = prefix + "pretrained_vocab/glove.6B/glove.6B.100d.txt"
    data_path = prefix + "yelp-recsys-2013/data/"

    # ======= Define Dataset =========
    train_dataset = ReviewData(train_csv_path, glove_path, data_path)
    train_loader = get_dataloader(train_dataset, batch_size=batch_size)
    test_dataset = ReviewData(test_csv_path, glove_path, data_path)
    test_loader = get_dataloader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"Reading train data from {train_csv_path}, test data from {test_csv_path}")
    print(f"Training dataset size {len(train_dataset)},\t test dataset size: {len(test_dataset)}")


    # ===== Define Model ===========
    model = CNNDLGA(doc_len)
    if torch.cuda.is_available():
        model.cuda()
    else:
        print("model cannot be put into cuda")

    # ====== Define Optimizer =======
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ====== Define Loss ========
    loss_func = nn.MSELoss()


    # ====== Start Trainnig ========
    print("---------------- start training -------------------")
    batch_loss = 0.0
    start_time = time.time()
    total_step = len(train_loader)
    for epoch in range(epoch_num):
        for i, (u_batch, i_batch, labels) in enumerate(train_loader):

            # put data into cuda
            u_batch = to_cuda(u_batch.float())
            i_batch = to_cuda(i_batch.float())
            labels = to_cuda(labels.float())

            # forward + backward + optimize
            optimizer.zero_grad()
            y_pred = model(u_batch, i_batch)
            loss = loss_func(y_pred, labels)
            loss.backward()
            optimizer.step()

            batch_loss += loss.data.cpu().numpy()
            if (i+1) % 10 == 0:
                batch_loss /= 10
                print(f"epoch: {epoch+1}/{epoch_num}\t step: {i+1}/{total_step}\t batch loss: {batch_loss}\
                time/batch: {(time.time() - start_time) / 10}")
                batch_loss = 0.
                start_time = time.time()

            if (i+1) % 1000 == 0:
                torch.save(model.state_dict(), 'cnndlga_conv1d/model_'+str(epoch)+'.pkl')

            #======== Evaluation ===========
            if (i+1) % 2000 == 0:
                test_time_start = time.time()
                print("------- start testing, just see one batch ----------")
                # start testint
                rmse = 0.0
                test_bz = len(test_loader)
                for j, (u_batch, i_batch, labels) in enumerate(test_loader):
                    u_batch = to_cuda(u_batch.float())
                    i_batch = to_cuda(i_batch.float())
                    labels = to_cuda(labels.float())

                    with torch.no_grad():
                        outputs = model(u_batch, i_batch)
                        loss = loss_func(outputs, labels)
                        rmse += torch.sqrt(loss).item()


                print(f"epoch: {epoch+1}/{epoch_num}\t test rmse: {rmse/test_bz}")
