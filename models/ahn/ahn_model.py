import torch 
import torch.nn as nn 
import torch.nn.functional as F 


from .ahn_layers import WordEmbedding, Seq2SeqEncoder, UnbalancedCoAttentionAggregator, UnbalancedCoAttentionAggregatorReview, TorchFM, Embedding
class AHN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, k_factor, user_size, item_size, word_vocab_size, 
                pretrained_word_embeddings, rnn_dropout=0., dropout=0.5,
                item_review_num=None, word_encoder_str="LSTM"):
        super(AHN, self).__init__()
        self.hidden_dim = hidden_dim

        #self.user_word_embeddings = WordEmbedding(user_word_vocab_size, embedding_dim, user_pretrained_word_embeddings)
        #self.item_word_embeddings = WordEmbedding(item_word_vocab_size, embedding_dim, item_pretrained_word_embeddings)
        self.word_embeddings = WordEmbedding(word_vocab_size, embedding_dim, pretrained_word_embeddings)

        #self.user_word_encoder = Seq2SeqEncoder(nn.LSTM, embedding_dim, hidden_dim // 2, dropout=rnn_dropout, bidirectional=True)
        #self.item_word_encoder = Seq2SeqEncoder(nn.LSTM, embedding_dim, hidden_dim // 2, dropout=rnn_dropout, bidirectional=True)
        self.word_encoder = Seq2SeqEncoder(nn.LSTM, embedding_dim, hidden_dim // 2, dropout=rnn_dropout, bidirectional=True)

        self.unbalanced_sentence_aggregator = UnbalancedCoAttentionAggregator(hidden_dim, hidden_dim, item_review_num)
        self.user_review_trans_layer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                                    nn.ReLU())
        self.item_review_trans_layer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                                        nn.ReLU())

        self.unbalanced_review_aggregator = UnbalancedCoAttentionAggregatorReview(hidden_dim, hidden_dim)

        self.user_embeddings = Embedding(user_size, hidden_dim)
        self.item_embeddigns = Embedding(item_size, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)

        self.fm = TorchFM(n=hidden_dim*4, k=k_factor)


    def forward(self, user_reviews, item_reviews, user_sent_masks, item_sent_masks, user_sent_lengths,
                        item_sent_lenghts, user_review_masks, item_review_masks, user_ids, item_ids):
        """
        Args:
            user_reviews: [bz, ur_num, us_num, uw_num]
            item_reviews: [bz, ir_num, is_num, iw_num]
            user_sent_masks: [bz, ur_num, us_num]
            item_sent_masks: [bz, ir_num, is_num]
            user_sent_lengths: [bz, ur_num, us_num]
            item_sent_lenghts: [bz, ir_num, is_num]
            user_review_masks: [bz, ur_num]
            item_review_masks: [bz, ir_num]
            user_ids: [bz]
            item_ids: [bz]
        """ 
        # encode and aggreate word --> sentence
        bz, ur_num, us_num, uw_num = list(user_reviews.size())
        _, ir_num, is_num, iw_num = list(item_reviews.size())
        user_words = self.word_embeddings(user_reviews)
        item_words = self.word_embeddings(item_reviews)

        user_words = user_words.view(bz*ur_num*us_num, uw_num, self.hidden_dim)
        item_words = item_words.view(bz*ir_num*is_num, iw_num, self.hidden_dim)
        user_sent_lengths = user_sent_lengths.view(bz*ur_num*us_num)
        item_sent_lenghts = item_sent_lenghts.view(bz*ir_num*is_num)
        user_words = self.word_encoder(user_words, user_sent_lengths) #[bz*rn*sn, wn, hdz]
        item_words = self.word_encoder(item_words, item_sent_lenghts)

        user_sents, _ = torch.max(user_words, dim=1)
        user_sents = user_sents.view(bz, ur_num, us_num, self.hidden_dim)
        item_sents, _ = torch.max(item_words, dim=1)
        item_sents = item_sents.view(bz, ir_num, is_num, self.hidden_dim)

        # aggregate sentence --> review
        user_reviews, item_reviews, \
        user_sent_weights, item_sent_weights, \
            item_all_sent_weights = self.unbalanced_sentence_aggregator(user_sents, item_sents, user_sent_masks, item_sent_masks)
                            #[bz, rn, hdz]
        user_reviews = self.user_review_trans_layer(user_reviews)
        item_reviews = self.item_review_trans_layer(item_reviews)

        # aggregate review  --> user/item
        rv_users, rv_items, \
        user_review_weights, item_review_weights = self.unbalanced_review_aggregator(user_reviews, item_reviews, user_review_masks, item_review_masks)

        # final output
        id_users = self.user_embeddings(user_ids)
        id_items = self.item_embeddigns(item_ids)
        users = torch.cat([rv_users, id_users], dim=-1)
        items = torch.cat([rv_items, id_items], dim=-1)

        final_features = torch.cat([users, items], dim=-1)
        final_features = self.dropout(final_features)
        out_logits = self.fm(final_features).view(-1)

        return out_logits, user_sent_weights, item_sent_weights, user_review_weights, item_review_weights
