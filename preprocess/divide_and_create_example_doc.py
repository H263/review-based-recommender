import json 
import argparse
from collections import defaultdict
import os
import pickle
import gzip
import re

import pandas as pd
from tqdm import tqdm
import numpy as np

from preprocess._tokenizer import Vocab, Indexlizer
from preprocess._stop_words import ENGLISH_STOP_WORDS


"""
NOTE:
    - Concatenate all reviews from each user or item to form a giant document for models like DeepCoNN, D-Att.
"""
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default='/raid/hanszeng/datasets/amazon_dataset/reviews_Toys_and_Games_5.json.gz')
    parser.add_argument("--dest_dir", default="./datasets/Toys_and_Games_5/doc_split/")
    parser.add_argument("--rv_num_keep_prob", default=0.9, type=float)
    parser.add_argument("--max_doc_len", default=500, type=int)
    parser.add_argument("--random_shuffle", default=False)

    args = parser.parse_args() 

    return args 

def truncate_pad_tokens(tokens, max_seq_len, pad_token):
    # truncate 
    if len(tokens) > max_seq_len:
        tokens = tokens[:max_seq_len]
    # pad
    res_length = max_seq_len - len(tokens)
    tokens = tokens + [pad_token] * res_length
    return tokens

def write_pickle(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)

def split_data(args):
    path = args.data_path
    dest_dir = args.dest_dir

    f = gzip.open(path)

    users = [] 
    items = [] 
    ratings = [] 
    reviews = [] 
    times = []

    for line in f:
        js_dict = json.loads(line)
        if str(js_dict['reviewerID'])=='unknown':
            print("unknown user")
            continue
        if str(js_dict['asin'])=='unknown':
            print("unknown item")
            continue

        users.append(js_dict["reviewerID"])
        items.append(js_dict["asin"])
        ratings.append(js_dict["overall"])
        reviews.append(js_dict["reviewText"])
        times.append(js_dict["unixReviewTime"])

    df = pd.DataFrame({"user_id": pd.Series(users),
                        "item_id": pd.Series(items),
                        "rating": pd.Series(ratings),
                        "review": pd.Series(reviews),
                        "time": pd.Series(times)})

    # sort df by `user_id` and `time`  and split
    df = df.sort_values(by=["user_id", "time"]).reset_index(drop=True)
    print(df.iloc[:10])

    # randomly divide train, validation, test set by 0.8, 0.1, 0.1.
    np.random.seed(20200616)
    num_samples = len(df)
    train_idx = np.random.choice(num_samples, int(num_samples*0.8), replace=False)
    remain_idx = list(set(range(num_samples)) - set(train_idx))
    train_idx = list(train_idx)
    num_remain = len(remain_idx)
    valid_idx = remain_idx[:int(num_remain * 0.5)]
    test_idx = remain_idx[int(num_remain * 0.5):]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    if args.random_shuffle:
        train_df = train_df.sample(frac=1).reset_index(drop=True)
    valid_df = df.iloc[valid_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # postprocessing: remove user and item in training that only have one review
    item_id_counts = train_df.groupby("item_id")["review"].agg(["count"])
    item_id_counts[item_id_counts.index.name] = item_id_counts.index
    item_id_counts = item_id_counts.reset_index(drop=True)

    user_id_counts = train_df.groupby("user_id")["review"].agg(["count"])
    user_id_counts[user_id_counts.index.name] = user_id_counts.index
    user_id_counts = user_id_counts.reset_index(drop=True)

    remove_uids =  set(list(user_id_counts[user_id_counts["count"] ==1].user_id))
    remove_iids = set(list(item_id_counts[item_id_counts["count"] == 1].item_id))

    print(f"remove uids: {len(remove_uids)}, iids:  {len(remove_iids)}")
    print(f"len train, valid, test df: {len(train_df)}, {len(valid_df)}, {len(test_df)}")
    for rmuid in remove_uids:
        train_df = train_df[train_df.user_id != rmuid]
        valid_df = valid_df[valid_df.user_id != rmuid]
        test_df = test_df[test_df.user_id != rmuid]
        
    for rmiid in remove_iids:
        train_df = train_df[train_df.item_id != rmiid]
        valid_df = valid_df[valid_df.item_id != rmiid]
        test_df = test_df[test_df.item_id != rmiid]

    train_df = train_df.reset_index(drop=True)
    valid_df = test_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    del df

    # postprocessing: remove user and item for valid and test dataset if they not contains in train
    def get_unique_user_item_ids(_df):
        return set(_df.user_id.unique()), set(_df.item_id.unique())
    train_user_ids, train_item_ids = get_unique_user_item_ids(_df=train_df)
    valid_user_ids, valid_item_ids = get_unique_user_item_ids(_df=valid_df)
    test_user_ids, test_item_ids = get_unique_user_item_ids(_df=test_df)
    
    remove_uids = (valid_user_ids.union(test_user_ids)).difference(train_user_ids)
    remove_iids = (valid_item_ids.union(test_item_ids)).difference(train_item_ids)
    
    for rmuid in remove_uids:
        valid_df = valid_df[valid_df.user_id != rmuid]
        test_df = test_df[test_df.user_id != rmuid]
    for rmiid in remove_iids:
        valid_df = valid_df[valid_df.item_id != rmiid]
        test_df = test_df[test_df.item_id != rmiid]

    print(f"remove uids: {len(remove_uids)}, iids:  {len(remove_iids)}")
    print(f"len train, valid, test df: {len(train_df)}, {len(valid_df)}, {len(test_df)}")

    # numerize user and item
    users = list(train_df["user_id"])
    items = list(train_df["item_id"])
    user2id = {u:i+1 for i, u in enumerate(np.unique(users))}
    item2id = {it:i+1 for i, it in enumerate(np.unique(items))}
    print(f"user2id: {list(user2id.items())[:10]}, item2id: {list(item2id.items())[:10]}")
    user2id["<pad>"] = 0 
    item2id["<pad>"] = 0
    print(f"user2id: {list(user2id.items())[:10]}, item2id: {list(item2id.items())[:10]}")

    train_df["user_id"] = train_df["user_id"].apply(lambda x: user2id[x])
    train_df["item_id"] = train_df["item_id"].apply(lambda x: item2id[x])
    valid_df["user_id"] = valid_df["user_id"].apply(lambda x: user2id[x])
    valid_df["item_id"] = valid_df["item_id"].apply(lambda x: item2id[x])
    test_df["user_id"] = test_df["user_id"].apply(lambda x: user2id[x])
    test_df["item_id"] = test_df["item_id"].apply(lambda x: item2id[x])
    
    write_pickle(os.path.join(args.dest_dir, "raw_train_df.pkl"), train_df)
    write_pickle(os.path.join(args.dest_dir, "raw_valid_df.pkl"), valid_df)
    write_pickle(os.path.join(args.dest_dir, "raw_test_df.pkl"), test_df)

    return train_df, valid_df, test_df



def create_meta(df, args):
    meta = {}
    # statistics
    reviews = list(df.review)
    indexlizer = Indexlizer(reviews, special_tokens=["<pad>", "<unk>", "<sep>"], preprocessor=clean_str, mode="word",
                        stop_words=ENGLISH_STOP_WORDS, max_len=args.max_doc_len)
    #indexlized_reviews = indexlizer.transform(reviews)
    #df["idxed_review"] = indexlized_reviews
    meta["user_num"] = df.user_id.max() + 1
    meta["item_num"] = df.item_id.max() + 1 # 加上 pad_idx 0, 并且考虑了空隙
    print(df.user_id.max(), df.item_id.max())

    # create meta for reviews and rids
    user_docs = {}
    item_docs = {}
    
    # first: form giant document
    train_users = list(df["user_id"])
    train_items = list(df["item_id"])
    train_reviews = list(df["review"])

    for user, item, review in zip(train_users, train_items, train_reviews):
        if user not in user_docs:
            user_docs[user] = review + " <sep> "
        else:
            user_docs[user] += review + " <sep> "
        if item not in item_docs:
            item_docs[item] = review + " <sep> "
        else:
            item_docs[item] += review + " <sep> "
    
    # second: indexlize giant document
    for user, doc in tqdm(user_docs.items()):
        idxed_doc = indexlizer.transform([doc])[0]
        user_docs[user] = idxed_doc
    for item, doc in tqdm(item_docs.items()):
        idxed_doc = indexlizer.transform([doc])[0]
        item_docs[item] = idxed_doc
    
    meta["user_docs"] = user_docs
    meta["item_docs"] = item_docs
    meta["indexlizer"] = indexlizer
    meta["doc_len"] = args.max_doc_len

    # test 
    t_uid, t_iid = 1, 45 
    t_doc = meta["user_docs"][t_uid]
    print("uid: ", t_uid)
    print("idxed decoded doc: ",  t_doc)
    print("decoded doc: ", indexlizer.transform_idxed_review(t_doc))
    print(len(meta["user_docs"][t_uid]))

    t_doc = meta["user_docs"][t_iid]
    print("iid: ", t_iid)
    print("decoded doc: ",  t_doc)
    print("decoded doc: ", indexlizer.transform_idxed_review(t_doc))
    print(len(meta["item_docs"][t_uid]))

    return meta 
    

def create_examples(df, meta, set_name, args):
    examples = []
    for _, row in tqdm(df.iterrows()):
        uid = row.user_id 
        iid = row.item_id 
        rating = row.rating
        u_doc = meta["user_docs"][uid]
        i_doc = meta["item_docs"][iid]

        exp = [uid, iid, rating, u_doc, i_doc]
        examples.append(exp)
    
    return examples

if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.dest_dir):
        os.makedirs(args.dest_dir)

    train_df, valid_df, test_df = split_data(args)
    meta = create_meta(train_df, args)

    train_examples = create_examples(train_df, meta, "train", args)
    valid_examples = create_examples(valid_df, meta, "valid", args)
    test_examples = create_examples(test_df, meta, "test", args)

    # print meta 
    for k, v in meta.items():
        if isinstance(v, dict):
            print(k)
        else:
            print(k, v)

    write_pickle(os.path.join(args.dest_dir, "meta.pkl"), meta)
    write_pickle(os.path.join(args.dest_dir, "train_exmaples.pkl"), train_examples)
    write_pickle(os.path.join(args.dest_dir, "valid_exmaples.pkl"), valid_examples)
    write_pickle(os.path.join(args.dest_dir, "test_exmaples.pkl"), test_examples)
