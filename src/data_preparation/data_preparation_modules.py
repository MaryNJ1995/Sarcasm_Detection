import re
from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split

from models.pretrain_t5 import Classifier


def create_dataset():
    df_train = pd.read_csv("../../../data/SarcasmTweetKaggle/train.csv")
    df_test = pd.read_csv("../../../data/SarcasmTweetKaggle/test.csv")
    tweets = []
    labels = []
    for tweet, lbl in zip(df_train["tweets"], df_train["class"]):
        if lbl in ["regular", "sarcasm"]:
            labels.append(str(lbl))
            tweets.append(str(tweet))

    print(len(tweets))

    for tweet, lbl in zip(df_test["tweets"], df_test["class"]):
        if lbl in ["regular", "sarcasm"]:
            labels.append(str(lbl))
            tweets.append(str(tweet))

    df = pd.DataFrame({"tweets": tweets, "labels": labels})
    df.to_csv("total.csv", index=False)


def pre_process(input_text):
    input_text = re.sub(r"http\S+", "", input_text)
    input_text = re.sub("http://[a-zA-z./\d]*", "", input_text)

    input_text = re.sub("@[^\s]+", "", input_text)
    input_text = re.sub("@([^@]{0,30})\s", "", input_text)
    input_text = re.sub("@([^@]{0,30})）", "", input_text)
    # input_text = input_text.replace("#Irony", " ")
    # input_text = input_text.replace("#irony", " ")
    # input_text = input_text.replace("#sarcasm", " ")
    # input_text = input_text.replace("#Sarcasm", " ")
    input_text = re.sub("\s\s+", " ", input_text)

    return input_text


def remove_hashtags(input_text):
    new_string = ''
    for i in input_text.split():
        if i[:1] != '#':
            new_string = new_string.strip() + ' ' + i
    return new_string


def is_larger_than_value(input_text):
    if len(input_text.split(" ")) < 4:
        return False
    return True


def runner():
    df = pd.read_csv("total.csv")
    print(len(df))
    my_tweets = []
    my_labels = []
    for tweet, lbl in zip(df.tweets, df.labels):
        tweet = pre_process(tweet)
        tweet = remove_hashtags(tweet)
        if is_larger_than_value(tweet):
            my_tweets.append(tweet)
            if lbl == "sarcasm":
                my_labels.append("1")
            elif lbl == "regular":
                my_labels.append("0")

    a = Counter()
    a.update(my_labels)
    print(a)

    new_df = pd.DataFrame({"tweets": my_tweets, "labels": my_labels})
    print(len(new_df))

    new_df.to_csv("final_data_1.csv", index=False)


def split_data():
    df = pd.read_csv("final_data_1.csv")
    train, test = train_test_split(df, test_size=0.25, shuffle=True,
                                   random_state=1234)
    train.to_csv("pretrain_train.csv", index=False)
    test.to_csv("pretrain_test.csv", index=False)


def save_lm_model():
    model_path = "../assets/saved_models/t5_large/version_3/checkpoints/QTag-epoch=04-val_loss=0.31.ckpt"

    model = Classifier.load_from_checkpoint(model_path)
    model.t5_model.save_pretrained("./finetune_t5")


def create_multi_label_data_set():
    df = pd.read_csv("../data/SemEval/all_data.csv")
    texts, sarcasm_lbls, irony_lbls, satire_lbls, understatement_lbls, overstatement_lbls, \
    rhetorical_question_lbls = [], [], [], [], [], [], []
    for txt, sarcastic, sarcasm, irony, satire, understatement, overstatement, rhetorical_question in zip(
            df["tweet"], df["sarcastic"], df["sarcasm"], df["irony"], df["satire"],
            df["understatement"], df["overstatement"], df["rhetorical_question"]
    ):
        if sarcastic == 1:
            texts.append(txt)
            sarcasm_lbls.append(int(sarcasm))
            irony_lbls.append(int(irony))
            satire_lbls.append(int(satire))
            understatement_lbls.append(int(understatement))
            overstatement_lbls.append(int(overstatement))
            rhetorical_question_lbls.append(int(rhetorical_question))

    new_df = pd.DataFrame({"tweets": texts, "sarcasm": sarcasm_lbls,
                           "irony": irony_lbls, "satire": satire_lbls,
                           "understatement": understatement_lbls,
                           "overstatement": overstatement_lbls,
                           "rhetorical_question": rhetorical_question_lbls})
    train, test = train_test_split(new_df, test_size=0.25, shuffle=True,
                                   random_state=1234)
    train.to_csv("../data/SemEval/train_data.csv", index=False)
    test.to_csv("../data/SemEval/dev_data.csv", index=False)


def create_pair_data_set():
    df = pd.read_csv("../data/SemEval/all_data.csv")
    texts, pair_texts, labels = [], [], []
    flag = True
    for text, pair_text, lbl in zip(df.tweet, df.rephrase, df.sarcastic):
        if lbl == 1:
            if flag:
                texts.append(text)
                pair_texts.append(pair_text)
                labels.append(0)
                flag = not flag
            else:
                texts.append(pair_text)
                pair_texts.append(text)
                labels.append(1)
                flag = not flag

    new_df = pd.DataFrame({"tweets": texts, "pair_tweets": pair_texts, "labels": labels})
    train, test = train_test_split(new_df, test_size=0.25, shuffle=True,
                                   random_state=1234)
    train.to_csv("../data/SemEval/train_data.csv", index=False)
    test.to_csv("../data/SemEval/dev_data.csv", index=False)


create_multi_label_data_set()
