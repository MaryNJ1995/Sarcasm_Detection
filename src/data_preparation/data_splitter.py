import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def splite_dataset(dataframe):
    # print(tabulate(dataframe[:50], headers='keys', tablefmt='psql'))
    dataframe = dataframe.sample(frac=1).reset_index(drop=True)
    dataframe = shuffle(dataframe)
    dataframe = dataframe[["id", "tweet", "sarcastic", "rephrase", "dialect"]]
    train, test = train_test_split(dataframe, test_size=0.20, random_state=345, stratify=dataframe['sarcastic'])
    # train, dev = train_test_split(train, test_size=0.15, random_state=123, stratify=train['sarcastic'])

    train = train.reset_index(drop=True)
    # dev = dev.reset_index(drop=True)
    test = test.reset_index(drop=True)

    train.to_csv("train_data.csv", index=False,
                 header=["id", "tweet", "sarcastic", "rephrase", "dialect"])
    # dev.to_csv("/mnt/hd/file/AA/data/Processed/news_chunk15_part_test_unnorm.csv", index=False,
    #            header=["text", "label", "pos"])
    print(len(train))
    test.to_csv("test_data3.csv", index=False,
                header=["id", "tweet", "sarcastic", "rephrase", "dialect"])
    print(len(test))


# print(len(set(dataframe["Author"])))
if __name__ == "__main__":
    dataframe = pd.read_csv("../../../data/SemEval/Task_A_AR/all_data.csv")
    # dataframe = dataframe.dropna()

    # read_symbols_file(dataframe, "aa_data_part/normal_mut_tweets.xlsx")
    splite_dataset(dataframe)

    # dataframe = pd.read_csv("/mnt/hd/file/AA/data/Processed/news_chunk15_part_dev_unnorm.csv")
    # dataframe.to_excel("/mnt/hd/file/AA/data/Processed/news.xlsx", index=False)
