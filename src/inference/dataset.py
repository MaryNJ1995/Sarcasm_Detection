from torch.utils.data import Dataset
from typing import List
import torch
import pandas as pd


class InferenceDataset(Dataset):
    def __init__(self, texts: List[list], tokenizer, max_length: int):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item_index):
        inputs = self.tokenizer.encode_plus(
            text=self.texts[item_index],
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True
        )

        return {"inputs_ids": inputs["input_ids"].flatten(),
                "attention_mask": inputs["attention_mask"].flatten()}


class PairSarcasmDataset(Dataset):
    def __init__(self, texts: list, text_pairs: list, targets: list, tokenizer, max_len):
        self.texts = texts
        self.text_pairs = text_pairs
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item_index):
        text = self.texts[item_index]
        text_pair = self.text_pairs[item_index]
        target = self.targets[item_index]

        inputs_ids = self.tokenizer.encode_plus(text=text,
                                                text_pair=text_pair,
                                                add_special_tokens=True,
                                                max_length=2 * self.max_len,
                                                return_tensors="pt",
                                                padding="max_length",
                                                truncation=True,
                                                return_token_type_ids=True).input_ids

        inputs_ids = inputs_ids.flatten()

        return {"inputs_ids": inputs_ids, "targets": torch.tensor(target)}


class MultiSarcasmDataset(Dataset):
    def __init__(self, data: pd.DataFrame, label_columns, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_columns = label_columns

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item_index):
        data_row = self.data.iloc[item_index]

        text = data_row.tweets
        target = data_row[self.label_columns]

        inputs_ids = self.tokenizer.encode_plus(text=text,
                                                add_special_tokens=True,
                                                max_length=self.max_len,
                                                return_tensors="pt",
                                                padding="max_length",
                                                truncation=True,
                                                return_token_type_ids=True).input_ids

        inputs_ids = inputs_ids.flatten()

        return {"inputs_ids": inputs_ids, "label_sarcasm": torch.tensor(target[0]),
                "label_irony": torch.tensor(target[1]),
                "label_satire": torch.tensor(target[2]),
                "label_understatement": torch.tensor(target[3]),
                "label_overstatement": torch.tensor(target[4]),
                "label_rhetorical_question": torch.tensor(target[5])}
