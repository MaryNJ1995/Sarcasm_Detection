from abc import ABC, abstractmethod
import torch
import pytorch_lightning as pl


class CustomDataset(ABC, torch.utils.data.Dataset):
    def __init__(self, texts: list, targets: list, tokenizer, max_len):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.targets)

    @abstractmethod
    def __getitem__(self, item_index):
        """

        :param item_index:
        :return:
        """


class SarcasmDataset(CustomDataset):
    def __init__(self, texts: list, targets: list, tokenizer, max_len):
        super().__init__(texts, targets, tokenizer, max_len)

    def __getitem__(self, item_index):
        text = self.texts[item_index]

        target = self.targets[item_index]

        inputs_ids = self.tokenizer.encode_plus(text=text,
                                                add_special_tokens=True,
                                                max_length=self.max_len,
                                                return_tensors="pt",
                                                padding="max_length",
                                                truncation=True,
                                                return_token_type_ids=True).input_ids

        inputs_ids = inputs_ids.flatten()

        return {"inputs_ids": inputs_ids, "targets": torch.tensor(target)}


class DataModule(pl.LightningDataModule):

    def __init__(self, data: dict, batch_size, num_workers, tokenizer,
                 max_len):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None

    def setup(self):
        self.train_dataset = SarcasmDataset(texts=self.data["train_data"][0],
                                            targets=self.data["train_data"][1],
                                            tokenizer=self.tokenizer,
                                            max_len=self.max_len)
        self.val_dataset = SarcasmDataset(texts=self.data["val_data"][0],
                                          targets=self.data["val_data"][1],
                                          tokenizer=self.tokenizer,
                                          max_len=self.max_len)
        self.test_dataset = SarcasmDataset(texts=self.data["test_data"][0],
                                           targets=self.data["test_data"][1],
                                           tokenizer=self.tokenizer,
                                           max_len=self.max_len)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                           num_workers=self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size,
                                           num_workers=self.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size,
                                           num_workers=self.num_workers)
