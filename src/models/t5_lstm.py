# -*- coding: utf-8 -*-
# pylint: disable-msg=import-error
# pylint: disable-msg=too-many-ancestors
# pylint: disable-msg=arguments-differ
# ========================================================
"""This module is written for write t5_cnn classifier."""
# ========================================================


# ========================================================
# Imports
# ========================================================

import torch
import pytorch_lightning as pl
import torchmetrics
import transformers


class Classifier(pl.LightningModule):
    def __init__(self, n_classes: int, class_weights, arg):
        super().__init__()

        self.accuracy = torchmetrics.Accuracy()
        self.F_score = torchmetrics.F1(average="none", num_classes=n_classes)
        self.F_score_total = torchmetrics.F1(average="weighted", num_classes=n_classes)
        self.max_len = arg.max_length
        self.lr = arg.lr

        self.mt5_model = transformers.T5EncoderModel.from_pretrained(
            arg.lm_model_path, return_dict=True)

        self.lstm = torch.nn.LSTM(input_size=self.mt5_model.config.hidden_size,
                                  hidden_size=arg.hidden_size,
                                  num_layers=arg.lstm_layers,
                                  bidirectional=arg.bidirectional,
                                  dropout=arg.dropout,
                                  batch_first=True)

        self.output = torch.nn.Linear(in_features=2*arg.hidden_size,
                                      out_features=n_classes)
        self.dropout = torch.nn.Dropout(arg.dropout)

        self.loss = torch.nn.CrossEntropyLoss(weight=class_weights)
        self.save_hyperparameters()

    def forward(self, batch):
        mt5_output = self.mt5_model(batch["inputs_ids"]).last_hidden_state
        # mt5_output.size() = [batch_size, sequence_length, hidden_size]

        output, (hidden, _) = self.lstm(mt5_output)
        # hidden.size() = [2 * lstm_layers, batch_size, hidden_size]

        hidden_concat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        # hidden_concat.size() = [batch_size, 2*hidden_size]

        return self.output(hidden_concat)

    def training_step(self, batch, batch_idx):
        label = batch["targets"].flatten()
        outputs = self.forward(batch)
        loss = self.loss(outputs, label)

        metric2value = {"train_loss": loss,
                        "train_acc": self.accuracy(torch.softmax(outputs, dim=1), label),
                        "train_f1_first_class": self.F_score(torch.softmax(outputs, dim=1), label)[0],
                        "train_f1_second_class": self.F_score(torch.softmax(outputs, dim=1), label)[1],
                        "train_total_F1": self.F_score_total(torch.softmax(outputs, dim=1), label)}

        self.log_dict(metric2value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": label}

    def validation_step(self, batch, batch_idx):
        label = batch["targets"].flatten()
        outputs = self.forward(batch)
        loss = self.loss(outputs, label)

        metric2value = {"val_loss": loss,
                        "val_acc": self.accuracy(torch.softmax(outputs, dim=1), label),
                        "val_f1_first_class": self.F_score(torch.softmax(outputs, dim=1), label)[0],
                        "val_f1_second_class": self.F_score(torch.softmax(outputs, dim=1), label)[1],
                        "val_total_F1": self.F_score_total(torch.softmax(outputs, dim=1), label)}

        self.log_dict(metric2value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        label = batch["targets"].flatten()
        outputs = self.forward(batch)
        loss = self.loss(outputs, label)

        metric2value = {"test_loss": loss,
                        "test_acc": self.accuracy(torch.softmax(outputs, dim=1), label),
                        "test_f1_first_class": self.F_score(torch.softmax(outputs, dim=1), label)[0],
                        "test_f1_second_class": self.F_score(torch.softmax(outputs, dim=1), label)[1],
                        "test_total_F1": self.F_score_total(torch.softmax(outputs, dim=1), label)}

        self.log_dict(metric2value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return [optimizer]
