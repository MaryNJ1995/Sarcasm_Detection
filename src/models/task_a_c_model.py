# -*- coding: utf-8 -*-
"""
This module is written to write Classifier for task A
"""

# ============================ Third Party libs ============================
import pytorch_lightning as pl
import torch
import torch.nn.functional as function
import torchmetrics
from transformers import T5EncoderModel

# ============================ My packages ============================
from .attention import ScaledDotProductAttention
from .manual_transformers import EncoderLayer


class Classifier(pl.LightningModule):
    """
    The classifier for Task A
    """

    def __init__(self, n_classes: int, class_weights, arg):
        super().__init__()
        self.accuracy = torchmetrics.Accuracy()
        self.F_score = torchmetrics.F1(average="none", num_classes=n_classes)
        self.F_score_total = torchmetrics.F1(average="weighted", num_classes=n_classes)
        self.lr = arg.lr
        self.num_layers = arg.num_layers
        self.hidden_size = arg.hidden_size
        self.n_epochs = arg.n_epochs
        self.out_features = 512
        self.criterion = torch.nn.CrossEntropyLoss()  # CrossEntropyLoss()
        self.model = T5EncoderModel.from_pretrained(arg.lm_model_path)
        self.lstm = torch.nn.LSTM(input_size=self.model.config.hidden_size,
                                  hidden_size=self.hidden_size,
                                  num_layers=self.num_layers,
                                  bidirectional=True,
                                  batch_first=True)
        self.attention = ScaledDotProductAttention(
            dim=self.model.config.hidden_size + (2 * self.hidden_size))
        self.enc_layer = EncoderLayer(hid_dim=self.model.config.hidden_size,
                                      n_heads=16, pf_dim=self.model.config.hidden_size * 2,
                                      dropout=arg.dropout)
        self.fully_connected_layers_cat = \
            torch.nn.Linear(in_features=self.model.config.hidden_size + (2 * self.hidden_size),
                            out_features=self.out_features)
        self.fully_connected_layers_last = \
            torch.nn.Linear(in_features=self.out_features,
                            out_features=n_classes)
        self.loss = torch.nn.CrossEntropyLoss(weight=class_weights)
        self.tanh = torch.nn.Tanh()
        self.dropout = torch.nn.Dropout(arg.dropout)
        self.save_hyperparameters()

    def forward(self, batch):
        mt5_output = self.model(batch["inputs_ids"]).last_hidden_state
        # mt5_output.last_hidden_state.size() = [batch_size, sen_len, 768]

        # embedded.size() =  [sen_len, batch_size, 768]
        enc_out = self.enc_layer(mt5_output)
        # enc_out.shape:[batch_size, seq_len, 2*emb_dim] = ([32, 150, 1024])

        output, (_, _) = self.lstm(enc_out)
        # output.size() = [sent_len, batch_size, hid_dim * num_directions]
        # hidden.size() = [num_layers * num_directions, batch_size, hid_dim]
        # cell.size() = [num_layers * num_directions, batch_size, hid_dim]

        output = torch.cat([output, enc_out], dim=2)
        # output.size() = [batch_size, sent_len, embedding_dim+2*hid_dim]===>(64,150,1024)

        context, attn = self.attention(output, output, output)

        output = self.tanh(self.fully_connected_layers_cat(context))
        output = output.permute(0, 2, 1)
        output = function.max_pool1d(output, output.shape[2]).squeeze(2)
        # output.size() = [batch_size, out_units]===>(64,256)

        output = self.fully_connected_layers_last(output)
        # output.size() = [batch_size,output_size]===>(64,2)
        # print(output.size())

        return output

    def training_step(self, batch, batch_idx):
        label = batch["targets"].flatten()
        outputs = self.forward(batch)
        loss = self.loss(outputs, label)

        metric2value = {"train_loss": loss,
                        "train_acc": self.accuracy(torch.softmax(outputs, dim=1), label),
                        "train_f1_first_class": self.F_score(torch.softmax(outputs, dim=1), label)[
                            0],
                        "train_f1_second_class": self.F_score(torch.softmax(outputs, dim=1), label)[
                            1],
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
                        "val_f1_second_class": self.F_score(torch.softmax(outputs, dim=1), label)[
                            1],
                        "val_total_F1": self.F_score_total(torch.softmax(outputs, dim=1), label)}

        self.log_dict(metric2value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        label = batch["targets"].flatten()
        outputs = self.forward(batch)
        loss = self.loss(outputs, label)

        metric2value = {"test_loss": loss,
                        "test_acc": self.accuracy(torch.softmax(outputs, dim=1), label),
                        "test_f1_first_class": self.F_score(torch.softmax(outputs, dim=1), label)[
                            0],
                        "test_f1_second_class": self.F_score(torch.softmax(outputs, dim=1), label)[
                            1],
                        "test_total_F1": self.F_score_total(torch.softmax(outputs, dim=1), label)}

        self.log_dict(metric2value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return [optimizer]
