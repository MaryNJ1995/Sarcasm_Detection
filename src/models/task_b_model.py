# -*- coding: utf-8 -*-
"""
This module is written to write task B classifier.
"""

# ============================ Third Party libs ============================
import torch
import pytorch_lightning as pl
import torchmetrics
import transformers

# ============================ My packages ============================
from models.attention import ScaledDotProductAttention
from models.manual_transformers import EncoderLayer


class Classifier(pl.LightningModule):
    def __init__(self, class_weights: dict, all_class2weights: dict, arg):
        super().__init__()

        self.accuracy = torchmetrics.Accuracy()
        self.all_class2weights = all_class2weights
        self.F_score = torchmetrics.F1(average="none", num_classes=2)
        self.F_score_total = torchmetrics.F1(average="weighted", num_classes=2)
        self.max_len = arg.max_length
        self.lr = arg.lr
        self.out_features = 512
        self.tanh = torch.nn.Tanh()

        self.t5_model = transformers.T5EncoderModel.from_pretrained(
            arg.lm_model_path, return_dict=True)
        self.lstm = torch.nn.LSTM(input_size=self.t5_model.config.hidden_size,
                                  hidden_size=arg.hidden_size,
                                  num_layers=arg.lstm_layers,
                                  bidirectional=True,
                                  batch_first=True)
        self.attention = ScaledDotProductAttention(
            dim=self.t5_model.config.hidden_size + (2 * arg.hidden_size))
        self.enc_layer = EncoderLayer(hid_dim=self.t5_model.config.hidden_size,
                                      n_heads=8, pf_dim=self.t5_model.config.hidden_size * 2,
                                      dropout=arg.dropout)
        self.fully_connected_layers_cat = \
            torch.nn.Linear(in_features=self.t5_model.config.hidden_size + (2 * arg.hidden_size),
                            out_features=self.out_features)

        self.pooler = torch.nn.MaxPool1d(self.max_len)

        self.output_sarcasm = torch.nn.Linear(in_features=self.out_features,
                                              out_features=2)
        self.output_irony = torch.nn.Linear(in_features=self.out_features,
                                            out_features=2)
        self.output_satire = torch.nn.Linear(in_features=self.out_features,
                                             out_features=2)
        self.output_understatement = torch.nn.Linear(in_features=self.out_features,
                                                     out_features=2)
        self.output_overstatement = torch.nn.Linear(in_features=self.out_features,
                                                    out_features=2)
        self.output_rhetorical_question = torch.nn.Linear(in_features=self.out_features,
                                                          out_features=2)

        self.loss_sarcasm = torch.nn.CrossEntropyLoss(weight=class_weights["sarcasm"])
        self.loss_irony = torch.nn.CrossEntropyLoss(weight=class_weights["irony"])
        self.loss_satire = torch.nn.CrossEntropyLoss(weight=class_weights["satire"])
        self.loss_understatement = torch.nn.CrossEntropyLoss(weight=class_weights["understatement"])
        self.loss_overstatement = torch.nn.CrossEntropyLoss(weight=class_weights["overstatement"])
        self.loss_rhetorical_question = torch.nn.CrossEntropyLoss(
            weight=class_weights["rhetorical_question"])
        self.save_hyperparameters()

    def forward(self, batch):
        t5_output = self.t5_model(batch["inputs_ids"]).last_hidden_state
        # mt5_output.size() = [batch_size, sequence_length, hidden_size]

        enc_out = self.enc_layer(t5_output)
        # enc_out.size() = [batch_size, sequence_length, hidden_size]

        output, (_, _) = self.lstm(enc_out)

        # output.size() = [sent_len, batch_size, hid_dim * num_directions]
        # hidden.size() = [num_layers * num_directions, batch_size, hid_dim]
        # cell.size() = [num_layers * num_directions, batch_size, hid_dim]

        output = torch.cat([output, enc_out], dim=2)
        output, attn = self.attention(output, output, output)

        # output.size() = [batch_size, sent_len, em

        output = self.tanh(self.fully_connected_layers_cat(output))
        output = output.permute(0, 2, 1)

        pooler = self.pooler(output).squeeze(2)

        # pooler.size() = [batch_size, hidden_size]

        return self.output_sarcasm(pooler), self.output_irony(pooler), \
               self.output_satire(pooler), self.output_understatement(pooler), \
               self.output_overstatement(pooler), self.output_rhetorical_question(pooler)

    def training_step(self, batch, batch_idx):
        label_sarcasm = batch["label_sarcasm"].flatten()
        label_irony = batch["label_irony"].flatten()
        label_satire = batch["label_satire"].flatten()
        label_understatement = batch["label_understatement"].flatten()
        label_overstatement = batch["label_overstatement"].flatten()
        label_rhetorical_question = batch["label_rhetorical_question"].flatten()

        output_sarcasm, output_irony, output_satire, output_understatement, \
        output_overstatement, output_rhetorical_question = self.forward(batch)

        loss_sarcasm = self.loss_sarcasm(output_sarcasm, label_sarcasm)
        loss_irony = self.loss_irony(output_irony, label_irony)
        loss_satire = self.loss_satire(output_satire, label_satire)
        loss_understatement = self.loss_understatement(output_understatement, label_understatement)
        loss_overstatement = self.loss_overstatement(output_overstatement, label_overstatement)
        loss_rhetorical_question = self.loss_rhetorical_question(output_rhetorical_question,
                                                                 label_rhetorical_question)

        loss = (((1 - self.all_class2weights["sarcasm"]) * loss_sarcasm) +
                ((1 - self.all_class2weights["irony"]) * loss_irony) +
                ((1 - self.all_class2weights["satire"]) * loss_satire) +
                ((1 - self.all_class2weights["understatement"]) * loss_understatement) +
                ((1 - self.all_class2weights["overstatement"]) * loss_overstatement) +
                ((1 - self.all_class2weights[
                    "rhetorical_question"]) * loss_rhetorical_question)) / 6

        metric2value = {
            "train_loss": loss,
            "train_f1_sarcasm_class": self.F_score_total(torch.softmax(output_sarcasm, dim=1),
                                                         label_sarcasm),
            "train_f1_sarcasm_class_not":
                self.F_score(torch.softmax(output_sarcasm, dim=1), label_sarcasm)[0],
            "train_f1_sarcasm_class_is":
                self.F_score(torch.softmax(output_sarcasm, dim=1), label_sarcasm)[1],
            "train_f1_irony_class": self.F_score_total(torch.softmax(output_irony, dim=1),
                                                       label_irony),
            "train_f1_irony_class_not":
                self.F_score(torch.softmax(output_irony, dim=1), label_irony)[0],
            "train_f1_irony_class_is":
                self.F_score(torch.softmax(output_irony, dim=1), label_irony)[1],
            "train_f1_satire_class": self.F_score_total(torch.softmax(output_satire, dim=1),
                                                        label_satire),
            "train_f1_satire_class_not":
                self.F_score(torch.softmax(output_satire, dim=1), label_satire)[0],
            "train_f1_satire_class_is":
                self.F_score(torch.softmax(output_satire, dim=1), label_satire)[1],
            "train_f1_understatement_class": self.F_score_total(
                torch.softmax(output_understatement, dim=1), label_understatement),
            "train_f1_understatement_class_not":
                self.F_score(torch.softmax(output_understatement, dim=1), label_understatement)[0],
            "train_f1_understatement_class_is":
                self.F_score(torch.softmax(output_understatement, dim=1), label_understatement)[1],
            "train_f1_overstatement_class": self.F_score_total(
                torch.softmax(output_overstatement, dim=1), label_overstatement),
            "train_f1_overstatement_class_not":
                self.F_score(torch.softmax(output_overstatement, dim=1), label_overstatement)[0],
            "train_f1_overstatement_class_is":
                self.F_score(torch.softmax(output_overstatement, dim=1), label_overstatement)[1],
            "train_f1_rhetorical_question_class": self.F_score_total(
                torch.softmax(output_rhetorical_question, dim=1), label_rhetorical_question),
            "train_f1_rhetorical_question_class_not":
                self.F_score(torch.softmax(output_rhetorical_question, dim=1),
                             label_rhetorical_question)[0],
            "train_f1_rhetorical_question_class_is":
                self.F_score(torch.softmax(output_rhetorical_question, dim=1),
                             label_rhetorical_question)[1],
        }

        self.log_dict(metric2value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        label_sarcasm = batch["label_sarcasm"].flatten()
        label_irony = batch["label_irony"].flatten()
        label_satire = batch["label_satire"].flatten()
        label_understatement = batch["label_understatement"].flatten()
        label_overstatement = batch["label_overstatement"].flatten()
        label_rhetorical_question = batch["label_rhetorical_question"].flatten()

        output_sarcasm, output_irony, output_satire, output_understatement, \
        output_overstatement, output_rhetorical_question = self.forward(batch)

        loss_sarcasm = self.loss_sarcasm(output_sarcasm, label_sarcasm)
        loss_irony = self.loss_irony(output_irony, label_irony)
        loss_satire = self.loss_satire(output_satire, label_satire)
        loss_understatement = self.loss_understatement(output_understatement, label_understatement)
        loss_overstatement = self.loss_overstatement(output_overstatement, label_overstatement)
        loss_rhetorical_question = self.loss_rhetorical_question(output_rhetorical_question,
                                                                 label_rhetorical_question)

        loss = (((1 - self.all_class2weights["sarcasm"]) * loss_sarcasm) +
                ((1 - self.all_class2weights["irony"]) * loss_irony) +
                ((1 - self.all_class2weights["satire"]) * loss_satire) +
                ((1 - self.all_class2weights["understatement"]) * loss_understatement) +
                ((1 - self.all_class2weights["overstatement"]) * loss_overstatement) +
                ((1 - self.all_class2weights[
                    "rhetorical_question"]) * loss_rhetorical_question)) / 6

        metric2value = {
            "val_loss": loss,
            "val_f1_sarcasm_class": self.F_score_total(torch.softmax(output_sarcasm, dim=1),
                                                       label_sarcasm),
            "val_f1_sarcasm_class_not":
                self.F_score(torch.softmax(output_sarcasm, dim=1), label_sarcasm)[0],
            "val_f1_sarcasm_class_is":
                self.F_score(torch.softmax(output_sarcasm, dim=1), label_sarcasm)[1],
            "val_f1_irony_class": self.F_score_total(torch.softmax(output_irony, dim=1),
                                                     label_irony),
            "val_f1_irony_class_not": self.F_score(torch.softmax(output_irony, dim=1), label_irony)[
                0],
            "val_f1_irony_class_is": self.F_score(torch.softmax(output_irony, dim=1), label_irony)[
                1],
            "val_f1_satire_class": self.F_score_total(torch.softmax(output_satire, dim=1),
                                                      label_satire),
            "val_f1_satire_class_not":
                self.F_score(torch.softmax(output_satire, dim=1), label_satire)[0],
            "val_f1_satire_class_is":
                self.F_score(torch.softmax(output_satire, dim=1), label_satire)[1],
            "val_f1_understatement_class": self.F_score_total(
                torch.softmax(output_understatement, dim=1), label_understatement),
            "val_f1_understatement_class_not":
                self.F_score(torch.softmax(output_understatement, dim=1), label_understatement)[0],
            "val_f1_understatement_class_is":
                self.F_score(torch.softmax(output_understatement, dim=1), label_understatement)[1],
            "val_f1_overstatement_class": self.F_score_total(
                torch.softmax(output_overstatement, dim=1), label_overstatement),
            "val_f1_overstatement_class_not":
                self.F_score(torch.softmax(output_overstatement, dim=1), label_overstatement)[0],
            "val_f1_overstatement_class_is":
                self.F_score(torch.softmax(output_overstatement, dim=1), label_overstatement)[1],
            "val_f1_rhetorical_question_class": self.F_score_total(
                torch.softmax(output_rhetorical_question, dim=1), label_rhetorical_question),
            "val_f1_rhetorical_question_class_not":
                self.F_score(torch.softmax(output_rhetorical_question, dim=1),
                             label_rhetorical_question)[0],
            "val_f1_rhetorical_question_class_is":
                self.F_score(torch.softmax(output_rhetorical_question, dim=1),
                             label_rhetorical_question)[1],
        }
        self.log_dict(metric2value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        label_sarcasm = batch["label_sarcasm"].flatten()
        label_irony = batch["label_irony"].flatten()
        label_satire = batch["label_satire"].flatten()
        label_understatement = batch["label_understatement"].flatten()
        label_overstatement = batch["label_overstatement"].flatten()
        label_rhetorical_question = batch["label_rhetorical_question"].flatten()

        output_sarcasm, output_irony, output_satire, output_understatement, \
        output_overstatement, output_rhetorical_question = self.forward(batch)

        loss_sarcasm = self.loss_sarcasm(output_sarcasm, label_sarcasm)
        loss_irony = self.loss_irony(output_irony, label_irony)
        loss_satire = self.loss_satire(output_satire, label_satire)
        loss_understatement = self.loss_understatement(output_understatement, label_understatement)
        loss_overstatement = self.loss_overstatement(output_overstatement, label_overstatement)
        loss_rhetorical_question = self.loss_rhetorical_question(output_rhetorical_question,
                                                                 label_rhetorical_question)

        loss = (((1 - self.all_class2weights["sarcasm"]) * loss_sarcasm) +
                ((1 - self.all_class2weights["irony"]) * loss_irony) +
                ((1 - self.all_class2weights["satire"]) * loss_satire) +
                ((1 - self.all_class2weights["understatement"]) * loss_understatement) +
                ((1 - self.all_class2weights["overstatement"]) * loss_overstatement) +
                ((1 - self.all_class2weights[
                    "rhetorical_question"]) * loss_rhetorical_question)) / 6

        metric2value = {
            "test_loss": loss,
            "test_f1_sarcasm_class": self.F_score_total(torch.softmax(output_sarcasm, dim=1),
                                                        label_sarcasm),
            "test_f1_sarcasm_class_not":
                self.F_score(torch.softmax(output_sarcasm, dim=1), label_sarcasm)[0],
            "test_f1_sarcasm_class_is":
                self.F_score(torch.softmax(output_sarcasm, dim=1), label_sarcasm)[1],
            "test_f1_irony_class": self.F_score_total(torch.softmax(output_irony, dim=1),
                                                      label_irony),
            "test_f1_irony_class_not":
                self.F_score(torch.softmax(output_irony, dim=1), label_irony)[0],
            "test_f1_irony_class_is": self.F_score(torch.softmax(output_irony, dim=1), label_irony)[
                1],
            "test_f1_satire_class": self.F_score_total(torch.softmax(output_satire, dim=1),
                                                       label_satire),
            "test_f1_satire_class_not":
                self.F_score(torch.softmax(output_satire, dim=1), label_satire)[0],
            "test_f1_satire_class_is":
                self.F_score(torch.softmax(output_satire, dim=1), label_satire)[1],
            "test_f1_understatement_class": self.F_score_total(
                torch.softmax(output_understatement, dim=1), label_understatement),
            "test_f1_understatement_class_not":
                self.F_score(torch.softmax(output_understatement, dim=1), label_understatement)[0],
            "test_f1_understatement_class_is":
                self.F_score(torch.softmax(output_understatement, dim=1), label_understatement)[1],
            "test_f1_overstatement_class": self.F_score_total(
                torch.softmax(output_overstatement, dim=1), label_overstatement),
            "test_f1_overstatement_class_not":
                self.F_score(torch.softmax(output_overstatement, dim=1), label_overstatement)[0],
            "test_f1_overstatement_class_is":
                self.F_score(torch.softmax(output_overstatement, dim=1), label_overstatement)[1],
            "test_f1_rhetorical_question_class": self.F_score_total(
                torch.softmax(output_rhetorical_question, dim=1), label_rhetorical_question),
            "test_f1_rhetorical_question_class_not":
                self.F_score(torch.softmax(output_rhetorical_question, dim=1),
                             label_rhetorical_question)[0],
            "test_f1_rhetorical_question_class_is":
                self.F_score(torch.softmax(output_rhetorical_question, dim=1),
                             label_rhetorical_question)[1],
        }
        self.log_dict(metric2value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"test_loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return [optimizer]
