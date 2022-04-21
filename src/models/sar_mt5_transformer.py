import pytorch_lightning as pl
import torch
import torchmetrics
from torch import nn
from transformers import T5EncoderModel

from configuration.config import BaseConfig
from models.attention import ScaledDotProductAttention
from models.manual_transformers import EncoderLayer


class Classifier(pl.LightningModule):
    def __init__(self, n_classes: int, class_weights, arg):
        super().__init__()
        self.accuracy = torchmetrics.Accuracy()
        self.F_score = torchmetrics.F1(average="none", num_classes=n_classes)
        self.F_score_total = torchmetrics.F1(average="weighted", num_classes=n_classes)
        self.lr = arg.lr
        self.hidden_size = arg.hidden_size
        self.n_epochs = arg.n_epochs
        self.num_layers = arg.num_layers
        self.criterion = torch.nn.CrossEntropyLoss()  # CrossEntropyLoss()
        self.model = T5EncoderModel.from_pretrained(arg.lm_model_path)
        self.enc_layer = EncoderLayer(hid_dim=self.model.config.hidden_size,
                                      n_heads=16, pf_dim=self.model.config.hidden_size * 2,
                                      dropout=arg.dropout)
        self.output_layer = torch.nn.Linear(in_features=4 * self.model.config.hidden_size,
                                            out_features=n_classes)
        self.max_pool1d = nn.MaxPool1d(arg.max_length)
        self.attention = ScaledDotProductAttention(dim=4 * self.model.config.hidden_size)

        self.dropout = torch.nn.Dropout(arg.dropout)
        self.save_hyperparameters()
        self.loss = torch.nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, batch):
        token_features = self.model(batch["inputs_ids"])
        # mt5_output.last_hidden_state.size() = [batch_size, sen_len, emb_dim]

        token_features = token_features.last_hidden_state

        token_features = self.dropout(token_features)
        # embedded.size() =[batch_size, sen_len, emb_dim]

        enc_out = self.enc_layer(token_features)
        # enc_out.shape:[batch_size, seq_len, 2*emb_dim] = ([32, 150, 1024])

        out_features = torch.cat((enc_out, enc_out, enc_out, token_features), dim=2)
        # out_features.shape: [batch_size, seq_len, 4*emb_dim] = ([32, 150, 4 * 1024])
        context, attn = self.attention(out_features, out_features, out_features)
        context = context.permute(0, 2, 1)
        # context.size= [batch_size, feature_dim, sent_len] ===>torch.Size([32,  4 * 1024], 150])

        maxed_pool = self.max_pool1d(context).squeeze(2)
        # maxed_pool.size= [batch_size, feature_dim] ===>torch.Size([32,  8 * 1024])

        output = self.output_layer(maxed_pool)
        # output.size= [batch_size, n_classes] ===>torch.Size([32, 2])

        return output

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


if __name__ == '__main__':
    CONFIG_CLASS = BaseConfig()
    CONFIG = CONFIG_CLASS.get_config()

    MODEL = Classifier(class_weights=0.2, n_classes=2, arg=CONFIG)
    x = dict()
    x["inputs_ids"] = torch.rand((64, 150))

    MODEL.forward(x)
