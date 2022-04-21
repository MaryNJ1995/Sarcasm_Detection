import pytorch_lightning as pl
import torch
import torch.nn.functional as function
import torchmetrics
from transformers import MT5EncoderModel

from configuration.config import BaseConfig
from .attention_model import Attention


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
        self.model = MT5EncoderModel.from_pretrained(arg.lm_model_path)
        self.lstm_1 = torch.nn.LSTM(input_size=self.model.config.hidden_size,
                                    hidden_size=self.hidden_size,
                                    num_layers=self.num_layers,
                                    bidirectional=True)
        self.lstm_2 = torch.nn.LSTM(input_size=(2 * self.hidden_size) + self.model.config.hidden_size,
                                    hidden_size=self.hidden_size,
                                    num_layers=self.num_layers,
                                    bidirectional=True)
        self.lstm_3 = torch.nn.LSTM(input_size=(4 * self.hidden_size) + self.model.config.hidden_size,
                                    hidden_size=self.hidden_size,
                                    num_layers=self.num_layers,
                                    bidirectional=True)

        self.attention = Attention(rnn_size=3 * (2 * self.hidden_size) + self.model.config.hidden_size)
        self.fully_connected_layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=(2 * ((3 * (2 * self.hidden_size)) + self.model.config.hidden_size)),
                            out_features=750),
            torch.nn.Linear(in_features=750, out_features=n_classes)
        )
        self.batchnorm = torch.nn.BatchNorm1d(
            num_features=2 * ((3 * (2 * self.hidden_size)) + self.model.config.hidden_size))
        self.loss = torch.nn.CrossEntropyLoss(weight=class_weights)

        self.dropout = torch.nn.Dropout(0.2)
        self.save_hyperparameters()

    def forward(self, batch):
        mt5_output = self.model(batch["inputs_ids"])
        # mt5_output.last_hidden_state.size() = [batch_size, sen_len, 768]

        embedded = mt5_output.last_hidden_state.permute(1, 0, 2)
        # embedded.size() =  [sen_len, batch_size, 768]

        embedded = self.dropout(embedded)
        # embedded.size() = [sent_len, batch_size, embedding_dim]===>torch.Size([150, 64, 768])

        output_1, (hidden, cell) = self.lstm_1(embedded)
        # output_1.size() = [sent_len, batch_size, hid_dim*num_directions]===>torch.Size([150, 64, 2*256])
        # hidden.size() = [num_layers * num_directions, batch_size, hid_dim]
        # cell.size() = [num_layers * num_directions, batch_size, hid_dim]

        lstm_2_input = torch.cat((output_1, embedded), dim=2)
        # lstm_2_input.size() = [sent_len, batch_size, hid_dim*num_directions+embedding_dim]
        # torch.Size([150, 64, (2*256+300)])

        output_2, (hidden, cell) = self.lstm_2(lstm_2_input, (hidden, cell))
        # output_2.size() = [sent_len, batch_size, hid_dim*num_directions]===>torch.Size([150, 64, 2*256])
        # hidden.size() = [num_layers * num_directions, batch_size, hid_dim]
        # cell.size() = [num_layers * num_directions, batch_size, hid_dim]

        lstm_3_input = torch.cat((output_1, output_2, embedded), dim=2)
        # lstm_3_input.size() = [sent_len, batch_size, 2*hid_dim*num_directions+embedding_dim]
        # torch.Size([150, 64, (2*2*256+300)])

        output_3, (_, _) = self.lstm_3(lstm_3_input, (hidden, cell))
        # output_3.size() = [sent_len, batch_size, hid_dim * num_directions]
        # _.size() = [num_layers * num_directions, batch_size, hid_dim]
        # _.size() = [num_layers * num_directions, batch_size, hid_dim]

        all_features = torch.cat((output_3, output_2, output_1, embedded), dim=2)
        # all_features.size= [sent_len, batch_size, (3*hid_dim * num_directions)+768]==>torch.Size([150, 64, 1536])

        attention_score = self.attention(all_features)
        # this score is the importance of each word in 30 different way
        # attention_score.size() = [batch_size, num_head_attention, sent_len]===>([64, 30, 150])

        all_features = all_features.permute(1, 0, 2)
        # all_features.size() = [batch_size, sent_len, (3*hid_dim * num_directions)+ pos_embedding_dim + embedding_dim]

        hidden_matrix = torch.bmm(attention_score, all_features)
        hidden_matrix = hidden_matrix.permute(0, 2, 1)
        # hidden_matrix is each word's importance * each word's feature
        # hidden_matrix.size=(batch_size,num_head_attention,((3*hid_dim*num_directions)+768)===>(64,30,1536)
        max_pool_output = function.max_pool1d(hidden_matrix, hidden_matrix.shape[2]).squeeze(2)
        avg_pool_output = function.avg_pool1d(hidden_matrix, hidden_matrix.shape[2]).squeeze(2)

        pooled_cat = torch.cat((max_pool_output, avg_pool_output), dim=1)
        # batchnorm = self.batchnorm(pooled_cat)
        final_output = self.fully_connected_layers(pooled_cat)
        return final_output

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
    x = torch.rand((64, 150))
    y = torch.rand((64, 150))
    z = torch.rand((64, 150))

    MODEL.forward(x.long(), y.long(), z.long())
