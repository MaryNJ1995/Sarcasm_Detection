# -*- coding: utf-8 -*-
# ========================================================
"""config module is written for write parameters."""
# ========================================================


# ============================ Third Party libs ============================
import argparse
from pathlib import Path


class BaseConfig:
    """
    BaseConfig class is written to write configs in it
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--model_name", type=str, default="SarcasmDetection")

        self.parser.add_argument("--processed_data_dir", type=str,
                                 default=Path(__file__).parents[
                                             2].__str__() + "/data/SemEval/Task_B_EN/")
        self.parser.add_argument("--assets_dir", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/assets/")

        self.parser.add_argument("--saved_model_path", type=str,
                                 default=Path(__file__).parents[
                                             2].__str__() + "/assets/SarcasmDetection/")
        self.parser.add_argument("--lm_model_path", type=str,
                                 default=Path(__file__).parents[3].__str__() +
                                         "/pretrained_models/t5_en_large/",
                                 help="Path of the multilingual lm model dir")

        self.parser.add_argument("--csv_logger_path", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/assets")

        self.parser.add_argument("--train_file", type=str, default="train_data.csv")
        self.parser.add_argument("--test_file", type=str, default="test_data.csv")
        self.parser.add_argument("--val_file", type=str, default="test_data.csv")

        self.parser.add_argument("--data_headers", type=list, default=["tweet", "sarcastic"])
        self.parser.add_argument("--customized_headers", type=list, default=["texts", "labels"])
        self.parser.add_argument("--multi_data_headers", type=list,
                                 default=["tweets", "sarcasm", "irony", "satire", "understatement",
                                          "overstatement", "rhetorical_question"])
        self.parser.add_argument("--multi_customized_headers", type=list,
                                 default=["tweets", "sarcasm", "irony", "satire", "understatement",
                                          "overstatement", "rhetorical_question"])
        self.parser.add_argument("--save_top_k", type=int, default=1, help="...")

        self.parser.add_argument("--num_workers", type=int,
                                 default=10,
                                 help="...")

        self.parser.add_argument("--max_length", type=int,
                                 default=100,
                                 help="...")

        self.parser.add_argument("--n_epochs", type=int,
                                 default=100,
                                 help="...")

        self.parser.add_argument("--batch_size", type=int,
                                 default=2,
                                 help="...")

        self.parser.add_argument("--lr", default=2e-5,
                                 help="...")

        self.parser.add_argument("--hidden_size", type=int,
                                 default=128,
                                 help="...")
        self.parser.add_argument("--filter_sizes", type=int,
                                 default=[2, 3, 4],
                                 help="...")
        self.parser.add_argument("--n_filters", type=int,
                                 default=100,
                                 help="...")
        self.parser.add_argument("--lstm_layers", type=int,
                                 default=2,
                                 help="...")
        self.parser.add_argument("--bidirectional", type=bool,
                                 default=True,
                                 help="...")
        self.parser.add_argument("--dropout", type=float,
                                 default=0.15,
                                 help="...")
        self.parser.add_argument("--embedding_dim", type=int,
                                 default=256,
                                 help="...")

        self.parser.add_argument("--num_layers", type=float,
                                 default=1,
                                 help="...")
        self.parser.add_argument("--alpha", type=float,
                                 default=50.0,
                                 help="...")
        self.parser.add_argument("--alpha_warmup_ratio", type=float,
                                 default=0.1,
                                 help="...")

    def get_config(self):
        """

        :return:
        """
        return self.parser.parse_args()
