# -*- coding: utf-8 -*-
# ========================================================
"""trainer module is written for train model"""
# ========================================================


# ========================================================
# Imports
# ========================================================

import logging
import os

import numpy as np
import pytorch_lightning as pl
import sklearn.utils.class_weight as class_weight
import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
import transformers

from configuration import BaseConfig
from data_loader import read_csv, write_json
from utils import normalizer
from models import MultiDataModule, build_checkpoint_callback
from models.multilabel_tmp import Classifier

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    CONFIG = CONFIG_CLASS.get_config()
    TOKENIZER = transformers.T5Tokenizer.from_pretrained(
        CONFIG.lm_tokenizer_path)
    LOGGER = CSVLogger(CONFIG.csv_logger_path, name=CONFIG.model_name)

    # load raw data
    RAW_TRAIN_DATA = read_csv(path=os.path.join(CONFIG.processed_data_dir, CONFIG.train_file),
                              columns=CONFIG.multi_data_headers,
                              names=CONFIG.multi_customized_headers).dropna()
    RAW_TRAIN_DATA.tweets = RAW_TRAIN_DATA.tweets.apply(
        lambda x: normalizer(x))

    RAW_VAL_DATA = read_csv(path=os.path.join(CONFIG.processed_data_dir, CONFIG.val_file),
                            columns=CONFIG.multi_data_headers,
                            names=CONFIG.multi_customized_headers).dropna()
    RAW_VAL_DATA.tweets = RAW_VAL_DATA.tweets.apply(lambda x: normalizer(x))

    RAW_TEST_DATA = read_csv(path=os.path.join(CONFIG.processed_data_dir, CONFIG.test_file),
                             columns=CONFIG.multi_data_headers,
                             names=CONFIG.multi_customized_headers).dropna()
    RAW_TEST_DATA.tweets = RAW_TEST_DATA.tweets.apply(lambda x: normalizer(x))

    logging.debug(RAW_TRAIN_DATA.head(), RAW_VAL_DATA.head(), RAW_TEST_DATA.head())
    logging.debug("length of Train data is: {}".format(len(RAW_TRAIN_DATA)))
    logging.debug("length of Val data is: {}".format(len(RAW_VAL_DATA)))
    logging.debug("length of Test data is: {}".format(len(RAW_TEST_DATA)))

    logging.debug("Maximum length is: {}".format(CONFIG.max_length))

    LABEL_COLUMNS = RAW_TRAIN_DATA.columns.tolist()[1:]
    logging.debug("Label columns: {}".format(LABEL_COLUMNS))

    # Calculate class_weights
    class2weights = {}
    allclass2weights = {}
    for cls in LABEL_COLUMNS:
        num_pos = 0
        class_weights = class_weight.compute_class_weight(
            "balanced",
            classes=np.unique(RAW_TRAIN_DATA[cls]),
            y=np.array(RAW_TRAIN_DATA[cls]))
        class2weights[cls] = torch.Tensor(class_weights)
        for lbl in RAW_TRAIN_DATA[cls]:
            if lbl == 1:
                num_pos+=1
        allclass2weights[cls] = num_pos/len(RAW_TRAIN_DATA)
    logging.debug("class_weights is: {}".format(class2weights))

    DATA_MODULE = MultiDataModule(data={"train_data": RAW_TRAIN_DATA,
                                        "val_data": RAW_VAL_DATA,
                                        "test_data": RAW_TEST_DATA
                                        },
                                  tokenizer=TOKENIZER, batch_size=CONFIG.batch_size,
                                  max_len=CONFIG.max_length, num_workers=CONFIG.num_workers,
                                  label_columns=LABEL_COLUMNS)

    DATA_MODULE.setup()

    CHECKPOINT_CALLBACK = build_checkpoint_callback(CONFIG.save_top_k)
    # CHECKPOINT_CALLBACK_F1 = build_checkpoint_callback(CONFIG.save_top_k, monitor="val_f1_second_class",
    #                                                    mode="max")
    EARLY_STOPPING_CALLBACK = EarlyStopping(monitor="val_loss", patience=50, mode="min")

    STEPS_PER_EPOCH = len(RAW_TRAIN_DATA) // CONFIG.batch_size
    TOTAL_TRAINING_STEPS = STEPS_PER_EPOCH * 30
    WARMUP_STEPS = TOTAL_TRAINING_STEPS // 5

    # Instantiate the Model Trainer
    TRAINER = pl.Trainer(max_epochs=CONFIG.n_epochs, gpus=[0],  # CONFIG.num_of_gpu,
                         callbacks=[CHECKPOINT_CALLBACK,
                                    EARLY_STOPPING_CALLBACK],
                         progress_bar_refresh_rate=60, logger=LOGGER)
    N_CLASSES = len(LABEL_COLUMNS)

    # Train the Classifier Model
    MODEL = Classifier(class_weights=class2weights,
                       allclass2weights=allclass2weights,
                       n_warmup_steps=WARMUP_STEPS,
                       n_training_steps=TOTAL_TRAINING_STEPS,
                       arg=CONFIG)
    TRAINER.fit(MODEL, DATA_MODULE)
    TRAINER.test(ckpt_path="best", datamodule=DATA_MODULE)
    #
    # # save best model path
    # write_json(path=os.path.join(CONFIG.saved_model_path, CONFIG.model_name,
    #                              "b_model_path.json"),
    #            data={"best_model_path": CHECKPOINT_CALLBACK.best_model_path})
