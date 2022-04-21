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
from indexer import Indexer
from utils import normalizer
from models import DataModule, build_checkpoint_callback
from models.new_RCNN import Classifier

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    CONFIG = CONFIG_CLASS.get_config()
    TOKENIZER = transformers.T5Tokenizer.from_pretrained(
        CONFIG.lm_tokenizer_path)
    LOGGER = CSVLogger(CONFIG.csv_logger_path, name=CONFIG.model_name)

    # load raw data
    RAW_TRAIN_DATA = read_csv(path=os.path.join(CONFIG.processed_data_dir, CONFIG.train_file),
                              columns=CONFIG.data_headers,
                              names=CONFIG.customized_headers).dropna()
    RAW_TRAIN_DATA.tweets = RAW_TRAIN_DATA.tweets.apply(lambda x: normalizer(x))
    RAW_TRAIN_DATA.pair_tweets = RAW_TRAIN_DATA.pair_tweets.apply(lambda x: normalizer(x))

    RAW_VAL_DATA = read_csv(path=os.path.join(CONFIG.processed_data_dir, CONFIG.val_file),
                            columns=CONFIG.data_headers,
                            names=CONFIG.customized_headers).dropna()
    RAW_VAL_DATA.tweets = RAW_VAL_DATA.tweets.apply(lambda x: normalizer(x))
    RAW_VAL_DATA.pair_tweets = RAW_VAL_DATA.pair_tweets.apply(lambda x: normalizer(x))

    RAW_TEST_DATA = read_csv(path=os.path.join(CONFIG.processed_data_dir, CONFIG.test_file),
                             columns=CONFIG.data_headers,
                             names=CONFIG.customized_headers).dropna()
    RAW_TEST_DATA.tweets = RAW_TEST_DATA.tweets.apply(lambda x: normalizer(x))
    RAW_TEST_DATA.pair_tweets = RAW_TEST_DATA.pair_tweets.apply(lambda x: normalizer(x))


    logging.debug(RAW_TRAIN_DATA.head(), RAW_VAL_DATA.head(), RAW_TEST_DATA.head())
    logging.debug("length of Train data is: {}".format(len(RAW_TRAIN_DATA)))
    logging.debug("length of Val data is: {}".format(len(RAW_VAL_DATA)))
    logging.debug("length of Test data is: {}".format(len(RAW_TEST_DATA)))

    TARGET_INDEXER = Indexer(RAW_TRAIN_DATA[CONFIG.customized_headers[2]])
    TARGET_INDEXER.build_vocab2idx()

    TRAIN_TARGETS = [[target] for target in RAW_TRAIN_DATA[CONFIG.customized_headers[2]]]

    TRAIN_TARGETS = TARGET_INDEXER.convert_samples_to_indexes(TRAIN_TARGETS)

    VAL_TARGETS = [[target] for target in RAW_VAL_DATA[CONFIG.customized_headers[2]]]
    VAL_TARGETS = TARGET_INDEXER.convert_samples_to_indexes(VAL_TARGETS)

    TEST_TARGETS = [[target] for target in RAW_TEST_DATA[CONFIG.customized_headers[2]]]
    TEST_TARGETS = TARGET_INDEXER.convert_samples_to_indexes(TEST_TARGETS)

    logging.debug("Maximum length is: {}".format(CONFIG.max_length))

    # Calculate class_weights
    class_weights = class_weight.compute_class_weight(
        "balanced",
        classes=np.unique(RAW_TRAIN_DATA[CONFIG.customized_headers[2]]),
        y=np.array(RAW_TRAIN_DATA[CONFIG.customized_headers[2]]))

    logging.debug("class_weights is: {}".format(class_weights))

    TRAIN_DATA = [list(RAW_TRAIN_DATA[CONFIG.customized_headers[0]]),
                  list(RAW_TRAIN_DATA[CONFIG.customized_headers[1]]),
                  TRAIN_TARGETS]

    VAL_DATA = [list(RAW_VAL_DATA[CONFIG.customized_headers[0]]),
                list(RAW_VAL_DATA[CONFIG.customized_headers[1]]),
                VAL_TARGETS]

    TEST_DATA = [list(RAW_TEST_DATA[CONFIG.customized_headers[0]]),
                 list(RAW_TEST_DATA[CONFIG.customized_headers[1]]),
                 TEST_TARGETS]

    DATA_MODULE = DataModule(data={"train_data": TRAIN_DATA,
                                   "val_data": VAL_DATA,
                                   "test_data": TEST_DATA
                                   },
                             tokenizer=TOKENIZER, batch_size=CONFIG.batch_size,
                             max_len=CONFIG.max_length, num_workers=CONFIG.num_workers)

    DATA_MODULE.setup()

    CHECKPOINT_CALLBACK = build_checkpoint_callback(CONFIG.save_top_k)
    CHECKPOINT_CALLBACK_F1 = build_checkpoint_callback(CONFIG.save_top_k, monitor="val_f1_second_class",
                                                       mode="max")
    EARLY_STOPPING_CALLBACK = EarlyStopping(monitor="val_loss", patience=30, mode="min")

    STEPS_PER_EPOCH = len(RAW_TRAIN_DATA) // CONFIG.batch_size
    TOTAL_TRAINING_STEPS = STEPS_PER_EPOCH * 40
    WARMUP_STEPS = TOTAL_TRAINING_STEPS // 5

    # Instantiate the Model Trainer
    TRAINER = pl.Trainer(max_epochs=CONFIG.n_epochs, gpus=[0],  # CONFIG.num_of_gpu,
                         callbacks=[CHECKPOINT_CALLBACK, CHECKPOINT_CALLBACK_F1,
                                    EARLY_STOPPING_CALLBACK],
                         progress_bar_refresh_rate=60, logger=LOGGER)
    N_CLASSES = len(TARGET_INDEXER.vocabs)

    # Train the Classifier Model
    MODEL = Classifier(n_classes=N_CLASSES, class_weights=torch.Tensor(class_weights),
                       arg=CONFIG, n_warmup_steps=WARMUP_STEPS,
                       n_training_steps=TOTAL_TRAINING_STEPS)
    TRAINER.fit(MODEL, DATA_MODULE)
    TRAINER.test(ckpt_path="best", datamodule=DATA_MODULE)

    # save best model path
    write_json(path=os.path.join(CONFIG.saved_model_path, CONFIG.model_name,
                                 "b_model_path.json"),
               data={"best_model_path": CHECKPOINT_CALLBACK.best_model_path})
