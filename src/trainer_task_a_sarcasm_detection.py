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
from hazm import Normalizer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
# ========================================================
# Imports
# ========================================================
from tabulate import tabulate
from transformers import MT5Tokenizer

from configuration import BaseConfig
from data_loader import read_csv, write_json
from data_preparation import normalize_text
from indexer import Indexer
from models import DataModule, build_checkpoint_callback
from models.Msar_RCNN import Classifier

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    CONFIG = CONFIG_CLASS.get_config()
    TOKENIZER = MT5Tokenizer.from_pretrained(CONFIG.mlm_tokenizer_path)
    LOGGER = CSVLogger(CONFIG.csv_logger_path, name="2")
    NORMALIZER = Normalizer()
    # load raw data
    RAW_TRAIN_DATA = read_csv(path=os.path.join(CONFIG.processed_data_dir, CONFIG.train_file),
                              columns=CONFIG.data_headers,
                              names=CONFIG.customized_headers).dropna()
    RAW_TRAIN_DATA.texts = RAW_TRAIN_DATA.texts.apply(lambda x: normalize_text(x, NORMALIZER))
    print(tabulate(RAW_TRAIN_DATA[:10], headers='keys', tablefmt='psql'))

    RAW_VAL_DATA = read_csv(path=os.path.join(CONFIG.processed_data_dir, CONFIG.val_file),
                            columns=CONFIG.data_headers,
                            names=CONFIG.customized_headers).dropna()
    RAW_VAL_DATA.texts = RAW_VAL_DATA.texts.apply(lambda x: normalize_text(x, NORMALIZER))
    print(tabulate(RAW_VAL_DATA[:10], headers='keys', tablefmt='psql'))

    RAW_TEST_DATA = read_csv(path=os.path.join(CONFIG.processed_data_dir, CONFIG.test_file),
                             columns=CONFIG.data_headers,
                             names=CONFIG.customized_headers).dropna()
    RAW_TEST_DATA.texts = RAW_TEST_DATA.texts.apply(lambda x: normalize_text(x, NORMALIZER))
    print(tabulate(RAW_TEST_DATA[:10], headers='keys', tablefmt='psql'))

    logging.debug(RAW_TRAIN_DATA.head(), RAW_VAL_DATA.head(), RAW_TEST_DATA.head())
    logging.debug("length of Train data is: {}".format(len(RAW_TRAIN_DATA)))
    logging.debug("length of Val data is: {}".format(len(RAW_VAL_DATA)))
    logging.debug("length of Test data is: {}".format(len(RAW_TEST_DATA)))

    TARGET_INDEXER = Indexer(RAW_TRAIN_DATA[CONFIG.customized_headers[1]])
    TARGET_INDEXER.build_vocab2idx()

    TRAIN_TARGETS = [[target] for target in RAW_TRAIN_DATA[CONFIG.customized_headers[1]]]

    TRAIN_TARGETS = TARGET_INDEXER.convert_samples_to_indexes(TRAIN_TARGETS)

    VAL_TARGETS = [[target] for target in RAW_VAL_DATA[CONFIG.customized_headers[1]]]
    VAL_TARGETS = TARGET_INDEXER.convert_samples_to_indexes(VAL_TARGETS)

    TEST_TARGETS = [[target] for target in RAW_TEST_DATA[CONFIG.customized_headers[1]]]
    TEST_TARGETS = TARGET_INDEXER.convert_samples_to_indexes(TEST_TARGETS)

    logging.debug("Maximum length is: {}".format(CONFIG.max_length))

    # Calculate class_weights
    class_weights = class_weight.compute_class_weight(
        "balanced",
        classes=np.unique(RAW_TRAIN_DATA[CONFIG.customized_headers[1]]),
        y=np.array(RAW_TRAIN_DATA[CONFIG.customized_headers[1]]))

    logging.debug("class_weights is: {}".format(class_weights))

    TRAIN_DATA = [list(RAW_TRAIN_DATA[CONFIG.customized_headers[0]]),
                  TRAIN_TARGETS]

    VAL_DATA = [list(RAW_VAL_DATA[CONFIG.customized_headers[0]]),
                VAL_TARGETS]

    TEST_DATA = [list(RAW_TEST_DATA[CONFIG.customized_headers[0]]),
                 TEST_TARGETS]

    DATA_MODULE = DataModule(data={"train_data": TRAIN_DATA,
                                   "val_data": VAL_DATA,
                                   "test_data": TEST_DATA
                                   },
                             tokenizer=TOKENIZER, batch_size=CONFIG.batch_size,
                             max_len=CONFIG.max_length, num_workers=CONFIG.num_workers)

    DATA_MODULE.setup()

    CHECKPOINT_CALLBACK = build_checkpoint_callback(CONFIG.save_top_k)
    CHECKPOINT_CALLBACK_F1 = build_checkpoint_callback(CONFIG.save_top_k, monitor="val_total_F1",
                                                       mode="max")  # train_f1_second_class
    EARLY_STOPPING_CALLBACK = EarlyStopping(monitor="val_loss", patience=50, mode="min")

    # Instantiate the Model Trainer
    TRAINER = pl.Trainer(max_epochs=CONFIG.n_epochs, gpus=[0],  # CONFIG.num_of_gpu,
                         callbacks=[CHECKPOINT_CALLBACK, CHECKPOINT_CALLBACK_F1,
                                    EARLY_STOPPING_CALLBACK],
                         progress_bar_refresh_rate=60, logger=LOGGER)
    N_CLASSES = len(TARGET_INDEXER.vocabs)
    STEPS_PER_EPOCH = len(RAW_TRAIN_DATA) // CONFIG.batch_size
    TOTAL_TRAINING_STEPS = STEPS_PER_EPOCH * 40
    WARMUP_STEPS = TOTAL_TRAINING_STEPS // 5

    # MODEL_PATH = "../assets/2/version_45_finetunedonkaggle/checkpoints/QTag-epoch=21-val_loss=0.70.ckpt"
    # MODEL = Classifier.load_from_checkpoint(MODEL_PATH, map_location="cuda:0")

    # Train the Classifier Model
    MODEL = Classifier(n_classes=N_CLASSES, class_weights=torch.Tensor(class_weights),
                       arg=CONFIG)  # n_warmup_steps=WARMUP_STEPS, n_training_steps=TOTAL_TRAINING_STEPS)

    TRAINER.fit(MODEL, DATA_MODULE)
    TRAINER.test(ckpt_path="best", datamodule=DATA_MODULE)

    # save best model path
    write_json(path=os.path.join(CONFIG.assets_dir, CONFIG.model_name,
                                 "b_model_path.json"),
               data={"best_model_path": CHECKPOINT_CALLBACK.best_model_path})
