# -*- coding: utf-8 -*-
"""
trainer module is written for train model in task B
"""

# ============================ Third Party libs ============================
import logging
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
import transformers

# ============================ My packages ============================
from configuration import BaseConfig
from data_loader import read_csv, write_json
from data_preparation import normalize_text
from models import build_checkpoint_callback
from models.task_b_model import Classifier
from dataset import MultiDataModule
from utils import calculate_class_weights

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    CONFIG = CONFIG_CLASS.get_config()
    TOKENIZER = transformers.T5Tokenizer.from_pretrained(
        CONFIG.lm_model_path)
    LOGGER = CSVLogger(CONFIG.csv_logger_path, name=CONFIG.model_name)

    # load raw data
    RAW_TRAIN_DATA = read_csv(path=os.path.join(CONFIG.processed_data_dir, CONFIG.train_file),
                              columns=CONFIG.multi_data_headers,
                              names=CONFIG.multi_customized_headers).dropna()
    RAW_TRAIN_DATA.tweets = RAW_TRAIN_DATA.tweets.apply(normalize_text)

    RAW_VAL_DATA = read_csv(path=os.path.join(CONFIG.processed_data_dir, CONFIG.val_file),
                            columns=CONFIG.multi_data_headers,
                            names=CONFIG.multi_customized_headers).dropna()
    RAW_VAL_DATA.tweets = RAW_VAL_DATA.tweets.apply(normalize_text)

    RAW_TEST_DATA = read_csv(path=os.path.join(CONFIG.processed_data_dir, CONFIG.test_file),
                             columns=CONFIG.multi_data_headers,
                             names=CONFIG.multi_customized_headers).dropna()
    RAW_TEST_DATA.tweets = RAW_TEST_DATA.tweets.apply(normalize_text)

    logging.debug(RAW_TRAIN_DATA.head(), RAW_VAL_DATA.head(), RAW_TEST_DATA.head())
    logging.debug("length of Train data is: %s", len(RAW_TRAIN_DATA))
    logging.debug("length of Val data is: %s", len(RAW_VAL_DATA))
    logging.debug("length of Test data is: %s", len(RAW_TEST_DATA))

    logging.debug("Maximum length is: %s", CONFIG.max_length)

    LABEL_COLUMNS = RAW_TRAIN_DATA.columns.tolist()[1:]
    logging.debug("Label columns: %s", LABEL_COLUMNS)

    # Calculate class_weights
    class2weights, all_class2weights = calculate_class_weights(RAW_TRAIN_DATA, LABEL_COLUMNS)
    logging.debug("class_weights is: %s", class2weights)
    logging.debug("all_class2weights is: %s", all_class2weights)

    DATA_MODULE = MultiDataModule(data={"train_data": RAW_TRAIN_DATA,
                                        "val_data": RAW_VAL_DATA,
                                        "test_data": RAW_TEST_DATA
                                        },
                                  tokenizer=TOKENIZER, batch_size=CONFIG.batch_size,
                                  max_len=CONFIG.max_length, num_workers=CONFIG.num_workers,
                                  label_columns=LABEL_COLUMNS)

    DATA_MODULE.setup()

    CHECKPOINT_CALLBACK = build_checkpoint_callback(CONFIG.save_top_k)
    EARLY_STOPPING_CALLBACK = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    # Instantiate the Model Trainer
    TRAINER = pl.Trainer(max_epochs=CONFIG.n_epochs, gpus=[0],
                         callbacks=[CHECKPOINT_CALLBACK,
                                    EARLY_STOPPING_CALLBACK],
                         progress_bar_refresh_rate=60, logger=LOGGER)

    # Train the Classifier Model
    MODEL = Classifier(class_weights=class2weights,
                       all_class2weights=all_class2weights,
                       arg=CONFIG)
    TRAINER.fit(MODEL, DATA_MODULE)
    TRAINER.test(ckpt_path="best", datamodule=DATA_MODULE)

    # save best model path
    write_json(path=os.path.join(CONFIG.saved_model_path, CONFIG.model_name,
                                 "b_model_path.json"),
               data={"best_model_path": CHECKPOINT_CALLBACK.best_model_path})
