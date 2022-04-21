import os
import numpy as np
from torch.utils.data import DataLoader
import transformers
from sklearn.metrics import classification_report

from configuration import BaseConfig
from data_loader import read_csv
from models.sar_t5_transformers import Classifier as model_1
from models.t5_cnn import Classifier as model_2
from models.t5_lstm import Classifier as model_3
from inference import Inference, InferenceDataset

# from utils import progress_bar


if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    CONFIG = CONFIG_CLASS.get_config()

    TOKENIZER = transformers.T5Tokenizer.from_pretrained(CONFIG.lm_tokenizer_path)

    # load raw data
    RAW_DATA = read_csv(path=os.path.join(CONFIG.processed_data_dir, CONFIG.test_file),
                        columns=CONFIG.data_headers,
                        names=CONFIG.customized_headers).dropna()

    DATASET = InferenceDataset(texts=list(RAW_DATA.texts),
                               tokenizer=TOKENIZER,
                               max_length=CONFIG.max_length)

    DATALOADER = DataLoader(DATASET, batch_size=1,
                            shuffle=False, num_workers=4)

    MAIN_PATH = "../assets/saved_models/t5_large"

    MODEL_PATH = ["t5_transformers/checkpoints/QTag-epoch=25-val_loss=0.83.ckpt",
                  "t5_cnn/checkpoints/QTag-epoch=05-val_loss=0.60.ckpt",
                  "version_0/checkpoints/QTag-epoch=05-val_loss=0.63.ckpt"]

    ALL_SCORES = []

    for index, path in enumerate(MODEL_PATH):
        print(path)
        print(index)
        print("OK")
        if index == 0:
            print(path)
            MODEL = model_1.load_from_checkpoint(os.path.join(MAIN_PATH, path), map_location="cuda:1")
        elif index == 1:
            print(path)
            MODEL = model_2.load_from_checkpoint(os.path.join(MAIN_PATH, path), map_location="cuda:1")
        else:
            print(path)
            MODEL = model_3.load_from_checkpoint(os.path.join(MAIN_PATH, path), map_location="cuda:1")

        MODEL.eval().to("cuda:1")

        INFER = Inference(MODEL, TOKENIZER)

        PREDICTED_LABELS = []
        SCORES = []
        for i_batch, sample_batched in enumerate(DATALOADER):
            sample_batched["inputs_ids"] = sample_batched["inputs_ids"].to("cuda:1")
            PRED, OUTPUTS = INFER.predict(sample_batched)
            SCORES.extend(OUTPUTS)
            PREDICTED_LABELS.extend(PRED)

        ALL_SCORES.append(SCORES)

        # progress_bar(i_batch, len(DATALOADER), "testing ....")

        REPORT = classification_report(y_true=list(RAW_DATA.labels), y_pred=PREDICTED_LABELS,
                                       target_names=["not_sarcastic", "sarcastic"])

        print()
        print(index)
        print(REPORT)
        print()

    print(ALL_SCORES[0])
    print(ALL_SCORES[1])
    print(ALL_SCORES[2])
    x = sum([np.array(ALL_SCORES[0]), np.array(ALL_SCORES[1]), np.array(ALL_SCORES[2])])

    print(x.tolist())
    x = np.argmax(x, axis=1)
    print(x)
    print(list(RAW_DATA.labels))

    REPORT = classification_report(y_true=list(RAW_DATA.labels), y_pred=x.tolist(),
                                   target_names=["not_sarcastic", "sarcastic"])
    print(REPORT)
    # print(x)
    # print(len(x))
    # print(ALL_SCORES[0])
    # print(len(ALL_SCORES))
    # print(len(ALL_SCORES[0]))
    # print(len(ALL_SCORES[1]))

    # FILE = open("task_a_en.txt", "w")
    #
    # FILE.write("task_a_en")
    # FILE.write("\n")
    # for pred in PREDICTED_LABELS:
    #     FILE.write(str(pred))
    #     FILE.write("\n")
    # FILE.close()
