import os

import transformers
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from configuration import BaseConfig
from data_loader import read_csv
from inference import Inference, InferenceDataset
from models.sar_RCNN import Classifier
from utils import progress_bar

if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    CONFIG = CONFIG_CLASS.get_config()

    MODEL_PATH = "/home/maryam.najafi/Project_Sarcasm_Detection/assets/SarcasmDetection/version_46_best_en/checkpoints/" \
                 "QTag-epoch=16-val_loss=1.61.ckpt"

    MODEL = Classifier.load_from_checkpoint(MODEL_PATH, map_location="cuda:0")
    MODEL.eval().to("cuda:0")

    TOKENIZER = transformers.MT5Tokenizer.from_pretrained(CONFIG.lm_tokenizer_path)

    RAW_DATA = read_csv(path=os.path.join(CONFIG.processed_data_dir, CONFIG.test_file),
                        columns=["text", "sarcastic"],
                        names=["texts", "labels"]).dropna()
    print(RAW_DATA.head())
    INFER = Inference(MODEL, TOKENIZER)

    DATASET = InferenceDataset(texts=list(RAW_DATA.texts),
                               tokenizer=TOKENIZER,
                               max_length=CONFIG.max_length)

    DATALOADER = DataLoader(DATASET, batch_size=1,
                            shuffle=False, num_workers=4)

    PREDICTED_LABELS = []
    for i_batch, sample_batched in enumerate(DATALOADER):
        sample_batched["inputs_ids"] = sample_batched["inputs_ids"].to("cuda:0")
        OUTPUT, _ = INFER.predict(sample_batched)
        PREDICTED_LABELS.extend(OUTPUT)

        progress_bar(i_batch, len(DATALOADER), "testing ....")

    FILE2WRITE = open("task_a_en.txt", "w")
    FILE2WRITE.write("task_a_en")
    FILE2WRITE.write("\n")
    for pred in PREDICTED_LABELS:
        FILE2WRITE.write(str(pred))
        FILE2WRITE.write("\n")
    FILE2WRITE.close()
    report = classification_report(y_true=list(RAW_DATA.labels), y_pred=PREDICTED_LABELS,
                                   target_names=["not_sarcastic", "sarcastic"], digits=4)
    print(report)
