import os
from torch.utils.data import DataLoader
import transformers
from sklearn.metrics import classification_report

from configuration import BaseConfig
from data_loader import read_csv
from models.multilabel_tmp import Classifier
from inference import Inference, InferenceDataset, PairSarcasmDataset, \
    MultiSarcasmDataset

# from utils import progress_bar


if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    CONFIG = CONFIG_CLASS.get_config()

    MODEL_PATH = "../assets/saved_models/t5_large/version_8/checkpoints/QTag-epoch=06-val_loss=0.53.ckpt"

    MODEL = Classifier.load_from_checkpoint(MODEL_PATH, map_location="cuda:1")
    MODEL.eval().to("cuda:1")

    TOKENIZER = transformers.T5Tokenizer.from_pretrained(CONFIG.lm_tokenizer_path)

    # load raw data
    RAW_DATA = read_csv(path=os.path.join(CONFIG.processed_data_dir, CONFIG.test_file),
                        columns=CONFIG.multi_data_headers,
                        names=CONFIG.multi_customized_headers).dropna()
    print(len(RAW_DATA))

    INFER = Inference(MODEL, TOKENIZER)
    LABEL_COLUMNS = RAW_DATA.columns.tolist()[1:]

    DATASET = MultiSarcasmDataset(data=RAW_DATA,
                                  label_columns=LABEL_COLUMNS,
                                  tokenizer=TOKENIZER,
                                  max_len=CONFIG.max_length)

    DATALOADER = DataLoader(DATASET, batch_size=1,
                            shuffle=False, num_workers=4)

    FILE = open("task_a_en.txt", "w")

    PREDICTED_SARCASM = []
    PREDICTED_IRONY = []
    PREDICTED_SATIRE = []
    PREDICTED_UNDERSTATEMENT = []
    PREDICTED_OVERSTATEMENT = []
    PREDICTED_RHETORICAL_QUESTION = []
    for i_batch, sample_batched in enumerate(DATALOADER):
        sample_batched["inputs_ids"] = sample_batched["inputs_ids"].to("cuda:1")
        # OUTPUT, _ = INFER.predict(sample_batched)
        output_sarcasm, output_irony, output_satire, output_understatement, \
        output_overstatement, output_rhetorical_question = INFER.predict_multi(sample_batched)
        PREDICTED_SARCASM.extend(output_sarcasm)
        PREDICTED_IRONY.extend(output_irony)
        PREDICTED_SATIRE.extend(output_satire)
        PREDICTED_UNDERSTATEMENT.extend(output_understatement)
        PREDICTED_OVERSTATEMENT.extend(output_overstatement)
        PREDICTED_RHETORICAL_QUESTION.extend(output_rhetorical_question)

        # progress_bar(i_batch, len(DATALOADER), "testing ....")

    report = classification_report(y_true=list(RAW_DATA.sarcasm), y_pred=PREDICTED_SARCASM,
                                   target_names=["not_sarcasm", "is_sarcasm"])
    print(report)
    report = classification_report(y_true=list(RAW_DATA.irony), y_pred=PREDICTED_IRONY,
                                   target_names=["not_irony", "is_irony"])
    print(report)
    report = classification_report(y_true=list(RAW_DATA.satire), y_pred=PREDICTED_SATIRE,
                                   target_names=["not_satire", "is_satire"])
    print(report)
    report = classification_report(y_true=list(RAW_DATA.understatement), y_pred=PREDICTED_UNDERSTATEMENT,
                                   target_names=["not_understatement", "is_understatement"])
    print(report)
    report = classification_report(y_true=list(RAW_DATA.overstatement), y_pred=PREDICTED_OVERSTATEMENT,
                                   target_names=["not_overstatement", "is_overstatement"])
    print(report)
    report = classification_report(y_true=list(RAW_DATA.rhetorical_question), y_pred=PREDICTED_RHETORICAL_QUESTION,
                                   target_names=["not_rhetorical_question", "is_rhetorical_question"])
    print(report)

    # FILE.write("task_a_en")
    # FILE.write("\n")
    # for pred in PREDICTED_LABELS:
    #     FILE.write(str(pred))
    #     FILE.write("\n")
    # FILE.close()
