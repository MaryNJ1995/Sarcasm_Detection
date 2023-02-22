import re


def normalize_text(input_data: str):
    input_text = input_data.rstrip('\r\n').strip()
    input_text = re.sub(r"http\S+", "", input_text)
    input_text = re.sub("@[^\s]+", "", input_text)
    input_text = re.sub("@([^@]{0,30})\s", "", input_text)
    input_text = re.sub("@([^@]{0,30})）", "", input_text)
    input_text = re.sub("#([^#]{0,30})）", "", input_text)
    input_text = re.sub("\s\s+", " ", input_text)

    return input_text
