import re


def normalize_text(input_data: str, normalizer):
    input_text = input_data.rstrip('\r\n').strip()
    # input_text = normalizer.normalize(input_text)
    input_text = re.sub(r"http\S+", "", input_text)
    # input_text = re.sub("https://[a-zA-z./\d]*", "", input_text)
    input_text = re.sub("http://[a-zA-z./\d]*", "", input_text)
    # input_text = emoji.get_emoji_regexp().sub(u'', input_text)
    input_text = re.sub("@[^\s]+", "", input_text)
    input_text = re.sub("@([^@]{0,30})\s", "", input_text)
    input_text = re.sub("@([^@]{0,30})）", "", input_text)
    input_text = re.sub("#([^#]{0,30})）", "", input_text)
    input_text = input_text.replace('"', "")
    # input_text = remove_hashtags(input_text)
    # input_text = re.sub(r"[،,:.)(|-»;@#?؟!&$]+\*", " ", input_text)
    # input_text = input_text.replace("#Irony", " ")
    # input_text = input_text.replace("#irony", " ")
    # input_text = input_text.replace("#sarcasm", " ")
    # input_text = input_text.replace("#Sarcasm", " ")
    input_text = re.sub("\s\s+", " ", input_text)

    return input_text


def remove_hashtags(input_text):
    new_string = ''
    for i in input_text.split():
        if i[:1] != '#':
            new_string = new_string.strip() + ' ' + i
    return new_string
if __name__ == '__main__':
    from hazm import Normalizer
    NORM=Normalizer()
    print(normalize_text("That shitty feeling we all love so much 😊", NORM))
