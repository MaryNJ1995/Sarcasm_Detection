# -*- coding: utf-8 -*-
# ========================================================
"""This module is written for write useful function."""
# ========================================================


# ========================================================
# Imports
# ========================================================

import sys
from typing import List
import sklearn.utils.class_weight as class_weight
import pandas as pd
import numpy as np
import torch


def find_max_length_in_list(data: List[list]) -> int:
    """

    :param data: [["item_1", "item_2", "item_3", "item_4"], ["item_1", "item_2"]]
    :return: 4
    """
    return max(len(sample) for sample in data)


def ignore_pad_index(true_labels: List[list], pred_labels: List[list],
                     pad_token: str = "[PAD]") -> [List[list], List[list]]:
    """

    :param true_labels: ["item_1", "item_2", "pad", "pad"], ["item_3", "pad", "pad", "pad"]]
    :param pred_labels: [["item_1", "item_2", "item_3", "pad"], ["item_1", "pad", "item_2", "item_3"]]
    :param pad_token: pad
    :return: ([["item_1", "item_2"], ["item_3"]], [["item_1", "item_2"], ["item_1"]])
    """
    for idx, sample in enumerate(true_labels):
        true_ = list()
        pred_ = list()
        for index, item in enumerate(sample):
            if item != pad_token:
                true_.append(sample[index])
                pred_.append(pred_labels[idx][index])
        true_labels[idx] = true_
        pred_labels[idx] = pred_
    return true_labels, pred_labels


def convert_index_to_tag(data: List[list], idx2tag: dict) -> List[list]:
    """

    :param data: [["item_1", "item_2", "item_3"], ["item_1", "item_2"]]
    :param idx2tag: {"pad_item": 0, "item_1": 1, "item_2": 2, "item_3": 3}
    :return: [[1, 2, 3], [1, 2]]
    """
    return [[idx2tag[item] for item in sample] for sample in data]


def convert_subtoken_to_token(tokens: List[list], labels: List[list]) -> \
        [List[list], List[list]]:
    """

    :param tokens:
    :param labels:
    :return:
    """
    for idx, (tkns, lbls) in enumerate(zip(tokens, labels)):
        new_tokens, new_labels = list(), list()
        token_, label_ = None, None
        for token, label in zip(tkns, lbls):
            if token.startswith("##"):
                token_ = token_ + token[2:]
            elif label == "X":
                token_ = token_ + token
            else:
                if token_:
                    new_tokens.append(token_)
                    new_labels.append(label_)
                token_ = token
                label_ = label
        new_tokens.append(token_)
        new_labels.append(label_)

        tokens[idx] = new_tokens
        labels[idx] = new_labels

    return tokens, labels


def convert_predict_tag(tags: list, subtoken_check: list) -> list:
    """

    :param tags:
    :param subtoken_check:
    :return:
    """
    for idx, (tag, sbt_chk) in enumerate(zip(tags, subtoken_check)):
        processed_tags = []
        for index, (label, check) in enumerate(zip(tag, sbt_chk)):
            if check == 1:
                if label == "X":
                    tmp = tag[index - 1].replace('B-', 'I-')
                    processed_tags.append(tmp)
                else:
                    processed_tags.append(label)
        tags[idx] = processed_tags
    return tags


def handle_subtoken_labels(entities: list, subtoken_check: list) -> list:
    """

    :param entities:
    :param subtoken_check:
    :return:
    """
    return [entities[idx] for idx, item in enumerate(subtoken_check) if item == "1"]


def convert_x_label_to_true_label(predicted_tags: list, unexpected_entity: str) -> list:
    """

    :param predicted_tags:
    :param unexpected_entity:
    :return:
    """
    for tag_index, tag in enumerate(predicted_tags):
        if tag is unexpected_entity:
            predicted_tags[tag_index] = predicted_tags[tag_index - 1].replace("B-", "I-")
    return predicted_tags


def progress_bar(index, max, postText):
    """

    """
    n_bar = 50  # size of progress bar
    j = index / max
    sys.stdout.write('\r')
    sys.stdout.write(f"[{'=' * int(n_bar * j):{n_bar}s}] {int(100 * j)}%  {postText}")
    sys.stdout.flush()


def calculate_class_weights(data: pd.DataFrame, lbl_names: List[str]) -> [dict, dict]:
    """
    function to calculate class weight in multi-label dataset
    Args:
        data: dataset
        lbl_names: column name of labels

    Returns:
        class weight for each class
        total class weight


    """
    class2weights = {}
    all_class2weights = {}
    for cls in lbl_names:
        num_pos = 0
        class_weights = class_weight.compute_class_weight(
            "balanced",
            classes=np.unique(data[cls]),
            y=np.array(data[cls]))
        class2weights[cls] = torch.Tensor(class_weights)
        for lbl in data[cls]:
            if lbl == 1:
                num_pos += 1
        all_class2weights[cls] = num_pos / len(data)
    return class2weights, all_class2weights
