#!/usr/bin/env python3
import torch
import numpy as np
from torch.autograd import Variable


def word_function_dict():
    """Add functional words by Prof. Fernandez to dictionary."""
    path="function_words/"
    word_functions = ["adverbs", "articles", "auxiliary_verbs", "conjunctions",\
                 "impersonal_pronouns", "prepositions", "quantifiers"]
    d = {}
    for word_function in word_functions:
        d[word_function] = []
        with open(path + word_function + ".txt", "r") as f:
            for word in f.read().splitlines():
                d[word_function] += [word]
    return d


def LSM_loss_1(TGA, loss):

    history = TGA.history.history_strings

    # Do not use LSM without context
    if len(history) == 1 and history[0] == "__SILENCE__":
        return loss

    label = TGA.observation["labels_choice"]

    word_function_dict = TGA.word_function_dict
    functions = TGA.word_functions

    lsm = compute_LSM_score(history, label, word_function_dict, functions)

    # learning rate
    alpha = 0.2

    loss = (1 - alpha) * loss + alpha * 2 * (1 - lsm) * loss

    return loss


def LSM_loss_2(TGA, loss, preds):

    history = TGA.history.history_strings

    # Do not use LSM without context
    if len(history) == 1 and history[0] == "__SILENCE__":
        return loss

    preds = [TGA._v2t(p) for p in preds][0]

    word_function_dict = TGA.word_function_dict
    functions = TGA.word_functions

    lsm = compute_LSM_score(history, preds, word_function_dict, functions)

    # learning rate
    #alpha = Variable(torch.Tensor([TGA.lr_alpha]), requires_grad=True)
    alpha = TGA.lr_alpha

    # loss = (1 - alpha) * loss + alpha * 2 * (1 - lsm) * loss
    loss = loss * (1 + alpha * (1 - 2 * lsm))
    return loss


def compute_LSM_score(history, label, word_function_dict, functions):

    history = reconstruct_history(history)
    label = reconstruct_string(label)

    M = []
    for f in functions:
        function_words = word_function_dict[f]

        # function-word count in history
        hist_c = np.array([function_word_count(turn, function_words) for turn in history])

        # function-word count in label
        label_c = function_word_count(label, function_words)

        # example lsm formula
        lsm_score = 1 - np.absolute(hist_c - label_c) / (hist_c + label_c + 0.000000001)

        M.append(lsm_score)
    M = np.array(M)

    # TODO: Adding weight usage on history and/or functions
    lsm = np.mean(M)
    return lsm


def function_word_count(s, function_words):
    c = 0
    for w in s:
        if w in function_words:
            c += 1
    return c


def reconstruct_history(history):
    return [reconstruct_string(turn) for turn in history]


def reconstruct_string(s):
    s = s.replace("n't", " n't")
    s = s.replace("'", " '")
    s = s.split()
    return s