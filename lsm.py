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


def word_function_dict_by_id(dictionary):
    """Add the vocabulary id of functional words by Prof. Fernandez to dictionary."""
    path="function_words/"
    word_functions = ["adverbs", "articles", "auxiliary_verbs", "conjunctions",\
                 "impersonal_pronouns", "prepositions", "quantifiers"]
    d = {}
    for word_function in word_functions:
        d[word_function] = []
        with open(path + word_function + ".txt", "r") as f:
            for word in f.read().splitlines():
                idx = dictionary[word]
                if idx != 3:
                    d[word_function] += [idx]
    return d


def LSM_loss_1(TGA, loss):
    # LSM on golden response

    history = TGA.history.history_strings

    # Do not use LSM without context
    if len(history) == 1 and history[0] == "__SILENCE__":
        return loss

    label = TGA.observation["labels_choice"]

    word_function_dict = TGA.word_function_dict
    functions = TGA.word_functions

    lsm = compute_LSM_score_1(history, label, word_function_dict, functions)

    # learning rate
    #alpha = TGA.lr_alpha
    alpha = 0.2

    loss = loss * (1 + alpha * (1 - 2 * lsm))

    return loss


def compute_LSM_score_1(history, label, word_function_dict, functions):

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


def LSM_loss_2(TGA, loss, preds):
    # LSM on proposed response

    history = TGA.history.history_strings

    # Do not use LSM without context
    if len(history) == 1 and history[0] == "__SILENCE__":
        return loss

    label = TGA.observation["labels_choice"]
    preds = [TGA._v2t(p) for p in preds][0]

    word_function_dict = TGA.word_function_dict
    functions = TGA.word_functions

    lsm = compute_LSM_score_2(history, label, preds, word_function_dict, functions)

    # learnable learning rate
    alpha = TGA.lr_alpha

    # loss = (1 - alpha) * loss + alpha * 2 * (1 - lsm) * loss
    loss = loss * (1 + alpha * (1 - 2 * lsm))

    return loss


def compute_LSM_score_2(history, label, preds, word_function_dict, functions):

    history = reconstruct_history(history)
    label = reconstruct_string(label)
    preds = reconstruct_string(preds)

    M_label = []
    M_preds = []
    for f in functions:
        function_words = word_function_dict[f]

        # function-word count in history
        hist_c = np.array([function_word_count(turn, function_words) for turn in history])

        # function-word count in label
        label_c = function_word_count(label, function_words)

        # function-word count in preds
        preds_c = function_word_count(preds, function_words)

        # example lsm formula
        lsm_score_label = 1 - np.absolute(hist_c - label_c) / (hist_c + label_c + 0.000001)
        lsm_score_preds = 1 - np.absolute(hist_c - preds_c) / (hist_c + preds_c + 0.000001)

        M_label.append(lsm_score_label)
        M_preds.append(lsm_score_preds)

    M_label = np.array(M_label)
    M_preds = np.array(M_preds)

    # function distance between label and preds
    M = abs(M_label - M_preds)

    # TODO: Adding weight usage on history and/or functions

    # distance to lsm score
    M = abs(1 - M)
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


def LSM_beam_score(BS, scores, lsm_dict):
    print("---LSM---")


    #skip initialisation
    scores = scores.tolist()
    if len(scores) <= 1:
        return torch.tensor(scores)


    context = BS.context
    context_lsm_scores, _ = LSM_scores_from_idxs(context, lsm_dict)

    # List of current beam sentences
    beam_sentences = []
    for i in range(BS.beam_size):
        beam_sentence = []
        for output in BS.outputs:
            beam_sentence.append(output[i].tolist())
        beam_sentences += [beam_sentence]


    function_list = []
    sentence_lsm_scores = []
    for beam_sentence in beam_sentences:
        sentence_lsm_score, functions = LSM_scores_from_idxs(beam_sentence, lsm_dict)
        sentence_lsm_scores += [sentence_lsm_score]
        function_list += [functions]

    S = np.array(sentence_lsm_scores)
    F = np.array(function_list)
    C = np.array(context_lsm_scores)
    Cstack = np.vstack([C]*BS.beam_size)

    M = S / (C + 0.000001)

    # iterate over beam index with y
    for y in range(len(F)):
        for x in range(len(F[i])):
            word_function = F[y,x]
            function_idxs = lsm_dict[word_function]
            lsm_score = M[y,x]

            for idx in function_idxs:

                # TODO: Fix lsm score
                scores[y][idx] = lsm_score
                continue


    scores = torch.tensor(scores)
    print(scores)
    print(scores.shape)
    print("---End LSM---")
    return scores


def LSM_scores_from_idxs(l, lsm_dict):
    length = len(l)
    scores = []
    functions = []
    for function, indices in lsm_dict.items():
        functions += [function]
        f_count = 0
        for t in l:
            if t in indices:
                f_count += 1
        scores += [f_count / length]
    return scores, functions