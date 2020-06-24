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


def LSM_loss_1(TGA, batch, loss):
    # LSM on golden response

    losses = []

    # Iterate over batches
    for b in batch.observations:

        # Get batch history
        history = b["full_text"].split("\n")

        # Do not use LSM without context
        if len(history) == 1 and history[0] == "__SILENCE__":
            losses += [loss]
            continue

        # Get batch golden response
        label = b["labels_choice"]

        word_function_dict = TGA.word_function_dict
        functions = TGA.word_functions

        lsm = compute_LSM_score_1(history, label, word_function_dict, functions)

        # learning rate
        #alpha = TGA.lr_alpha
        alpha = 0.2

        loss = loss * (1 + alpha * (1 - 2 * lsm))

        losses += [loss]

    # Average Batch loss
    loss = torch.mean(torch.stack(losses))
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


def LSM_beam_score(scores, lsm_dict, context, outputs, beam_size, device):

    #skip initialisation
    new_scores = scores.tolist()
    if len(new_scores) <= 1:
        return scores

    # Skip contextless dialogue
    context = context
    if len(context) <= 1:
        return scores

    context_lsm_scores, _ = LSM_scores_from_idxs(context, lsm_dict)

    # List of current beam sentences
    beam_sentences = []
    for i in range(beam_size):
        beam_sentence = []
        for output in outputs:
            beam_sentence.append(output[i].tolist())
        beam_sentences += [beam_sentence]


    # Find the lsm-score for each function in each beam 
    function_list = []
    beam_lsm_scores = []
    for beam_sentence in beam_sentences:
        beam_lsm_score, functions = LSM_scores_from_idxs(beam_sentence, lsm_dict)
        beam_lsm_scores += [beam_lsm_score]
        function_list += [functions]

    F = np.array(function_list)
    B = np.array(beam_lsm_scores)
    C = np.array(context_lsm_scores)
    Cstack = np.vstack([C]*beam_size)

    # Calculate LSM scores
    M = (Cstack - B) / (np.abs(Cstack + B) + 0.000001)

    # iterate over beam index with y
    for y in range(len(F)):
        for x in range(len(F[i])):
            word_function = F[y,x]
            function_idxs = lsm_dict[word_function]
            lsm_score = M[y,x]

            for idx in function_idxs:
                new_scores[y][idx] += abs(new_scores[y][idx]) * lsm_score

    new_scores = torch.tensor(new_scores, device=torch.device(device))

    #alpha = TGA.lr_alpha
    alpha = 0.2

    # Change Score based on alpha
    scores = (1 - alpha) * scores + alpha * new_scores

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


def document_loss(path, loss):
    if path != None:
        f = open(path + '_loss_documentation.txt', "a")
        f.write(str(loss.item()) + '\n')
        f.close()
    return