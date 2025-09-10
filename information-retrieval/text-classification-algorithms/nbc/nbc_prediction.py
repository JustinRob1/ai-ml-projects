import json
import math
import os
import sys

from pprint import pprint
from normalize import normalize
from collections import defaultdict


def main():
    try:
        model_file = os.path.abspath(sys.argv[1])
        test_file = os.path.abspath(sys.argv[2])
    except IndexError:
        print("Error: Invalid number of arguments")
        print("Usage: python nbc_prediction.py <model_file> <test_file>",)
        return 1

    C, V, prior, cond_prob = read_model(model_file)
    test_data = get_json_data(test_file)
    test_categories = [data['category'] for data in test_data]

    predictor = MultinomialNBPredictor(C, V, prior, cond_prob)
    predicted_categories = [
        predictor.apply(data['text'])
        for data in test_data
    ]

    counts = {c: {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0} for c in C}

    for orig, pred in zip(test_categories, predicted_categories):
        for c in C:
            if orig == c and pred == c:
                counts[c]['TP'] += 1
            elif orig != c and pred == c:
                counts[c]['FP'] += 1
            elif orig == c and pred != c:
                counts[c]['FN'] += 1
            elif orig != c and pred != c:
                counts[c]['TN'] += 1

    stats = {c: {'TP_sum': 0, 'FP_sum': 0, 'FN_sum': 0} for c in C}
    stats['f1_sum'] = 0

    # table for each class
    max_class_len = max([len(c) for c in C])
    print(f"{'Class':<{max_class_len}}\tTP\tFP\tFN\tTN\tPrecision\tRecall\tF1")
    for c in C:
        tp = counts[c]['TP']
        fp = counts[c]['FP']
        fn = counts[c]['FN']
        tn = counts[c]['TN']
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * precision * recall / \
            (precision + recall) if precision + recall > 0 else 0
        stats['f1_sum'] += f1
        stats[c]['TP_sum'] += tp
        stats[c]['FP_sum'] += fp
        stats[c]['FN_sum'] += fn
        print(
            f"{c:<{max_class_len}}\t{tp}\t{fp}\t{fn}\t{tn}\t{precision:.5f}\t\t{recall:.5f}\t{f1:.5f}")

    macroavg_f1 = stats['f1_sum'] / len(C)
    print('Macro-average F1: {:.5f}'.format(macroavg_f1))

    aggr_tp = sum([stats[c]['TP_sum'] for c in C])
    aggr_fp = sum([stats[c]['FP_sum'] for c in C])
    aggr_fn = sum([stats[c]['FN_sum'] for c in C])
    aggr_precision = aggr_tp / \
        (aggr_tp + aggr_fp) if aggr_tp + aggr_fp > 0 else 0
    aggr_recall = aggr_tp / (aggr_tp + aggr_fn) if aggr_tp + aggr_fn > 0 else 0
    micro_f1 = 2 * aggr_precision * aggr_recall / \
        (aggr_precision + aggr_recall) if aggr_precision + aggr_recall > 0 else 0
    print('Micro-average F1: {:.5f}'.format(micro_f1))

    return 0


def read_model(model_file):
    '''
    Writes a train model to a tsv file with the following format:

    For each prior:
    prior\t<class_name>\t<prior_prob>\n

    For each likelihood:
    likelihood\t<class_name>\t<term>\t<likelihood_prob>\n
    '''
    prior = {}
    cond_prob = defaultdict(dict)
    with open(model_file, "r") as f:
        for line in f:
            # get prior probabilities
            if line.startswith("prior"):
                _, class_name, prior_prob = line.strip().split("\t")
                prior[class_name] = float(prior_prob)
            # get likelihood probabilities
            elif line.startswith("likelihood"):
                _, class_name, term, likelihood_prob = line.strip().split("\t")
                cond_prob[class_name][term] = float(likelihood_prob)

    C = list(prior.keys())
    V = set()
    for class_name in C:
        V.update(cond_prob[class_name].keys())

    return C, V, prior, cond_prob


def get_json_data(json_file):
    '''
    Returns a list of json objects.
    '''
    if not (os.path.exists(json_file)):
        raise ValueError(f"Error: {json_file} does not exist")

    if (os.path.isdir(json_file)):
        raise ValueError(
            f"Error: {json_file} should be a file, but is a directory")

    with open(json_file, "r") as f:
        try:
            return json.load(f)
        except json.decoder.JSONDecodeError as e:
            raise ValueError(
                f"Error: {json_file} is not a valid json file") from e


class MultinomialNBPredictor:
    def __init__(self, C, V, prior, cond_prob):
        self.C = C
        self.V = V
        self.prior = prior
        self.cond_prob = cond_prob

    def extract_tokens_from_doc(self, d):
        normalized_d = normalize(d)
        # only keep tokens that are in the vocabulary
        return [term for term in normalized_d if term in self.V]

    def apply(self, d):
        '''
        Returns the predicted class for a given document.
        '''
        W = self.extract_tokens_from_doc(d)
        scores = {}
        for c in self.C:
            scores[c] = math.log(self.prior[c])
            for term in W:
                scores[c] += math.log(self.cond_prob[c][term])
        return max(scores, key=scores.get)


if __name__ == "__main__":
    sys.exit(main())
