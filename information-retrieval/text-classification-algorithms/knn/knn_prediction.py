import json
import math
import heapq
import os
import sys

from collections import Counter
from normalize import normalize
from doc_vec import DocumentVector


def main():
    try:
        model_file = os.path.abspath(sys.argv[1])
        test_file = os.path.abspath(sys.argv[2])
        k = int(sys.argv[3])
        if k <= 0:
            raise ValueError
    except IndexError:
        print("Error: Invalid number of arguments")
        print("Usage: python knn_prediction.py <model_file> <test_file> <k>")
        return 1
    except ValueError:
        print("k must be a positive integer")
        return 1

    idf, vectors, C = read_model(model_file)
    V = idf.keys()
    test_data = get_json_data(test_file)
    to_doc_vecs(test_data, idf)

    predictor = KNNPredictor(vectors, k, V)

    test_categories = []
    predicted_categories = []
    for d in test_data:
        test_categories.append(d['category'])
        predicted_categories.append(predictor.apply(d['text']))

    print_stats(test_categories, predicted_categories, C)

    return 0


def print_stats(test_categories, predicted_categories, C):

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


def to_doc_vecs(train_data, idf):
    '''
    Converts the text in train_data to DocumentVectors using ltc tf-idf weights.
    '''
    # normalize text
    for d in train_data:
        tf_dict = get_tf_dict_from(normalize(d['text']))
        # Compute the ltc tf-idf weight for each term
        doc_v = DocumentVector()
        for term in tf_dict:
            if term not in idf:
                # ignore terms not in the model
                continue
            doc_v.add(term, tf_weight(term, tf_dict) * idf[term])
        doc_v.cosine_normalize()
        d['text'] = doc_v


def tf_weight(term, tf_dict):
    '''
    Returns the tf weight of term in doc.

    term: term to compute tf weight for
    tf_dict: dict of term -> raw term frequency
    '''
    return 1 + math.log10(tf_dict[term])


def get_tf_dict_from(text):
    '''
    Returns a dictionary of term -> raw term frequency.
    '''
    return Counter(text)


def read_model(model_file):
    '''
    Read the model file and return the following values:
    idf: dict of term to idf weight
    doc_vectors: dict of doc_id to DocumentVector
    '''
    idf = {}
    doc_vectors = []
    C = set()
    with open(model_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("idf"):
                _, term, idf_weight = line.split('\t')
                idf[term] = float(idf_weight)
            elif line.startswith("vector"):
                try:
                    _, c, vec_str = line.split('\t')
                    doc_vectors.append((c, DocumentVector(vec_str)))
                    C.add(c)
                except ValueError:
                    pass
    return idf, doc_vectors, C


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


class KNNPredictor:
    def __init__(self, vectors, k, V):
        '''
        vectors: dict of class names to DocumentVector
        k: number of nearest neighbors 
        '''
        self.vectors = vectors
        self.k = k
        self.V = V

    def apply(self, d):
        class_probs = {}
        for _, c in self.__compute_nearest_neighbors(d):
            class_probs[c] = class_probs.get(c, 0) + 1
        for c in class_probs:
            class_probs[c] /= self.k

        # return the class with the highest probability
        return max(class_probs, key=class_probs.get)

    def __compute_nearest_neighbors(self, d):
        dists = []
        for c, v in self.vectors:
            # euclidean distance
            dist = 0
            for term in d:
                if term not in self.V:
                    continue
                dist += (d[term] - v.get(term, 0)) ** 2
            dist = math.sqrt(dist)
            heapq.heappush(dists, (dist, c))

        # compute k nearest neighbors
        return [heapq.heappop(dists) for _ in range(self.k)]


if __name__ == "__main__":
    sys.exit(main())
