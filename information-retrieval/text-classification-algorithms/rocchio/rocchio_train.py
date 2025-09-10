import math
import os
import sys
import json

from collections import Counter
from normalize import normalize
from doc_vec import DocumentVector


def main():
    try:
        train_data_file = os.path.abspath(sys.argv[1])
        model_file = os.path.abspath(sys.argv[2])
    except IndexError:
        print("Error: Invalid number of arguments")
        print("Usage: python rocchio_train.py <train_data_file> <model_file>",)
        return 1

    try:
        train_data = get_json_data(train_data_file)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    idf, D = idfs_and_doc_vecs(train_data)
    rocchio = RocchioTrainer(D.keys(), D)
    centroids = rocchio.train()
    write_idfs_and_centroids(idf, centroids, model_file)


def idfs_and_doc_vecs(train_data):
    '''
    Returns a tuple of (idf, D) where
    idf is a dictionary of term -> idf weight
    D is a dictionary of category -> list of DocumentVector
    '''

    N = len(train_data)
    # normalize text
    for d in train_data:
        d['text'] = normalize(d['text'])

    # df for all documents
    df = Counter()
    for d in train_data:
        df.update(set(d['text']))

    idf = {}
    for term in df:
        idf[term] = idf_weight(term, df, N)

    D = {}

    for d in train_data:
        tf_dict = get_tf_dict_from(d['text'])
        # Compute the ltc tf-idf weight for each term
        doc_v = DocumentVector()
        for term in tf_dict:
            doc_v.add(term, tf_weight(term, tf_dict) * idf[term])
        doc_v.cosine_normalize()
        c = d['category']
        D[c] = D.get(c, []) + [doc_v]

    return idf, D


def idf_weight(term, df, N):
    '''
    Returns the idf weight of term in doc.

    term: term to compute idf weight for
    df: dict of term -> document frequency
    N: total number of documents
    '''
    return math.log10(N / df[term])


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


def write_idfs_and_centroids(idf, centroids, model_file):
    '''
    Writes the idf and centroids to a file.
    Formats:
    idf\t<term>\t<idf weight>
    centroid\t<class>\t<centroid vector>
    '''

    # ask if we should overwrite the file
    if os.path.exists(model_file):
        overwrite = input(
            f"Warning: {model_file} already exists. Overwrite? (y/n): ")
        if overwrite != 'y':
            print("Exiting...")
            return

    # create directories if they don't exist
    os.makedirs(os.path.dirname(model_file), exist_ok=True)

    # written in tsv
    with open(model_file, "w") as f:
        # write idf
        for term in idf:
            f.write(f"idf\t{term}\t{idf[term]}\n")
        # write centroids
        for c in centroids:
            f.write(f"centroid\t{c}\t{centroids[c]}\n")


class RocchioTrainer:
    def __init__(self, C, D):
        '''
        C: list of class names
        D: dict of class names to list of DocumentVector
        '''
        self.C = C
        self.D = D

    def train(self):
        centroids = {}
        for c in self.C:
            # create doc vectors
            D_j = self.D[c]
            centroid = DocumentVector()
            for d in D_j:
                centroid += d
            for term in centroid:
                centroid[term] /= len(D_j)
            centroids[c] = centroid
        return centroids


if __name__ == "__main__":
    sys.exit(main())
