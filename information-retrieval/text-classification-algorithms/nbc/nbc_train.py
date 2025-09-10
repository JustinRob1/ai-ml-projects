from collections import Counter, defaultdict
import json
import os
import sys

from normalize import normalize


def main():
    try:
        train_data_file = os.path.abspath(sys.argv[1])
        model_file = os.path.abspath(sys.argv[2])
    except IndexError:
        print("Error: Invalid number of arguments")
        print("Usage: python nbc_train.py <train_data_file> <model_file>",)
        return 1

    try:
        train_data = get_json_data(train_data_file)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    try:
        nb_trainer = MultinomialNBTrainer(train_data)
        V, prior, cond_prob = nb_trainer.train()
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    write_model(model_file, prior, cond_prob)
    return 0


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


def write_model(model_file, prior, cond_prob):
    '''
    Writes a train model to a tsv file with the following format:

    For each prior:
    prior\t<class_name>\t<prior_prob>\n

    For each likelihood:
    likelihood\t<class_name>\t<term>\t<likelihood_prob>\n
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

    with open(model_file, "w") as f:
        # write tsv
        for c in prior:
            f.write(f"prior\t{c}\t{prior[c]}\n")
        for c in cond_prob:
            for t in cond_prob[c]:
                f.write(f"likelihood\t{c}\t{t}\t{cond_prob[c][t]}\n")


class MultinomialNBTrainer:

    def __init__(self, train_data):
        self.D = train_data
        self.C = set()   # unique class names
        self.V = set()   # vocabulary
        self.N = 0       # number of documents
        self.Nc = {}     # number of documents in class c
        self.cond_prob = defaultdict(dict)

    def extract_vocabulary(self):
        '''
        Extracts the vocabulary from the training data.
        '''
        for doc in self.D:
            self.V.update(doc['text'])

    def normalize_all_texts_in_docs(self):
        '''
        Normalizes all texts in the training data.
        '''
        for doc in self.D:
            doc['text'] = normalize(doc['text'])

    def extract_classes_and_docs(self):
        '''
        Extracts the classes and number of documents in each class.
        '''
        try:
            for doc in self.D:
                self.C.add(doc['category'])
                self.N += 1
                self.Nc.setdefault(doc['category'], 0)
                self.Nc[doc['category']] += 1
        except KeyError as e:
            raise KeyError(
                f"Error: train_data does not have keys: 'category', 'text'") from e

    def concat_text_of_all_docs_in_class(self, class_name):
        '''
        Returns a list of all tokens in all documents in class class_name.
        '''
        text_c = []
        for doc in self.D:
            if doc['category'] == class_name:
                text_c.extend(doc['text'])
        return text_c

    def train(self):
        '''
        Trains the nbc model and return the vocabulary, prior, and conditional probabilities.
        '''
        self.normalize_all_texts_in_docs()
        self.extract_vocabulary()
        self.extract_classes_and_docs()
        prior = {}
        T = defaultdict(dict)
        for c in self.C:
            prior[c] = self.Nc[c] / self.N
            text_c = self.concat_text_of_all_docs_in_class(c)
            T[c] = Counter(text_c)
            T_c_total = len(text_c)
            for t in self.V:
                self.cond_prob[c][t] = (T[c][t] + 1) / \
                    (T_c_total + len(self.V))

        return self.V, prior, self.cond_prob


if __name__ == "__main__":
    sys.exit(main())
