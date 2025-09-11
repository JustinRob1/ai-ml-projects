# This program receives the tagger type and the path to a test file
# as command line parameters and outputs the POS tagged version of that file.
import nltk
from nltk import TaggerI
from nltk.tag import hmm, brill_trainer
from nltk.tag.brill import Template, Pos, Word
from nltk.probability import LidstoneProbDist, LaplaceProbDist, MLEProbDist
from collections import defaultdict
import argparse
import os

class InitialBrillTagger(TaggerI):
    '''
    Initial tagger that assigns the most common tag for each word in the training data.
    '''
    def __init__(self, word_dict):
        super().__init__()
        self.word_dict = word_dict
        
    def tag(self, tokens):
        tagged_tokens = [(token, self.word_dict[token] if token in self.word_dict.keys() else 'NN') for token in tokens]
        return tagged_tokens
    
def lidstone(freqdist, bins):
    return LidstoneProbDist(freqdist, gamma=0.0001)

def laplace(freqdist, bins):
    return LaplaceProbDist(freqdist, bins)

def mle(freqdist, bins):
    return MLEProbDist(freqdist)

def tag(tagger, train_path, test_path, output_path):
    '''
    Trains a POS tagger on the training data, tags the test data and writes the tagged data to the output file.
    '''
    train_data = process_data(train_path)
    test_data = process_data(test_path)
    
    if tagger == 'hmm':
        trainer = hmm.HiddenMarkovModelTrainer()
        tagger = trainer.train(train_data, estimator=lidstone)
    else:
        # Get the most common tag for each word in the training data.
        word_dict = brill_process_data(train_data)
        initial_tagger = InitialBrillTagger(word_dict)
        # Define the templates to be used by the BrillTaggerTrainer.
        templates = [ 
            Template(Pos([0])),
            Template(Pos([-1])), 
            Template(Pos([1])), 
            Template(Pos([-2])), 
            Template(Pos([2])), 
            Template(Pos([-2, -1])),
            Template(Pos([-1,0,1])), 
            Template(Pos([1, 2])),
            Template(Pos([-2,-1,0,1])), 
            Template(Pos([-2,-1,0,1,2])),
            Template(Pos([-3, -2, -1])), 
            Template(Pos([1, 2, 3])), 
            Template(Pos([-1]), Pos([1])),
            Template(Word([0])),
            Template(Word([-1])), 
            Template(Word([1])), 
            Template(Word([-2])), 
            Template(Word([2])), 
            Template(Word([-2, -1])), 
            Template(Word([-1,0,1])),
            Template(Word([1, 2])),
            Template(Word([-2,-1,0,1])),
            Template(Word([-2,-1,0,1,2])),
            Template(Word([-3, -2, -1])), 
            Template(Word([1, 2, 3])), 
            Template(Word([-1]), Word([1])), 
            ]
        trainer = brill_trainer.BrillTaggerTrainer(initial_tagger=initial_tagger, templates=templates)
        tagger = trainer.train(train_data)
        
    accuracy = tagger.accuracy(test_data)
    print(f"Accuracy: {accuracy:.4f}")
    # Tag the test data using the trained tagger.
    test_data = [tagger.tag([word for word, _ in sentence]) for sentence in test_data]
    write_output(output_path, test_data)
    
def process_data(file_path):
    '''
    Reads the training or test data from the file and returns it as a list of sentences as required by the NLTK taggers.
    '''
    train_data = []
    sentence = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            # If the line is not empty, split it into word and tag and add it to the current sentence.
            if line:
                word, tag = line.split()
                sentence.append((word, tag))
            # If the line is empty, the sentence is complete and we add it to the list of all sentences.
            else:
                if sentence:
                    train_data.append(sentence)
                    sentence = []
        # Add the last sentence
        if sentence:
            train_data.append(sentence)
    # Sentences is returned as a list of lists of tuples.
    # Each inner list represents a sentence and each tuple represents a word and its tag.
    # Example: [[('I', 'PRP'), ('like', 'VBP'), ('to', 'TO'), ('read', 'VB'), ('books', 'NNS'), ('.', '.')], ...]
    return train_data

def brill_process_data(train_data):
    '''
    Returns a dictionary with the most common tag for each word in the training data.
    Used to initialize the BrillTaggerTrainer.
    '''
    
    # Creates a dictionary that stores the number of times each word has been tagged with each tag.
    word_tags = defaultdict(lambda: defaultdict(int))
    for sentence in train_data:
        for word, tag in sentence:
            word_tags[word][tag] += 1
         
    # Stores the most common tag for each word in the training data.
    word_dict = {}
    for item in word_tags.keys():
        value = max(word_tags[item], key=word_tags[item].get)
        word_dict[item] = value
        
    return word_dict
    
def write_output(output_path, test_data):
    '''
    Writes the tagged test data to the output file in the required format.
    
    In IN
    a DT
    medium JJ
    saucepan NN
    ...
    
    '''
    with open(output_path, 'w') as file:
        for sentence in test_data:
            for word, tag in sentence:
                file.write(f'{word} {tag}\n')
            file.write('\n')
            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tagger', required=True, choices=['hmm', 'brill'], help='Type of tagger to use (hmm or brill)')
    parser.add_argument('--train', required=True, help='Path to the training file')
    parser.add_argument('--test', required=True, help='Path to the test file')
    parser.add_argument('--output', required=True, help='Path to the output file')
    args = parser.parse_args()
    
    if not os.path.exists(args.train):
        print('Training file does not exist.')
        return
    if not os.path.exists(args.test):
        print('Test file does not exist.')
        return
    
    tag(args.tagger, args.train, args.test, args.output)
    
if __name__ == "__main__":
    main()
