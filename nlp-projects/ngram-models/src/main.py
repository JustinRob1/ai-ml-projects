from pathlib import Path
from collections import Counter, defaultdict
import argparse
import random
import math
    
# Source: https://docs.python.org/3/library/pathlib.html
# Source: https://switowski.com/blog/pathlib/
def data_split():
    '''
    Reads the transformed files in the transformed directory and splits the data into a training set and a dev set.
    '''
    transformed_dir = Path('./transformed')
    data_dir = Path('./data')
    training_file = data_dir / 'training.txt'
    dev_file = data_dir / 'dev.txt'
    
    # Recursively gets the paths of all the txt files in the transformed directory
    transformed_files = list(transformed_dir.rglob('*.txt'))
    
    # Source: https://www.geeksforgeeks.org/writing-to-file-in-python/
    # Open our two dataset files simultaneously
    with open (training_file, 'w') as training, open (dev_file, 'w') as dev:
        # Iterates through all the transformed files and writes utterance to either the
        # training.txt file or dev.txt file
        for file in transformed_files:
            with open(file, 'r') as f:  
                for line in f:
                    # Strip whitespace and add our own newline charcter
                    # Add <s> and </s> tags to the beginning and end of each utterance
                    new_line = f"<s> {line.strip()} </s>\n"
                    
                    # With 80% probability, write the utterance to the training set
                    if random.random() < 0.8:
                        training.write(new_line)
                    # With 20% probability, write the utterance to the dev set
                    else:
                        dev.write(new_line)
                        
def unigram(args):
    '''
    Trains a unigram model and measures perplexity on a held out corpus. 
    '''

    # Train Stage
    with open(args.train_path, "r") as f: 
        utterances = f.read().split()

    # Source : https://stackoverflow.com/questions/12488722/counting-bigrams-pair-of-two-words-in-a-file-using-python
    # Get count of unigrams in corpus
    unigrams = Counter(utterances)
    vocab = list(unigrams.keys())
    count_unigrams = sum(unigrams.values())

    # Compute probabilities for each unigram
    probs = defaultdict(dict)
    for w in vocab:
        probs[w] = unigrams[w] / count_unigrams

    # Dev Stage
    with open(args.dev_path, "r") as f:
        dev_utterances = f.read().split()

    # Get all the unigrams from the dev corpus
    dev_unigrams = Counter(dev_utterances)
    N = sum(dev_unigrams.values())

    # Computing perplexity for the dev set
    prod_probs = 0
    for w, count in dev_unigrams.items():
        # Check for OOV utterances
        tok = '<UNK>' if w not in vocab else w
        prod_probs += count * -math.log(probs[tok])

    perplexity = math.exp(prod_probs/N)
    print(round(perplexity, 4))

def bigram(args):
    '''
    Trains a bigram model and measures perplexity on a held out corpus. 
    '''
    
    # Train Stage
    with open(args.train_path, "r") as f: 
        utterances = f.read().split()
        
    # Source : https://stackoverflow.com/questions/12488722/counting-bigrams-pair-of-two-words-in-a-file-using-python
    # Get count of unigrams and bigrams in corpus
    unigrams = Counter(utterances)
    bigrams = Counter(zip(utterances, utterances[1:]))
    vocab = list(unigrams.keys())
    vocab_size = len(vocab)
    
    # Remove unnecessary case
    del bigrams[('</s>', '<s>')]
    
    # Compute conditional probabilities for each bigram
    probs = defaultdict(dict)
    for w1 in vocab:
        for w2 in vocab:
            count_bigrams = 0 if (w1, w2) not in bigrams.keys() else bigrams[(w1, w2)]
            # Add 1 to each bigram count for Laplace Smoothing
            if args.laplace:
                probs[w1][w2] = (count_bigrams + 1)/ (unigrams[w1] + vocab_size)
            else:
                try:
                    # probs[w1][w2] = max(count_bigrams/ unigrams[w1], 1e-10)
                    probs[w1][w2] = count_bigrams / unigrams[w1]
                except ZeroDivisionError:
                    probs[w1][w2] = 0

    # Dev Stage
    with open(args.dev_path, "r") as f:
        dev_utterances = f.read().split()
        
    # Get all the bigrams from the dev corpus
    dev_bigrams = Counter(zip(dev_utterances, dev_utterances[1:]))
    del dev_bigrams[('</s>', '<s>')]  # Remove unnecessary case 
    N = sum(dev_bigrams.values())  # Number of bigrams
    
    # Computing perplexity for dev set
    prod_probs = 0
    for (w1, w2), count in dev_bigrams.items():
        # Handle OOV utterances
        tok1 = '<UNK>' if w1 not in vocab else w1
        tok2 = '<UNK>' if w2 not in vocab else w2
        # Compute -ve log probability of bigram (times how many times it occurs)
        try: 
            prod_probs += count * -math.log(probs[tok1][tok2])
        except ValueError:
            prod_probs = math.inf
            break
        
    perplexity = math.exp(prod_probs / N)    
    print(round(perplexity, 4))

def trigram(args):
    '''
    Trains a trigram model and measures perplexity on a held out corpus. 
    '''

    # Train Stage
    with open(args.train_path, "r") as f: 
        utterances = f.read().split()

    # Source : https://stackoverflow.com/questions/12488722/counting-bigrams-pair-of-two-words-in-a-file-using-python
    # Get count of unigrams, bigrams, and trigrams in corpus
    unigrams = Counter(utterances)
    bigrams = Counter(zip(utterances, utterances[1:]))
    trigrams = Counter(zip(utterances, utterances[1:], utterances[2:]))
    vocab = list(unigrams.keys())
    vocab_size = len(vocab)

    # Remove unnecessary case, not an utterance
    # These trigrams indicate the end of a sentence and the start of a new one
    for w in vocab:
        del trigrams[(w, '</s>', '<s>')]
        del trigrams[('</s>', '<s>', w)]

    # Compute conditional probabilities
    probs = defaultdict(dict)
    for w1 in vocab:
        for w2 in vocab:
            for w3 in vocab:
                count_trigrams = 0 if (w1, w2, w3) not in trigrams.keys() else trigrams[(w1, w2, w3)]
                # Add 1 to each trigram count for Laplace Smoothing
                if args.laplace:
                    probs[(w1, w2)][w3] = (count_trigrams + 1) / (bigrams[(w1, w2)] + vocab_size)
                else:
                    try: 
                        probs[(w1, w2)][w3] = count_trigrams/ bigrams[(w1, w2)]
                    except ZeroDivisionError:
                        probs[(w1, w2)][w3] = 0
                    # probs[(w1, w2)][w3] = max(true_prob, 1e-10)    
                    
    # Dev Stage
    with open(args.dev_path, "r") as f:
        dev_utterances = f.read().split()

    # Get all the trigrams from the dev corpus
    dev_unigrams = Counter(dev_utterances)
    dev_trigrams = Counter(zip(dev_utterances, dev_utterances[1:], dev_utterances[2:]))
    dev_vocab = list(dev_unigrams.keys())

    # Remove unnecessary case
    for w in dev_vocab:
        del dev_trigrams[(w, '</s>', '<s>')]
        del dev_trigrams[('</s>', '<s>', w)]
    N = sum(dev_trigrams.values())  # Number of bigrams

    # Computing perplexity for dev set
    prod_probs = 0
    for (w1, w2, w3), count in dev_trigrams.items():
        # Handle OOV utterances
        tok1 = '<UNK>' if w1 not in vocab else w1
        tok2 = '<UNK>' if w2 not in vocab else w2
        tok3 = '<UNK>' if w3 not in vocab else w3
        # Skip unseen trigrams
        if probs[(tok1, tok2)].get(tok3) is None:
            continue
        # Compute -ve log probability of trigram (times how many times it occurs)
        try:
            prod_probs += count * -math.log(probs[(tok1, tok2)] [tok3])
        except ValueError:
            prod_probs = math.inf
            break


    perplexity = math.exp(prod_probs/N)    
    print(round(perplexity, 4))
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_type', choices=['unigram', 'bigram', 'trigram'], help="Type of n-gram model (uni, bi, tri)")
    parser.add_argument('train_path', help="Path to training data")
    parser.add_argument('dev_path', help = "Path to dev data")
    parser.add_argument('--laplace', action='store_true', help='Enable Laplace Smoothing')
    parser.add_argument("--partition", action="store_true", help="Re-partition the training and dev sets.")
    args = parser.parse_args()
    
    if args.partition:
        data_split()

    # Validate that Laplace smoothing is not used with unigram
    if args.model_type == 'unigram' and args.laplace:
        parser.error("--laplace option is invalid for unigram model.")
    
    # Call the model function based on the model type
    if args.model_type == 'unigram':
        unigram(args)
    elif args.model_type == 'bigram':
        bigram(args)
    elif args.model_type == 'trigram':
        trigram(args)
    
if __name__ == "__main__":
    main()