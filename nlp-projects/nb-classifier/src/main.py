import argparse
import os
import math
import numpy as np
import pandas as pd

def preprocess(args):
    """
    Preprocesses the training and test data
    """
    df = pd.read_csv(args.train)
    df['tokens'] = df['tokens'].str.lower()  # Just so we don't repeat counts a lot
    df['tokens'] = df['tokens'].str.replace(r'[^\w\s]', '', regex=True)  # Remove punctuation

    test_df = pd.read_csv(args.test)
    test_df['tokens'] = test_df['tokens'].str.lower()  # Just so we don't repeat counts a lot
    test_df['tokens'] = test_df['tokens'].str.replace(r'[^\w\s]', '', regex=True)  # Remove punctuation
    
    return df, test_df


def cross_validation(args, df):
    """
    Trains the Naive Bayes model using 3-fold cross validation
    """
    # Number of folds
    num_folds = 3
    num_rows = len(df.index)

    # Shuffle the indices
    indices = np.arange(num_rows)

    # Calculate fold sizes
    fold_sizes = [num_rows // num_folds + (1 if i < num_rows % num_folds else 0) for i in range(num_folds)] # [667,667,666]

    # Generate the splits
    splits = []
    current = 0
    for fold_size in fold_sizes:
        splits.append(indices[current:current + fold_size])
        current += fold_size

    # Accuracies to average
    accuracies = []

    # Combine two folds for training and leave one out for validation
    for i in range(num_folds):
        # Validation indices
        val_indices = splits[i]
        # Training indices: concatenate all splits except the one for validation
        train_indices = np.concatenate([splits[j] for j in range(num_folds) if j != i])
        
        # Get training and validation data
        train_df = df.iloc[train_indices].copy()
        val_df = df.iloc[val_indices].copy()

        # Naive Bayes Model based on bag-of-words features
        vocab, bag, word_counter, relation_counts = create_naive_bayes(train_df=train_df)

        # Naive Bayes on validation set with Laplpace Smoothing
        val_df['predictions'] = val_df['tokens'].apply(classify_tokens, vocab=vocab, bag=bag, word_counter=word_counter, relation_counts=relation_counts)
        correct_preds = val_df['predictions'] == val_df['relation']
        accuracies.append(correct_preds.mean().item())

    print(f"Cross Validation Accuracy: {100 * np.mean(accuracies):.4f}%")


def create_naive_bayes(train_df):
    bag = {}  # Counts how many times a word is found in relation
    word_counter = {}  # Counts the total number of words in a relation
    relation_counts = train_df['relation'].value_counts() / len(train_df.index)  # Counts prob dist of relations
    vocab = set((' '.join(train_df['tokens'])).split())  # Stores all unique words in the corpus

    # Iterate through each group and get bag of words for it
    for relation, group in train_df.groupby('relation'):
        tokens = ' '.join(group['tokens'])
        words = tokens.split()
        word_counts = pd.Series(words).value_counts()
        bag[relation] = word_counts  # c
        word_counter[relation]= sum(word_counts)  # N 

    return vocab, bag, word_counter, relation_counts


def classify_tokens(tokens, vocab, bag, word_counter, relation_counts):
    """
    Performs classification on a set of tokens using the Naive Bayes model
    """
    tokens = tokens.split()
    scores = {}

    # Calculate the probability of each relation given the tokens
    for relation in bag.keys(): 
        prob = 0
        for token in tokens:
            # conditional probability of each token
            if token not in vocab:
                continue    # drop OOV words
            count = bag[relation][token] if token in bag[relation].keys() else 0
            cond_prob = (count + 1) / (word_counter[relation] + len(vocab))
            # Use log probabilities to avoid underflow
            prob += math.log(cond_prob)
        # also the probability of the class
        prob += math.log(relation_counts[relation])
        scores[relation] = math.exp(prob)

    # Return the relation with the highest probability
    pred = max(scores, key=scores.get)    
    return pred

def test_naive_bayes(args, train_df, test_df):
    """
    Classifies the test data using the trained Naive Bayes model
    """
    # Train the Naive Bayes model
    vocab, bag, word_counter, relation_counts = create_naive_bayes(train_df)
    
    # Applies the classify_tokens function to each test case in the test data
    # Uses the 3-fold cross validation Naive Bayes model to predict the relation
    test_df['predictions'] = test_df['tokens'].apply(classify_tokens, 
                                                     vocab=vocab, 
                                                     bag=bag, 
                                                     word_counter=word_counter, 
                                                     relation_counts=relation_counts)
    
    # Calculate the accuracy of the model on the test data
    correct_preds = test_df['predictions'] == test_df['relation']
    accuracy = correct_preds.mean().item()
    print(f"Test Accuracy: {100 * accuracy:.4f}%")
    
    # Output the predictions to the csv file according to the format
    output_df = test_df[['relation', 'predictions', 'row_id']]
    output_df.columns = ['original_label', 'output_label', 'row_id']
    output_df.to_csv(args.output, index=False)

    # Output the confusion matrix and metrics
    true_labels = test_df['relation']
    pred_labels = test_df['predictions']
    relations = true_labels.unique()
    
    conf_matrix = get_conf_matrix(true_labels, pred_labels, relations)
    print("\nTrue\\Pred", end="\t")
    for label in relations:
        print(label, end="\t")
    print()
    
    # Output the precision and recall for each relation
    for true_label in relations:
        print(true_label, end="\t")
        for pred_label in relations:
            print(conf_matrix[true_label][pred_label], end="\t\t")
        print()
        
    # Output the precision and recall for each relation, micro-, macro-averaged precision and recall
    print_metrics(relations, conf_matrix)
    
        
def get_conf_matrix(true_labels, pred_labels, labels):
    """
    Returns a confusion matrix from the true and predicted labels
    """
    
    conf_matrix = {}
    # Initialize the confusion matrix
    for label in labels:
        conf_matrix[label] = {}
        for label2 in labels:
            conf_matrix[label][label2] = 0
            
    # Increment the correct cell in the confusion matrix for each set of true and predicted labels
    for true_label, pred_label in zip(true_labels, pred_labels):
        conf_matrix[true_label][pred_label] += 1
    
    return conf_matrix 
        
        
def print_metrics(labels, conf_matrix):
    """
    Outputs the relation precision, recall, micro-averaged precision/recall, macro-averaged precision/recall
    """
    precision = {}
    recall = {}
    
    # For calculating micro-averages
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    # For each label, calculate the precision and recall
    for label in labels:
        # tp when predicted = true
        tp = conf_matrix[label][label]
        # Sum all of the other values in the row 
        fp = sum(conf_matrix[other][label] for other in labels if other != label)
        # Sum all of the other values in the column 
        fn = sum(conf_matrix[label][other] for other in labels if other != label)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        # Calculate precision and recall for each relation
        precision[label] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[label] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
    # Print precision/recall row for each relation
    print(f"\n\t\tPrecision\tRecall")
    for label in sorted(labels):
        print(f"{label}\t{precision[label]:.4f}\t\t{recall[label]:.4f}")
        
    # Calculate micro-averaged metrics
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    print(f"\nMicro-averaged Precision: {micro_precision:.4f}")
    print(f"Micro-averaged Recall: {micro_recall:.4f}")
    
    # Calculate macro-averaged metrics
    macro_precision = sum(precision[label] for label in labels) / len(labels)
    macro_recall = sum(recall[label] for label in labels) / len(labels)
    print(f"\nMacro-averaged Precision: {macro_precision:.4f}")
    print(f"Macro-averaged Recall: {macro_recall:.4f}")
    

def main():
    parser = argparse.ArgumentParser()
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
    
    # Preprocess the data
    df, test_df = preprocess(args)
    # 3-fold cross validation
    cross_validation(args, df)
    # Test the Naive Bayes model (make predictions on the test data)
    test_naive_bayes(args, df, test_df)
    
if __name__ == "__main__":
    main()