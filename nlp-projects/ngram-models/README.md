# N-Gram Models

## Description

This project trains and evaluates n-gram probabilistic language models on a subset of the CHILDES corpus, which contains transcripts of conversations between children and adults. The raw data is cleaned to remove punctuation, non-word utterances, etc. After cleaning, the data is transformed to use [ArpaBET](https://en.wikipedia.org/wiki/ARPABET) to represent the utterances. For example, the sentence "What's that" is transformed to "W AH T S TH AE T". Each utterance is marked with a start tag `<s>` and an end tag `</s>` to indicate the beginning and end of an utterance. Out-of-vocabulary words are replaced with the `<UNK>` token if the word is not found in the [CMU dictionary](http://www.speech.cs.cmu.edu/cgi-bin/cmudict).

The data cleaning and transformation process was completed in a previous project and the transformed data is stored in the [transformed](transformed) directory. Before training, the transformed data is split into a training and development set, with 80% of the utterances in the training set and 20% in the development set. Model training works by counting the word occurrences in the training set and then computing conditional probabilities based on these counts. After training the [n-gram model](https://en.wikipedia.org/wiki/Word_n-gram_language_model#Unigram_model) (unigram, bigram, or trigram), the trained model is evaluated on the development set by computing the perplexity of the given data.

[Perplexity](https://en.wikipedia.org/wiki/Perplexity#Perplexity_of_a_probability_model) is the standard intrinsic metric for measuring language model performance and measures how well the model predicts the data. The lower the perplexity, the better the model predicts the development set. The program supports unigram, bigram, and trigram models (see Program Execution below). The bigram and trigram models can be trained with or without [Laplace smoothing](https://en.wikipedia.org/wiki/Additive_smoothing) (see Program Execution below).

## Instructions

### 1. Setup Virtual Environment (Optional)

To run the program you can set up a virtual environment by executing the following commands:

`python3 -m venv venv`

Activate the virtual environment:

`source venv/bin/activate`

### 2. Packages

Our program only uses packages that are part of the Python Standard Library. No external libraries are allowed so there is no need to install any additional packages.

The following packages (from the Python Standard Library) are used in the program:

- `collections`
- `pathlib`
- `math`
- `random`
- `argparse`

### 3. Python Version

The Python version we used on the lab machine is 3.8.10

### 4. Program Execution

Here is the general format for running the program:

`python3 src/main.py <model> <training_set> <dev_set> [--laplace] [--partition]`

- The first command line argument (`<model>`) is the n-gram model to train. There are three possible options: `unigram`, `bigram`, and `trigram`.
- The second command line argument (`<training_set>`) is the path to the training data. (e.g., `data/training.txt`)
- The third command line argument (`<dev_set>`) is the path to the data for which perplexity will be computed. (e.g., `data/dev.txt`)
- The `--laplace` flag is optional and indicates that the model should be trained with Laplace smoothing. If the flag is not provided, the model will be trained without smoothing.
- The `--partition` flag is optional and will re-partition the transformed data and write it into the `data/training.txt` and `data/dev.txt` files. If the flag is not provided, the program will not partition the dataset and will use the existing `data/training.txt` and `data/dev.txt` files.
- If the model is unigram, it is invalid to provide the `--laplace` flag.
- The program MUST be run from the root directory of the project.

For example, to run the program to train a bigram model with Laplace smoothing, execute the following command in the root directory of the project:

`python3 src/main.py bigram data/training.txt data/dev.txt --laplace`

To run an unsmoothed trigram model with a re-partitioned dataset, execute the following command:

`python3 src/main.py trigram data/training.txt data/dev.txt --partition`

Attempting to run the program with an invalid command line argument will result in an error message which will indicate the correct usage of the program.

## Input

- `data/training.txt`: The training set for the language model.
  - Contains 80% of the utterances from the transformed dataset.
  - Each line contains an utterance with ArpaBET symbols.
  - `<s>` and `</s>` are used to denote the start and end of an utterance.
  - Example:
    ```
    ...
    <s> M AH NG K IY </s>
    <s> W AH T S DH IH S </s>
    <s> L EH T S S IY </s>
    ...
    ```
- `data/dev.txt`: The development set for evaluating the language model.
  - Contains 20% of the utterances from the transformed dataset.
  - Each line contains an utterance with ArpaBET symbols.
  - `<s>` and `</s>` are used to denote the start and end of an utterance.
  - Example:
    ```
    ...
    <s> M AH NG K IY </s>
    <s> W AH T S DH IH S </s>
    <s> L EH T S S IY </s>
    ...
    ```

## Output

The program outputs the calculated perplexity of the dev set using the trained n-gram model. 

For example, when we run the program with the following command:

`python3 src/main.py bigram data/training.txt data/dev.txt --laplace`

The output will be the calculated perplexity value rounded to four decimal places:

```
14.9888
```

For unsmoothed bigram and trigram models, calculating the perplexity on the dev set will result in infinite perplexity due to zero probabilities. In this case, the program will output:

```
inf
```

## Files

- `src/main.py`: The main program that partitions the dataset, trains the n-gram model and evaluates the model on the dev set.
- `data/training.txt`: The training set for the language model (see Input Data).
- `data/dev.txt`: The development set for evaluating the language model (see Input Data).
- `report.md`: Contains implementation details with justification for the design choices.
- `transformed/`: A directory containing the transformed data which is partitioned into training and dev sets.

## Evaluation

| Model   | Smoothing  | Training set PPL | Dev set PPL |
| ------- | ---------- | ---------------- | ----------- |
| unigram | -          | 30.9461          | 30.9561     |
| bigram  | unsmoothed | 14.9937          | inf         |
| bigram  | Laplace    | 14.9953          | 14.9888     |
| trigram | unsmoothed | 7.8604           | inf         |
| trigram | Laplace    | 7.9452           | 7.9922      |

## Sources

- https://docs.python.org/3/library/pathlib.html
- https://switowski.com/blog/pathlib/
- https://www.geeksforgeeks.org/writing-to-file-in-python/
- https://stackoverflow.com/questions/12488722/counting-bigrams-pair-of-two-words-in-a-file-using-python
- https://web.stanford.edu/~jurafsky/slp3/3.pdf
