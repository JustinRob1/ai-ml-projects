# CFG Evaluator

This project implements a context-free grammar (CFG) evaluator for detecting grammatical errors in English sentences. The goal is to parse sentences using a custom CFG and identify whether each sentence is grammatically correct, supporting the development and evaluation of grammar checkers for NLP applications.

## Summary

The program reads a dataset of sentences with part-of-speech (POS) tags, parses each sentence using a chart parser and a user-defined CFG, and outputs predictions indicating whether each sentence is grammatically correct. It also computes and prints evaluation metrics such as confusion matrix, precision, and recall.

## Instructions

### 1. Setup Virtual Environment (Optional)

To run the program you can set up a virtual environment by executing the following commands:

`python3 -m venv venv`

Activate the virtual environment:

`source venv/bin/activate`

### 2. Libraries

Our program requires the use of the `nltk` module. Once the virtual environment is activated, install the required libraries by executing the following command:

`pip install nltk`

### 3. Program Execution

To run the program, execute the following command in the root directory of the project:

`python3 src/main.py data/train.tsv grammars/toy.cfg output/train.tsv`

After execution, the results will be store in the file [output/train.tsv](output/train.tsv). The tsv file contains the following columns:
- `id`: The id of the input sentence
- `ground_truth`: The ground truth label of the input sentence, copied from the dataset.
- `prediction`: 1 if the sentence has grammar errors, 0 if not. In other words, whether the POS sequence can be parsed successfully with our grammar and parser.

Example: 
```
id	ground_truth	prediction
457	0	0
193	0	0
...
```

Each row in the output file corresponds to a sentence in the input file. Additionally, a confusion matrix and precision and recall scores will be printed to the console to evaluate the performance of the grammar.

## Data

The assignment's train data can be found in [data/train.tsv](data/train.tsv).

The input data is a TSV file with the following columns:
- `id`: The unique id of the sentence.
- `label`: indicates whether a sentence contains grammar errors (1 means having errors and 0 means error-free).
- `sentence`: The original sentence which is already tokenized and space separated.
- `pos`: The part of speech tags for each token in the sentence, space separated.

Example:
```
id	label	sentence	pos
457	0	It 's very confusing without the right signs .	PRP VBZ RB JJ IN DT JJ NNS . 
193	0	We are restricted in doing nearly everything .	PRP VBP JJ IN VBG RB NN . 
...
```

## Files

- [data/train.tsv](data/train.tsv): Contains the training data (see above).
- [output/train.tsv](output/train.tsv): Contains the results of the program execution (see above).
- [src/main.py](src/main.py): The python program that reads the input data, parses the sentences using `nltk` chart parser with our toy grammar ([grammars/toy.cfg](grammars/toy.cfg)), and writes the results to the output file ([output/train.tsv](output/train.tsv)).
- REPORT.md: Inclues the precision and recall scores, result analysis and answers to the questions.
- [grammars/toy.cfg](grammars/toy.cfg): The toy grammar used to parse the sentences.
    - Example: 
    ```
    S  -> NP VP PUNCT
    PP -> ADP NP
    NP -> DET Noun | NP PP
    VP -> Verb NP | VP PP
    Verb -> VB | VBD | VBG | VBN | VBP | VBZ
    ......

    DET   -> 'DT'
    Noun  -> 'NN'
    VB    -> 'VB'
    ADP   -> 'ADP'
    PUNCT -> 'PUNCT'
    ......
    ```

## Python Version

The Python version we used on the lab machine is 3.8.10.

## Sources

- [https://www.nltk.org/howto/parse.html](https://www.nltk.org/howto/parse.html)
- [https://www.nltk.org/howto/grammar.html](https://www.nltk.org/howto/grammar.html)
- [https://www.nltk.org/data.html](https://www.nltk.org/data.html)
- [https://people.cs.umass.edu/~brenocon/cs485_f23/ex6_cfg.pdf](https://people.cs.umass.edu/~brenocon/cs485_f23/ex6_cfg.pdf)
- [https://www.geeksforgeeks.org/simple-ways-to-read-tsv-files-in-python/](https://www.geeksforgeeks.org/simple-ways-to-read-tsv-files-in-python/)