# Naive Bayes Classifier

## Description

This program builds a Naive Bayes model based on bag-of-word (BoW) features to classify the relation of a sentence. Given a sentence containing two entities (called head and tail), the goal of our model is to classify the relation between the head entity and the tail. For example, from the sentence “Newton served as the president of the Royal Society”, the relation “is a member of” between the head entity “Newton” and the tail entity “the Royal Society” can be extracted. The sentences are classified into one of the following 4 relations: `publisher`, `director`, `performer`, and `characters`.

Training data is read from a CSV file (`data/train.csv`), preprocessed, and used to train the Naive Bayes model. The Naive Bayes model is trained on the data using a [3-fold cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)) and the training accuracy is reported. The trained NB model is then used to predict the relations between the head and tail entities in the test data, which is read from a CSV file (`data/test.csv`). The test accuracy is reported and the predictions are written to an output CSV file (`output/test.csv`). In addition, a confusion matrix is generated, along with the precision and recall for each relation. The micro-average precision and macro-average precision are also reported.

## Instructions

### 1. Setup Virtual Environment

To run the program you must set up a virtual environment by executing the following command:

`python3 -m venv venv`

Activate the virtual environment:

`source venv/bin/activate`

### 2. Python Version

The Python version we used on the lab machine is 3.8.10

### 3. Install Required Libraries

To install the required libraries, execute the following commands:

`pip install pandas numpy`

### 4. Program Execution

The program `src/main.py` receives the following three command line arguments (flag followed by the command line argument):
1. `--train`: Path to the training data file. (e.g. `data/train.csv`)
2. `--test`: Path to the test data file. (e.g. `data/test.csv`)
3. `--output`: Path to the output file. (e.g. `output/test.csv`)

The specified command line arguments must follow the flag and must be separated by a space.
The command line arguments can be in any order as long as the command line argument follows the flag.
All command line arguments are required.
The commands must be executed in the root directory of the project.

General format:

`python3 src/main.py --train <train_file> --test <test_file> --output <output_file>`

Example usage:

`python3 src/main.py --train data/train.csv --test data/test.csv --output output/test.csv`

This command will train a Naive Bayes model on the training data (using 3-fold cross validation), predict the relations between the head and tail entities in the test data, and write the predictions to the output file.
## Input

The input data is adapted from the [FewRel](https://aclanthology.org/D18-1514/) dataset. The training data is in the form of a CSV file (`data/train.csv`) and the test data is in the form of a CSV file (`data/test.csv`). The training and test data files contain the following columns:
| Column      | Example         | Description       |
| ----------- | --------------- | ----------------- |
| row_id      | 435             | The unique row id |
| tokens      | Trapped and Deceived ) is a 1994 television film directed by Robert Iscove . | The tokenized sentence, separated by a single space. |
| relation    | director        | The correct relation (original label) |
| head_pos    | 0 1 2           | Position of the head entity (Trapped and Deceived). Indices start with 0 and are separated by a single space. |
| tail_pos    | 11 12           | Position of the tail entity (Robert Iscove) |


Example training/test data:
```
row_id,tokens,relation,head_pos,tail_pos
1046,"He openly thanked and acknowledged her in the liner notes of OneRepublic 's debut album , "" Dreaming Out Loud "" .",performer,17 18 19,11
817,"Actress Ava Gardner had 14 dresses created for her in 1956 by Christian Dior for the Mark Robson film "" The Little Hut "" .",director,20 21 22,16 17
1165,"It also features a song by Melanie Blatt and Artful Dodger called "" TwentyFourSeven "" .",performer,13,6 7
...
```

## Output

The program’s output file is a CSV file (`output/test.csv`) that contains the following columns: `original_label`, `output_label`, and `row_id`. The `original_label` column contains the correct relation from the test data, the `output_label` column contains the predicted relation from our trained Naive Bayes model, and the `row_id` column contains the unique row id from the test data.

Example output:
```
original_label,output_label,row_id
director,director,766
director,director,1621
characters,characters,524
...
```

The program prints the training accuracy, test accuracy, confusion matrix, precision and recall for each relation, micro-averaged precision, and macro-averaged precision to the console.

For example, the following command:
`python3 src/main.py --train data/train.csv --test data/test.csv --output output/test.csv`

Will output to the console:
```
Cross Validation Accuracy: 88.9997%
Test Accuracy: 89.2500%

True\Pred       director        characters      performer       publisher
director        83              6               3               2
characters      6               88              5               4
performer       5               5               91              2
publisher       1               2               2               95

                Precision       Recall
characters      0.8713          0.8544
director        0.8737          0.8830
performer       0.9010          0.8835
publisher       0.9223          0.9500

Micro-averaged Precision: 0.8925
Micro-averaged Recall: 0.8925

Macro-averaged Precision: 0.8921
Macro-averaged Recall: 0.8927
```

## Files

- `src/main`: The main program that trains the Naive Bayes model, predicts the relations between the head and tail entities in the test data, and writes the predictions to an output file.
- `data/train.txt`: The training data file that contains the row id, tokens, relation labels, and head and tail entity positions. See [Input](#input) for more details.
- `data/test.txt`: The test data file that contains the row id, tokens, relation labels, and head and tail entity positions. See [Input](#input) for more details.
- `output/test.txt`: The output file that contains the original relation labels, predicted relation labels, and row ids. See [Output](#output) for more details.
- `report.md`: Contains decision choices, confusion matrix, justification and error analysis.

## 3rd Party Libraries

* `main.py L:indices = np.arange(num_rows)` used `numpy` for shuffling the data.
* `main.py L:train_indices = np.concatenate([splits[j] for j in range(num_folds) if j != i])` used `numpy` for concatenating the splits in the cross-validation.
* `main.py L:print(f"Cross Validation Accuracy: {100 * np.mean(accuracies):.4f}%")` used `numpy` for averaging the accuracies in cross-validation.
* `main.py L:accuracies.append(correct_preds.mean().item())` used `numpy` for calculating the accuracy in cross-validation.

## Sources

- [https://stackoverflow.com/questions/22588316/pandas-applying-regex-to-replace-values](https://stackoverflow.com/questions/22588316/pandas-applying-regex-to-replace-values)
- [https://www.w3schools.com/python/pandas/pandas_csv.asp](https://www.w3schools.com/python/pandas/pandas_csv.asp)
- [https://www.w3schools.com/python/pandas/ref_df_apply.asp](https://www.w3schools.com/python/pandas/ref_df_apply.asp)
- [https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html](https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html)
- [https://www.w3schools.com/python/pandas/ref_df_iloc.asp](https://www.w3schools.com/python/pandas/ref_df_iloc.asp)
- [https://www.geeksforgeeks.org/naive-bayes-classifiers/](https://www.geeksforgeeks.org/naive-bayes-classifiers/)