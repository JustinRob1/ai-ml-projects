|   Student name   |   CCID   |
|------------------|----------|
| Justin Robertson | jtrober1 |
| Jejoon Ryu       | jejoon   |

## Assumptions

- As per the specification, when any error is encountered, the error message will be printed and the program will exit.

- We assume the training data is in the same format as per the specification: { "category": "class_name", "text": "text in one news story"} and consists of valid strings

## Implementation Details

- We created the program normalize.py to normalize the data. This program is imported into our classifier programs and is used to normalize the data before it is used in the classifier. The normalization includes removal of punctuation, converting words to lowercase, tokenization and using the WordNetLemmatizer to lemmatize the words.

- As per the specificatiom, when training the classifier, the program will ask the user if they want to overwrite the existing csv file. If the user enters 'y' or 'Y', the program will overwrite the existing csv file. If the user enters 'n' or 'N', the program will not overwrite the existing tsv file. If the user enters anything else the program will exit.

- We created the program doc_vec.py for the rocchio and knn classifiers that will create the document vectors for the training and test data. This program is imported into our classifier programs and is used to create the document vectors.

- The vectors that are written to the tsv file are in the format: {term1:idf_score1, term2:idf_score2, ...}. This is the format that the rocchio and knn prediction programs expect the vectors to be in.


## Algorithms and Data Structures

- In the nbc_train.py program we create an instance of the class MultinomialNBTrainer using the training data. The method train() is called on the instance which first normalizes the training data using the normalize.py program. We then iterate through the training data and extract the class and text for each class and store them in a set. A default dictionary is created to store conditional probabilities for the terms for each class. We then iterate through the set of classes and for each class we iterate through the text associated with that class and calculate the conditional probabilities for each term in the text, using add-one smoothing to avoid zero probabilities. After the train() method is compeleted, the conditional probabilities are written to the specificed tsv file.
We use a set to store the classes and text because we want to ensure that there are no duplicates. We use a default dictionary to store the conditional probabilities because we want to ensure that there are no missing keys. 

- In the nbc_prediction.py program we first read in the tsv file that was created by the nbc_train.py program by using the read_model() function. The function iterates through the tsv file and stores the class names into a list and the unique terms into a set. The conditional probabilities of the terms and the priors are both stored in dictionaries. Next we create an instance of the class MultinomialNBCPredictor using our extract data. Next we call apply on the instance for each document in the test data which applies the Naive Bayes algorithm to predict the class of the document. The predicted class and actual class are stored in a list. Finally we use the list of predicted and actual classes to calculate and print the necessary metrics for the classifier.

- In the rocchio_train.py program we first normalize the training data and the iterate through the text and calculate the df for each term. The idf is then calculate and add to our instance DocumentVector where each DocumentVector is a dictionary of term to tf-idf weight. The document vectors are added to a dictionary and the idf wights are added to a dictionary. An instance of our class RocchioTrainer is create and then we apply the method train() which returns a dictionary of the centroids for each class. The idf weight for every class along with the centroids for each class are written to the tsv file.

- In the rocchio_prediction.py program we first read in the tsv file that was created by the rocchio_train.py program by using the read_model() function and store the centroids and the idf scores of the terms into lists. The test data is retrieved, normalized, stored as a list, and coverted to DocumentVectors using ltc tf-idf weights. We then create an instance of our class RocchioPredictor using our centroids and idf scores. We then iterate through our test data, storing the expected class into a list and applying the function apply() which returns the class with the smallest euclidean distance to the centroid. The predicted class is store in a list and we use these two lists to calculate and print the necessary metrics for the classifier.

- The knn_create_vectors.py program is nearly identical to the rocchio_train.py program but instead of calculating the centroids, we use a dictionary that stores the DocumentVectors for each document in the training data. The apply method is applied which loops through each class and for each class it checks whether the DocumentVector has been added to the dictionary so we don't add duplicates. The idf weights are written to the tsv file and we iterate through the dictionary and write the vectors to the tsv file.

- The knn_prediction.py program is nearly identical to the rocchio_prediction.py program but instead of computing the nearest centroid, we compute the k nearest neighbour of each document in the test data. This is achieved by calculating the Euclidean distance between the test document vector and each training document vector and then returning the k training document vectors with the shortest distance. The results are store in a heap to quickly retrieve the k nearest neighbours. The predicted class is then determined by calculating the class probailities of these neighbours and returns the class with the highest probability.

## Error Handling

- Our programs check if the correct number of arguments are passed in.

- Our programs check if the file paths are valid and exist.

- Our programs check if the input file is a valid tsv file.

- Our programs check if the input file is a valid json file.

- Our programs check if the user wants to overwrite the existing tsv file.

- The knn_prediction.py program checks if the user enters positive integer for k.

## Sources

- https://realpython.com/knn-python/
- https://www.geeksforgeeks.org/-k-nearest-neighbor-algorithm-in-python/

## Instructions

### 1. Install Dependencies

If you don't have pipenv installed, install pipenv:
```sh
$ pip install pipenv
```

Install dependencies from Pipfile:
```sh
$ pipenv install
```

At this point, a Python virtualenv is automatically created. To activate the virtualenv, run:
```sh
$ pipenv shell
```
Download nltk data for normalization:
```sh
$ python -m nltk.downloader popular
```

### 2. Run the Program

Make sure that you are in the root directory of the project to run these examples. Absolute or relative paths can also be used.

#### 2.1. Naive Bayes Classifier

To run the Naive Bayes Classifier Training Program, run the following command:
```sh
python3 ./nbc/nbc_train.py ./data/train.json ./nbc_bbc_model.tsv
```

The tsv file nb_bbc_model.tsv will be created in the root directory of the project which will be used by the prediction program.

To run the Naive Bayes Classifier Prediction Program, run the following command:
```sh
python3 ./nbc/nbc_prediction.py ./nbc_bbc_model.tsv ./data/test.json 
```

A table will be printed to the console with the true positive, false positive, false negative and true negative counts, as well as the precision, recall and F1 score for each class. Below the table the micro-averaged F1 and macro-averaged F1 scores will be printed.

#### 2.2. Rocchio Classifier

To run the Rocchio Classifier Training Program, run the following command:
```sh
python3 ./rocchio/rocchio_train.py ./data/train.json ./rocchio_bbc_model.tsv
```

The tsv file rocchio_bbc_model.tsv will be created in the root directory of the project which will be used by the prediction program.

To run the Rocchio Classifier Prediction Program, run the following command:
```sh
python3 ./rocchio/rocchio_prediction.py ./rocchio_bbc_model.tsv ./data/test.json 
```

A table will be printed to the console with the true positive, false positive, false negative and true negative counts, as well as the precision, recall and F1 score for each class. Below the table the micro-averaged F1 and macro-averaged F1 scores will be printed.

#### 2.3. kNN Classifier

To run the kNN Classifier Training Program, run the following command:
```sh
python3 ./knn/knn_create_vectors.py ./data/train.json ./knn_bbc_vectors.tsv
```

The tsv file knn_bbc_vectors.tsv will be created in the root directory of the project which will be used by the prediction program.

To run the kNN Classifier Prediction Program, run the following command:
```sh
python3 ./knn/knn_prediction.py ./knn_bbc_vectors.tsv ./data/test.json k
```
where k is the number of nearest neighbours to use. A table will be printed to the console with the true positive, false positive, false negative and true negative counts, as well as the precision, recall and F1 score for each class. Below the table the micro-averaged F1 and macro-averaged F1 scores will be printed.

## kNN analysis

Report the F1 for each class separately, as well as the micro and macro averages for all classes.

| Class         | k=1     | k=5     | k=11    | k=23    |
|---------------|---------|---------|---------|---------|
| tech          | 0.93333 | 0.93333 | 0.95455 | 0.91304 |
| business      | 0.96000 | 0.96000 | 0.96000 | 0.96000 |
| sport         | 1.00000 | 0.98113 | 1.00000 | 0.98113 |
| entertainment | 0.97436 | 0.97436 | 1.00000 | 0.94737 |
| politics      | 1.00000 | 0.97561 | 1.00000 | 0.97561 |
|---------------|---------|---------|---------|---------|
| ALL-micro     | 0.97368 | 0.96491 | 0.98246 | 0.95614 |
| ALL-macro     | 0.97354 | 0.96489 | 0.98291 | 0.95543 |

Report the Precision for each class separately.

| Class         | k=1     | k=5     | k=11    | k=23    |
|---------------|---------|---------|---------|---------|
| tech          | 0.87500 | 0.87500 | 0.91304 | 0.84000 |
| business      | 1.00000 | 1.00000 | 1.00000 | 1.00000 |
| sport         | 1.00000 | 0.96296 | 1.00000 | 0.96296 |
| entertainment | 1.00000 | 1.00000 | 1.00000 | 1.00000 |
| politics      | 1.00000 | 1.00000 | 1.00000 | 1.00000 |
|---------------|---------|---------|---------|---------|

Report the Recall for each class separately.

| Class         | k=1     | k=5     | k=11    | k=23    |
|---------------|---------|---------|---------|---------|
| tech          | 1.00000 | 1.00000 | 1.00000 | 1.00000 |
| business      | 0.92308 | 0.92308 | 0.92308 | 0.92308 |
| sport         | 1.00000 | 1.00000 | 1.00000 | 1.00000 |
| entertainment | 0.95000 | 0.95000 | 1.00000 | 0.90000 |
| politics      | 1.00000 | 0.95238 | 1.00000 | 0.95238 |
|---------------|---------|---------|---------|---------|


## Classifier Comparison

Report the F1 for each class separately, as well as the micro and macro averages for all classes.

| Class         | NBC     | Rocchio | kNN (k=11) |
|---------------|---------|---------|------------|
| tech          | 0.93333 | 0.87500 | 0.95455    |
| business      | 0.96000 | 0.93878 | 0.96000    |
| sport         | 1.00000 | 1.00000 | 1.00000    |
| entertainment | 0.97436 | 0.91892 | 1.00000    |
| politics      | 1.00000 | 1.00000 | 1.00000    |
|---------------|---------|---------|------------|
| ALL-micro     | 0.98246 | 0.94737 | 0.98246    |
| ALL-macro     | 0.98291 | 0.94654 | 0.98291    |

Report the Precision for each class separately.

| Class         | NBC     | Rocchio | kNN (k=11) |
|---------------|---------|---------|------------|
| tech          | 0.91304 | 0.77778 | 0.91304    |
| business      | 1.00000 | 1.00000 | 1.00000    |
| sport         | 1.00000 | 1.00000 | 1.00000    |
| entertainment | 1.00000 | 1.00000 | 1.00000    |
| politics      | 1.00000 | 1.00000 | 1.00000    |
|---------------|---------|---------|------------|


Report the Recall for each class separately.

| Class         | NBC     | Rocchio | kNN (k=11) |
|---------------|---------|---------|------------|
| tech          | 1.00000 | 1.00000 | 1.00000    |
| business      | 0.92308 | 0.88462 | 0.92308    |
| sport         | 1.00000 | 1.00000 | 1.00000    |
| entertainment | 1.00000 | 0.85000 | 1.00000    |
| politics      | 1.00000 | 1.00000 | 1.00000    |
|---------------|---------|---------|------------|
