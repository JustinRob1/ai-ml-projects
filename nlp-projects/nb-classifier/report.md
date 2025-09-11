## Design Choices

The main design choices we made are concerned with how to handle unknown words, apply smoothing techniques and preprocess the data. Any out-of-vocabulary (OOV) words in the test data are entirely skipped, which aligns with the lecture notes on this topic. We observed that some words might not be present in the training corpus for specific relations during the test phase, resulting in zero probabilities. To address this issue, we employed Laplace Smoothing, as discussed in our lectures. This method ensures that every word has a non-zero probability. For instance, the word "Metro" appears only once in the training corpus for the relation "performer." With smoothing, if "Metro" appears in the test data, it will receive a non-zero probability. We utilized Pandas to read the CSV files, making data manipulation and preprocessing straightforward. 

During preprocessing, we removed punctuation marks, as they did not contribute any context to the classification task and could distort the importance of meaningful tokens. In our testing, removing stop-words from the training data resulted in either worse performance or no improvement in accuracy, so we opted to keep stopwords in the training data. We used [NLTK's stopwords](https://www.geeksforgeeks.org/removing-stop-words-nltk-python/) but found that performance in terms of precision and recall for specific characters decreased significantly. All tokens were converted to lowercase, as maintaining a single count for each word yielded better results than accommodating different cases in the corpus. This adjustment also helps reduce the dimensionality of the feature space by eliminating duplicate words. Additionally, we did not tokenize the words because the Bag-of-Words model suffices for our Naive Bayes approach. These preprocessing steps are standard for a Bag-of-Words model, ensuring a consistent data representation.

For 3-fold cross-validation, we divided the training corpus into three separate splits, using two for training and one for validation in each iteration. When testing during inference, we utilized the entire training corpus to train the model. The 3-fold cross-validation method helps prevent overfitting and enhances the model's ability to generalize to unseen data. It provides a more conservative estimate of performance, especially when the data is imbalanced across different relations. We decided not to use the head and tails for preprocessing or any other tasks because we believe they do not add value for training a Naive Bayes Bag-of-Words model. To avoid numerical underflow, we employed log probabilities, as they can become very small. Using log probabilities enables us to perform summations instead of multiplications, which is computationally more efficient. Lastly, we do not tune any model parameters since our Naive Bayes model, based on Bag-of-Words features, is straightforward and lacks hyperparameters that require tuning.

## Results

### Accuracy

| Dataset      | Accuracy     |
| ------------ | ------------ | 
| Training     | 88.9997%     |
| Test         | 89.2500%     |

### Confusion Matrix

| True\Pred   | director | characters | performer | publisher |
| ----------- | -------- | ---------- | --------- | --------- |
| director    | 83       | 6          | 3         | 2         |
| characters  | 6        | 88         | 5         | 4         |
| performer   | 5        | 5          | 91        | 2         |
| publisher   | 1        | 2          | 2         | 95        |

### Precision and Recall

| Relation   | Precision | Recall  |
| ---------- | --------- | ------- |
| director   | 0.8737    | 0.8830  |
| characters | 0.8713    | 0.8544  |
| performer  | 0.9010    | 0.8835  |
| publisher  | 0.9223    | 0.9500  |

### Micro-Averaged, Macro-Averaged Precision and Recall

| Metric              | Precision     | Recall    |
| ------------------- | ------------- | --------- |
| Micro-Averaged      | 0.8925        | 0.8925    |
| Macro-Averaged      | 0.8921        | 0.8927    |

## Error Analysis

Out of the test corpus consisting of 400 test cases, only 43 were misclassified. The most common misclassifications occurred between characters and directors. Many characters that were misclassified as directors included either a director's or an author's name in the tokens, which explains the errors. For example, two of the six misclassifications involved the name "Jean," which is popular among directors; it appeared ten times in the training corpus, with the majority being directors. Conversely, directors misclassified as characters often had tokens that included the name of a character or actor, along with the role that they played in addition to being a director. A character sentence that discusses a movie may reference the director's name, leading to misclassification. For instance, the sentence "In Ridley Scott 's 1979 horror film "" Alien "" , Ellen Ripley and her crew attempt to capture an escaped Xenomorph with the help of a cattle prod ." was misclassified as a director when the correct label was character.

Characters and performers are also often confused, as the tokens for misclassified samples typically contain both the name of a performer and the character they portrayed, alongside a brief description of that character. We believe that the Naive Bayes model should ideally produce higher probabilities for both classes in most cases. An example is the sentence, "Odell in particular has mentioned James Hetfield of Metallica as his biggest influence in his guitar - work , mostly notably the track "" Sad But True "" .". which was misclassified as a performer when the correct label was character. Such examples, particularly those with music-related references, are more likely to pertain to performers. However, in the absence of distinct keywords, the model might struggle to classify them correctly. 

In general, some texts contain multiple roles, names, and descriptions that could be relevant to multiple classes. For example, "Following the release of "" Spider - Man 3 "" , Sony Pictures Entertainment had announced a May 5 , 2011 , release date for Sam Raimi 's next film in the earlier series .". This was classified as a publisher when the true label was director. The sentence includes both the name of a publishing studio and a director, likely contributing to the misclassification.

Most misclassifications where the predicted label was director included either a mention of the director's name and the film or contained the word director explicitly in the tokens. This can be observed in the sentence, "ring on the night is also a 1985 documentary directed by michael apted covering the formative stages of sting's solo career released as dvd in 2005" whose correct label is performer. Sentences lacking clear, relation-specific verbs (e.g., "directed by," "played," or "published") can be challenging to classify accurately. For characters misclassified as publishers, there isn't an obvious pattern for the misclassifications, except for one instance that mentioned both the publisher's name and the license under which it was released. Overall, some data consists of ambiguous tokens that can belong to multiple classes. For example, in the test set, the word "Queen" appears in all four categories.

Additionally, the training data consists of only 2000 samples, which we believe is insufficient to capture the full context of the relations. The [FewRel](https://aclanthology.org/D18-1514/) dataset contains very specific data from movies, books, and music, making accurate classification difficult. For instance, one misclassified sample— "In the "" One Piece "" manga , Luffy cited but changed it to "" Instead of saying what you like , say what you hate ! """ — was classified as a performer when the correct label was character. This specific example illustrates that the key character "Luffy" is not represented in the training data. Although simple, the bag-of-words model lacks syntactic structure, limiting its ability to capture the context of relations.