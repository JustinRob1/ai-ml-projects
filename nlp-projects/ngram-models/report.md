# Design Choices

## Transformations

The transformations used were directly obtained from Assignment 1, with a few modifications. Based on the feedback received, the data was cleaned to remove empty lines to make the data consistent and more usable. Additionally, all previous `<NF>` tokens, used to identify symbols that were not found in the original CMU dictionary, were replaced by `<UNK>` tokens to standardize the handling of unknown words. This serves as the unknown symbol which is used to handle Out-Of-Vocabulary words that may appear in the dev set.

To distinguish different utterances, we use special begin-of-utterance symbols and end-of-utterance symbols, specifically `<s>` and `</s>` respectively. The processed data features one utterance per line, for example, the utterance `HH AY F AH N EY T IH D K AA M P AW N D W ER D`, becomes the modified utterance with the special symbols `<s> HH AY F AH N EY T IH D K AA M P AW N D W ER D </s>`. This approach helps avoid excessive special symbols in the training corpus, which could inflate their counts unnecessarily.

## Zero Probabilities

For the unsmoothed bigram and trigram model, we decided that it is appropriate to only catch division by zero errors, that occur when calculating perplexity on the dev set. We inevitably run into these errors, as we may find sequences in the dev set that were not found in the training set, which would be assigned zero probability, resulting in infinite perplexity. Since a perfectly unsmoothed model, would never be able to handle these, we decided that catching the error and reporting infinite perplexity would be the most appropriate solution. The training set perplexity for the unsmoothed bigram and trigram models will be finite, since we are testing on the same data that the model was trained on, there will be no unseen sequences.

We also consider three other methods to handle zero probabilities for the unsmoothed bigram and trigram models. However, these methods inevitably resulted in some form of smoothing, which would not be a true representation of an unsmoothed model. We decided to stick with the true unsmoothed model, as it provides a clear comparison between the smoothed and unsmoothed models. The methods we considered are as follows:

- Initially, for the unsmoothed bigrams and trigrams models, the unseen contexts were assigned a small non-negative (`1e-10`) probability. Although this would hinder the actual probability distribution, it prevented any division-by-zero errors and ensured that zero probabilities did not negatively impact the downstream perplexity calculations. However, we decided against this approach, as it would not provide a true unsmoothed model and would result in a finite perplexity value. 

- Another method we tried to handle zero probabilities was skipping the unseen context and not including it in the perplexity calculation. This approach would also not provide a true perplexity value thus affecting the overall probability distribution. Skipping unseen contexts would bias the model towards the training set and thus would be an inaccurate representation of the model's performance on unseen data.

- Finally, we attempted a more definited smoothing techniques with a backoff method. Introducing this smoothing method yielded similar results to Laplace smoothing, but it would defeat the purpose of distinguishing between the two methods.


## Additional Design Choices

We used the collections module to count occurrences of various ARPAbet symbols. We read the entire training corpus and provide it as a sequence to the Counter(). Doing so will create unnecessary bigrams of the form (`</s>`, `<s>`), and unnecessary trigrams such as (w, `</s>`, `<s>`) and (`</s>`, `<s>`, w). We delete these contexts while counting the probabilities as they do not exist in the actual corpus and would unintentionally decrease the model's perplexity.

Any OOV (Out-Of-Vocabulary) symbols found in the dev corpus are replaced by the `<UNK>` symbol, and probabilities are calculated after the replacement. This is the main method described in the lectures, and we use this method to address Key Errors when getting probabilities from dictionaries. We feel this method is simpler to implement then a backoff mechanism and still allows for proper probabilities.

To prevent numerical underflow that can occur when multiplying small probabilities, we utilized log probabilities, as discussed in the lectures. We rounded the perplexity values to four decimal places to display the results in a more readable format.

## Program Execution

We decided to add the optional flag (`--partition`) to the program, allowing users to re-partition the data from the `transformed/` directory. This enables users to reset the dataset partition if there are any changes to the dataset. The re-partitioned data will be written to the `data/training.txt` and `data/dev.txt` files. If the flag is not provided, the program will not re-partition the dataset and will use the existing `data/training.txt` and `data/dev.txt` files. The addition of the `--partition` flag displays the completion of Task 1 in the assignment. 
