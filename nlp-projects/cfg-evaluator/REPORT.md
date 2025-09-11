# Report

## Precision and Recall

Precision: 0.3013 \
Recall: 0.8129

---

|                | ground_truth = 1 | ground_truth = 0 |
| -------------- | ---------------- | ---------------- |
| prediction = 1 | TP = 113         | FP = 262         |
| prediction = 0 | FN = 26          | TN = 202         |

---

## Analysis

- Our grammar prioritizes higher recall over precision. For a grammar checker, we believe it is more important to identify as many grammatically incorrect sentences as possible, even if it means occasionally flagging some grammatically correct sentences as incorrect. In practice, users would likely only activate the grammar checker when they are unsure about the correctness of a few of their sentences. It wouldn't matter if grammatically correct sentences are flagged if the user is already confident in their correctness. However, if the grammar checker prioritized precision over recall, it would be less useful, as it would miss many grammatically incorrect sentences and fail to do its job. We believe it is less important for a grammar checker to correctly identify grammatically correct sentences that the user already knows to be grammatically correct.

- We decided against adding any rules that would increase precision at the cost of recall. We experimented with various general rules, such as `VP -> VP VP`, which would only marginally improve precision while significantly reducing recall. We tried other grammars that aimed to increase the total number of true negatives, but they were unsuccessful; increasing true negatives also raised the number of false negatives, resulting in decreased recall.

- Additionally, we implemented rules designed to generalize better to any future additions to the corpus. Given that our training data is relatively small (603 sentences), these rules did not alter the precision or recall. However, they would likely be more beneficial if the grammar were applied to a larger corpus. For example, the rule `VBZ PP` does not increase precision or recall, but it can successfully parse sentences like "She studies at night". 

## Questions

### Why did our grammar produce false negatives?

- Many false negatives can be attributed to overly permissive rules in our grammar. While these rules are needed to correctly label true negatives, they also lead to a significant number of sentences being incorrectly labelled as false negatives.
For example, having the rule `NP -> CC PRP`, allows grammatically correct sentences like "So you have wasted hours buying nothing." to be predicted as valid, but it also permits grammatically incorrect sentences such as "or you are going to get nervous!" to be incorrectly labelled as grammatically correct. Other overly permissive rules include `NP -> NP NP` or `VP -> VP VP`, which can incorrectly parse ungrammatical sentences.

- The second reason for the occurrence of false negatives is the use of certain tag sequences that can work both ways. We found some sequences, such as `PRP Verb RB RB`, which appear in both grammatically correct and incorrect sentences. For instance, "I don't really like staying in tents" is grammatically correct, but "I am not also keen on preparing it" is grammatically incorrect. The choice of words and the order of adverbs can make a sentence grammatically incorrect.

- A third reason for false negatives is mislabelling. Some sentences that are seemingly grammatically correct can be semantically ambiguous, leading to incorrect labelling. Ambiguous rules can lead to multiple parse trees for a sentence. For example, the combination of rules `NP -> PRP NP` and `NP -> NP PP` can create ambiguities and result in incorrect parsing. Consider the sentence "But it was worthy", this sentence is grammatically correct, but has been labelled incorrect.

### Why did our grammar produce false positives?

- The first reason is that our grammar does not cover enough cases. There are numerous uncommon sentence structures in the corpus, and identifying all of them would be very tedious. As a result, many grammatically correct sentences are not found. For example, the 
sentence "furthermore I simply love water sports and their challenges" is grammatically correct, but has a complex structure that begins with the POS tags `RB PRP RB` which is hard to detect and requires specific rules to cover.

- The second reason is that some grammatically correct sentences are too niche. Including the appropriate rules to cover these sentences would increase the number of false negatives. This is apparent for sentences that begin with adverbial phrases and prepositional phrases, which is why these rules were left out. The omitted rules are: \
`S -> ADVP NP VP PUNCT | ADVP COM NP VP PUNCT | ADVP VP NP PUNCT| PP NP VP PUNCT | ADVP COM PP PUNCT | ADVP COM NP PUNCT` 

- The third reason we encounter false positives is due to punctuations. Many grammatically correct sentences include punctuation, such as commas between phrases. For example, the sentence "Thirdly, the ad mentioned possible discounts", begins with an adverbial phrase followed by punctuation. "Her boss was a rich man, really rich", has a punctuation in the middle of the sentence. Trying to find punctuation in between phrases would require many more rules, adding complexity to the grammar and also introducing more false positives as a result. 

### With our current design, is it possible to build a perfect grammar checker?

- No, it is not possible to build a perfect grammar with the current design.

### If not, briefly justify your answer.

- English is not a context-free language. Natural languages are inherently complex and require correct context-sensitive parsing. They often contain ambiguities, meaning the same sentence structure can convey different meanings based on the context. These ambiguities make it difficult to capture all of the grammatical sentence structures needed to correctly parse the dataset. To build a context-free grammar, we would need to include many rules to generalize perfectly, which is not feasible. We would almost certainly find false positives if our grammar covered a large corpus. And if our grammar does not cover a large enough corpus, then there would be too many false negatives. The only way to build such grammar would be to represent every possible sentence in the corpus with a rule like  `S -> sentence `. Either the grammar will be too permissive, allowing grammatically incorrect sentences to be parsed, or it will be overly restrictive, rejecting grammatically correct but less common sentences. This creates a constant trade-off between being too permissive or too restrictive, which makes it impossible to build a perfect grammar. Additionally, there may be semantically incorrect and grammatically correct sentences, which would mean that a perfect grammar checker would have to understand the meaning of the sentence as well.

