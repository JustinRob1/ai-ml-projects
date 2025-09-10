|        Student Name         |   CCID   |
|-----------------------------|----------|
|student 1: Justin Robertson  | jtrober1 |

Assumptions:
    - If the user wants to enter words with ' or " then they must enter the escape character \ before the ' or ".

    - The text associated with the zones and queries are in English

    - As per the specification, when any error is encountered the error message will be printed and the program will exit

    - Each document has must have a field called doc_id that must be an integer that is of type string

Implementation Details:
    - If a mixed query is entered and a phrase query is not in the index then it will be ignored

    - If a phrase query is entered but it is not in the index, the program will print no k results found.

    - For phrase queries, the query vector will contain all terms that appear in the query either as a keyword or inside a phrase.
    The query vector is then passed to the cosineScore function which will calculate the cosine score for each document in the pool.

    - The inverted index file will always be called index.tsv and the document lengths file will always be called doc.tsv

    - When indexing the files will automatically be overwritten if they already exist

    - To avoid conflicts when indexing put the files to be indexed in their own directory, not in the src or data directory

    - The program will automatically create the output directory if it does not exist

    - The index.tsv and doc.tsb file contain the doc_id as integers and not strings

    - Both the query and the inverted index are normalizaed and stemmed using the same process

    - The inverted index file is a dictionary where the keys are the tokens and the values are a list of tuples. Each tuple

    contains the document id and the positions of the token in the document. The positions are stored in a list.

    - We decided to keep stopwords since the example query of line:you is a stopword and it is useful to to the users to be able to query for stopwords

    - We decided to replace all punctuation with spaces to allow the user to query for tokens with punctuation such as "I'm", 'co-education', '3/20/91', etc.

    - We applied case folding to all words since users are likely to query in all lowercase

    - We used the Unidecode library to convert accented characters to their English equivalent to allow querying of these words with accents

    - We used the Porter Stemmer as it is the most common algorithm for stemming in English and in most cases it will improve querying



Algorithms and Data Structure: 
    - The algorthim used for indexing will iterate through the data in the JSON file and create a dictionary where the keys are the tokens and the values are a list of tuples. Each tuple contains the document id and the positions of the token in the document. The positions are stored in a list.

    - For the data structure of querying the inverted index is built into a dictionary where the keys are the tokens and the values are a list of tuples. Each tuple contains the document id and the positions of the token in the document. The positions are stored in a list.

    - The data structure for the document lengths is a dictionary where the keys are the document ids and the values are the document lengths.

    - For calculating the cosine score we iterate through the queries and calculate its weight. We then iterate through the documents and calculate its weight. The weighting is according to the apc.btn scheme. The score is then calculated by multiplying the query weight by the document weight.

    - For calculating the cosine score we store the top k scores in a min heap. The heap is implemented using a list. The list is initialized with k elements where each element is a tuple containing the document id and the score. The heap is then iterated through and if the score of the current document is greater than the score of the smallest element in the heap, the smallest element is removed and the current document is added to the heap. This process is repeated until all the documents have been processed. The heap is then sorted in descending order by score and then by document id.

    - The heap is the most efficient data structure since construction of the heap is O(N log k) where N is the number of documents and k is the number of results to return. The heap is also easy to convert to a list of tuples containing the document id and the score. The heap is also easy to sort in descending order by score and then by document id. It takes O(k log k) to get the top k results.

    - For the intersection of the two postings list we use the postingIntersection function. This function takes two postings lists and returns the intersection of the two lists. The function uses two pointers to iterate through the two lists. The function will iterate through the two lists until one of the pointers reaches the end of the list. The function will then compare the document ids of the two pointers. If the document ids are equal, the document id is added to the intersection list and the pointers are incremented. If the document id of the first pointer is less than the document id of the second pointer, the first pointer is incremented. If the document id of the second pointer is less than the document id of the first pointer, the second pointer is incremented. This process is repeated until one of the pointers reaches the end of the list. The function returns the intersection list.

    - This function is the most efficient since it only iterates through the two lists once. The function also uses the least amount of memory since it only stores the intersection list and not the two lists.

    - The dictionary is the most efficient data structure for indexing since it allows for fast lookups and insertion of new tokens. The dictionary is also easy to convert to a tsv file.


Sources: 
    - https://docs.python.org/3/library/heapq.html (Min Heap)
    - https://realpython.com/list-comprehension-python/ (List Comprehension)
    - https://stackoverflow.com/questions/49118619/outputting-a-list-to-a-tsv-file-in-python (for writing to tsv file)
    - https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string (for removing punctuation)
    - https://nlp.stanford.edu/IR-book/html/htmledition/positional-indexes-1.html (for positional indexes)
    - https://www.geeksforgeeks.org/python-stemming-words-with-nltk/ (for Porter Stemmer)
    - https://stackoverflow.com/questions/517923/what-is-the-best-way-to-remove-accents-normalize-in-a-python-unicode-string (for removing accents)

## 1. Setup environment
We use three third-party libraries unidecode, nltk and boolean.py.
They can be installed using the following instructions.

1. python3 -m pip install --user virtualenv
2. python3 -m venv env
3. source env/bin/activate
4. Install the libraries
    - pip3 install unidecode
    - pip3 install nltk
5. When you are done type: deactivate

### 2. Examples
1. How to run the index_creation.py program.
First change to the directory where the index.py file is located.
Usage: python3 index_creation.py <path to JSON file> <path to output directory>
```
python3 index_creation.py ../data/dr_seuss_lines.json ../dr_seuss_lines_index
```
After executing these instructions, the index files index.tsv and doc.tsv are generated in ../dr_seuss_lines_index.
The directory ../dr_seuss_lines_index is created if it does not exist.
The index files should be put into their own directory to avoid conflicts with other index files
Absolute or relative paths can be used for the output directory

2. How to run the query.py program.
First change to the directory where the query.py file is located.
Usage: python3 query.py <path to index directory> <path to query file> <path to output file> <number of results to return>
```
python3 query.py ../dr_seuss_lines_index 5 :get better: going :like you: lot
```

After executing these instructions, the following will be printed to  STDOUT, in this order:
```
1. One line with the number of documents that were considered for the query (see Step 1 below).
2. One line with the number of documents with non-zero scores among those that were considered for the query.
3. Up to k lines each with a document id and a non-zero similarity score (one besides the other, with the id first and a tab in between), sorted by decreasing score.
```

The above example will produce the following:
```
1
1
0 	 0.04604093769761876
```
Only 1 document id and score was printed since only 1 document had a non-zero score.

There are three possible queries allowed:
Keyword Queries contain only query terms separated by spaces (e.g. Lorax, cares better, etc.). 
Phrase Queries contain phrases delimited by colons (e.g., :someone like you cares:).
```
Eamples:
Keyword query: lot like you someone
Phrase query: :The Lorax:
Mixed query: :get better: going :like you: lot
```

3. Index creation
3.1. Error checking Our program checks whether the number of command line parameters equals 3. If not, the program output errors like this.

Error: Wrong number of arguments
Usage: python3 index_creation.py <path to JSON file> <path to output directory>
Our program checks whether the given file exist. If not, the program outputs Error: The file does not exist..

Our program checks whether the given file is a valid JSON file. If not, the program outputs Error: Not a valid JSON file..

Our program checks whether the file was open properly. If not, the program outputs Error: Could not read/open the file..

Our program checks if any doc_id is missing. If not, the program outputs Error: Missing doc_id..

Our program checks if any doc_id is duplicated. If not, the program outputs Error: Duplicate doc_id..

Our program checks that each zone only consists of alphanumeric characters. If not, the program outputs Error: Invalid zone name..

Our program checks that every document has at least one zone. If not, the program outputs Error: Missing zone..

3.2. Normalization/Tokenization

Iterate through each document in the JSON file and for every document, iterate through each zone. For the text in each zone, we do the following:

Replace all punctuation with spaces
Tokenize the text
Convert all accented characters to their English equivalent
Stem the words using the Porter Stemmer
Convert all characters to lowercase
Remove any non-alphanumeric characters

3.3. Indexing

Once each document in each zone is tokenized and it is then added to the index. The index is a dictionary where the key is the token and the value is a list of tuples. Each tuple contains the doc_id and the position of the token in the document. The index is then written to a tsv file. The tsv file is formatted as follows:
token \t doc_id \t postings_list

4. Querying
4.1. Error checking

Our program checks whether the number of command line parameters is valid. If not, the program output errors like this.

Error: Wrong number of arguments
Usage: python3 querying.py <path to output directory> <boolean query>

Our program checks whether the given index directory exist. If not, the program outputs Error: The given index directory does not exist.

Our program checks whether the index file exists. If not, the program outputs Error: The given index file does not exist.

Our program checks whether the files exists. If not, the program outputs Error: The given doc file does not exist.

Our program checks whether the query string is empty. If not, the program outputs Error: The query string is empty.

Our program checks whether k is a integer. If not, the program outputs Error: Error: Invalid k must be an int of arguments

Our program checks whether k is an positive integer. If not, the program outputs Error: Invalid k must be greater than 0

4.2. Parse the boolean query

We read the index.tsv file and doc.tsv file and store them in two dictionaries. The index dictionary has the token as the key and the postings list as the value. The doc dictionary has the doc_id as the key and the doc_name as the value.

4.3. Execute the query

The query is executed by first parsing the query string into a list of tokens. We then iterate though the tokens and create a pool of documents. We collect all documents containing at least one of the phrases in the query into a pool. For queries without phrases (i.e., only keywords) the pool should be all documents in the system. We then score the documents in the pool. The scoring should be according to Algorithm 7.1 in the textbook and follows the apc.btn "scheme". The query vector contains all terms that appear in the query either as a keyword or inside a phrase.


