|        Student Name         |   CCID   |
|-----------------------------|----------|
|student 1: Justin Robertson  | jtrober1 |
|student 2: Zhiyuan Yu        |   zyu6   |

Assumptions:
    - Each zone name is a single token (i.e., no spaces or punctuation are allowed in a zone name)<br>
    - The text associated with the zones are in English<br>
    - As per the specification, when any error is encountered the error message will be printed 
    and the program will exit<br>
    - Each document has must have a field called doc_id that must be an integer that is of type string<br>

Implementation Details:
    - We decided to keep stopwords since the example query of line:you is a stopword and it is useful to 
    to the users to be able to query for stopwords<br>
    - We decided to replace all punctuation with spaces to allow the user to query for tokens with punctuation
    such as "I'm", 'co-education', '3/20/91', etc.<br>
    - We applied case folding to all words since users are likely to query in all lowercase<br>
    - We used the Unidecode library to convert accented characters to their English equivalent to allow querying 
    of these words with accents<br>
    - We used the Porter Stemmer as it is the most common algorithm for stemming in English and in most cases it
    will improve querying<br>
    - Pharse queries are not allowed<br>
    - For querying.py the user must enter their query in single quotation marks in order for the program to
    parse the query correctly. If it is not entered in double quotation marks, the program will crash<br>
    - If the user enters a query that contains a token that does not exist in the index, the program will
    print nothing<br>
    - The index tokens in the index file are sorted in ascending order starting with numbers<br>
    - Our implementation does not support pharse queries<br>
    - To avoid conflicts, put each new index into it's own directory

Sources: 
    - https://stackoverflow.com/questions/49118619/outputting-a-list-to-a-tsv-file-in-python
    (for writing to tsv file)<br>

    - https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string
    (for removing punctuation)<br>

    - https://www.geeksforgeeks.org/python-stemming-words-with-nltk/
    (for Porter Stemmer)<br>

    - https://stackoverflow.com/questions/517923/what-is-the-best-way-to-remove-accents-normalize-in-a-python-unicode-string
    (for removing accents)<br>

## 1. Setup environment
We use three third-party libraries unidecode, nltk and boolean.py.
They can be installed using the following instructions.

1. python3 -m pip install --user virtualenv
2. python3 -m venv env
3. source env/bin/activate
4. Install the libraries
    - pip3 install unidecode
    - pip3 install nltk
    - pip3 install boolean.py
5. When you are done type: deactivate

### 2. Examples
1. How to run the index_creation.py program.
Usage: python3 index_creation.py <path to JSON file> <path to output directory>
```
cd w23-hw1-JustinRob1-master
cd src
python3 index_creation.py ../data/dr_seuss_lines.json ../dr_seuss_lines_index
```
After executing these instructions, the index files are generated in w23-hw1-JustinRob1-master/dr_seuss_lines_index.
The index files should be put into their own directory to avoid conflicts with other index files
Absolute or relative paths can be used for the output directory

2. How to run the querying.py program
Usage: python3 querying.py <path to output directory> <boolean query>
```
cd w23-hw1-JustinRob1-master
cd src
python3 querying.py  ../dr_seuss_lines_index "book:Lorax OR (book:you AND line:you)"
```
After executing these instructions, the related document ids are output to standard output in the ascending order.<br>
For the above example, the output is as follows.<br>
```
0
2
4
```

### 3. Index creation
**3.1. Error checking**
Our program checks whether the number of command line parameters equals 3. If not, the program output errors like this.
```
Error: Wrong number of arguments
Usage: python3 index_creation.py <path to JSON file> <path to output directory>
```

Our program checks whether the given file exist. If not, the program outputs `Error: The file does not exist.`.<br>

Our program checks whether the given file is a valid JSON file. If not, the program outputs `Error: Not a valid JSON file.`.<br>

Our program checks whether the file was open properly. If not, the program outputs `Error: Could not read/open the file.`.<br>

Our program asks the user if they want to overwrite the tsv file if it already exists. If the user does not want to overwrite the file, the program exits.<br>

Our program checks if any doc_id is missing. If not, the program outputs `Error: Missing doc_id.`.<br>

Our program checks if any doc_id is duplicated. If not, the program outputs `Error: Duplicate doc_id.`.<br>

Our program checks that each zone only consists of alphanumeric characters. If not, the program outputs `Error: Invalid zone name.`.<br>

Our program checks that every document has at least one zone. If not, the program outputs `Error: Missing zone.`.<br>


**3.2. Normalization/Tokenization**

Iterate through each document in the JSON file and for every document, iterate through each zone. For the text in each zone, we do the following:
- Replace all punctuation with spaces
- Tokenize the text
- Convert all accented characters to their English equivalent
- Stem the words using the Porter Stemmer
- Convert all characters to lowercase
- Remove any non-alphanumeric characters

**3.3. Indexing**

Once each document in each zone is tokenized and it is then added to the index. The index is a dictionary where the key is the token and the value is a dictionary where the key is the doc_id and the value is the postings list which contains all the doc_id's that the token appears in. The index is written to a tsv file in the following format:
```
token \t doc_id \t postings_list
```

### 4. Querying
**4.1. Error checking**

Our program checks whether the number of command line parameters equals 3. If not, the program output errors like this.
```
Error: Wrong number of arguments
Usage: python3 querying.py <path to output directory> <boolean query>
```   
Our program checks whether the given index directory exist. If not, the program outputs `Error: The given index directory does not exist.`.<br>

Our program checks whether the given index directory is indeed a directory. If not, the program outputs `Error: The given index directory is not a directory.`.<br>

Our program checks whether the query string is formed well. If not, the program outputs `Error: The query is not properly formed.`.<br>

Our program checks whether each term has the zone prefix. If not, the program outputs `Error: Invalid boolean query.` 
`Boolean query should be in the form of <zone>:<query>`.<br>

Our program checks whether the index file exists. If not, the program outputs `Error: The given index file does not exist.`.<br>

In addition, if the query string contains unsupported boolean predicate, it outputs `ERROR: The boolean query only supports And, OR, Not and ()`.<br>

**4.2. Parse the boolean query**

We use boolean.py to parse the boolean query string. This is allowed to use in this assignment. Then we use the return value to build our custom boolean tree.

**4.3. Execute the query**

In order to the query, we define a boolean tree. It has four types of nodes, which are derived classes of `Component`. <br>
- The first type is `Leaf`, which represents the atomic query. <br>
- The second type is `AndQuery`, which represents the intersection of two query result. <br>
- The third type is `OrQuery`, which represents the union of two query result. <br>
- The fourth type is `NotQuery`, which represents the complement of query result. <br>   

The program calls the `query` method of root node. Then the code recursively calls the `query` method of children. By this way, the program gets the query result.<br>
Except for the document index, the `query` method takes one extra parameter called max_set. It is used in the NOT operation. <br>
In order to calculate complement set, we need a complete set which is all documents. However, AND operator shrinks the complete set. Hence, this parameter is used to accelerate the speed of NOT operation.
   
