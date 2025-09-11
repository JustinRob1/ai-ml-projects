# Data Preprocessing and Transformation

This project is an exercise in data preprocessing and transformation for Natural Language Processing (NLP), using a subset of the CHILDES corpus. The goal is to clean, normalize, and phonetically transcribe child-adult conversational transcripts to prepare them for downstream NLP tasks such as language modeling.

## Summary

The code processes raw `.cha` files from the CHILDES corpus, performing the following steps:

1. **Cleaning**: Extracts and cleans dialogue lines, removing non-linguistic content, punctuation, and irrelevant metadata. The cleaned data is saved in the `clean/` directory, preserving the original folder structure.
2. **Transformation**: Converts cleaned text into phonetic representations using the CMU Pronouncing Dictionary (CMUDict). Out-of-vocabulary words are handled and logged. The transformed data is saved in the `transformed/` directory.
3. **Documentation**: All regex commands, justifications, and words not found in the dictionary are documented in the respective `.txt` files.

## Instructions

### 1. Setup Virtual Environment (Optional)

To run the program you can set up a virtual environment by executing the following commands:

`python3 -m venv venv`

Activate the virtual environment:

`source venv/bin/activate`


### 2. Libraries

Our program only uses libraries that are part of the Python Standard Library. No additional libraries are required.

### 3. Program Execution

To run the program, execute the following command in the root directory of the project:

`python3 src/main.py`

After execution, the program will generate two directories in the root directory, `/clean` and `/transformed`, containing the cleaned and transformed data, respectively. Both of these directories maintain the same directory strcuture as the input data, `/Data`.

## Data

The data can be found inside [Data](Data).

## Files

All the justifications for our project is present in justifications.txt .\
All the words not found in our transformations are present in not_found.txt .\
All the regex commands used and their justifications are present in regex_commands.txt .

## Sources

### [https://www.w3schools.com/python/python_regex.asp](https://www.w3schools.com/python/python_regex.asp)
### [https://docs.python.org/3/library/pathlib.html](https://docs.python.org/3/library/pathlib.html)
### [https://switowski.com/blog/pathlib/](https://switowski.com/blog/pathlib/)
