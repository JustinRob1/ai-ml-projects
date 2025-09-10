import json
import sys
import os
import csv
import string
from unidecode import unidecode
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer  

# Description: Tokenizes the data in the JSON file by removing stopwords, punctuation and only
# keeping words that are all alphabetic characters.
# Arguments: file - an array of strings for a given zone in the JSON file
# Returns: an array of the tokenized words
def tokenization(file):
    stemmer = PorterStemmer()
    tokens = []  # Array of tokenized words
    new_file = file.translate(str.maketrans('', '', string.punctuation)) # Remove punctuation from the file
    token = word_tokenize(new_file) # Tokenize the file
    # For each token only append it to the array tokens if it is not a stopword, is all alphabetic characters
    # and is not already in the array tokens
    for word in token:
        word = unidecode(word) # Remove accents from the word
        word = stemmer.stem(word) # Stem the word
        # Check to see if the word is not a stopword, is all alphabetic characters
        # and is not already in the array tokens
        if word.isalnum():
            tokens.append(word.casefold()) 
    return tokens

# Description: Writes the token, the document frequency of the token and the array of the doc_id's to the tsv file
# Arguments: zone - the zone of the inverted index
#            dict - the dictionary of the index 
# Returns: None
def output_file(dict, file):
    # Open the file and write the token, the document frequency of the token
    with open(file, 'w', newline='') as f_output:
        tsv_output = csv.writer(f_output, delimiter='\t')
        file_name = file.split("/")[-1]
        if file_name == "index.tsv":
            tsv_output.writerow(["token", "DF", "postings"])
            # Sort the keys in the dictionary and write the token, the document frequency of the token
            # and the array of the doc_id's to the tsv file
            for key in sorted(dict.keys()):
                tsv_output.writerow([key, len(dict[key]), dict[key]])
        else:
            tsv_output.writerow(["doc_id", "length"])
            # Sort the keys in the dictionary and write the doc_id and the length of the document
            # to the tsv file
            for key in sorted(dict.keys()):
                tsv_output.writerow([key, dict[key]])

# Description: Creates the inverted index for each zone in the JSON file
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Error: Invalid number of arguments")
        print("Usage: python3 index_creation.py <path to JSON file> <path to output directory>")
        sys.exit(1)

    # Check to see if the file exists
    if not os.path.exists(sys.argv[1]):
        print("Error: File does not exist")
        sys.exit(1)

    # Try to open the JSON file
    try:
        file = open(sys.argv[1], 'r', encoding="utf8")
    except OSError:
        print("Error: Could not open/read file:")
        sys.exit(1)

    # returns JSON object as
    # a dictionary
    try:
        corpus = json.load(file)
    except Exception:
        print("Error: Not a valid JSON file")
        sys.exit(1)

    index_dict = {}  # Dictionary for the inverted indexes
    doc_dict = {}    # Dictionary for the documents
    id_arr = []      # Array of the doc_id's to check for duplicates
    prev = ""        # Variable to check if a zone is missing
    doc_str = ""     # A string to hold all the tokens of a document (tab seperated for each zone)
    doc_id = ""       # Variable to hold the doc_id  

    # Iterate through each document in the JSON file for the given zone and tokenize the data
    for i in range(len(corpus)):
        dFlag = True  # Flag to check for the doc_id
        # For each document in the JSON file, iterate through each zone of the document and tokenize the data
        for j in corpus[i].items():
            # Check to see if the zone name is alphanumeric
            # Check to see if the doc_id exists and is not empty
            if dFlag == True:
                if j[0] != "doc_id":
                    print("Error: doc_id is missing")
                    sys.exit(1)
                else:
                    dFlag = False
            # Check to see if the doc_id is not empty and there
            # are no duplicate doc_id's
            if j[0] == "doc_id":
                # If the previous field was a doc_id, then the zone is missing
                if prev == "doc_id":
                    print("Error: Zone is missing")
                    sys.exit(1)
                # 
                if j[1] == "":
                    print("Error: doc_id is empty")
                    sys.exit(1)
                # If the doc_id is already in the array then it is a duplicate
                if j[1] in id_arr:
                    print("Error: Duplicate doc_id's")
                    sys.exit(1)
                # Else it is not a duplicate so append it to the array
                else:
                    prev = "doc_id"  # Set the previous zone to doc_id
                    if doc_id == "":
                        doc_id = int(j[1])  # Set the doc_id
                    else:
                        id_arr.append(j[1])  # Append the doc_id to the array
                        # Tokenize the data in the zone if the zone exists in the dictionary
                        tokens = tokenization(doc_str)
                        doc_dict[doc_id] = len(tokens) # Add the doc_id and the length of the document to the dictionary
                        # Iterate through each token and add it to the dictionary if it is not already in the dictionary
                        # If it is already in the dictionary, append the doc_id to the array of the token
                        # Store the position of the token in an array
                        for i in range(len(tokens)):
                            if tokens[i] not in index_dict:
                                index_dict[tokens[i]] = [[doc_id, [i]]]
                            else:
                                # Check if the last array in the postings array has the same doc_id
                                # If it does, increment the frequency
                                # Else append the doc_id and the frequency to the postings array
                                if index_dict[tokens[i]][-1][0] == doc_id:
                                    index_dict[tokens[i]][-1][1].append(i)
                                else:
                                    index_dict[tokens[i]].append([doc_id, [i]])
                        doc_id = int(j[1])  # Set the doc_id to the current doc_id
                        # Reset the doc_str to empty
                        doc_str = ""
            # If field is not a doc_id, check to see if the zone exists
            else:
                if not j[0].isalnum():
                    print("Error: Invalid zone name")
                    print("Only alphanumeric characters are allowed")
                    sys.exit(1)
                prev = ""  # Set the previous encountered field to empty
                # Append the contents of the zone to the doc_str 
                doc_str = doc_str + " " + j[1]

    # Tokenize the data in the zone if the zone exists in the dictionary
    # Iterate through each token and add it to the dictionary if it is not already in the dictionary
    # If it is already in the dictionary, append the doc_id to the array of the token
    tokens = tokenization(doc_str)
    doc_dict[doc_id] = len(tokens) # Add the doc_id and the length of the document to the dictionary

    # Iterate through each token and add it to the dictionary if it is not already in the dictionary
    for i in range(len(tokens)):
        # If the token is not in the dictionary, add it to the dictionary
        if tokens[i] not in index_dict:
            index_dict[tokens[i]] = [[doc_id, [i]]]
        else:
        # Check if the last array in the postings array has the same doc_id
        # If it does, increment the frequency
        # Else append the doc_id and the frequency to the postings array
            if index_dict[tokens[i]][-1][0] == doc_id:
                index_dict[tokens[i]][-1][1].append(i)
            else:
                index_dict[tokens[i]].append([doc_id, [i]])
            
    # Sort the postings array for each token
    for key in index_dict.keys():
        index_dict[key].sort()
    
    # For each element in the postings array, convert the array to a tuple
    for key in index_dict.keys():
        for i in range(len(index_dict[key])):
            index_dict[key][i] = tuple(index_dict[key][i])

    # Create the index directory
    if not os.path.exists(sys.argv[2]):
        os.mkdir(sys.argv[2])    

    # Create the index.tsv and doc.tsv files
    index_path = (os.path.join(sys.argv[2], "index.tsv"))
    doc_path = (os.path.join(sys.argv[2], "doc.tsv"))

    # Output the dictionaries to the files
    output_file(index_dict, index_path)
    output_file(doc_dict, doc_path)
    print("Index file created successfully")

    # Closing file
    file.close()