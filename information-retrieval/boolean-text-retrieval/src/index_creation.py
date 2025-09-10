import json
import sys
import os
import csv
import string
import nltk
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
        if word.isalnum() and word.casefold() not in tokens:
            tokens.append(word.casefold()) 
    return tokens

# Description: Writes the token, the document frequency of the token and the array of the doc_id's to the tsv file
# Arguments: zone - the zone of the inverted index
#            index_dict - the dictionary of the inverted index for each zone
# Returns: None
def output_file(zone, index_dict, file):
        # Open the file and write the token, the document frequency of the token
            with open(file, 'w', newline='') as f_output:
                tsv_output = csv.writer(f_output, delimiter='\t')
                tsv_output.writerow(["token", "DF", "postings"])
                # Sort the keys in the dictionary and write the token, the document frequency of the token
                # and the array of the doc_id's to the tsv file
                for key in sorted(index_dict[zone].keys()):
                    tsv_output.writerow([key, len(index_dict[zone][key]), index_dict[zone][key]])

# Reads the JSON file
# Create an inverted index for each zone
def main():
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
    id_arr = []  # Array of the doc_id's to check for duplicates
    prev = ""  # Variable to check if a zone is missing

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
                    id_arr.append(j[1])  # Append the doc_id to the array
            # If field is not a doc_id, check to see if the zone exists
            else:
                if not j[0].isalnum():
                    print("Error: Invalid zone name")
                    print("Only alphanumeric characters are allowed")
                    sys.exit(1)
                prev = ""  # Set the previous encountered field to empty
                # If the zone hasn't been indexed yet, create a dictionary for the zone
                # Where the values is a dictionary of the tokens and the postings list
                if j[0] not in index_dict:
                    index_dict[j[0]] = {}
                if j[0] in index_dict:
                    # Tokenize the data in the zone if the zone exists in the dictionary
                    token = tokenization(j[1])
                    # Iterate through each token and add it to the dictionary if it is not already in the dictionary
                    # If it is already in the dictionary, append the doc_id to the array of the token
                    for word in token:
                        if word not in index_dict[j[0]]:
                            index_dict[j[0]][word] = []
                        if word in index_dict[j[0]]:
                            index_dict[j[0]][word].append(corpus[i]["doc_id"])
            
    # Create the index directory
    if not os.path.exists(sys.argv[2]):
        os.mkdir(sys.argv[2])

    # Iterate through each zone in the dictionary and write the token, the document frequency of the token
    # and the array of the doc_id's to the tsv file
    # Fix the alignment of the rows in the tsv file
    for i in index_dict.keys():
        # Check to see if the file already exists
        path = (os.path.join(sys.argv[2], i + ".tsv"))
        # If the file already exists, ask the user if they want to overwrite the file
        if os.path.exists(path):
            print("Error: File " + i + ".tsv already exists")
            # Ask the user if they want to overwrite the file
            user_input = input("Do you want to overwrite the file? (y/n): ")
            # Continue to ask the user if they want to overwrite the file until they enter a valid input
            while True:
                if user_input.lower() == "y":
                    print("File overwritten")
                    # Call the output_file function to write the tsv file
                    output_file(i, index_dict, path)
                    break
                elif user_input.lower() == "n":
                    print("File not overwritten")
                    break
                else:
                    user_input = input("Invalid input. Do you want to overwrite the file? (y/n): ")
        else:
            # Call the output_file function to write the tsv file
            output_file(i, index_dict, path)

    # Closing file
    file.close()

main()
