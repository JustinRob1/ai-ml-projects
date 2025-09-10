import sys
import os
import math
import csv
import string
import heapq
from unidecode import unidecode
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
csv.field_size_limit(sys.maxsize)

# Description: This function reads the index from the index.tsv file and returns the rebuilds the index
# Input: index_path - the path to the directory containing the index
# Output: index - a dictionary with the index
def get_index(index_path):
    index = {}
    # Open the file and read the index from the file
    # Join with the extension index.tsv
    with open(os.path.join(index_path, 'index.tsv'), 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)    # Skip the header
        # Read the index line by line
        for line in reader:
            token = line[0]
            df = int(line[1])
            postings = eval(line[2])
            # Add the token, the document frequency of the token and the array of the doc_id's to the index
            index[token] = {'DF': df, 'postings': postings}
    return index

# Description: This function reads the document lengths from the doc.tsv file and returns a dictionary with the document lengths
# Input: index_path - the path to the directory containing the index
# Output: lengths - a dictionary with the document lengths
def get_lengths(index_path):
    lengths = {}
    # Open the file and read the document lengths from the file
    with open(os.path.join(index_path, 'doc.tsv'), 'r') as f:
        next(f)    # Skip the header
        # Read the document lengths line by line
        for line in f:
            doc_id, length = line.strip().split('\t')
            lengths[int(doc_id)] = int(length)
    return lengths

# Description: This function performs the query normalization and stemming
# Input: query - the query string
# Output: processed_tokens - a list of the processed tokens
def process_query(query):
    # Create stemmer
    stemmer = PorterStemmer()
    # Split query into individual terms
    tokens = word_tokenize(query)
    # Normalize and stem each term
    processed_tokens = []

    # Normalize and stem each term
    for token in tokens:
        token = token.lower().strip()
        token = token.translate(str.maketrans('', '', string.punctuation))
        token = unidecode(token)
        token = stemmer.stem(token)
        # Check if the token is alphanumeric
        if token.isalnum():
            processed_tokens.append(token)
        # Else the token is not value
        else:
            print("Invalid query item")
            sys.exit(1)
    
    # Return list of processed terms
    return processed_tokens

# Description: This returns a list of all of the keywords in the query
def get_keywords(query):
    # Split the query string into individual words
    words = query.split()
    # Create an array of keywords by filtering out any words that are surrounded by colons
    bFlag = False  # Flag to indicate if the word is surrounded by colons
    keywords = []
    # Iterate through the words in the query
    for word in words:
        # Check if the word is surrounded by colons
        if word[0] == ":":
            bFlag = True
        if word[-1] == ":":
            bFlag = False 
        # If the word is not surrounded by colons, add it to the list of keywords
        elif bFlag == False:
            keywords.append(word)
    # Return the list of keywords
    return keywords

# Description: This returns a list of all of the phrases in the query 
# Input: query - the query string 
# Output: phrases - a list of the phrases in the query
def get_phrases(query):
    phrases = []
    # Iterate through the query string
    while True:
        # Find the start and end of the phrase
        start = query.find(":")
        # Check if the phrase is not found, break out of the loop
        if start == -1:
            break
        # Find the end of the phrase
        end = query.find(":", start+1)
        # Check if the phrase is not found, break out of the loop
        if end == -1:
            break
        # Add the phrase to the list of phrases
        phrases.append(query[start+1:end])
        # Remove the phrase from the query
        query = query[:start] + query[end+1:]
    return phrases

# Description: This function returns a list of the top k documents
# Input: query_vector - a dictionary with the query vector
#        index - the inverted index
#        lengths - a dictionary with the document lengths
#        k - the number of documents to return
# Output: top_k - a list of the top k documents
def cosineScore(query, index, lengths, k, N):
    # Initialize heap
    heap = []
    heap_size = 0

    # Calculate scores
    scores = {}
    doc_weights = {}
    # Initialize scores and weights
    for doc_id in lengths.keys():
        scores[int(doc_id)] = 0.0
        doc_weights[int(doc_id)] = 0.0

    # Calculate the score for each document
    for term in query:
        # Check if term is in the index
        if term in index:
            # Calculate weightings for query
            postings = index[term]["postings"]
            # Remove all of the postings if its doc_id is not in the lengths dictionary
            postings = [p for p in postings if p[0] in lengths]
            # Calculate the weightings for the term
            wtq = math.log10(N / index[term]['DF'])

            # Calculate the max tf for the term
            max_tf = 0
            for doc in postings:
                if len(doc[1]) > max_tf:
                    max_tf = len(doc[1])
            
            # Calculate the weightings for each document
            for doc in postings:
                # According to the apc.btn weighting if the tf for the document is 0, then the weight is 0
                # Therefore, we just need to calculate wtq = math.log10(N / index[term]['DF']) for the weight
                # Since we are iterating through the postings we do not consider any document that has a tf of 0

                # Calculate the term frequency for the document
                tfd = 0.5 + (0.5 * (len(doc[1])) / max_tf)
                # Calculate the document frequency for the document
                dfd = max(0, math.log10(N - index[term]['DF'] / index[term]['DF']))
                # Calculate the weight for the document
                doc_weights[doc[0]] = tfd * dfd

            # Calculate the score for the document
            for doc in doc_weights:
                # Normalize the weights using cosine normalization
                doc_weights[doc] = doc_weights[doc] / math.sqrt(sum([w**2 for w in doc_weights.values()]))
                # The score is equal to wtq * wtd (where doc_weights[doc] = wtd)
                scores[doc] += wtq * doc_weights[doc]

                # Update the heap
                # The heap can only contain the top k documents with the highest scores
                # Check if the heap already contains a node with the same document id
                if doc in [d[1] for d in heap]:
                    # If it does, update the score for the node
                    node = None
                    # Find the index of the node in the heap
                    for i in range(len(heap)):
                        if heap[i][1] == doc:
                            node= i
                            break
                    # Update the score for the node
                    heap[node] = (scores[doc], doc)
                    # Reheapify the heap
                    heapq.heapify(heap)
                # Else the heap does not contain a node with the same document id
                else:
                    # If it doesn't, add the node to the heap if the heap is not full
                    if heap_size < k:
                        heapq.heappush(heap, (scores[doc], doc))
                        heap_size += 1
                    # Else determine if the score for the document is greater than the smallest score in the heap
                    else:
                        smallest = heapq.nsmallest(1, heap)[0][0]
                        if scores[doc] > smallest:
                            heapq.heapreplace(heap, (scores[doc], doc))
    
    # Normalize scores and update heap
    for i in range(len(heap)):
        doc_id = heap[i][1]
        score = heap[i][0] / lengths[doc_id]
        heap[i] = (score, doc_id)
    # Reheapify the heap
    heapq.heapify(heap)

    # Calculate the number of documents with a score greater than 0
    num_docs = 0
    for doc in scores:
        if scores[doc] > 0:
            num_docs += 1

    # Return top k scores, reverse the heap so that the highest scores are at the beginning of the list
    top_scores = sorted(heap, reverse=True)[-k:]
    return (top_scores, num_docs)

# Description: This function intersects two postings lists for phrase queries
# Input: p1 - the first postings list
#        p2 - the second postings list
# Output: result - the intersection of the two postings lists
def phrase_intersect(p1, p2):
    i, j = 0, 0
    result = []
    # Iterate through the postings lists
    while i < len(p1) and j < len(p2):
        # Check if the document ids are equal
        if p1[i][0] == p2[j][0]:
            # Iterate through the positions in the postings lists
            for pos1 in p1[i][1]:
                # Check if the positions are adjacent
                for pos2 in p2[j][1]:
                    if pos2 - pos1 == 1:
                        result.append(p1[i][0])
                        break
                # If the positions are adjacent, break out of the loop
                if result and result[-1] == p1[i][0]:
                    break
            # Increment the pointers
            i += 1
            j += 1
        # Check if the document id in the first postings list is less than the document id in the second postings list
        elif p1[i][0] < p2[j][0]:
            i += 1
        # Else the document id in the second postings list is less than the document id in the first postings list
        else:
            j += 1
    # Return the intersection of the two postings lists
    return result

if __name__ == '__main__':
    # Print error message if the number of arguments is not at least 3
    if len(sys.argv) < 3:
        print("Error: Invalid number of arguments")
        print("Usage: python3 qyeru.py <path> <k> <query>")
        sys.exit(1)
    
    # Print error message if sys.argv[2] is not an integer
    try:
        int(sys.argv[2])
    except ValueError:
        print("Error: Invalid k must be an int of arguments")
        print("Usage: python3 search.py <path> <k> <query>")
        sys.exit(1)

    # Print error message if sys.argv[2] is less than 1
    if int(sys.argv[2]) < 1:
        print("Error: Invalid k must be greater than 0")
        print("Usage: python3 search.py <path> <k> <query>")
        sys.exit(1)

    # Print error message if sys.argv[1] does not contain the required files
    if not os.path.isfile(sys.argv[1] + '/index.tsv') or not os.path.isfile(sys.argv[1] + '/doc.tsv'):
        print("Error: The files index.tsv and doc.tsv must be in the directory")
        print("Usage: python3 search.py <path> <k> <query>")
        sys.exit(1)
    
    # Print error message if sys.argv[1] is not a directory
    if not os.path.isdir(sys.argv[1]):
        print("Error: The path must be a directory")
        print("Usage: python3 search.py <path> <k> <query>")
        sys.exit(1)

    # Print error message if sys.argv[3] is empty
    if len(sys.argv) < 4:
        print("Error: The query must not be empty")
        print("Usage: python3 search.py <path> <k> <query>")
        sys.exit(1)

    # Get the index directory, k, and query from the command line
    index_dir = sys.argv[1] 
    k = int(sys.argv[2])
    query = ' '.join(sys.argv[3:])

    # Get the index and lengths
    index = get_index(index_dir)
    lengths = get_lengths(index_dir)
    N = len(lengths)

    # Get the keywords and phrases from the query
    keywords = get_keywords(query)
    phrases = get_phrases(query)

    # If there are no phrases, just calculate the cosine score
    # The pool of documents is all documents
    if len(phrases) == 0:
        # Process the query
        tokens = process_query(query)
        # Calculate the cosine score
        result = cosineScore(tokens, index, lengths, k, N)
        # Print the number of documents considered for the query
        print(len(lengths))
        # Print the number of documents with a score greater than 0
        print(result[1])
        # Print the top k documents and their scores
        for score, doc_id in result[0]:
            if score > 0:
                print(doc_id, "\t", score)
    # Othwerwise, the query contains a phrase
    # The pool of documents is the documents that contain at least one phrase
    else:
        # Initialize the list of document ids to an empty list
        doc_ids = []
        # Initialize the list of terms to an empty list
        terms = []
        # Iterate through the phrases
        for phrase in phrases:
            tokens = process_query(phrase)
            # Add the terms to the list of terms
            for token in tokens:
                terms.append(token)
            # Get postings lists for each word and find the intersection
            # Check if each word is in the index
            # If it is not then ignore the phrase
            postings_lists = []
            for token in tokens:
                if token in index:
                    postings_lists.append(index[token]['postings'])
            # If there are no postings lists, ignore the phrase
            if len(postings_lists) == 0:
                continue
            else:
                # Initialize the intersection to the first postings list
                p1 = postings_lists[0]
                # Iterate through the postings lists
                for i in range(1, len(postings_lists)):
                    result = phrase_intersect(p1, postings_lists[i])
                    # If there is an intersection, add the document ids to the list of document ids
                    if result:
                        for doc_id in result:
                            if doc_id not in doc_ids:
                                doc_ids.append(doc_id)
                    # Set intersect to the next postings list
                    p1 = postings_lists[i]

        # Get the pool of documents that contain at least one phrase
        new_lengths = {}
        for doc_id in doc_ids:
            new_lengths[doc_id] = lengths[doc_id]
        lengths = new_lengths

        # Append the phrases to the keywords
        for word in keywords:
            terms.append(word)
        # Process the query
        result = cosineScore(terms, index, lengths, k, N)
        # Print the number of documents considered for the query
        print(len(lengths)) 
        # Print the number of documents with a score greater than 0
        print(result[1])
        # Print the top k documents and their scores
        for score, doc_id in result[0]:
            if score > 0:
                print(doc_id, "\t", score)