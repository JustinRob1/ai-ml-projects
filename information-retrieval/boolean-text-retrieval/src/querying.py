import csv
import re
import sys
import boolean
import string
from unidecode import unidecode
import os
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# It store all documents
all_documents = set([])

# process the query item
def process_query_item(item):
    new_item = item.translate(str.maketrans('', '', string.punctuation))
    word = unidecode(new_item)
    if word.isalnum():
        return ps.stem(word.casefold())
    else:
        print("Invalid query item")
        sys.exit(1)

# Component is base class of nodes in the boolean tree
class Component:
    def query(self, document_index, max_set=None):
        """
        Execute the query in this node
        :param document_index: created document index
        :param max_set: It means that the final result is subset of max_set.
            This is used in NotQuery.
        :return: query result
        """

# Leaf represents the atomic query
class Leaf(Component):
    def __init__(self, item):
        """
        Constructor
        :param item: query item in this node
        """
        split = item.split(':')
        self.zone = split[0]
        self.query_item = split[1]

    def query(self, document_index, max_set=None):
        """
        Execute the query in this node
        :param document_index: created document index
        :param max_set: It means that the final result is subset of max_set.
            This is used in NotQuery.
        :return: query result
        """
        if len(max_set) == 0:
            # The result is empty
            return set([])

        docs = []
        # First, find the corresponding zone
        zone_items = document_index.get(self.zone, None)
        if zone_items is not None:
            # Then find the documents containing this word
            result = zone_items.get(self.query_item, None)
            if result is not None:
                docs = result['postings']

        # Convert list to set in order to perform set operation
        return set(docs)

# AndQuery represents the AND query
class AndQuery(Component):
    def __init__(self, children):
        """
        Constructor
        :param children: children of this node
        """
        self.children = children

    def query(self, document_index, max_set=None):
        """
        Execute the query in this node
        :param document_index: created document index
        :param max_set: It means that the final result is subset of max_set.
            This is used in NotQuery.
        :return: query result
        """
        if len(max_set) == 0:
            # The result is empty
            return set([])

        # Query the documents in subtree
        docs = None
        result = max_set
        for child in self.children:
            # AND query shrink the size of final query result.
            # Because the query result is common documents.
            result = child.query(document_index, result)
            if docs is None:
                # This is the first query
                docs = result
            else:
                # Calculate the intersection of the query results
                docs = docs.intersection(result)

            # If the result is empty, directly return empty list
            if len(docs) == 0:
                return set([])

        return docs

# OrQuery represents the OR query
class OrQuery(Component):
    def __init__(self, children):
        """
        Constructor
        :param children: children of this node
        """
        self.children = children

    def query(self, document_index, max_set=None):
        """
        Execute the query in this node
        :param document_index: created document index
        :param max_set: It means that the final result is subset of max_set.
            This is used in NotQuery.
        :return: query result
        """
        if len(max_set) == 0:
            # The result is empty
            return set([])

        # Query the documents in subtree
        docs = None
        for child in self.children:
            result = child.query(document_index, max_set)
            if docs is None:
                # This is the first query
                docs = result
            else:
                # Calculate the intersection of the query results
                docs = docs.union(result)

        return docs

# NotQuery represents the NOT query
class NotQuery(Component):
    def __init__(self, child):
        """
        Constructor
        :param child: child of this node
        """
        self.child = child

    def query(self, document_index, max_set=None):
        """
        Execute the query in this node
        :param document_index: created document index
        :param max_set: It means that the final result is subset of max_set.
            This is used in NotQuery.
        :return: query result
        """
        if len(max_set) == 0:
            # The result is empty
            return set([])

        # First find those documents containing this term
        containing_docs = self.child.query(document_index, max_set)

        # Then Exclude these documents
        docs = max_set.difference(containing_docs)

        # Convert list to set in order to perform set operation
        return set(docs)

def build_boolean_tree_helper(boolean_expr):
    """
    Recursively build the boolean tree
    :param boolean_expr: the boolean expression
    :return: boolean tree
    """
    if isinstance(boolean_expr, boolean.Symbol):
        # Ensure each token has the corresponding zone
        if len(boolean_expr.obj.split(':')) < 2:
            print('Error: A token appears without the zone')
            sys.exit(1)

        # Create the atomic query
        return Leaf(boolean_expr.obj)

    if isinstance(boolean_expr, boolean.NOT):
        # Create the NOT query
        leaf = build_boolean_tree_helper(boolean_expr.args[0])
        return NotQuery(leaf)

    if isinstance(boolean_expr, boolean.AND):
        # Create the AND query
        children = []
        for arg in boolean_expr.args:
            children.append(build_boolean_tree_helper(arg))
        return AndQuery(children)

    if isinstance(boolean_expr, boolean.OR):
        # Create the AND query
        children = []
        for arg in boolean_expr.args:
            children.append(build_boolean_tree_helper(arg))
        return OrQuery(children)

    print('ERROR: The boolean query only supports And, OR, Not and ()')
    sys.exit(1)


def build_boolean_tree(boolean_query):
    """
    Build a boolean tree by the boolean query
    :param boolean_query: the boolean query
    :return: the boolean tree
    """
    

    # Use boolean module to parse this boolean query
    algebra = boolean.BooleanAlgebra()
    try:
        boolean_expr = algebra.parse(boolean_query)
    except:
        print('Error: The query is not properly formed.')
        sys.exit(1)

    # Build the boolean tree
    boolean_tree = build_boolean_tree_helper(boolean_expr)

    return boolean_tree


def read_index_files(index_dir):
    """
    Load document index from the files
    :param index_dir: parent directory of index files
    :return: document index
    """
    global all_documents

    document_index = {}
    for filename in os.listdir(index_dir):
        path = os.path.join(index_dir, filename)
        with open(path, 'r') as fp:
            reader = csv.reader(fp, delimiter='\t')
            # skip the header
            next(reader)

            # Read the content of index
            zone = filename.split('.')[0]
            document_index[zone] = {}
            for row in reader:
                try:
                    token, df, postings = row[0], int(row[1]), re.findall(r'\d+', row[2])
                    document_index[zone][token] = {
                        'DF': df, 'postings': postings
                    }
                    all_documents = all_documents.union(set(postings))
                except:
                    continue

    return document_index

def main():
    global all_documents

    # Check the command line parameters
    if len(sys.argv) != 3:
        print("Error: Wrong number of arguments")
        print(
            "Usage: python3 querying.py <path to output directory> <boolean query>")
        sys.exit(1)

    index_dir = sys.argv[1]
    boolean_query = sys.argv[2]

    # Iterate through the query and for when a a colon is found, call the
    # function process_query_item to process the query. The query is the string
    # from the colon to the next space or the end of the string boolean_query.
    boolean_query += ' '
    parsed_query = ''
    i = 0
    # Iterate through the query and for when a a colon is found, call the
    # function process_query_item to process the query. The query is the string
    # from the colon to the next space or the end of the string boolean_query.
    while i < len(boolean_query):
        if boolean_query[i] == ':':
            parsed_query += boolean_query[i]
            for j in range(i + 1, len(boolean_query)):
                if boolean_query[j] == ' ' or boolean_query[j] == ')':
                    item = boolean_query[i + 1:j]
                    try:
                        parsed_query += process_query_item(item)
                    except TypeError:
                        print("Error: Zone missing from query.")
                        print('Boolean query should be in the form of "<zone>:<query>"')
                        sys.exit(1)
                    i += len(item)
                    break
            i += 1
        else:
            parsed_query += boolean_query[i]
            i += 1

    # Check if the query is properly formatted
    if ":" not in parsed_query:
        print("Error: Improperly formatted query.")
        print('Boolean query should be in the form of "<zone>:<query>"')
        sys.exit(1)

    # Check the validity of command line parameters
    if not os.path.exists(index_dir):
        print('Error: The given index directory does not exist.')
        sys.exit(1)
    if not os.path.isdir(index_dir):
        print('Error: The given index directory is not a directory.')
        sys.exit(1)

    # Build a boolean tree by the boolean query
    boolean_tree = build_boolean_tree(parsed_query)

    # Load document index from the files
    document_index = read_index_files(index_dir)

    # Execute the boolean query
    docs = boolean_tree.query(document_index, all_documents)

    # sort the document id in ascending order
    sorted_docs = [str(v) for v in list(sorted([int(item) for item in docs]))]

    # Print the result to the standard output
    for doc_id in sorted_docs:
        print(doc_id)

main()