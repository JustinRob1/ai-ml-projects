import sys
import nltk


def normalize(text):
    '''
    Given a string, returns a list of normalized tokens.

    Normalization includes:
        - tokenization 
        - removing punctuations
        - lower the case
        - lemmatization
    '''

    tokens = nltk.wordpunct_tokenize(text)
    # remove punctuations
    tokens = [token for token in tokens if token.isalpha()]
    # lower the case
    tokens = [token.lower() for token in tokens]

    lemmatizer = nltk.stem.WordNetLemmatizer()
    # lemmatize verbs
    tokens = [lemmatizer.lemmatize(token, pos='v') for token in tokens]
    # lemmatize nouns
    tokens = [lemmatizer.lemmatize(token, pos='n') for token in tokens]

    return tokens


if __name__ == '__main__':
    text = sys.argv[1]
    print(normalize(text))
