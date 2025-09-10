import math


class DocumentVector:
    '''
    A document vector is a dict of term to tf-idf weight.
    '''

    def __init__(self, s=None):
        '''
        s: string of the form "term1:weight1,term2:weight2,term3:weight3"
        If s is None, self.vec is an empty dict.
        Otherwise, self.vec is a dict of term to weight.
        '''
        self.__vec = {}
        if s is not None:
            for term_weight in s.split(","):
                term, weight = term_weight.split(":")
                self.__vec[term] = float(weight)

    def cosine_normalize(self):
        '''
        Cosine normalizes the document vector.
        '''
        magnitude = math.sqrt(sum([self[term]**2 for term in self]))
        for term in self:
            self[term] /= magnitude

    def __add__(self, other):
        '''
        Returns a new DocumentVector that is the sum of self and other.
        '''
        new_vec = DocumentVector()
        new_vec.set_vec(self.copy_vec())
        for term in other:
            new_vec[term] = new_vec.get(term, 0) + other[term]
        return new_vec

    def __sub__(self, other):
        '''
        Returns a new DocumentVector that is the difference of self and other.
        '''
        new_vec = DocumentVector()
        new_vec.set_vec(self.copy_vec())
        for term in other:
            new_vec[term] = new_vec.get(term, 0) - other[term]
        return new_vec

    def __getitem__(self, term):
        '''
        Returns the weight of term in the document vector.
        '''
        return self.__vec[term]

    def __setitem__(self, term, value):
        '''
        Sets the weight of term in the document vector to value.
        '''
        self.__vec[term] = value

    def __iter__(self):
        return iter(self.__vec)

    def set_vec(self, vec):
        '''
        Sets the document vector to vec.
        '''
        self.__vec = vec

    def copy_vec(self):
        '''
        Returns a copy of the document vector.
        '''
        return self.__vec.copy()

    def get(self, term, default=0):
        '''
        Returns the weight of term in the document vector.
        If term is not in the document vector, returns default.
        '''
        return self.__vec.get(term, default)

    def add(self, term, value):
        '''
        Adds value to the weight of term in the document vector.
        '''
        self.__vec[term] = self.__vec.get(term, 0) + value

    def values(self):
        '''
        Returns a list of the weights in the document vector.
        '''
        return self.__vec.values()

    def __str__(self):
        '''
        Returns a string of the form "term1:weight1,term2:weight2,term3:weight3"
        '''
        s = []
        for term in self.__vec:
            s.append(f"{term}:{self.__vec[term]}")
        return ",".join(s)
