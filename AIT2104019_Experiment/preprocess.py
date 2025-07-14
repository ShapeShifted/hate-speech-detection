import string
import re
import nltk
from textblob import TextBlob
from nltk.corpus import wordnet
from nltk.metrics import edit_distance
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize


import spacy

nlp = spacy.load('en_core_web_sm')

stopword_extra = ["always","want","even","still","ever","also","already","yet","basically","actually","need","please","ago","probable","probably","however","instead","quite","nt","na","u","gon","lol","im","ca","us","cnt","wo","em","etc","ll","aint","r","cant","shouldnt","wont","lah","dont","never"]

stopwords = nltk.corpus.stopwords.words('english')
stopwords += stopword_extra

negative_list = ['not','never','ain','aint','no','neither','nor','nt','cant','dont',"cnt",'wont',"shouldnt"]

def preprocess(input):
    preprocessed_input = []

    input = sent_tokenize(input)

    for text in input:

        #1. Generating the list of words in the tweet (hastags and other punctuations removed)
        text_blob = TextBlob(text)
        text = ' '.join(text_blob.words)

        # remove number
        text = re.sub(r'[0-9]', '', text)

        # lowercase
        text = text.lower()

        text = text.replace('/',' ')

        for punctuation in string.punctuation:
            text = text.replace(punctuation, '')

        text = word_tokenize(text)

        #keep tokens that are alphabet characters
        text = [t for t in text if t.isalpha()]

        # replace the negation token
        replacer  = AntonymReplacer()
        text = replacer.replace_negations(text)

        # remove the stopwords
        text = [i for i in text if i not in stopwords]

        #remove empty token
        text = [t for t in text if len(t) > 0]

        preprocessed_input.append(text)


    return preprocessed_input


def lemmatization(sent, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    doc = nlp(" ".join(sent))
    texts_out = [token.lemma_ for token in doc if token.pos_ in allowed_postags]
    return texts_out

class AntonymReplacer(object):
    def replace(self, word, pos=None):
        antonyms = set()

        for syn in wordnet.synsets(word, pos=pos):
            for lemma in syn.lemmas():
                for antonym in lemma.antonyms():
                    antonyms.add(antonym.name())

        if len(antonyms) >= 1:
            return antonyms.pop()
        else:
            return None

    def replace_negations(self, sent):
        i, l = 0, len(sent)
        words = []

        while i < l:
            word = sent[i]

            if word in negative_list and i+1 < l:
                ant = self.replace(sent[i+1])

                if ant:
                    words.append(ant)
                    i += 2
                    continue

            words.append(word)
            i += 1

        return words