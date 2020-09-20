from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from string import punctuation
import re

def tokenization(text):
    tokens = word_tokenize(text)

def toLower(text):
    return [sentence.lower() for sentence in text]

def remove_html(htmlTxt):
    return [" ".join(BeautifulSoup(sentence).findAll(text=True)) for sentence in htmlTxt ]

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return [url_pattern.sub(r'', word) for word in text ]

def remove_stopwords(text):
    fullList=[]
    STOPWORDS = set(stopwords.words('english'))
    for sentence in text:
        sentence = " ".join([word for word in str(sentence).split() if word not in STOPWORDS])
        fullList.append(sentence)
    return fullList

def remove_specialChar(text):
    return [sentence.replace(punctuation,' ') for sentence in text ]

def strip_digits(text):
    fullList=[]
    for sentence in text:
        sentence = " ".join([word for word in str(sentence).split() if not word.isdigit()])
        fullList.append(sentence)
    return fullList

def strip_punctuation(text):
    fullList=[]
    for sentence in text:
        sentence = " ".join([word for word in str(sentence).split() if word not in punctuation])
        fullList.append(sentence)
    return fullList

def preprocessingText(textSet):
    textSet = textSet.dropna()
    textSet['text'] = toLower(textSet['text']).copy()
    textSet['text'] = strip_digits(textSet['text'])
    textSet['text'] = strip_punctuation(textSet['text']).copy()
    textSet = textSet.dropna()
    return textSet
