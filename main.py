import nltk
import heapq
import bs4 as bs
import urllib.request
import re


def add(key, counts, count):
    if key in counts.keys():
        counts[key] += count
    else:
        counts[key] = count


def summarize(text, sentencePercent):
    word_tokens = nltk.word_tokenize(text)
    sentences = nltk.sent_tokenize(text)
    sentenceCount = sentencePercent*len(sentences)/100
    # print(word_tokens)
    stopwords = nltk.corpus.stopwords.words('english')
    # print(stopwords)
    word_freq = {}
    for word in word_tokens:
        if word not in stopwords:
            add(word, word_freq, 1)
    if len(word_freq) <= 0:
        return "No words"
    # print(word_freq)
    max_freq = max(word_freq.values())
    for word in word_freq.keys():
        word_freq[word] /= max_freq
    # print(word_freq)
    sentence_scores = {}
    for sentence in sentences:
        for token in nltk.word_tokenize(sentence.lower()):
            if token in word_freq.keys():
                if len(sentence.split(' ')) < 25:
                    add(sentence, sentence_scores, word_freq[token])
    # print(sentence_scores)
    selected_sentences = heapq.nlargest(int(sentenceCount), sentence_scores, key=sentence_scores.get)
    return " ".join(selected_sentences)


climate_change = "The climate change in real. Glaciers are melting. With climate change world temperature is rising. " \
                 "Weathers have become extreme. Climate change has affected agriculture and industry equally. The " \
                 "seasons are changing their cycle. Measures are needed to control this phenomenon."

file_path = "/Users/sowmysrinivasan/Documents/Purview Architecture.txt"
heading1s = []
with open(file_path, 'r') as file:
    # Read the contents of the file
    contents = file.read()
    contents = re.sub(r'\[[0-9]*\]', ' ', contents)
    contents = re.sub(r'\s+', ' ', contents)
    contents = re.sub(r'[^a-zA-Z]', ' ', contents)
    contents = re.sub(r'\s+', ' ', contents)
    contents = contents.lower()
    heading1s = contents.split("H1:")
    for heading1 in heading1s:
        heading1_parts = heading1.split("\n", 1)
        print(heading1_parts[0])
        if len(heading1_parts) > 1:
            for heading2 in heading1_parts[1].split("H2:"):
                heading2_parts = heading2.split("\n", 1)
                print(heading2_parts[0])
                if len(heading2_parts) > 1:
                    print(summarize(heading2_parts[1], 33))
