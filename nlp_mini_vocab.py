import random
from typing import List, Set, Any, Generator

import numpy as np
from numpy import ndarray

sentence_words: list[list[str]] = [
    ["a"],
    ["bat", "cat", "dog", "rat"],
    ["crawls", "walks", "runs", "jumps"],
    ["slowly", "quickly"],
    ["in", "on", "under"],
    ["a"],
    ["car", "room", "table"]
]
EPOCH_COUNT: int = 3
EMBEDDING_DIM: int = 4
PREFIX_SIZE: int = 2
BATCH_SIZE: int = 10
HIDDEN_UNITS: int = 8


def generate_sentences(count: int) -> list[str]:
    sentences: list[str] = []
    word_indexes: ndarray = np.zeros((len(sentence_words)), dtype=int)
    while True:
        sentence: str = ""
        for j in range(0, len(word_indexes)):
            sentence = sentence + " " + sentence_words[j][word_indexes[j]]
        sentences.append(sentence.strip())

        for j in range(len(sentence_words) - 1, -1, -1):
            word_indexes[j] += 1
            if word_indexes[j] == len(sentence_words[j]):
                word_indexes[j] = 0
            else:
                break

        if sum(word_indexes) == 0:
            break

    return random.sample(sentences, count)


def get_vocab_words(sentences: list[str]) -> list[str]:
    result: list[str] = list(set([word for sentence in sentences for word in sentence.split()]))
    result.sort()
    return result


def get_word_ids(words: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for idx, word in enumerate(words):
        result[word] = idx
    return result


def get_x_y_trains(sentences: list[str], word_ids: dict[str, int]) -> (ndarray, ndarray):
    x_train: list[list[int]] = []
    y_train: list[list[int]] = []
    for training_sentence in sentences:
        sentence_words = training_sentence.split()
        for i in range(0, len(sentence_words)):
            if i + 2 >= len(sentence_words):
                break

            x_1_word_id = word_ids[sentence_words[i]]
            x_2_word_id = word_ids[sentence_words[i + 1]]
            y_word_id = word_ids[sentence_words[i + 2]]
            x_train.append([x_1_word_id, x_2_word_id])
            y_train.append([y_word_id])

    return np.array(x_train), np.array(y_train)


def lookup_embeddings(embedding_matrix: ndarray, input_batch: ndarray):
    input_embeddings_batch = ndarray((len(input_batch), PREFIX_SIZE * EMBEDDING_DIM))
    for item_index, input_item in enumerate(input_batch):
        input_embeddings = np.zeros((PREFIX_SIZE, EMBEDDING_DIM))
        for word_index, input_word_index in enumerate(input_item):
            input_word_embedding = np.array(embedding_matrix[input_word_index]).reshape((1, EMBEDDING_DIM))
            input_embeddings[word_index] = input_word_embedding

        input_embeddings_batch[item_index] = input_embeddings.reshape(1, PREFIX_SIZE * EMBEDDING_DIM)

    return input_embeddings_batch


def log_softmax(x: ndarray) -> ndarray:
    x_max = np.max(x, axis=-1, keepdims=True)
    x_exp = np.exp(x - x_max)
    x_sum = np.sum(x_exp, axis=-1, keepdims=True)
    return x - x_max - np.log(x_sum)


training_sentences = generate_sentences(50)
print(training_sentences)
vocab_words = get_vocab_words(training_sentences)
print(vocab_words)
VOCAB_SIZE = len(vocab_words)
word_ids = get_word_ids(vocab_words)
x_train, y_train = get_x_y_trains(training_sentences, word_ids)
print(x_train.shape)
print(y_train.shape)

embedding_matrix = np.random.randn(VOCAB_SIZE, EMBEDDING_DIM)
layer1_w = np.random.randn(2*EMBEDDING_DIM, HIDDEN_UNITS)
layer1_b = np.random.randn(1, HIDDEN_UNITS)
layer2_w = np.random.randn(HIDDEN_UNITS, VOCAB_SIZE)
layer2_b = np.random.randn(1, VOCAB_SIZE)


for epochs in range(0, 1):
    epoch_loss: float = 0.0
    batch_starts = np.arange(0, len(x_train), BATCH_SIZE)
    for batch_start in batch_starts:
        x_train_batch: ndarray = x_train[batch_start: batch_start + BATCH_SIZE]
        y_train_batch: ndarray = y_train[batch_start: batch_start + BATCH_SIZE]
        x_embeddings = lookup_embeddings(embedding_matrix, x_train_batch)
        layer1_output = np.matmul(x_embeddings, layer1_w) + layer1_b
        layer1_activation_output = np.tanh(layer1_output)
        layer2_output = np.matmul(layer1_activation_output, layer2_w) + layer2_b
        layer2_softmax_output = log_softmax(layer2_output)

        print(layer2_softmax_output.shape)

