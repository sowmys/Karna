import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# PyTorch
import torch
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report, confusion_matrix

# Data preprocessing
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from cb_bert_model import bert_tokenizer, init_bert_model, bert_train, bert_predict, device
from cb_text_cleanup import deep_clean

# Text cleaning
# Naive Bayes

cyber_bullying_tweet_file = "/Users/sowmysrinivasan/Library/Mobile Documents/com~apple~CloudDocs/Cyber " \
                            "Bullying/cyberbullying_tweets.csv"
cyber_bullying_model_file = "/Users/sowmysrinivasan/Library/Mobile Documents/com~apple~CloudDocs/Cyber " \
                       "Bullying/cb_tweets.zip"
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
EPOCHS = 2


def conf_matrix(y, y_pred, title, labels):
    sns.set_style("whitegrid")
    sns.despine()
    plt.style.use("seaborn-whitegrid")
    plt.rc("figure", autolayout=True)
    plt.rc("axes", labelweight="bold", labelsize="large", titleweight="bold", titlepad=10)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax = sns.heatmap(confusion_matrix(y, y_pred), annot=True, cmap="Purples", fmt='g', cbar=False,
                     annot_kws={"size": 30})
    plt.title(title, fontsize=25)
    ax.xaxis.set_ticklabels(labels, fontsize=16)
    ax.yaxis.set_ticklabels(labels, fontsize=14.5)
    ax.set_ylabel('Test', fontsize=25)
    ax.set_xlabel('Predicted', fontsize=25)
    plt.show()

print("Reading data file ...")
df = pd.read_csv(cyber_bullying_tweet_file)
df = df.rename(columns={'tweet_text': 'text', 'cyberbullying_type': 'sentiment'})

print("deduping ...")
duplicated_df = df.duplicated()
if duplicated_df.sum() > 0:
    df = df[~duplicated_df]
print(df.sentiment.value_counts())

print("cleaning up input text ...")
texts_new = []
for t in df.text:
    texts_new.append(deep_clean(t))
df['text_clean'] = texts_new
df.sentiment.value_counts()

print("pruning the input text ...")
# We can see that lots of tweets of the class "other_cyberbullying" have been removed. Since the class is very
# unbalanced compared to the other classes and looks too "generic", we decide to remove the tweets labeled belonging
# to this class.By performing some tests, the f1 score for predicting the "other_cyberbullying" resulted to be
# around 60%, a value far lower compared to the other f1 scores (around 95% using LSTM model). This supports the
# decision of removing this generic class.
df = df[df["sentiment"] != "other_cyberbullying"]

# Then we also define a list of the classes names, which will be useful for the future plots.
sentiments = ["religion", "age", "ethnicity", "gender", "not bullying"]

# Prune long (> 100 words and short (<4) tweets)
text_len = []
for text in df.text_clean:
    tweet_len = len(text.split())
    text_len.append(tweet_len)

df['text_len'] = text_len
df = df[df['text_len'] > 3]
df = df[df['text_len'] < 100]
df.sort_values(by=['text_len'], ascending=False)
print(df.sentiment.value_counts())

# replace target col by ordinal numbers
df['sentiment'] = df['sentiment'].replace(
    {'religion': 0, 'age': 1, 'ethnicity': 2, 'gender': 3, 'not_cyberbullying': 4})

print("preparing the X_train, Y_train, X_test, Y_test, X_valid, Y_valid...")
# split dataset to training, validating and testing sets
X = df['text_clean'].values
y = df['sentiment'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed_value)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train,
                                                      random_state=seed_value)

# Balance all sentiment classes with oversampling
ros = RandomOverSampler()
X_train_os, y_train_os = ros.fit_resample(np.array(X_train).reshape(-1, 1), np.array(y_train).reshape(-1, 1))
X_train_os = X_train_os.flatten()
y_train_os = y_train_os.flatten()

print("tokenizing inputs ...")
# Tokenize the inputs
train_inputs, train_masks = bert_tokenizer(X_train_os)
val_inputs, val_masks = bert_tokenizer(X_valid)
test_inputs, test_masks = bert_tokenizer(X_test)

# Convert target columns to pytorch tensors format
train_labels = torch.from_numpy(y_train_os)
val_labels = torch.from_numpy(y_valid)
test_labels = torch.from_numpy(y_test)

# Set batch size as specified by the BERT authors (16 or 32)
batch_size = 32

print("creating dataloaders for training, testing and validating data ...")
# Create the DataLoader for our training set
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# Create the DataLoader for our validation set
val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

# Create the DataLoader for our test set
test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

print("creating a bert model ...")
# Initialize the bert model
bert_classifier, optimizer, scheduler = init_bert_model(train_dataloader, epochs=EPOCHS)
print("saving model to " + cyber_bullying_model_file)
torch.save(bert_classifier.state_dict(), cyber_bullying_model_file)

if (os.path.exists(cyber_bullying_model_file)):
    print("loading model from " + cyber_bullying_model_file)
    bert_classifier.load_state_dict(copy.deepcopy(torch.load(cyber_bullying_model_file,device)))
else:
    print("training the model ...")
    # Train
    bert_train(bert_classifier, optimizer, scheduler, train_dataloader, val_dataloader, epochs=EPOCHS)
    print("saving model to " + cyber_bullying_model_file)
    torch.save(bert_classifier.state_dict(), cyber_bullying_model_file)

print("testing the model ...")
# Test
bert_preds = bert_predict(bert_classifier, test_dataloader)

print('Classification Report for BERT :\n', classification_report(y_test, bert_preds, target_names=sentiments))
conf_matrix(y_test, bert_preds, ' BERT Sentiment Analysis\nConfusion Matrix', sentiments)
