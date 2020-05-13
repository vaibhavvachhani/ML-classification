import json
import nltk
import numpy as np
import pandas as pd

from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from collections import defaultdict
from collections import Counter

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

# a list of stopwords provided by NLTK
stopwords = set(stopwords.words('english'))
# a list of words provided by NLTK
words = set(nltk.corpus.words.words())


def word_postag(word):
    tag = nltk.tag.pos_tag([word])[0][1][0]
    tag_set = {"J": wordnet.ADJ, "N": wordnet.NOUN,
               "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_set.get(tag, wordnet.NOUN)

# Tokenizing training documents (by obtaining lemmas)


def tokenize_documents(documents):
    dict_list = []
    for document in documents:
        tokens = nltk.word_tokenize(document)

        # Removing stop-words
        tokens = [word for word in tokens if word.lower() not in stopwords]

        dictionary = dict(Counter(tokens))
        dict_list.append(dictionary)
    return dict_list

# Reading (positive label) .json file


misinformationDocs = []
with open('train.json') as f:
    misinformationDict = json.load(f)

for each in misinformationDict.values():
    misinformationDocs.append(each["text"])

# Read (negative label) .csv file

csv_file = pd.read_csv('train-negative.csv')
informationDocs = csv_file['text']

# Tokenizing training documents (by obtaining lemmas)

misinformationTokens = tokenize_documents(misinformationDocs)
informationTokens = tokenize_documents(informationDocs)

# Combining the tokens to get the training data

X_train = misinformationTokens + informationTokens

Y_train_misinformation = (np.ones(len(misinformationDocs), dtype=int))
Y_train_information = (np.zeros(len(informationDocs), dtype=int))

Y_train = np.concatenate((Y_train_misinformation, Y_train_information))

devDocs = []
Y_dev = []
with open('dev.json') as f:
    dev_dict = json.load(f)

for each in dev_dict.values():
    devDocs.append(each["text"])
    Y_dev.append(each["label"])

X_dev = tokenize_documents(devDocs)

testDocs = []
with open('test-unlabelled.json') as f:
    test_dict = json.load(f)

for each in test_dict.values():
    testDocs.append(each["text"])

X_test = tokenize_documents(testDocs)

vectorizer = DictVectorizer(sparse=False)

# Vectorize datasets
X_train_vect = vectorizer.fit_transform(X_train)
X_dev_vect = vectorizer.transform(X_dev)
X_test_vect = vectorizer.transform(X_test)

# Question 6

# Naive Bayes Multinomial NB's hyperparameter is alpha value of the model...
# Optimal Alpha Value = 0.1
clf_multinomial = MultinomialNB(alpha=0.1)
clf_multinomial.fit(X_train_vect, Y_train)
y_pred = clf_multinomial.predict(X_dev_vect)
print("F-score: for Alpha-value: 0.1 with fit_prior TRUE is " +
      str(round(f1_score(Y_dev, y_pred, average='macro'), 3)))

count = 0
labels = {}
name = "dev-"
for p in y_pred:
    temp = {}
    temp["label"] = int(p)
    id = name + str(count)
    count += 1
    labels[id] = temp

with open("dev-nbmultinomial.json", 'w') as file:
    file.write(json.dumps(labels))

# Linear Logistic Regression's hyperparameters are the solver type & C value (inverse of regularisation strength!)
# Avoiding 'lbfgs' becuase of the convergance warnings and ‘sag’, ‘saga’ because of our data (small; 900 records!)

clf_regression = LogisticRegression(random_state=0, solver="newton-cg", C=10)
clf_regression.fit(X_train_vect, Y_train)
y_pred = clf_regression.predict(X_dev_vect)
print("F-score: for C-value: 10 with solver type newton-cg is " +
      str(round(f1_score(Y_dev, y_pred, average='macro'), 3)))

count = 0
labels = {}
name = "dev-"
for p in y_pred:
    temp = {}
    temp["label"] = int(p)
    id = name + str(count)
    count += 1
    labels[id] = temp

with open("dev-regression.json", 'w') as file:
    file.write(json.dumps(labels))

# Nearest Neughbor Classifier - Optimal K Value = 2
neigh = KNeighborsClassifier(n_neighbors=2)
neigh.fit(X_train_vect, Y_train)
y_pred = neigh.predict(X_dev_vect)
print("F-score: for 2 number of neighbors is " +
      str(round(f1_score(Y_dev, y_pred, average='macro'), 3)))

count = 0
labels = {}
name = "dev-"
for p in y_pred:
    temp = {}
    temp["label"] = int(p)
    id = name + str(count)
    count += 1
    labels[id] = temp

with open("dev-knn.json", 'w') as file:
    file.write(json.dumps(labels))

# Decision Tree Classifier - Optimal Min Samples Split Value = 0.1
decisionTreeClassifier = DecisionTreeClassifier(min_samples_split=0.1)
decisionTreeClassifier.fit(X_train_vect, Y_train)
y_pred = decisionTreeClassifier.predict(X_dev_vect)
print("F-score: for min_samples_split value: 0.1 is " +
      str(round(f1_score(Y_dev, y_pred, average='macro'), 3)))

count = 0
labels = {}
name = "dev-"
for p in y_pred:
    temp = {}
    temp["label"] = int(p)
    id = name + str(count)
    count += 1
    labels[id] = temp

with open("dev-decisiontree.json", 'w') as file:
    file.write(json.dumps(labels))

# Support Vector Classifier - Optimal C Value = 1
svc_model = SVC(gamma="auto", kernel="sigmoid", C=1.0)
svc_model.fit(X_train_vect, Y_train)
y_pred = svc_model.predict(X_dev_vect)
print("F-score: for C value: 1 is " +
      str(round(f1_score(Y_dev, y_pred, average='macro'), 3)))

count = 0
labels = {}
name = "dev-"
for p in y_pred:
    temp = {}
    temp["label"] = int(p)
    id = name + str(count)
    count += 1
    labels[id] = temp

with open("dev-svc.json", 'w') as file:
    file.write(json.dumps(labels))

# Random Forest Classifier - Optimal Estimators Value = 2 & Optimal Max Depth Value = 30
clf = RandomForestClassifier(n_estimators=2, max_depth=30, random_state=0)
clf.fit(X_train_vect, Y_train)
y_pred = clf.predict(X_dev_vect)
print("F-score for max-depth value: 30 is " +
      str(round(f1_score(Y_dev, y_pred, average='macro'), 3)))

count = 0
labels = {}
name = "dev-"
for p in y_pred:
    temp = {}
    temp["label"] = int(p)
    id = name + str(count)
    count += 1
    labels[id] = temp

with open("dev-randomforest.json", 'w') as file:
    file.write(json.dumps(labels))
