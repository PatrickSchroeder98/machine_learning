import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# nltk.download()

messages = [line.rstrip() for line in open('smsspamcollection/SMSSpamCollection', encoding='utf-8')]
print(len(messages))

for message_no, message in enumerate(messages[:10]):
    print(message_no, message)
    print('\n')

messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t',
                           names=["label", "message"])
print(messages.head())
print(messages.describe())
print(messages.groupby('label').describe())

messages['length'] = messages['message'].apply(len)
print(messages.head())

messages['length'].plot(bins=50, kind='hist')
plt.show()

print(messages.length.describe())

print(messages[messages['length'] == 910]['message'].iloc[0])

messages.hist(column='length', by='label', bins=50,figsize=(12,4))
plt.show()

# Text pre-processing

def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)

    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

print(messages['message'].head(5).apply(text_process))

bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])

# Print total number of vocab words
print(len(bow_transformer.vocabulary_))

message4 = messages['message'][3]
print(message4)

bow4 = bow_transformer.transform([message4])
print(bow4)
print(bow4.shape)

print(bow_transformer.get_feature_names_out()[4068])
print(bow_transformer.get_feature_names_out()[9554])

messages_bow = bow_transformer.transform(messages['message'])
print('Shape of Sparse Matrix: ', messages_bow.shape)
print('Amount of Non-Zero occurences: ', messages_bow.nnz)
sparsity = (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))
print('sparsity: {}'.format(sparsity))

tfidf_transformer = TfidfTransformer().fit(messages_bow)
tfidf4 = tfidf_transformer.transform(bow4)
print(tfidf4)

messages_tfidf = tfidf_transformer.transform(messages_bow)
print(messages_tfidf.shape)

spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])
print('predicted:', spam_detect_model.predict(tfidf4)[0])
print('expected:', messages.label[3])

all_predictions = spam_detect_model.predict(messages_tfidf)
print(all_predictions)
# For model trained on all data
print (classification_report(messages['label'], all_predictions))

# Training model on split data
msg_train, msg_test, label_train, label_test = \
train_test_split(messages['message'], messages['label'], test_size=0.2)

print(len(msg_train), len(msg_test), len(msg_train) + len(msg_test))

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

pipeline.fit(msg_train,label_train)
predictions = pipeline.predict(msg_test)

print(classification_report(predictions,label_test))
