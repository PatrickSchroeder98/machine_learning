import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

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
