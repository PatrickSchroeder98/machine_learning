import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
