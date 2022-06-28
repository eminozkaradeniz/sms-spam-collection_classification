import pandas as pd
messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t', names=['label', 'message'])
print(messages.groupby('label').describe(), '\n')


# Tokenization and Stemming 
from nltk.corpus import stopwords
STOPWORDS = stopwords.words('english')

import string
PUNCTUATIONS = string.punctuation

from nltk.stem import PorterStemmer
ps = PorterStemmer()


def text_process(message):
    message = [c for c in message if c not in PUNCTUATIONS]
    message = ''.join(message)
    message = [word for word in message.split() if word.lower() not in STOPWORDS]
    message = [ps.stem(word) for word in message]
    return message


# Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(analyzer=text_process).fit(messages['message'])
tfidf_messages = tfidf_vectorizer.transform(messages['message'])


from sklearn.model_selection import train_test_split
messages_train, messages_test, label_train, label_test = train_test_split(tfidf_messages,
                                                                          messages['label'],
                                                                          test_size=0.25)

# Model
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

clf = SVC()
parameters = {'kernel':('linear', 'poly', 'rbf', 'sigmoid'), 'C':[1, 5, 10, 20]}

cv = GridSearchCV(clf, parameters)
cv.fit(messages_train, label_train)
pred = cv.predict(messages_test)


from sklearn.metrics import classification_report
print(classification_report(label_test, pred))


def spam_detection(message):
    message = tfidf_vectorizer.transform([message])
    return cv.predict(message)[0]


sample_spam_message = "IMPORTANT - You could be entitled up to £3,160 in compensation from " \
                        "mis-sold PPI on a credit card or loan. Please reply PPI for info or STOP to opt out."
print(spam_detection(sample_spam_message))

sample_spam_message = "You’ve won a prize! Go to the link to claim your $500 Amazon gift card."
print(spam_detection(sample_spam_message))

sample_ham_message = "Sorry i cant come, i've an exam tomorrow"
print(spam_detection(sample_ham_message))

sample_ham_message = "Dear Caroline. Your appointment with Dr. Brown is scheduled for 2 p.m. today." \
                        "Please arrive at the clinic 10 mins prior."
print(spam_detection(sample_ham_message))
