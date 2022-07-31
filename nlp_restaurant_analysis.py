# Importing Libraries Numpy and Pandas

import numpy as np
import pandas as pd

# Saving the imported dataset in 'data' dataframe

data = pd.read_csv('C:\Users\Administrator\Desktop\Ishika\Work\NLP_ Restaurant_Review\Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

# Printing values of dataframe 'data'

data.head

# Importing Libraries NLTK - Natural Language Processing Toolkit and Regex

import nltk
import re

# Downloading and removing stopwords (words that do not add meaning to text) using Porter Stemmer

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Defining corpus, our set of words to be judged on

corpus = []

for i in range(0,1000):
    review = re.sub(pattern='[^a-zA-Z]', repl = ' ', string = data['Review'][i])
    review = review.lower()
    reviewgiven = review.split()
    reviewgiven = [word for word in reviewgiven if not word in set(stopwords.words('english'))]
    ps = PorterStemmer()
    reviewgiven = [ps.stem(word) for word in reviewgiven]
    review = ''.join(review)
    corpus.append(review)

# Printing and checking first few values of the corpus

corpus[0:20]

# Tokenizes words and creates a sparse matrix for the frequency

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=500)
x = cv.fit_transform(corpus)
y = data.iloc[:, 1].values

# Splitting the dataset into Training and Testing data

from sklearn.model_selection import train_test_split
xtrain, xtest = train_test_split, ytrain, ytest = train_test_split(x, y, test_size = 0.40, random_state = 0)

# Importing Multinomial Naive Bias Classifier

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(xtrain, ytrain)
ypredict = classifier.predict(xtest = train_test_split)

# Calculating Accuracy, Precision and Recall Percentage

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

score1 = accuracy_score(ytest,ypredict)
score2 = precision_score(ytest,ypredict)


# Printing Accuracy, Precision and Recall Percentage

print("Accuracy:", score1*100,"%")
print("Precision:", score2*100,"%")


# Creating Confusion Matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest, ypredict)
print(cm)

# Hyperparameter tuning the Naive Bayes Classifier

best_accuracy = 0.0
alpha_val = 0.0
for i in np.arange(0.1,1.1,0.1):
    temp_classifier = MultinomialNB(alpha=i)
    temp_classifier.fit(xtrain, ytrain)
    temp_y_pred = temp_classifier.predict(xtest = train_test_split)
    score = accuracy_score(ytest, temp_y_pred)
    print("Accuracy score for alpha={} is: {}%".format(round(i,1), round(score*100,4)))
    if score>best_accuracy:
      best_accuracy = score
      alpha_val = i
print('The best accuracy is {}% with alpha value as {}'.format(round(best_accuracy*100, 4), round(alpha_val,1)))

classifier = MultinomialNB(alpha=0.4)
classifier.fit(xtrain, ytrain)

# Predicting Reviews

def predict_sentiment(sample_review):
   sample_review = re.sub(pattern='[^a-zA-Z]',repl=' ', string = sample_review)
   sample_review = sample_review.lower()
   sample_review_words = sample_review.split()
   sample_review_words = [word for word in sample_review_words if not word in set(stopwords.words('english'))]
   ps = PorterStemmer()
   final_review = [ps.stem(word) for word in sample_review_words]
   final_review = ' '.join(final_review)
   temp = cv.transform([final_review]).toarray()
   return classifier.predict(temp)

sample_review = 'The food is really good here.'

if predict_sentiment(sample_review):
   print('This is a POSITIVE review.')
else:
   print('This is a NEGATIVE review!')

sample_review = 'Food was pretty bad and the service was very slow.'

if predict_sentiment(sample_review):
   print('This is a POSITIVE review.')
else:
   print('This is a NEGATIVE review!')