import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import nltk
from nltk.stem import WordNetLemmatizer 
from sklearn.metrics import f1_score, precision_score, recall_score
  

data=pd.read_csv('fetched_tweets_final.csv',sep = ",")

#To clean the tweets
#This was referred from https://www.geeksforgeeks.org/twitter-sentiment-analysis-using-python/
def clean_tweet(tweet): 
        '''
        Utility function to clean tweet text by removing links, special characters 
        using simple regex statements. 
        '''
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)", " ", tweet).split()) 


#To remove the retweets from the dataset
a=[]
for i in data["Tweets"]:
    if 'RT' in i:
        pass
    else:
        a.append(i)

#To assign the labels in the dataset for classification
z = []
for line in a:
    if 'lpc' in line:
        z.append(1)
    elif 'trudeau' in line:
        z.append(1)
    elif 'ndp' in line:
        z.append(-1)
    else:
        z.append(0)


#Create a new dataframe from the refined dataset
df = pd.DataFrame({'tweets':a , 'label':z})

#PErform cleaning of the tweets and removing the stopwords
cleaned_tweet = [] 
for i in df['tweets']:
    cleaned_tweet.append(clean_tweet(i))

df['cleaned_tweets'] = cleaned_tweet

stop = set(stopwords.words('english'))
df['cleaned_tweets_v1'] = df['cleaned_tweets'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)])) 

df['tokenized_tweets'] = df.apply(lambda row: nltk.word_tokenize(row["cleaned_tweets_v1"]), axis=1)

#Stemming the tweets to extract the root of the words
#This was referred from https://www.datacamp.com/community/tutorials/stemming-lemmatization-python
stemmer = PorterStemmer()
df["final_tweet_v1"] = df["tokenized_tweets"].apply(lambda x: [stemmer.stem(y) for y in x])
df['final_tweet_v1'] = df['final_tweet_v1'].apply(' '.join)

df['tokenized_tweets_v1'] = df.apply(lambda row: nltk.word_tokenize(row["final_tweet_v1"]), axis=1)

#lemmatizing the tweets to deduce the context of the words
#This was referred from: https://www.geeksforgeeks.org/python-lemmatization-with-nltk/
lemmatizer = WordNetLemmatizer()
df["final_tweet"] = df["tokenized_tweets_v1"].apply(lambda x: [lemmatizer.lemmatize(y) for y in x])
df['final_tweet'] = df['final_tweet'].apply(' '.join)


X = df['final_tweet']
y = df['label']

#Splitting the dataset into testing and training dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.33)

train_words = np.array(X_train)
test_words = np.array(X_test)

#Transfoming the training dataset by performing text processing
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(train_words)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#Transforming the test dataset
X_new_counts = vectorizer.transform(test_words)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

#Creating the models for the three algorithms and to train the model using th training data set and predict the test dataset
clf1 = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train_tfidf, y_train)
predicted = clf1.predict(X_new_tfidf)

clf2 = RandomForestClassifier().fit(X_train_tfidf, y_train)
predicted_rb = clf2.predict(X_new_tfidf)

clf3 = tree.DecisionTreeClassifier().fit(X_train_tfidf, y_train)
predicted_dc = clf3.predict(X_new_tfidf)

#Perform 5-fold cross validation for the three algorithms. 
#This code was referred from: http://rasbt.github.io/mlxtend/user_guide/classifier/EnsembleVoteClassifier/
print('5-fold cross validation:\n')

labels = ['Logistic Regression', 'Random Forest', 'Decision Tree Classifier']

for clf, label in zip([clf1, clf2, clf3], labels):

    scores = model_selection.cross_val_score(clf, X_train_tfidf,y_train,
                                              cv=5, 
                                              scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]"
          % (scores.mean(), scores.std(), label))

#Calculate the evaluation metrics for all the three algorithms

print("Decision Tree Classifier")
print ("F1-score",f1_score(y_test,predicted_dc, average="macro"))
print ("Precision",precision_score(y_test,predicted_dc, average="macro"))
print ("Recall",recall_score(y_test,predicted_dc, average="macro"))

print("Random Forest Classifier")
print ("F1-score",f1_score(y_test,predicted_rb, average="macro"))
print ("Precision",precision_score(y_test,predicted_rb, average="macro"))
print ("Recall",recall_score(y_test,predicted_rb, average="macro"))

print("Logistic Regression")
print ("F1-score",f1_score(y_test,predicted, average="macro"))
print ("Precision",precision_score(y_test,predicted, average="macro"))
print ("Recall",recall_score(y_test,predicted, average="macro"))

