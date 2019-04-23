import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn import tree
import numpy as np

#Read dataset
data=pd.read_csv('cleaned_version_v3.csv',sep = ",")

#Classify the labels
classifier1=[]
for i in data['Party']:
    if i=='Liberal':
        classifier1.append(0)
    elif i=='NDP-New Democratic Party':
        classifier1.append(1)
    elif i=='Bloc Québécois':
        classifier1.append(2)
    elif i=='Conservative':
        classifier1.append(3)
    elif i=='Green Party':
        classifier1.append(0)
data['Classifier']=classifier1


#PErform text processing using count vectorizer
vectorizer = CountVectorizer()
X= data['Province']
y= data['Classifier']

#Split into train and test data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.33)

#Train the model using triaining dataset and then pass test dataset through it.
vectorizer = CountVectorizer(stop_words='english').fit(X_train)
df_train = pd.DataFrame(vectorizer.transform(X_train).todense(), columns =vectorizer.get_feature_names())
df_test = pd.DataFrame(vectorizer.transform(X_test).todense(), columns =vectorizer.get_feature_names())

Z= data['Electoral District Name/Nom de circonscription']
t = data['Classifier']

#Same procedure for another set of features
Z_train, Z_test, t_train, t_test = train_test_split(Z,t,test_size = 0.33)
vectorizer = CountVectorizer(stop_words='english').fit(Z_train)
dfz_train = pd.DataFrame(vectorizer.transform(Z_train).todense(), columns =vectorizer.get_feature_names())
dfz_test = pd.DataFrame(vectorizer.transform(Z_test).todense(), columns =vectorizer.get_feature_names())


train = pd.concat([df_train,dfz_train], axis=1)
test = pd.concat([df_test,dfz_test], axis=1)

#Train the models
clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = tree.DecisionTreeClassifier(random_state=1)

#Perform 5-fold cross validation
print('5-fold cross validation:\n')

labels = ['Logistic Regression', 'Random Forest', 'Decision Tree Classifier']

for clf, label in zip([clf1, clf2, clf3], labels):

    scores = model_selection.cross_val_score(clf, train,y_train,
                                              cv=5, 
                                              scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]"
          % (scores.mean(), scores.std(), label))
    


