**General Analysis of 2015 Elections**

**Introduction**

Predicting the elections have always been one of the hot topics in the data mining field. There are different ways to predict the elections. One of the methods is using sentiment analysis of tweets to predict the election results. Sentiment analysis is the method of extracting the tweets and performing text processing on the tweets to convert the text into a machine-readable format using which we can predict the outcome or perform analysis on the tweets.  

In this project, we have used two datasets. The first dataset was obtained from Elections Canada - Elections Canada is the independent, non-partisan agency responsible for conducting federal elections and referendums [1]. This dataset contained the election information like the province, district, polling results, candidate's and party data and the other dataset used was the twitter tweets.  

After creating the model and training the model, three algorithms were applied to compare the accuracy of the results. The three algorithms used were Naïve Bayes, Random Forest and Decision tree classifier.  
 
**Libraries Used**

1. **Pandas:** Pandas Library was used to fetch the JSON and CSV files. 

2. **Google-translate:** Google-translate library was used in converting French text into English. 

3. **Scikit-learn:** This library was used to apply machine learning algorithms. 

4. **Numpy:** Numpy was used in array processing of the text. 

**Steps to execute**

*Dataset 1*

1. Extract the cleaned\_version\_v3 and run the code Model\_1.py to obtain the accuracy and the output from the first dataset.

*Dataset 2*

1. First run the fetch_tweets.py file to obtain the tweets. (Optional, if you want a complete dataset, else sample dataset can be used) 
2. Since the file was too large we could not upload the file hence we have put a sample dataset for evaluation purpose. 
3. Then run the code Final\_Model_Tweets.py to obtain the accuracy, precision, recall and f1-score for all the algorithms.

