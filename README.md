# AlternusVera
Alternus Vera Project

## Google drive project colab location
https://drive.google.com/open?id=1WavRqUOw8bNuxMlaqINGwcoNysih1A9C

## Project Video

Sprint 2: https://drive.google.com/file/d/1aFkR8U6Rwa74zaddxCiNoMpJsCqVuqPZ/view?usp=sharing  
Sprint 3: https://drive.google.com/file/d/1P1-xtOlhvRIpmpJbyKb3ZucZJFD5KT8T/view?usp=sharing  

## Datasets:

https://github.com/BuzzFeedNews/2016-10-facebookfact-check/tree/master/data 

https://www.cs.ucsb.edu/william/data/liardataset.zip 

https://www.kaggle.com/mrisdal/fake-news

https://github.com/bs-detector/bs-detector 

http://compsocial.github.io/CREDBANK-data/

http://www.politifact.com/



## Alternus Vera Final WorkBook  
Name: Sudha Amarnath  
Student ID: 013709956  
Business Problem  
The widespread propagation of false information online is not a recent phenomenon but its perceived impact in the 2016 U.S. presidential election has thrust the issue into the spotlight. Technology companies are already exploring machine learning-based approaches for solving the problem. In this project, we are using NLP based text classification to identify the different news categories.  

## Data
Liar Liar Data Set - https://drive.google.com/drive/folders/1IVl4Qt92LZwvMlnJGRcEZVKF-dpawhdz?usp=sharing  

## Description for Fake News Classification Selected
LIAR: A BENCHMARK DATASET FOR FAKE NEWS DETECTION

William Yang Wang, "Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection, to appear in Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (ACL 2017), short paper, Vancouver, BC, Canada, July 30-August 4, ACL.

## Description of the TSV format:

Column 1: the ID of the statement ([ID].json). Column 2: the label. Column 3: the statement. Column 4: the subject(s). Column 5: the speaker. Column 6: the speaker's job title. Column 7: the state info. Column 8: the party affiliation. Column 9-13: the total credit history count, including the current statement. 9: barely true counts. 10: false counts. 11: half true counts. 12: mostly true counts. 13: pants on fire counts. Column 14: the context (venue / location of the speech or statement).  

Note that we do not provide the full-text verdict report in this current version of the dataset, but you can use the following command to access the full verdict report and links to the source documents: wget http://www.politifact.com//api/v/2/statement/[ID]/?format=json  

Social acceptance = # of likes, # of comments (short term utility)  
Bias Score  
Spam Score  
Website credibility/ Domain Ranking  
Author credibility  
Political Affliation  
Occurance Location (Probability of announcing on Radio or Press release being fake is low)  
Sensationalism/Psychology Utility - agreeing with reader's prior beliefs  
Frequency Heuristic - Constant repetition makes them believe (Sensationlism)  
Echo Chamber - Forming groups and spreading opinions  
Visual - Images, Links, Videos  

## Feature 1 - News Coverage
The main idea is to find the integerity of the liar liar dataset topics against a source which could be the actual media like News Papers. There are high chances for hte positive corelation when the comparision is done with the more reliable source like the News Channels. For this task, I am considering the News Coverage Dataset from Kaggle [ https://www.kaggle.com/alvations/old-newspapers ]. This Old News Dataset from Kaggle, originally comes up from different languages. Since we are intersted only in English, this dataset is preprocessed to select only English News Articles [ https://drive.google.com/open?id=1S_GZ9xkRJ30HR9IYXMccF2UikWX-Zj1R ]. The liar liar dataset topics span over a decade , likely the news are from (2005-2015). Since the old news data setup after preprocessing is similar for the coverage in year wise, there could he high chances of co-relation. We then use this feature to perform Fake and not Fake classification for the Liar Liar Data set.

## The assignment explains 3 different approaches to classify text based on the news coverage information. The different approaches are as below
CountVectorizer  
Doc2Vec Model  
TF-IDF Vectorizer  


## The Performance of these approaches are evaluated based on the accuracy score using the following algorithms.
Multinomial Naive Bayes  
SVM  
SGD  
Random Forest  
Logistic Regression  


## Data Preprocessing
Remove non-letters/Special Characters and Punctuations  
Convert to lower case  
Remove punctuation  
Tokenize  
Remove stop words  
Lemmentize  
Stemming  
Remove small words of length < 3  

## What didn't work?
As for the News Coverage Assignment in the Sprint One, the Old News data set is way to big having nearly 1M rows of topics. Initially I tried it as the doc2vector train set and the test set being the liar liar data sets. The doc2vector model apporach for the complete train data set was too time consuming. From the initial estimate of parsing the vectors was taking more than 24 hours. Another method tried was to find the cosine similarity between the topics in the two files. But this being not so better than the doc2vector approach, ti wasn't considered for the news coverage topic modelling.  

## What worked later?
Since we are looking for a non biased labels in the test data set, I reduced the size of the Old News to randomly pick 100k news rows. This helped greatly in reducing the overall run time of the project to nearly 90 mintues. The accuracy of the predicted model for the News Coverage feature across all algorithms are nearly 50-60%.  

## Feature 2 - Sensational Feature Prediction
With the close look of the words, and when some of them are combined selectively together, there are cues which would lead to emotions in the way the speaker has said in a certain context. Words when used correctly can transform an “eh whatever” into “wow that’s it!”. Words can make you go from literally ROFL to fuming with fury to an uncontrollable-urge-to-take-action-NOW-or-the-earth-may-stop-swinging -on-its-axis. Highly emotional words are capable capable of transforming an absolute no into almost yes and a “perhaps” into “for sure”! Words that are used:  

When you are trying to sell people a solution  
When you are trying to get them to take an action (like, share, subscribe, buy)  
When you are trying to get people to click and read your article  
When you are trying to get someone to agree with you  
I am using a dataset from high emotion persiasive words [ https://www.thepersuasionrevolution.com/380-high-emotion-persuasive-words/ ] where there are 1400+ words that are both positive and negative emotions that will help to predict the sensational score for an article. The data enrichment is done using SentiNet library which provides polarity associated with 50,000 natural language concepts. A polarity is a floating number between -1 and +1. Minus one is extreme negativity, and plus one is extreme positivity. The knowledge base is free. It can be downloaded as XML file. SenticNet 5 reaches 100,000 commonsense concepts by employing recurrent neural networks to infer primitives by lexical substitution.

## Method used : 

By performing cosine similarity for each news in the Liar Liar Data set with the Sensational words results in a particular score for each topic. These topics are then given a sensational label based on the 50% sensataional score. For the score above 50% value, the sensational label is predicted as 1 otherwise its 0. Then I used TFIDF Vectorizer and Multinomial Naive Bayes algorithm. The Accuracy for this model achieved is improved to 60%.

## What is included in this assignment 2 compared to AV assignment 1?
Modular approach is being considered now for the team in a centralized directory.  
Sensational Feature is integrated in assignment 2  
Separate functions have been included for the features  
NewsCoverage() Class is defined based on TFIDF Vectorizer and Multinomial Naive Bayes algorithm to easily predict the headline text is fake or not fake.  
SensationalPrediction() Class is defined using TFIDF Vectorizer and Multinomial Naive Bayes algorithm to easily predict the headline text is fake or not fake.  
Sensational score is improved from Main Alternus Vera Git Hub Forked code (20% vs 60%).  

## What is included in this assignment 3 compared to AV assignment 2?
Modular approach is being considered now for the team in a centralized directory.  
Redefined the NewsCoverage() and SensationalPrediction() classes.  
Changed the algorithm for NewsCoverage Prediction to use the top document match from doc2vector output.  
For the NewsCoverage() Class Object pickle file is created at ../models/newscoverage_feature.pkl  
For the SensationalPrediction() Class Object pickle file is created at ../models/sensational_feature.pkl  
All the data sets and Models are located in AlternusVeraDataSets2019/Spartans/Sudha/input_data  
The Models are located in AlternusVeraDataSets2019/Spartans/Sudha/models  
Pickle load the NewsCoverage() Class Object and test the train_news head_line text for True and False Values.  
Pickle load the SensationalPrediction() Class Object and test the train_news head_line text for True and False Values.  

## Whats integrated in the final Workbook?
Integrated all other AV2 features  
Modularisation of features all classes  
Importing Classes as packages  
Importing Pickle Model  
Polynomial Equation  
