import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import *
from sklearn import metrics
import warnings

warnings.filterwarnings('ignore')


class SentimentAnalysis():

    def __init__(self):

        basedir = pickle.load(open('./models/basedir.pkl', 'rb'))
        trainfile = basedir + 'input_data/processed/trainnews_sensational_processed.csv'
        testfile = basedir + 'input_data/processed/testnews_sensational_processed.csv'

        global dataTrain
        global accscore

        dataTrain = pd.read_csv(trainfile, sep=',')
        dataTest = pd.read_csv(testfile, sep=',')


        tfidfV = TfidfVectorizer(stop_words='english', min_df=5, max_df=30, use_idf=True, smooth_idf=True,
                                 token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b')

        self.logR_pipeline = Pipeline([
            ('LogRCV', tfidfV),
            ('LogR_clf', LogisticRegression(solver='liblinear', C=32 / 100))
        ])

        self.logR_pipeline.fit(dataTrain['headline_text'], dataTrain['vader_polarity'])
        predicted_LogR = self.logR_pipeline.predict(dataTest['headline_text'])
        score = metrics.accuracy_score(dataTest['vader_polarity'], predicted_LogR)
        print("Sentiment Analysis Model Trained - accuracy:   %0.6f" % score)

    def predict(self, text):
        predicted = self.logR_pipeline.predict([text])
        predicedProb = self.logR_pipeline.predict_proba([text])[:, 1]
        return bool(predicted), float(predicedProb)

    def predictScore(self, text):
        predicedProb = self.logR_pipeline.predict_proba([text])[:,1]
        return float(predicedProb)

    def getScore(self):
        return accscore


if __name__ == "__main__":
    basedir = pickle.load(open('./models/basedir.pkl', 'rb'))
    sentimental_filename_pkl = basedir + 'models/sentimental_feature_av3.pkl'
    sp = SentimentAnalysis()
    text1 = "Says the Annies List political group supports third-trimester abortions on demand."
    text2 = "Most of the (Affordable Care Act) has already in some sense been waived or otherwise suspended."
    print(sp.predict(text1), text1)
    print(sp.predict(text2), text2)
    pickle.dump(sp, open(sentimental_filename_pkl, 'wb'))
    del sp
