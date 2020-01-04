from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import *
from sklearn import metrics
import warnings
import pickle

warnings.filterwarnings('ignore')



class SpamScoreFeature():
    def __init__(self):

        global accscore

        # load the dataset
        columnNames = ["encoded_label", "headline_text", "sensational_vector"]
        dataTrain = pd.read_csv('input_data/processed/trainnews_sensational_processed.csv', sep=',', header=None, names=columnNames)
        dataTest = pd.read_csv('input_data/processed/testnews_sensational_processed.csv', sep=',', header=None, names=columnNames)
        dataTrain = dataTrain.loc[1:]
        dataTest = dataTest.loc[1:]

        # load the spam dictionary
        spam_dict = pd.read_csv('input_data/spam/spam_dict.csv', usecols=[1], names=['spamword'], encoding='latin-1',
                                error_bad_lines=False)
        spam_dict = spam_dict.fillna(0)
        spam_dict = spam_dict.iloc[1:]
        spam_dict = spam_dict.drop_duplicates()

        # spam_dict.head(5)
        # Count vector for train data
        spamcountV = CountVectorizer(vocabulary=list(set(spam_dict['spamword'])))
        train_count = spamcountV.fit_transform(dataTrain['headline_text'].astype('U'))

        self.logR_pipeline = Pipeline([
            ('NBCV', spamcountV),
            ('nb_clf', MultinomialNB())])

        self.logR_pipeline.fit(dataTrain['headline_text'].astype('U'), dataTrain['encoded_label'])
        predicted_LogR = self.logR_pipeline.predict(dataTest['headline_text'].astype('U'))
        accscore = metrics.accuracy_score(dataTest['encoded_label'], predicted_LogR)
        print("Spam Score Model Trained - accuracy:   %0.6f" % accscore)

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
    spamscore_filename_pkl = basedir + 'models/spamscore_feature_av4.pkl'
    sp = SpamScoreFeature()
    text1 = "Says the Annies List political group supports third-trimester abortions on demand."
    print(sp.predict(text1), text1)
    pickle.dump(sp, open(spamscore_filename_pkl, 'wb'))
    del sp
