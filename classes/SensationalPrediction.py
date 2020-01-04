import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import accuracy_score
import pickle

class SensationalPrediction():

    def __init__(self):

        basedir = pickle.load(open('./models/basedir.pkl', 'rb'))
        trainfile = basedir + 'input_data/processed/trainnews_sensational_processed.csv'
        testfile = basedir + 'input_data/processed/testnews_sensational_processed.csv'

        global dataTrain
        global accscore

        dataTrain = pd.read_csv(trainfile, sep=',')
        dataTest = pd.read_csv(testfile, sep=',')

        tfidfV = TfidfVectorizer(ngram_range = (1,3), sublinear_tf = True)

        self.nb_pipeline_ngram = Pipeline([
        ('vector', tfidfV),
        ('mname',MultinomialNB())])

        self.nb_pipeline_ngram.fit(dataTrain['headline_text'], dataTrain['sensational_label'])
        predicted_nb_ngram = self.nb_pipeline_ngram.predict(dataTest['headline_text'])
        accscore = metrics.accuracy_score(dataTest['sensational_label'], predicted_nb_ngram)
        print("Sensational Feature Prediction - accuracy:   %0.6f" % accscore)

    def predict(self, text):
        predicted = self.nb_pipeline_ngram.predict([text])
        predicedProb = self.nb_pipeline_ngram.predict_proba([text])[:,1]
        return bool(predicted), float(predicedProb)

    def predictScore(self, text):
        predicedProb = self.nb_pipeline_ngram.predict_proba([text])[:,1]
        return float(predicedProb)

    def getScore(self):
        return accscore

if __name__ == "__main__":

    basedir = pickle.load(open('./models/basedir.pkl', 'rb'))
    sensational_filename_pkl = basedir + 'models/sensational_feature_av4.pkl'
    sp = SensationalPrediction()
    text1 = "Hillary Clinton agrees with John McCain \"by voting to give George Bush the benefit of the doubt on Iran.\""
    text2 = "Most of the (Affordable Care Act) has already in some sense been waived or otherwise suspended."
    print(sp.predict(text1), text1)
    print(sp.predict(text2), text2)
    pickle.dump(sp, open(sensational_filename_pkl, 'wb'))
    del sp

