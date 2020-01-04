import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import *
from sklearn import metrics
import pickle

class WordFrequency():

    def __init__(self):

        global accscore

        columnNames = ["id", "label", "statement", "subject", "speaker", "speaker_job_title", "state_info",
                       "party_affiliation", "barely_true_counts", "false_counts", "half_true_counts",
                       "mostly_true_counts", "pants_on_fire_counts", "context"]
        dataTrain = pd.read_csv('input_data/dataset/train.tsv', sep='\t', header=None, names=columnNames)
        dataValidate = pd.read_csv('input_data/dataset/valid.tsv', sep='\t', header=None, names=columnNames)
        dataTest = pd.read_csv('input_data/dataset/test.tsv', sep='\t', header=None, names=columnNames)

        # dropping columns
        columnsToRemove = ['id', 'subject', 'speaker', 'context', 'speaker_job_title', 'state_info',
                           'party_affiliation', 'barely_true_counts', 'false_counts', 'half_true_counts',
                           'mostly_true_counts', 'pants_on_fire_counts']
        dataTrain = dataTrain.drop(columns=columnsToRemove)
        dataValidate = dataValidate.drop(columns=columnsToRemove)
        dataTest = dataTest.drop(columns=columnsToRemove)

        def convertMulticlassToBinaryclass(r):
            v = r['label']
            if (v == 'true'):
                return 'true'
            if (v == 'mostly-true'):
                return 'true'
            if (v == 'half-true'):
                return 'true'
            if (v == 'barely-true'):
                return 'false'
            if (v == 'false'):
                return 'false'
            if (v == 'pants-fire'):
                return 'false'

        dataTrain['label'] = dataTrain.apply(convertMulticlassToBinaryclass, axis=1)
        dataValidate['label'] = dataValidate.apply(convertMulticlassToBinaryclass, axis=1)
        dataTest['label'] = dataTest.apply(convertMulticlassToBinaryclass, axis=1)

        tfidfV = TfidfVectorizer(stop_words='english', min_df=5, max_df=30, use_idf=True, smooth_idf=True,
                                 token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b')
        train_tfidf = tfidfV.fit_transform(dataTrain['statement'].values)
        test_tfidf = tfidfV.fit_transform(dataTest['statement'].values)

        #         print('TF-IDF VECTORIZER')

        ## Removing plurals for the tokens using PorterStemmer
        stemmer = PorterStemmer()
        tfidfVPlurals = tfidfV.get_feature_names()
        tfidfVSingles = [stemmer.stem(plural) for plural in tfidfVPlurals]

        # Applying Set to remove duplicates
        tfidfVTokens = list(set(tfidfVSingles))
        #         print('TFIDFV Tokens')
        #         print(tfidfVTokens)

        self.logR_pipeline = Pipeline([
            ('LogRCV', tfidfV),
            ('LogR_clf', LogisticRegression(solver='liblinear', C=32 / 100))
        ])

        self.logR_pipeline.fit(dataTrain['statement'], dataTrain['label'])
        predicted_LogR = self.logR_pipeline.predict(dataTest['statement'])
        accscore = metrics.accuracy_score(dataTest['label'], predicted_LogR)
        print("Word Frequency Model Trained - accuracy:   %0.6f" % accscore)

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
    wordfrequency_filename_pkl = basedir + 'models/sentimental_feature_av4.pkl'
    wf = WordFrequency()
    text1 = "Says the Annies List political group supports third-trimester abortions on demand."
    text2 = "Most of the (Affordable Care Act) has already in some sense been waived or otherwise suspended."
    print(wf.predict(text1), text1)
    print(wf.predict(text2), text2)
    pickle.dump(wf, open(wordfrequency_filename_pkl, 'wb'))
    del wf
