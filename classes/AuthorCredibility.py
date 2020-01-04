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

import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
import pandas as pd
import numpy as np
import os

class AuthorCredibility():

    def __init__(self):


        def loadJsonFiles(directory, veracity):
            shouldAppend = False
            for filename in os.listdir(directory):
                df2 = pd.read_json(directory + filename, lines=True)
                if (shouldAppend):
                    df = df.append(df2, ignore_index=True, sort=True)
                else:
                    df = df2
                df['veracity'] = veracity
                shouldAppend = True

            # removing nan values
            df['source'].fillna("", inplace=True)
            for index, row in df.iterrows():
                if (type(row['authors']) == float):
                    df.at[index, 'authors'] = []

            # removing unnecessary columns
            df = df.drop(columns=['keywords', 'meta_data', 'movies', 'keywords', 'summary', 'publish_date', 'top_img'])
            return df

        def loadDataset():
            dataFake = loadJsonFiles('input_data/politifact/FakeNewsContent/', 0)
            dataReal = loadJsonFiles('input_data/politifact/RealNewsContent/', 1)
            return dataReal, dataFake


        global accscore

        # load the dataset
        columnNames = ["encoded_label", "headline_text", "sensational_vector"]
        dataTrain = pd.read_csv('input_data/processed/trainnews_sensational_processed.csv', sep=',', header=None, names=columnNames)
        dataTest = pd.read_csv('input_data/processed/testnews_sensational_processed.csv', sep=',', header=None, names=columnNames)
        dataTrain = dataTrain.loc[1:]
        dataTest = dataTest.loc[1:]


        dataFake, dataReal = loadDataset()

        dataTrainFake = dataFake[:100]
        dataTrainReal = dataReal[:100]
        dataTestFake = dataFake[101:]
        dataTestReal = dataReal[101:]

        dataTest = dataTestFake.append(dataTestReal, ignore_index=True, sort=True)
        dataTrain = dataTrainFake.append(dataTrainReal, ignore_index=True, sort=True)
        dataAll = dataFake.append(dataReal, ignore_index=True, sort=True)
        dataAll.head()

        dataAllAuthorsVeracity = dataAll.copy()

        fakeZero = 0
        fakeOne = 0
        falseMoreThanOne = 0
        trueZero = 0
        trueOne = 0
        trueMoreThanOne = 0
        for index, row in dataAllAuthorsVeracity.iterrows():
            authorsCount = len(row['authors'])
            dataAllAuthorsVeracity.at[index, 'authors_count'] = len(row['authors'])
            if (authorsCount == 0):
                if (row['veracity'] == 1):
                    trueZero += 1
                else:
                    fakeZero += 1
            elif (authorsCount == 1):
                if (row['veracity'] == 1):
                    trueOne += 1
                else:
                    fakeOne += 1
            elif (authorsCount > 1):
                if (row['veracity'] == 1):
                    trueMoreThanOne += 1
                else:
                    falseMoreThanOne += 1

        print("trueZeroAuthors=", trueZero)
        print("fakeZeroAuthors=", fakeZero)
        print("trueOneAuthors=", trueOne)
        print("fakeOneAuthors=", fakeOne)
        print("trueMoreThanOneAuthors=", trueMoreThanOne)
        print("fakeMoreThanOneAuthors=", falseMoreThanOne)

        columnsToRemove = ['authors', 'canonical_link', 'images', 'source', 'url', 'text', 'title']
        dataAllAuthorsVeracity = dataAllAuthorsVeracity.drop(columns=columnsToRemove)
        dataAllAuthorsVeracity.head()

        dataTrainAuthorsVeracity = dataTrain.copy()
        dataTestAuthorsVeracity = dataTest.copy()

        for index, row in dataTrainAuthorsVeracity.iterrows():
            dataTrainAuthorsVeracity.at[index, 'authors_count'] = len(row['authors'])

        for index, row in dataTestAuthorsVeracity.iterrows():
            dataTestAuthorsVeracity.at[index, 'authors_count'] = len(row['authors'])

        X_train = dataTrainAuthorsVeracity['authors_count'].values.reshape(-1, 1)
        Y_train = dataTrainAuthorsVeracity['veracity'].values
        X_test = dataTestAuthorsVeracity['authors_count'].values.reshape(-1, 1)
        Y_test = dataTestAuthorsVeracity['veracity'].values.reshape(-1, 1)

        from sklearn import linear_model
        self.logClassifierAuthorsCount = linear_model.LogisticRegression(solver='liblinear', C=1, random_state=111)
        self.logClassifierAuthorsCount.fit(X_train, Y_train)
        predicted = self.logClassifierAuthorsCount.predict(X_test)
        from sklearn import metrics
        print("accuracy=", metrics.accuracy_score(Y_test, predicted))
        accscore = metrics.accuracy_score(Y_test, predicted)
        print("Author Credibility Trained - accuracy:   %0.6f" % accscore)

    def predict(self, numAuthors):
        x = np.array(numAuthors).reshape(-1, 1)
        predicted = self.logClassifierAuthorsCount.predict(x)
        predicedProb = self.logClassifierAuthorsCount.predict_proba(x)[:, 1]
        return bool(predicted), float(predicedProb)

    def predictScore(self, numAuthors):
        x = np.array(numAuthors).reshape(-1, 1)
        predicedProb = self.logClassifierAuthorsCount.predict_proba(x)[:,1]
        return float(predicedProb)

    def getScore(self):
        return accscore

if __name__ == "__main__":
    basedir = pickle.load(open('./models/basedir.pkl', 'rb'))
    authorcredibility_filename_pkl = basedir + 'models/authorcredibility_feature_av4.pkl'
    ac = AuthorCredibility()
    text1 = "Says the Annies List political group supports third-trimester abortions on demand."
    print(ac.predict(4), 4)
    pickle.dump(ac, open(authorcredibility_filename_pkl, 'wb'))
    del ac
