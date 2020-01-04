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

class SourceContent():

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

        dataAllBodyLength = dataAll.copy()
        for index, row in dataAllBodyLength.iterrows():
            textLength = len(row['text'])
            dataAllBodyLength.at[index, 'text_length'] = textLength

        import numpy as np
        import matplotlib.pyplot as plt

        from sklearn.linear_model import LinearRegression

        self.linearRegressionBodyLength = LinearRegression(fit_intercept=True)

        A = np.array(list(dataAllBodyLength.text_length))
        B = np.array(list(dataAllBodyLength.veracity))

        self.linearRegressionBodyLength.fit(A[:, np.newaxis], B)

        xfit = np.linspace(-1, max(dataAllBodyLength.text_length), 1000)
        yfit = self.linearRegressionBodyLength.predict(xfit[:, np.newaxis])

        plt.scatter(A, B, s=1, c="orange")
        plt.plot(xfit, yfit);

        print("Model slope:    ", self.linearRegressionBodyLength.coef_[0])
        print("Model intercept:", self.linearRegressionBodyLength.intercept_)
        print("R2 score:", self.linearRegressionBodyLength.score(A[:, np.newaxis], B))

        for index, row in dataTrain.iterrows():
            textLength = len(row['text'])
            dataTrain.at[index, 'text_length'] = textLength

        for index, row in dataTest.iterrows():
            textLength = len(row['text'])
            dataTest.at[index, 'text_length'] = textLength

        from sklearn import linear_model

        self.logClassifierBodyLength = linear_model.LogisticRegression(solver='liblinear', C=17 / 1000, random_state=111)
        self.logClassifierBodyLength.fit(dataTrain['text_length'].values.reshape(-1, 1), dataTrain['veracity'].values)

        predicted = self.logClassifierBodyLength.predict(dataTest['text_length'].values.reshape(-1, 1))

        from sklearn import metrics
        accscore = metrics.accuracy_score(dataTest['veracity'].values.reshape(-1, 1), predicted)
        print("Source Content - accuracy:   %0.6f" % accscore)

    def predict(self, length):
        x = np.array(length).reshape(-1, 1)
        predicted = self.logClassifierBodyLength.predict(x)
        predicedProb = self.logClassifierBodyLength.predict_proba(x)[:, 1]
        return bool(predicted), float(predicedProb)

    def predictScore(self, length):
        x = np.array(length).reshape(-1, 1)
        predicedProb = self.logClassifierBodyLength.predict_proba(x)[:,1]
        return float(predicedProb)

    def getScore(self):
        return accscore

if __name__ == "__main__":
    basedir = pickle.load(open('./models/basedir.pkl', 'rb'))
    sourcecontent_filename_pkl = basedir + 'models/sourcecontent_feature_av4.pkl'
    sc = SourceContent()
    print(sc.predict(12000), 12000)
    pickle.dump(sc, open(sourcecontent_filename_pkl, 'wb'))
    del sc
