import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import accuracy_score
import pickle

class SourceReputation():

    def __init__(self):

        basedir = pickle.load(open('./models/basedir.pkl', 'rb'))
        fakenewsfile = basedir + 'input_data/politifact-fakenews-sites.csv'

        global dataTrain
        global accscore

        self.dataFakeNewsSites = pd.read_csv(fakenewsfile, sep=',')
        display(self.dataFakeNewsSites.head())
        print(self.dataFakeNewsSites['type of site'].unique())

        for index, row in self.dataFakeNewsSites.iterrows():
            score = 1
            if (row['type of site'] == 'some fake stories'):
                score = 0.5
            self.dataFakeNewsSites.at[index, 'fake_score'] = score

        self.dataFakeNewsSites.head()


    def predictScore(self, source):
        if (source == ""):
            return 0
        d = self.dataFakeNewsSites[self.dataFakeNewsSites['site name'].str.match(source)]
        if (d['fake_score'].empty):
            return 0
        return int(d['fake_score'].values)


if __name__ == "__main__":

    basedir = pickle.load(open('./models/basedir.pkl', 'rb'))
    sourcereputation_filename_pkl = basedir + 'models/sourcereputation_feature_av4.pkl'
    sr = SourceReputation()
    text1 = "24wpn"
    print(sr.predictScore(text1), text1)
    pickle.dump(sr, open(sourcereputation_filename_pkl, 'wb'))
    del sr

