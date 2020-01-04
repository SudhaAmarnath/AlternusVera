import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import *
from sklearn import metrics
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
# from sklearn.cross_validation import KFold
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import itertools
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from string import punctuation
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import sparse



class Clickbait():
    question_words = ['who', 'whos', 'whose', 'what', 'whats', 'whatre', 'when', 'whenre', 'whens', 'couldnt',
                      'where', 'wheres', 'whered', 'why', 'whys', 'can', 'cant', 'could', 'will', 'would', 'is',
                      'isnt', 'should', 'shouldnt', 'you', 'your', 'youre', 'youll', 'youd', 'here', 'heres',
                      'how', 'hows', 'howd', 'this', 'are', 'arent', 'which', 'does', 'doesnt']

    contractions = ['tis', 'aint', 'amnt', 'arent', 'cant', 'couldve', 'couldnt', 'couldntve',
                    'didnt', 'doesnt', 'dont', 'gonna', 'gotta', 'hadnt', 'hadntve', 'hasnt',
                    'havent', 'hed', 'hednt', 'hedve', 'hell', 'hes', 'hesnt', 'howd', 'howll',
                    'hows', 'id', 'idnt', 'idntve', 'idve', 'ill', 'im', 'ive', 'ivent', 'isnt',
                    'itd', 'itdnt', 'itdntve', 'itdve', 'itll', 'its', 'itsnt', 'mightnt',
                    'mightve', 'mustnt', 'mustntve', 'mustve', 'neednt', 'oclock', 'ol', 'oughtnt',
                    'shant', 'shed', 'shednt', 'shedntve', 'shedve', 'shell', 'shes', 'shouldve',
                    'shouldnt', 'shouldntve', 'somebodydve', 'somebodydntve', 'somebodys',
                    'someoned', 'someonednt', 'someonedntve', 'someonedve', 'someonell', 'someones',
                    'somethingd', 'somethingdnt', 'somethingdntve', 'somethingdve', 'somethingll',
                    'somethings', 'thatll', 'thats', 'thatd', 'thered', 'therednt', 'theredntve',
                    'theredve', 'therere', 'theres', 'theyd', 'theydnt', 'theydntve', 'theydve',
                    'theydvent', 'theyll', 'theyontve', 'theyre', 'theyve', 'theyvent', 'wasnt',
                    'wed', 'wedve', 'wednt', 'wedntve', 'well', 'wontve', 'were', 'weve', 'werent',
                    'whatd', 'whatll', 'whatre', 'whats', 'whatve', 'whens', 'whered', 'wheres',
                    'whereve', 'whod', 'whodve', 'wholl', 'whore', 'whos', 'whove', 'whyd', 'whyre',
                    'whys', 'wont', 'wontve', 'wouldve', 'wouldnt', 'wouldntve', 'yall', 'yalldve',
                    'yalldntve', 'yallll', 'yallont', 'yallllve', 'yallre', 'yallllvent', 'yaint',
                    'youd', 'youdve', 'youll', 'youre', 'yourent', 'youve', 'youvent']

    def process_text(self, text):
        result = text.replace('/', '').replace('\n', '')
        result = re.sub(r'[1-9]+', 'number', result)
        result = re.sub(r'(\w)(\1{2,})', r'\1', result)
        result = re.sub(r'(?x)\b(?=\w*\d)\w+\s*', '', result)
        result = ''.join(t for t in result if t not in punctuation)
        result = re.sub(r' +', ' ', result).lower().strip()
        return result

    def cnt_stop_words(self, text):
        s = text.split()
        num = len([word for word in s if word in self.stop])
        return num

    def num_contract(self, text):
        s = text.split()
        num = len([word for word in s if word in self.contractions])
        return num

    def question_word(self, text):
        s = text.split()
        if s[0] in self.question_words:
            return 1
        else:
            return 0

    def part_of_speech(self, text):
        s = text.split()
        nonstop = [word for word in s if word not in self.stop]
        pos = [part[1] for part in nltk.pos_tag(nonstop)]
        pos = ' '.join(pos)
        return pos

    def __init__(self):

        global accscore

        df_ycb = pd.read_csv('input_data/clickbait/clickbait_data.txt', sep="\n", header=None, names=['text'])
        df_ycb['clickbait'] = 1

        df_ncb = pd.read_csv('input_data/clickbait/non_clickbait_data.txt', sep="\n", header=None, names=['text'])
        df_ncb['clickbait'] = 0

        df = df_ycb.append(df_ncb, ignore_index=True).reset_index(drop=True)

        self.stop = stopwords.words('english')

        # Creating some latent variables from the data
        df['text'] = df['text'].apply(self.process_text)
        df['question'] = df['text'].apply(self.question_word)

        df['num_words'] = df['text'].apply(lambda x: len(x.split()))
        df['part_speech'] = df['text'].apply(self.part_of_speech)
        df['num_contract'] = df['text'].apply(self.num_contract)
        df['num_stop_words'] = df['text'].apply(self.cnt_stop_words)
        df['stop_word_ratio'] = df['num_stop_words'] / df['num_words']
        df['contract_ratio'] = df['num_contract'] / df['num_words']

        df.drop(['num_stop_words', 'num_contract'], axis=1, inplace=True)

        df_train, df_test = train_test_split(df, test_size=0.2, random_state=0)

        self.tfidf = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode',
                                     analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 5),
                                     use_idf=1, smooth_idf=1, sublinear_tf=1)

        X_train_text = self.tfidf.fit_transform(df_train['text'])
        X_test_text = self.tfidf.transform(df_test['text'])

        self.cvec = CountVectorizer()

        X_train_pos = self.cvec.fit_transform(df_train['part_speech'])
        X_test_pos = self.cvec.transform(df_test['part_speech'])

        self.scNoMean = StandardScaler(with_mean=False)  # we pass with_mean=False to preserve the sparse matrix
        X_train_pos_sc = self.scNoMean.fit_transform(X_train_pos)
        X_test_pos_sc = self.scNoMean.transform(X_test_pos)

        X_train_val = df_train.drop(['clickbait', 'text', 'part_speech'], axis=1).values
        X_test_val = df_test.drop(['clickbait', 'text', 'part_speech'], axis=1).values

        self.sc = StandardScaler()
        X_train_val_sc = self.sc.fit(X_train_val).transform(X_train_val)
        X_test_val_sc = self.sc.transform(X_test_val)

        y_train = df_train['clickbait'].values
        y_test = df_test['clickbait'].values

        X_train = sparse.hstack([X_train_val_sc, X_train_text, X_train_pos_sc]).tocsr()
        X_test = sparse.hstack([X_test_val_sc, X_test_text, X_test_pos_sc]).tocsr()

        self.model = LogisticRegression(penalty='l2', C=98.94736842105263)
        self.model = self.model.fit(X_train, y_train)

        predicted_LogR = self.model.predict(X_test)
        accscore = metrics.accuracy_score(y_test, predicted_LogR)
        print("Clickbait Model Trained - accuracy:   %0.6f" % accscore)

    def predict(self, text):
        # creating the dataframe with our text so we can leverage the existing code
        dfrme = pd.DataFrame(index=[0], columns=['text'])
        dfrme['text'] = text

        # processing text
        dfrme['text'] = dfrme['text'].apply(self.process_text)

        # adding latent variables
        dfrme['question'] = dfrme['text'].apply(self.question_word)
        dfrme['num_words'] = dfrme['text'].apply(lambda x: len(x.split()))
        dfrme['part_speech'] = dfrme['text'].apply(self.part_of_speech)
        dfrme['num_contract'] = dfrme['text'].apply(self.num_contract)
        dfrme['num_stop_words'] = dfrme['text'].apply(self.cnt_stop_words)
        dfrme['stop_word_ratio'] = dfrme['num_stop_words'] / dfrme['num_words']
        dfrme['contract_ratio'] = dfrme['num_contract'] / dfrme['num_words']

        # removing latent variables that have high colinearity with other features
        dfrme.drop(['num_stop_words', 'num_contract'], axis=1, inplace=True)

        Xtxt_val = dfrme.drop(['text', 'part_speech'], axis=1).values
        Xtxt_val_sc = self.sc.transform(Xtxt_val)

        Xtxt_text = self.tfidf.transform(dfrme['text'])

        Xtxt_pos = self.cvec.transform(dfrme['part_speech'])
        Xtxt_pos_sc = self.scNoMean.transform(Xtxt_pos)
        Xtxt = sparse.hstack([Xtxt_val_sc, Xtxt_text, Xtxt_pos_sc]).tocsr()

        predicted = self.model.predict(Xtxt)
        predicedProb = self.model.predict_proba(Xtxt)[:, 1]
        return bool(predicted), float(predicedProb)

    def predictScore(self, text):
        # creating the dataframe with our text so we can leverage the existing code
        dfrme = pd.DataFrame(index=[0], columns=['text'])
        dfrme['text'] = text

        # processing text
        dfrme['text'] = dfrme['text'].apply(self.process_text)

        # adding latent variables
        dfrme['question'] = dfrme['text'].apply(self.question_word)
        dfrme['num_words'] = dfrme['text'].apply(lambda x: len(x.split()))
        dfrme['part_speech'] = dfrme['text'].apply(self.part_of_speech)
        dfrme['num_contract'] = dfrme['text'].apply(self.num_contract)
        dfrme['num_stop_words'] = dfrme['text'].apply(self.cnt_stop_words)
        dfrme['stop_word_ratio'] = dfrme['num_stop_words'] / dfrme['num_words']
        dfrme['contract_ratio'] = dfrme['num_contract'] / dfrme['num_words']

        # removing latent variables that have high colinearity with other features
        dfrme.drop(['num_stop_words', 'num_contract'], axis=1, inplace=True)

        Xtxt_val = dfrme.drop(['text', 'part_speech'], axis=1).values
        Xtxt_val_sc = self.sc.transform(Xtxt_val)

        Xtxt_text = self.tfidf.transform(dfrme['text'])

        Xtxt_pos = self.cvec.transform(dfrme['part_speech'])
        Xtxt_pos_sc = self.scNoMean.transform(Xtxt_pos)
        Xtxt = sparse.hstack([Xtxt_val_sc, Xtxt_text, Xtxt_pos_sc]).tocsr()

        predicedProb = self.model.predict_proba(Xtxt)[:, 1]
        return float(predicedProb)

    def getScore(self):
        return accscore

if __name__ == "__main__":
    basedir = pickle.load(open('./models/basedir.pkl', 'rb'))
    clickbait_filename_pkl = basedir + 'models/clickbait_feature_av4.pkl'
    cb = Clickbait()
    text1 = "Should You bring the money now"
    print(cb.predict(text1), text1)
    pickle.dump(cb, open(clickbait_filename_pkl, 'wb'))
    del cb
