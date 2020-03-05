import pandas as pd
import numpy as np
import csv
import gensim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from nltk.stem.porter import PorterStemmer
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from string import punctuation
import seaborn as sns
import pandas as pd
import numpy as np
import nltk
import re
import nltk
from collections import defaultdict
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
import matplotlib.pyplot as plt
from scipy import sparse
nltk.download('stopwords')
nltk.download('wordnet')
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import string
from plotly import tools  # to install $ pip install plotly
from plotly.subplots import make_subplots
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import pickle
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import (GridSearchCV, KFold, RandomizedSearchCV,
                                     learning_curve)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
# Code source: https://degravek.github.io/project-pages/project1/2017/04/28/New-Notebook/
# Dataset from Chakraborty et al. (https://github.com/bhargaviparanjape/clickbait/tree/master/dataset)

fake_news_file_name = "/content/drive/My Drive/AlternusVeraDataSets2019/Spartans/Doss/fake_news_Dataframe.pkl"
df_fake_news = pd.read_pickle(fake_news_file_name)
df_fake_news['result_clean_text']

pkl_liar_liar_file_name = "/content/drive/My Drive/AlternusVeraDataSets2019/Spartans/Doss/liar_liar_Dataframe.pkl"
pkl_liar_liar_test_file_name = "/content/drive/My Drive/AlternusVeraDataSets2019/Spartans/Doss/liar_liar_test_Dataframe.pkl"
fake_news_file_name = "/content/drive/My Drive/AlternusVeraDataSets2019/Spartans/Doss/fake_news_Dataframe.pkl"
pkl_svm_filename = '/content/drive/My Drive/AlternusVeraDataSets2019/Spartans/Doss/misinfo_svm.pkl'
pkl_rf_filename = '/content/drive/My Drive/AlternusVeraDataSets2019/Spartans/Doss/misinfo_rf.pkl'
pkl_nb_filename = '/content/drive/My Drive/AlternusVeraDataSets2019/Spartans/Doss/misinfo_nb.pkl'


class neural_misinformation():


  def __init__(self):
        print("Initializing")


  def load_fake_news_data(self):
    # Load the dataset
    df_fake_news = pd.read_pickle(fake_news_file_name)
    print(df_fake_news.columns)
    return df_fake_news
      
  def load_train_data(self):
    # Load the dataset
    df_train_data = pd.read_pickle(pkl_liar_liar_file_name)
    print(df_train_data)
    return df_train_data
  
  def load_test_data(self):
    # Load the dataset
    df_test_data = pd.read_pickle(pkl_liar_liar_test_file_name)
    return df_test_data
      
  def clean_str(self, string, Word):
      STOPWORDS = set(stopwords.words('english'))
      string = re.sub(r"^b", "", string)
      string = re.sub(r"\\n ", "", string)
      string = re.sub(r"\'s", "", string)
      string = re.sub(r"\'ve", "", string)
      string = re.sub(r"n\'t", "", string)
      string = re.sub(r"\'re", "", string)
      string = re.sub(r"\'d", "", string)
      string = re.sub(r"\'ll", "", string)
      string = re.sub(r",", "", string)
      string = re.sub(r"!", " ! ", string)
      string = re.sub(r"\(", "", string)
      string = re.sub(r"\)", "", string)
      string = re.sub(r"\?", "", string)
      string = re.sub(r"'", "", string)
      string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
      string = re.sub(r"[0-9]\w+|[0-9]", "", string)
      string = re.sub(r"\s{2,}", " ", string)
      string = ' '.join(Word(word).lemmatize() for word in string.split() if word not in STOPWORDS)  # delete stopwors from text
      return string.strip().lower()

  def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []
    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)
    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()


  def configure_plotly_browser_state(self):
    import IPython
    display(IPython.core.display.HTML('''
          <script src="/static/components/requirejs/require.js"></script>
          <script>
            requirejs.config({
              paths: {
                base: '/static/base',
                plotly: 'https://cdn.plot.ly/plotly-latest.min.js?noext',
              },
            });
          </script>
          '''))
#  df_real = train_data[train_data["label"]==1]
#  df_fake = train_data[train_data["label"]!=1]

  ## custom function for ngram generation ##
  def generate_ngrams(self, text, n_gram=1):
      token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]
      ngrams = zip(*[token[i:] for i in range(n_gram)])
      return [" ".join(ngram) for ngram in ngrams]

  ## custom function for horizontal bar chart ##
  def horizontal_bar_chart(self, df, color):
      trace = go.Bar(
          y=df["word"].values[::-1],
          x=df["wordcount"].values[::-1],
          showlegend=False,
          orientation = 'h',
          marker=dict(
              color=color,
          ),
      )
      return trace

  def plot_unigram(self, df_real, df_fake):
     ## Get the bar chart from sincere questions ##
     freq_dict = defaultdict(int)
     print(df_real.columns)
     for sent in df_real["clean"]:
         for word in self.generate_ngrams(sent, 1):
             freq_dict[word] += 1
     fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
     fd_sorted.columns = ["word", "wordcount"]
     trace0 = self.horizontal_bar_chart(fd_sorted.head(50), 'blue')


     ## Get the bar chart from insincere questions ##
     freq_dict = defaultdict(int)
     for sent in df_fake["clean"]:
         for word in self.generate_ngrams(sent, 1):
             freq_dict[word] += 1
     fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
     fd_sorted.columns = ["word", "wordcount"]
     trace1 = self.horizontal_bar_chart( fd_sorted.head(50), 'blue')

     # Creating two subplots
     fig = make_subplots(rows=1, cols=2, vertical_spacing=0.04,
                               subplot_titles=["Frequent words of real news", 
                                               "Frequent words of fake news"])
     fig.append_trace(trace0, 1, 1)
     fig.append_trace(trace1, 1, 2)
     fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")
     py.iplot(fig, filename='word-plots')


  def plot_bigram(self, df_real, df_fake):
    freq_dict = defaultdict(int)
    for sent in df_real["clean"]:
        for word in generate_ngrams(sent,2):
            freq_dict[word] += 1
    fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
    fd_sorted.columns = ["word", "wordcount"]
    trace0 = horizontal_bar_chart(fd_sorted.head(50), 'orange')


    freq_dict = defaultdict(int)
    for sent in df_fake["clean"]:
        for word in generate_ngrams(sent,2):
            freq_dict[word] += 1
    fd_sorted1 = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
    fd_sorted1.columns = ["word", "wordcount"]
    trace1 = horizontal_bar_chart(fd_sorted1.head(50), 'orange')

    # Creating two subplots
    fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04,horizontal_spacing=0.15,
                              subplot_titles=["Frequent bigrams of real news", 
                                              "Frequent bigrams of fake news"])
    fig.append_trace(trace0, 1, 1)
    fig.append_trace(trace1, 1, 2)
    fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Bigram Count Plots")
    py.iplot(fig, filename='word-plots')

  def train_models(self, train_data):
    svm_pipeline = Pipeline([
      ('svm_TF', TfidfVectorizer(lowercase=True, max_df= 0.7, ngram_range=(1,2 ), use_idf=True, smooth_idf=True))
        ,('svm_clf', SVC(gamma=0.7, kernel='rbf', random_state=20))
      ])
    rf_pipeline = Pipeline([
      ('rf_TF', TfidfVectorizer(lowercase=True, ngram_range=(1, 2), use_idf=True, smooth_idf=True)),
      ('rf_clf', RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1))
      ])
    nb_pipeline = Pipeline([
      ('nb_CV', CountVectorizer(ngram_range=(1,2))),
      ('nb_clf', MultinomialNB(alpha=6.5))
      ])
    
    svm_pipeline.fit(train_data['clean'], train_data['label'] )
    rf_pipeline.fit(train_data['clean'], train_data['label'] )
    nb_pipeline.fit(train_data['clean'], train_data['label'] )
    pickle.dump(svm_pipeline,  open(pkl_svm_filename, 'wb'))
    pickle.dump(rf_pipeline,  open(pkl_rf_filename, 'wb'))
    pickle.dump(nb_pipeline,  open(pkl_nb_filename, 'wb'))


  def predict_svm(self, train_data, pkl_svm_filename):
    loaded_model = pickle.load(open(pkl_svm_filename, 'rb'))
    test_data = self.load_test_data()
    y_predict = loaded_model.predict(test_data['clean'])
    result = loaded_model.score(train_data['headline_text'], train_data['label'])
    print('Support Vector Machine Result')
    print(result)

    self.show_eval_scores(y_predict, test_data, "Support Vector Machine")
   
  def predict_rf(self, train_data, pkl_rf_filename):
    loaded_model = pickle.load(open(pkl_rf_filename, 'rb'))
    test_data = self.load_test_data()
    y_predict = loaded_model.predict(test_data['clean'])
    result = loaded_model.score(train_data['headline_text'], train_data['label'])
    print('Random Forest Result')
    
    print(result)
    self.show_eval_scores(y_predict, test_data, "Random Forest Model")
  def predict_nb(self, train_data, pkl_nb_filename):
    loaded_model = pickle.load(open(pkl_nb_filename, 'rb'))
    test_data = self.load_test_data()
    y_predict = loaded_model.predict(test_data['clean'])
    result = loaded_model.score(train_data['headline_text'], train_data['label'])
    print('Naive Bayes Result')
    print(result)
    self.show_eval_scores(y_predict, test_data, "Naive Bayes")


  def show_eval_scores(self, y_pred, test_data, model_name):
    """Function to show to different evaluation score of the model passed
    on the test set.
    
    Parameters:
    -----------
    model: scikit-learn object
        The model whose scores are to be shown.
    test_set: pandas dataframe
        The dataset on which the score of the model is to be shown.
    model_name: string
        The name of the model.
    """
    # y_pred = model.predict(test_data['result_clean_text'])
    y_true = test_data['label']
    f1 = f1_score(y_true, y_pred, average='micro')
    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)

    print('Report for ---> {}'.format(model_name))
    print('Accuracy is: {}'.format(accuracy))
    print('F1 score is: {}'.format(f1))
    print('Precision score is: {}'.format(precision))
    print('Recall score is: {}'.format(recall))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')



