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
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle
import seaborn as sns
import pandas as pd
import numpy as np
import nltk
import re
import nltk
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from fastai import *
from fastai import *
from fastai.text import *
from fastai.tabular import *
from fastai.vision import *
from nltk.corpus import stopwords
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
import matplotlib.pyplot as plt
from scipy import sparse
# Code source: https://degravek.github.io/project-pages/project1/2017/04/28/New-Notebook/
# Dataset from Chakraborty et al. (https://github.com/bhargaviparanjape/clickbait/tree/master/dataset)

class trollness():


  def __init__(self):
        print("Initializing")

  def load_train_data(self, train_filename):
    # Load the dataset
    colnames = ['jsonid', 'label', 'headline_text', 'subject', 'speaker', 'speakerjobtitle', 'stateinfo','partyaffiliation', 'barelytruecounts', 'falsecounts','halftruecounts','mostlytrueocunts','pantsonfirecounts','context']
    train_news = pd.read_csv(train_filename, sep='\t', names = colnames, error_bad_lines=False)
    print(train_news)
    return train_news
  
  def load_test_data(self, test_filename):
    # Load the dataset
    colnames = ['jsonid', 'label', 'headline_text', 'subject', 'speaker', 'speakerjobtitle', 'stateinfo','partyaffiliation', 'barelytruecounts', 'falsecounts','halftruecounts','mostlytrueocunts','pantsonfirecounts','context']
    test_news = pd.read_csv(test_filename, sep='\t', names = colnames, error_bad_lines=False)
    return test_news

  def load_valid_news_data(self, valid_filename):
    # Load the dataset
    colnames = ['jsonid', 'label', 'headline_text', 'subject', 'speaker', 'speakerjobtitle', 'stateinfo','partyaffiliation', 'barelytruecounts', 'falsecounts','halftruecounts','mostlytrueocunts','pantsonfirecounts','context']
    valid_news = pd.read_csv(valid_filename, sep='\t', names = colnames, error_bad_lines=False)
    print(valid_news.columns)
    return valid_news

  def cyber_troll_data(self, troll_filename):
    # Load the dataset
    cyber_troll = pd.read_json(troll_filename, lines= True)
    print(cyber_troll)
    return cyber_troll
  
  def model (self):
    np.random.seed(2018)

    from nltk import PorterStemmer 
    stemmer = PorterStemmer()
    def lemmatize_stemming(text):
        return text
        return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
    def preprocess(text):
        result = []
        for token in gensim.utils.simple_preprocess(text):
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                if 'object' in token.strip() or 'dtype' in token.strip() or 'unknown' in token.strip():
                    continue
                result.append(lemmatize_stemming(token))
        return result

    word_vector_input_dataset = df.content.tolist()
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    from nltk.tokenize import word_tokenize
    data = word_vector_input_dataset
    tagged_data = []
    exception_count = 0
    for i, _d in enumerate(data):
        try:
            tagged_data.append(TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]))
        except:
            exception_count+=1
    
    print ("Total number of custom documents:",len(tagged_data))
    max_epochs = 10
    vec_size = 10
    alpha = 0.025

    model = Doc2Vec(size=vec_size,
                    alpha=alpha, 
                    min_alpha=0.00025,
                    min_count=10,
                    dm =1)
      
    model.build_vocab(tagged_data)

    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        model.alpha -= 0.0002
        model.min_alpha = model.alpha

    model.save("troll.model")
    print("Model Saved")   
      
  def corpus(self):
    corpus = []

    for i in range (0, len(df)):                              
        review = re.sub('[^a-zA-Z]',' ',df['content'][i])      
        review = review.lower()                                 
        review = review.split()                                 
        review = ' '.join(review)                               
        corpus.append(review)                                   

    corpus
    return corpus
    
  def bow_transformer(self, corpus):
    bow_transformer =  CountVectorizer()
    bow_transformer = bow_transformer.fit(corpus)
    return bow_transformer

  def wordcloud(self, corpus, bow_transformer):
    text = ""

    for doc in corpus:
        doc = word_tokenize(doc)
        text= text + " " + ' '.join(doc)
        
    wordcloud = WordCloud(max_font_size=50, max_words=1000, background_color="black").generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    print(len(bow_transformer.vocabulary_))
    messages_bow = bow_transformer.transform(corpus)  
    tfidf_transformer = TfidfTransformer().fit(messages_bow)  
    X = tfidf_transformer.transform(messages_bow)
    return X
  def naive_bayes(self, transformer, pkl_nb_filename):
    y = []
    for i in range(0,len(df)):
        y.append(df.annotation[i]['label'])
    X_train, X_test, y_train, y_test = train_test_split(transformer, y, test_size=0.25, random_state=42)
    
    classifier = MultinomialNB()
    classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)

    pickle.dump(classifier, open(pkl_nb_filename, 'wb'))
    loaded_model = pickle.load(open(pkl_nb_filename, 'rb'))
    result = loaded_model.predict(X_test)
    print(result)
    from sklearn.metrics import classification_report, accuracy_score
    print(accuracy_score(y_test, y_pred))

  def predict(self, train_filename, valid_filename, test_filename, pkl_svm_filename, factor_header_filename):
    headers = ['id','label','statement','subject',
              'speaker','job_title', 'state', 
              'affliation','barely_true','false',
              'half_true', 'mostly_true','pants_on_fire', 
              'venue']
    liar_train_df = pd.read_csv(train_filename, names=headers, delimiter='\t')
    liar_valid_df = pd.read_csv(valid_filename, names=headers, delimiter='\t')
    liar_test_df = pd.read_csv(test_filename,names=headers, delimiter='\t')
    statements = liar_train_df.statement.tolist()
    label = liar_train_df.label.tolist()
    training_vector = []
    text = "Lol! thats the dumbest"
    dvmodel= Doc2Vec.load("troll.model")
    test_sentence= [dvmodel.infer_vector(word_tokenize(text))]
    print (test_sentence)
    for statement in statements:
        training_vector.append(dvmodel.infer_vector(statement))
    troll_df= pd.DataFrame(training_vector)
    troll_df['label'] = pd.Series(label)
    troll_df['label'] = troll_df.label.apply(lambda x: 0 if 'barely' in str(x) or 'false' in str(x) else 1)
    troll_df.columns=['0','1','2','3','4','5','6','7','8','9','label']
    troll_df.head()
    features = ['0','1','2','3','4','5','6','7','8','9']
    X = troll_df[features]
    Y = troll_df[['label']]
    from sklearn import preprocessing
    scaler = preprocessing.MinMaxScaler()
    X = X.fillna(0);
    Y = Y.fillna(0);
    scaled_X = scaler.fit_transform(X)
    scaled_Y = scaler.fit_transform(Y)
    from sklearn.model_selection import train_test_split
    from sklearn import linear_model
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, scaled_Y, test_size=0.1, random_state=0)
    lm = linear_model.LogisticRegression(verbose=1)
    model = lm.fit(X_train, y_train)
    print (model)

    pickle.dump(model, open(pkl_svm_filename, 'wb'))
    loaded_model = pickle.load(open(pkl_svm_filename, 'rb'))
    result = loaded_model.predict(X_test)
    print(result)
    print ("Score:", loaded_model.score(X_test, y_test))
    predictions = lm.predict_proba(scaled_X)
    troll_predictions  = [p[1] for p in predictions]
    troll_predict_df = pd.DataFrame(troll_predictions)
    print (len(Y))
    print (len(troll_predictions))
    troll_predict_df['label_from_liar_liar_copy'] = pd.Series(liar_train_df.label.tolist())
    troll_predict_df.columns = ['troll','label_from_liar_liar_copy']
    troll_predict_df.head()
    troll_predict_df.to_csv(factor_header_filename)
    troll_predict_df['label'] = troll_predict_df.label_from_liar_liar_copy.apply(lambda x: 0 if "false" in str(x) or 'barely' in str(x) else 1)
    troll_predict_df.head()
    troll_predict_df.corr()