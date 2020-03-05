from nltk.corpus import stopwords
import nltk
import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
import gensim
from gensim.utils import simple_preprocess
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
stemmer = PorterStemmer()#Stemmers remove morphological affixes from words, leaving only the word
from gensim import corpora, models
from pprint import pprint
import lda
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from bokeh.plotting import save
from bokeh.plotting import figure, show, ColumnDataSource
from bokeh.io import  output_notebook
from bokeh.models import HoverTool

from gensim.models import Doc2Vec
from sklearn import utils
import gensim
from gensim.models.doc2vec import TaggedDocument
import re

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

color = sns.color_palette()
nltk.download('stopwords')
nltk.download('wordnet')

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

class Toxicity:
    def __init__(self):
        # vectorizer: ignore English stopwords & words that occur less than 5 times
        self.cvectorizer = CountVectorizer(min_df=5, stop_words='english')
        self.model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, min_count=1, alpha=0.065, min_alpha=0.065)
        self.logistic_regression_model = LogisticRegression(n_jobs=1, C=1e5)
        self.support_vector_machine_model = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
                ])

        countV_toxicity = CountVectorizer()
        self.random_forest = Pipeline([
        ('rfCV',countV_toxicity),
        ('rf_clf',RandomForestClassifier(n_estimators=200,n_jobs=3))
        ])

    def load_data(self, file_name):
        # Load the dataset
        print('Data File Name = ', file_name)
        dataset = pd.read_csv(file_name)
        return dataset

    # Corpus cleaning
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

    def clean_toxic_data(self, toxic_dataset, Word):
        X = toxic_dataset["comment_text"].fillna("DUMMY_VALUE").values
        possible_labels = [
            "toxic",
            "severe_toxic",
            "obscene",
            "threat",
            "insult",
            "identity_hate"]
        # Flag for toxic
        rowsums = toxic_dataset[possible_labels].sum(axis=1)
        # istoxic is target 0-Clean as number grows more severe toxic
        toxic_dataset['istoxic'] = rowsums
        toxic_dataset['clean'] = (rowsums == 0)  # Clean=1 Toxic =0
        y = toxic_dataset.istoxic
        toxic_dataset.head()
        toxic_dataset['comment_text_clean'] = toxic_dataset['comment_text'].apply(lambda x: self.clean_str(x, Word))  # calling clean for all rows

    def add_new_features_to_data(self, toxic_dataset):
        STOPWORDS = set(stopwords.words('english'))
        #Sentense count in each comment:
            #  '\n' can be used to count the number of sentences in each comment
        toxic_dataset['count_sent']=toxic_dataset["comment_text"].apply(lambda x: len(re.findall("\n",str(x)))+1)
        #Word count in each comment:
        toxic_dataset['count_word']=toxic_dataset["comment_text"].apply(lambda x: len(str(x).split()))
        #Unique word count
        toxic_dataset['count_unique_word']=toxic_dataset["comment_text"].apply(lambda x: len(set(str(x).split())))
        #Letter count
        toxic_dataset['count_letters']=toxic_dataset["comment_text"].apply(lambda x: len(str(x)))
        #punctuation count
        toxic_dataset["count_punctuations"] =toxic_dataset["comment_text"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
        #upper case words count
        toxic_dataset["count_words_upper"] = toxic_dataset["comment_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
        #title case words count
        toxic_dataset["count_words_title"] = toxic_dataset["comment_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
        #Number of stopwords
        toxic_dataset["count_stopwords"] = toxic_dataset["comment_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))
        #Average length of the words
        toxic_dataset["mean_word_len"] = toxic_dataset["comment_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

        #derived features
        #Word count percent in each comment:
        toxic_dataset['word_unique_percent']=toxic_dataset['count_unique_word']*100/toxic_dataset['count_word']
        #derived features
        #Punct percent in each comment:
        toxic_dataset['punct_percent']=toxic_dataset['count_punctuations']*100/toxic_dataset['count_word']

    def plot_toxic_distribution(self, toxic_dataset):
        #In our case we just need to categorize on toxicity. Different category of toxicity does not matter. 
        toxic_dataset.groupby('istoxic').comment_text_clean.count().plot.bar(ylim=0)

    def plot_heatmap(self, toxic_dataset):
        Cor = toxic_dataset[toxic_dataset.columns]
        #Calculate the correlation of the above variables
        cor = Cor.corr()
        #Plot the correlation as heat map
        fig, ax = plt.subplots(figsize=(10,10))
        sns.heatmap(cor,annot=True,linewidths=.5, ax=ax)

    def plot_heatmap_newfeatures(self, toxic_dataset):
        #To show heatmap data should be in matrix form.
        tc = toxic_dataset[['count_sent', 'count_word', 'count_unique_word', 'count_letters', 'count_punctuations', 'count_words_upper', 'count_words_title', 'count_stopwords', 'mean_word_len', 'word_unique_percent', 'punct_percent', 'istoxic']].corr()   #shows corelation in matrix form
        tc
        plt.figure(figsize = (16,5))
        sns.heatmap(tc, annot=True, cmap='coolwarm')

    def plot_wordcloud(self, text, mask=None, max_words=200, max_font_size=100, figure_size=(24.0,16.0), title = None, title_size=40, image_color=False):
        stopwords = set(STOPWORDS)
        more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}
        stopwords = stopwords.union(more_stopwords)
        wordcloud = WordCloud(background_color='white',
                        stopwords = stopwords,
                        max_words = max_words,
                        max_font_size = max_font_size, 
                        random_state = 42,
                        width=800, 
                        height=400,
                        mask = mask)
        wordcloud.generate(str(text))
        plt.figure(figsize=figure_size)
        if image_color:
            image_colors = wordcloud.ImageColorGenerator(mask)
            plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
            plt.title(title, fontdict={'size': title_size, 'verticalalignment': 'bottom'})
        else:
            plt.imshow(wordcloud)
            plt.title(title, fontdict={'size': title_size, 'color': 'green', 'verticalalignment': 'bottom'})
        plt.axis('off')
        plt.tight_layout()

    def toxicity_vs_text_length(self, toxic_dataset):
        toxic_dataset['count_sent'].loc[toxic_dataset['count_sent']>10] = 10
        plt.figure(figsize=(12,6))
        ## sentenses
        plt.subplot(121)
        plt.suptitle("Are longer comments more toxic?",fontsize=20)
        sns.violinplot(y='count_sent',x='clean', data=toxic_dataset,split=True)
        plt.xlabel('Clean?', fontsize=12)
        plt.ylabel('# of sentences', fontsize=12)
        plt.title("Number of sentences in each comment", fontsize=15)
        # words
        toxic_dataset['count_word'].loc[toxic_dataset['count_word']>200] = 200
        plt.subplot(122)
        sns.violinplot(y='count_word',x='clean', data=toxic_dataset,split=True,inner="quart")
        plt.xlabel('Clean?', fontsize=12)
        plt.ylabel('# of words', fontsize=12)
        plt.title("Number of words in each comment", fontsize=15)
        plt.show()

    def toxicity_vs_spam(self, toxic_dataset):
        #spammers - comments with less than 40% unique words
        spammers=toxic_dataset[toxic_dataset['word_unique_percent']<30]
        print("Clean Spam example:")
        print(spammers[spammers.clean==1].comment_text.iloc[1])
        print("Toxic Spam example:")
        print(spammers[spammers.clean<1].comment_text.iloc[2])

    def pos_clean_content(self, toxic_dataset):
        # Clear content POS
        tokens = nltk.word_tokenize(toxic_dataset[toxic_dataset["istoxic"]==0]["comment_text_clean"].tolist()[0])
        print(nltk.pos_tag(tokens))

    def pos_toxic_content(self, toxic_dataset):
        # Toxic content POS
        tokens = nltk.word_tokenize(toxic_dataset[toxic_dataset["istoxic"]>0]["comment_text_clean"].tolist()[0])
        print(nltk.pos_tag(tokens))
        tokens = nltk.word_tokenize(toxic_dataset[toxic_dataset["istoxic"]>0]["comment_text_clean"].tolist()[3])
        print(nltk.pos_tag(tokens))

    # perform lemmatize and stem preprocessing steps on the data set
    def lemmatize_stemming(self, text):
        return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
    
    def preprocess(self, text):
        result = []
        for token in gensim.utils.simple_preprocess(text):
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                result.append(self.lemmatize_stemming(token))
        return result

    def get_bag_of_words(self, dictionary):
        count = 0
        for k, v in dictionary.iteritems():
            print(k, v)
            count += 1
            if count > 10:
                break

    def get_bow_corpus(self, dictionary, processed_docs):
        return [dictionary.doc2bow(doc) for doc in processed_docs]

    def get_tfidf(self, bow_corpus):
        tfidf = models.TfidfModel(bow_corpus)
        corpus_tfidf = tfidf[bow_corpus]
        for doc in corpus_tfidf:
            pprint(doc)
            break
        return corpus_tfidf

    def get_lda_model(self, bow_corpus, dictionary):
        return gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)

    def lda_using_bow(self, lda_model):
        for idx, topic in lda_model.print_topics(-1):
            print('Topic: {} \nWords: {}'.format(idx, topic))
            topic.title

    def lda_tfidf(self, corpus_tfidf, dictionary):
        lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=dictionary, passes=2, workers=4)
        for idx, topic in lda_model_tfidf.print_topics(-1):
            print('Topic: {} Word: {}'.format(idx, topic))

    def get_lda_cv(self):
        n_topics = 20 # number of topics
        n_iter = 500 # number of iterations
        # train an LDA model
        lda_model_cv = lda.LDA(n_topics=n_topics, n_iter=n_iter)
        return lda_model_cv
    
    def lda_count_vectorizer(self, message_docs):
        cvz = self.cvectorizer.fit_transform(message_docs)
        return cvz

    def reduce_to_2d_tsne(self, X_topics):
        # a t-SNE model
        # angle value close to 1 means sacrificing accuracy for speed
        # pca initializtion usually leads to better results 
        tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
        # 20-D -> 2-D
        tsne_lda = tsne_model.fit_transform(X_topics)
        return tsne_lda

    def visualize_and_group(self, X_topics, message_docs, lda_model, tsne_lda):
        n_top_words = 5 # number of keywords we show
        # 20 colors
        colormap = np.array([
            "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
            "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
            "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
            "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"
        ])

        #Then we find the most likely topic for each news
        _lda_keys = []
        for i in range(X_topics.shape[0]):
            _lda_keys +=  X_topics[i].argmax(),

        # get top words for each topic
        topic_summaries = []
        topic_word = lda_model.topic_word_  # all topic words
        vocab = self.cvectorizer.get_feature_names()
        for i, topic_dist in enumerate(topic_word):
            topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1] # get!
            topic_summaries.append(' '.join(topic_words)) # append!

        output_notebook()

        # plot the news (each point representing one news)
        title = '20 newsgroups LDA viz'
        num_example = len(X_topics)

        plot_lda = figure(plot_width=1400, plot_height=1100,
                            title=title,
                            tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
                            x_axis_type=None, y_axis_type=None, min_border=1)
        source = ColumnDataSource(
                data=dict(
                    x= tsne_lda[:, 0], #tsne_lda.iloc[:, 0],
                    y= tsne_lda[:, 1], #tsne_lda.iloc[:, 1],
                    content= message_docs[:num_example],
                    topic_key= _lda_keys[:num_example],
                    c=colormap[_lda_keys][:num_example]
                )
            )

        plot_lda.circle('x', 'y', source=source, color='c')

        #plot the crucial words for each topic and tooltip
        # randomly choose a news (within a topic) coordinate as the crucial words coordinate
        topic_coord = np.empty((X_topics.shape[1], 2)) * np.nan
        for topic_num in _lda_keys:
            if not np.isnan(topic_coord).any():
                break
            topic_coord[topic_num] = tsne_lda[_lda_keys.index(topic_num)]

        # plot crucial words
        for i in range(X_topics.shape[1]):
            plot_lda.text(topic_coord[i, 0], topic_coord[i, 1], [topic_summaries[i]])

        # hover tools
        hover = plot_lda.select(dict(type=HoverTool))
        hover.tooltips = {"content": "@content - topic: @topic_key"}

        show(plot_lda)

    def label_sentences(self, corpus, label_type):
        """
        Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
        We do this by using the TaggedDocument method. The format will be "TRAIN_i" or "TEST_i" where "i" is
        a dummy index of the post.
        """
        labeled = []
        for i, v in enumerate(corpus):
            label = label_type + '_' + str(i)
            labeled.append(TaggedDocument(v.split(), [label]))
        return labeled

    def split_train_test(self, toxic_dataset):
        # Train Test Split
        X_train, X_test, y_train, y_test = train_test_split(toxic_dataset.comment_text_clean, toxic_dataset.istoxic, random_state=0, test_size=0.3)
        X_train = self.label_sentences(X_train, 'Train')
        X_test = self.label_sentences(X_test, 'Test')
        all_text = X_train + X_test
        return all_text, X_train, X_test, y_train, y_test

    def train_LR(self, tqdm, all_text):
        for epoch in range(30):
            self.model_dbow.train(utils.shuffle([x for x in tqdm(all_text)]), total_examples=len(all_text), epochs=1)
            self.model_dbow.alpha -= 0.002
            self.model_dbow.min_alpha = self.model_dbow.alpha

    def get_vectors(self, corpus_size, vectors_size, vectors_type):
        """
        Get vectors from trained doc2vec model
        :param doc2vec_model: Trained Doc2Vec model
        :param corpus_size: Size of the data
        :param vectors_size: Size of the embedding vectors
        :param vectors_type: Training or Testing vectors
        :return: list of vectors
        """
        vectors = np.zeros((corpus_size, vectors_size))
        for i in range(0, corpus_size):
            prefix = vectors_type + '_' + str(i)
            vectors[i] = self.model_dbow.docvecs[prefix]
        return vectors

    def get_accuracy_score(self, y_pred, y_test):
        print('accuracy %s' % accuracy_score(y_pred, y_test))
        print(classification_report(y_test, y_pred))

    def print_confusion_matrix(self, y_pred, y_test):
        conf_mat = confusion_matrix(y_test, y_pred)
        print('Confusion Matrix')
        print(conf_mat)

    def split_train_test_svm(self, toxic_dataset):
        X = toxic_dataset.comment_text_clean
        y = toxic_dataset.istoxic
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)
        return X_train, X_test, y_train, y_test

    def print_classification_report(self, y_test, y_pred):
        print(classification_report(y_test, y_pred))

    def run_mnb_algorithm(self, merged_df):
        X_body_text = merged_df['comment_text_clean'].values
        X_title_text = merged_df['comment_text'].values
        y = merged_df['label'].values
        tfidf = TfidfVectorizer(ngram_range=(1,2), max_df= 0.85, min_df= 0.01)
        X_body_tfidf = tfidf.fit_transform(X_body_text)
        X_title_tfidf = tfidf.fit_transform (X_title_text)
        indices = merged_df.index.values
        X_body_tfidf_train, X_body_tfidf_test, \
        y_body_train, y_body_test, \
        indices_body_train, indices_body_test = train_test_split(X_body_tfidf, y, indices, test_size = 0.2, random_state=42)
        merged_df.loc[indices_body_train].groupby('label').agg('count')
        merged_df.loc[indices_body_test].groupby('label').agg('count')
        nb_body = MultinomialNB()
        nb_body.fit(X_body_tfidf_train, y_body_train)
        y_body_train_pred = nb_body.predict(X_body_tfidf_train)
        y_body_pred = nb_body.predict(X_body_tfidf_test)
        self.print_classification_report(y_body_test, y_body_pred)

    def build_confusion_matrix_toxicity(self, classifier, train_news):
        k_fold = KFold(n_splits=5, random_state=None, shuffle=False)
        scores = []
        confusion = np.array([[0,0],[0,0]])

        for train_ind, test_ind in k_fold.split(train_news):
            train_text = train_news.iloc[train_ind]['comment_text_clean'] 
            train_y = train_news.iloc[train_ind]['istoxic']
        
            test_text = train_news.iloc[test_ind]['comment_text_clean']
            test_y = train_news.iloc[test_ind]['istoxic']
            
            classifier.fit(train_text,train_y)
            predictions = classifier.predict(test_text)
            
            confusion += confusion_matrix(test_y,predictions)
            score = f1_score(test_y,predictions)
            scores.append(score)
        
        print('Total statements classified:{}'.format(len(train_news)))
        print('Score:{}'.format(sum(scores)/len(scores)))
        print('Score length:{}'.format(len(scores)))
        cm = confusion
        sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
        plt.tight_layout()
        plt.title('Confusion matrix', y=1.1)
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')

    def run_rf_toxicity(self, train_news, test_news):
        self.random_forest.fit(train_news['comment_text_clean'],train_news['istoxic'])
        predicted_rf = self.random_forest.predict(test_news['comment_text_clean'])
        build_confusion_matrix_toxicity(self.random_forest)

    def predict_toxicity_rf(self, text):
        predicted = self.random_forest.predict([text])
        predicedProb = self.random_forest.predict_proba([text])[:,1]
        return bool(predicted), float(predicedProb)

    def DATAMINERS_getToxicityScore(self, text):
        binaryValue, probValue = predict_toxicity_rf(text)
        return (float(probValue))