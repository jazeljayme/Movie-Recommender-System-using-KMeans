import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.metrics.pairwise import euclidean_distances, linear_kernel
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from afinn import Afinn
from imblearn.under_sampling import RandomUnderSampler
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import en_core_web_sm

import spacy
import nltk
import unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem.snowball import SnowballStemmer
import re
from nltk import FreqDist

tokenizer = ToktokTokenizer()
nlp = en_core_web_sm.load()
nltk.download('stopwords', quiet=True)
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.append('nt')
data_dir='/home/msds2022/jjayme/DMW_3_Lab/jjayme/amazon-reviews'


def get_data():
    """Return the dataframe that contains the reviews."""

    data = '/amazon_reviews_us_Digital_Ebook_Purchase_v1_01.pkl'
    df = pd.read_pickle(data_dir+data)
    df = df.dropna()
    df['review_date']=pd.to_datetime(df['review_date'])
    df['star_rating'] = df['star_rating'].astype(int)
    
    return df


def stardist_plotter(df, f_num):
    """Plot the star ratings distribution."""
    
    fig = plt.figure(figsize = (10,5))
    ax = fig.gca()
    ax.set_title('Figure '+str(f_num)+'. Distribution of the'
                 ' star rating feature.', fontsize=16)
    ax.set_xlabel('Number of ratings', fontsize=14)
    ax.set_ylabel('Ratings', fontsize=14)
    df['star_rating'].hist(orientation='horizontal',
                           ax=ax, color='b', grid=False);

    
def helpful_votes_distplotter(df, f_num):
    """Plot the distribution of helpful votes."""
    
    fig = plt.figure(figsize = (10,5))
    ax = fig.gca()
    ax.grid(False)
    ax.set_title('Figure '+str(f_num)+'. Distribution of the'
                 ' helpful votes feature.', fontsize=16)
    ax.set_xlabel('Count of helpful votes', fontsize=14)
    ax.set_ylabel('Number of reviews', fontsize=14)
    df['helpful_votes'].hist(ax=ax, range=[0, 20],
                             color='b', grid=False);
    
    
def total_votes_distplotter(df, f_num):
    """Plot the distribution of total votes."""
    
    fig = plt.figure(figsize = (10,5))
    ax = fig.gca()
    ax.grid(False)
    ax.set_title('Figure '+str(f_num)+'. Distribution of the'
                 ' helpful votes feature.', fontsize=16)
    ax.set_xlabel('Count of total votes', fontsize=14)
    ax.set_ylabel('Number of reviews', fontsize=14)
    df['total_votes'].hist(ax=ax, range=[0, 20],
                          color='b', grid=False);

    
def pearson_heatmap(df, f_num):
    """Plot pearson correlational heatmap."""
    
    fig = plt.figure(figsize = (10,5))
    ax = fig.gca()
    ax.set_title('Figure '+str(f_num)+'. Feature'
                 ' Correlational Heatmap', fontsize=16)
    sns.heatmap(df.corr());

                
def plot_year_reviews(df, f_num):
    """Plot the distributions of reviews per year."""
    
    plt.figure(figsize = (10,5))
    sns.displot(df['review_date'], height=5, aspect=1.75, color='b')
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Counts', fontsize=14)
    plt.title('Figure '+str(f_num)+'. Number of Reviews per year',
              fontsize=16);

    
def year_customer_reviews(df, f_num):  
    """Plot customer counts and reviews per year."""
    
    df['year']=df['review_date'].dt.year    
    dummy = df['review_date'].dt.year.value_counts()
    df2 = pd.DataFrame(dummy).reset_index()
    df2.rename(columns={'index':'year', 'review_date':'Review counts'}, inplace=True)
    dummy = pd.DataFrame(df.groupby('year')['customer_id'].nunique()).reset_index()
    df3 = df2.merge(dummy, on='year')
    plt.figure(figsize=(10,5))

    plt.plot(df3['year'], df3['Review counts'], 'o-', color='b',
             label='Review Counts')
    plt.plot(df3['year'], df3['customer_id'], 'o-', color='orange',
            label='Customer Counts')

    plt.xlabel('year', fontsize=14)
    plt.ylabel('counts', fontsize=14)
    plt.title('Figure '+str(f_num)+'. Total number of reviews'
              ' and customers from year 1999-2013',
              fontsize=16)
    plt.legend(bbox_to_anchor=(1,1))

    
def bar_plot_year(df, f_num):
    """Plot the distribution of reviews per year."""
    
    fig = plt.figure(figsize = (10,5))
    ax = fig.gca()
    ax.set_title('Figure '+str(f_num)+'. Number of'
                 ' Reviews per year', fontsize=16)
    ax.set_xlabel('Number of reviews', fontsize=14)
    ax.set_ylabel('Year', fontsize=14)
    df['year'].value_counts().plot(kind='barh',
                                   color='b',ax=ax);

    
def grouper(df, year, f_num, twozeroten_mod=False):
    """Returns grouped data points per year and top 10 books per year."""
    
    grouped_years = df.groupby('year')
    if twozeroten_mod == False:
        fig = plt.figure()
        ax = fig.gca()
        ax.set_title('Figure '+str(f_num)+'. '+
                     'Top 10 books based on ratings and purchase/review'
                     ' count for '+str(year), 
                     fontsize=14)
        ax.set_xlabel('Number of reviews/purchases', fontsize=12)
        df_yr = grouped_years.get_group(year)
        df_yr_5 = df_yr[df_yr['star_rating'] == 5]
        df_yr_5['product_title'].value_counts().head(10).plot(kind='barh', 
                                                    color='b',ax=ax);
    else:
        fig = plt.figure()
        ax = fig.gca()
        ax.set_title('Figure '+str(f_num)+'. Top 10 books based on ratings'
                     ' and purchase/review count for 2010', 
                     fontsize=14)
        ax.set_xlabel('Number of reviews/purchases', fontsize=12)
        df_yr = grouped_years.get_group(2010)
        df_yr_5 = df_yr[df_yr['star_rating'] == 5]
        df_2010_5_sol = df_yr_5[(df_yr_5['star_rating'] == 5) & \
                        (df_yr_5['product_title'] != \
                         '2010 ABNA Quarterfinalist 11')]
        df_2010_5_sol['product_title'].value_counts().head(10)\
        .plot(kind='barh', ax=ax, color='b');
        
        
def truncate_df(start_date, df):
    """Return a subset of the amazon e-book data
    given `start_date`."""
    
    df = df[df['review_date'] >= start_date]
    return df


def remove_accented_chars(text):
    """Convert and standardize text into ASCII characters. It made
    sure characters which look identical actually are identical.
    Lastly, convert text into small letters.
    """
    
    text = unicodedata.normalize('NFKD', text).encode('ascii',
                              'ignore').decode('utf-8', 'ignore')
    return text.lower()


def remove_stopwords(text, is_lower_case=False):
    """Return the filtered text where stopwords are removed."""

    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in
                           stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() 
                           not in stopword_list]
    text = ' '.join(filtered_tokens)
    return text


def remove_special_characters(text, remove_digits=True):
    """Return the filtered text where special caharacters
    are removed."""
        
    text= re.sub(r'[\r|\n|\[|\]]+', '',text)
    text = text.replace('\\\\', '')
    #remove symbols
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    text = re.sub('br', '', text)
    text = re.sub(' +', ' ', text)
    return text
    
    
def lemmatize_text(text):
    """Return the the root forms of the text."""

    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-'
                     else word.text for word in text])
    return text


def process_doc(doc):
    """Return the processed text that will be used in the
    analysis."""
    
    doc = str(doc)
    doc = remove_accented_chars(doc)
    doc = remove_stopwords(doc)
    doc = remove_special_characters(doc)
    doc = lemmatize_text(doc)
    return doc


def get_sentiment_df(df, start_date='2013-01-01',
                     n=20000, target_col='review_headline'):
    """Return a dataframe with clean text and `score_rating`."""
    
    df = truncate_df(start_date, df)
    X, y = random_equal_sampler(df, n)
    df1 = X.copy()
    df1['cleaned_'+target_col] = df1[target_col].apply(lambda x:
                                                    process_doc(x))
    df1['star_rating'] = y
    df1['score_group'] = -1
    df1.loc[df1['star_rating'].isin([4,5]), 'score_group'] = 1 
    df1.loc[df1['star_rating'].isin([3]), 'score_group'] = 0 
    return df1


def random_equal_sampler(df, n, random_state=0):
    """Return the random subset of Dataframe with n samples for each
    star rating.
    Parameters
    ----------
    df: pandas DataFrame
        The dataframe that contains the reviews.
        
    n : int
        The sample size for each class.
        
    random_state : int
        Control the randomization of the algorithm
    
    Returns
    --------
    X_resampled : {array-like, dataframe, sparse matrix} 
                The array containing the resampled data.

    y_resampled: array-like
                The corresponding label of X_resampled.
    """

    strategy = {1:n, 2:n, 3:n, 4:n, 5:n}
    rus = RandomUnderSampler(random_state=random_state,
                             sampling_strategy= strategy)
    X = df[['product_title','review_headline','review_body']]
    y = df['star_rating'].values
    rus.fit(X, y)
    X_resampled, y_resampled = rus.fit_resample(X, y )
    return X_resampled, y_resampled


def plot_sentiments_count(df, f_num):
    """Plot the distribution of sentiments of the dataset."""
    
    f_num = str(f_num)
    plt.figure(figsize = (14,5))
    negative_reviews = df[df['score_group'] ==-1]
    positive_reviews = df[df['score_group'] == 1]
    neutral_reviews = df[df['score_group'] == 0]
    
    sns.barplot(x =['positive', 'neutral', 'negative'],
                y=[len(positive_reviews),len(neutral_reviews),
                   len(negative_reviews)],data = df)
    
    plt.xlabel("Sentiments", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.title("Figure "+f_num+'. Sentiment Counts',fontsize=16)
    plt.show()
    
    
def plot_freq_words(df, sentiment, f_num):
    """Plot the Top 50 terms of the sentiment groups."""
    
    f_num = str(f_num)
    if str.lower(sentiment) == 'positive':
        reviews = df[df['score_group'] == 1]
        
    elif str.lower(sentiment) == 'negative':
        reviews = df[df['score_group'] ==-1]
        
    elif str.lower(sentiment) == 'neutral':
        reviews = df[df['score_group'] == 0]
    

    
    freq_dist = FreqDist([word for review in 
                      reviews['cleaned_review_headline']
                       for word in str(review).split()])
    plt.figure(figsize = (14,6))
    plt.title('Figure '+f_num+ '. Cumulative counts of '+
              'Top 50 most frequent words belonging to '+
              sentiment+' Sentiment', fontsize=16)
    
    plt.xlabel('Word', fontsize=14)
    plt.ylabel('Cumulative Counts', fontsize=14)
    freq_dist.plot(50, cumulative = True);
   
    
def word_cloud(df, sentiment, f_num, afinn=False):
    """Plot the wordcloud of the sentiments using the star
    rating groups from the reviews."""
    
    f_num = str(f_num)
    if afinn:
        reviews = df[df['sentiment_category'] == \
                     str.lower(sentiment)]
    else:  
        if str.lower(sentiment) == 'positive':
            reviews = df[df['score_group'] == 1]
        elif str.lower(sentiment) == 'negative':
            reviews = df[df['score_group'] ==-1]
           
        elif str.lower(sentiment) == 'neutral':
            reviews = df[df['score_group'] == 0]

    color = {'Negative': {'colormap' :'viridis',
                           'background_color' : 'black'},
              'Positive': {'colormap' :'viridis',
                           'background_color' : 'white'},
              'Neutral': {'colormap' :'plasma',
                           'background_color' : 'white'}}
    
    
    text  =  ' '.join([word for review in
                       reviews['cleaned_review_headline']
                       for word in review.split()])
    
    wordcloud = WordCloud(background_color=color[sentiment]\
                         ['background_color'], 
                        stopwords = stopword_list, 
                        colormap = color[sentiment]['colormap'],
                        max_words = 100, 
                       collocations=False).generate(text)
    plt.figure(figsize = (8,6), dpi = 200)
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis("off")
    if afinn:
        plt.title("Figure "+f_num+'. AFINN Word Count belonging to '
                  + sentiment+' Sentiment', fontsize=12)
    else:
        plt.title('Figure '+f_num+'. Word Cloud '
              'belonging to '+ sentiment+' Sentiment', fontsize=12)
    plt.show()
    
    
def get_afinn_score(df, plot=True, f_num=0):
    """Return a dataframe with new sentiment score using AFINN
    Sentiment scores.
    
    If plot is True, then plot the distribution of the
    Afinn sentiment groups."""
    f_num = str(f_num)
    corpus = df['cleaned_review_headline'].values
    af = Afinn() #initializing Afinn
    
    #Generating scores for every review ( from 0 to +5)
    sentiment_scores = [af.score(review) for review in corpus]
    #generating categories
    sentiment_category = ['positive' if score > 0\
                         else 'negative' if score < 0\
                         else 'neutral' \
                         for score in sentiment_scores]

    #Plotting the sentiment group counts
    df['sentiment_score'] = sentiment_scores
    df['sentiment_category'] = sentiment_category
    negative_reviews = df[df['sentiment_category'] == 'negative']
    positive_reviews = df[df['sentiment_category'] == 'positive']
    neutral_reviews = df[df['sentiment_category'] == 'neutral']
    
    if plot:
        plt.figure(figsize = (14,5))
        sns.barplot(x = df['sentiment_category'].unique(),
                    y=[len(neutral_reviews),
                        len(positive_reviews),
                       len(negative_reviews)], data = df)
        plt.xlabel("Sentiments", fontsize=14)
        plt.ylabel("Counts", fontsize=14)
        plt.title("Figure "+f_num+'. AFINN Sentiment Counts',
                  fontsize=16)
        plt.show()
    return df


def project_svd(q, s, k):
    """Accept q, s and k and return the design matrix projected on to the
    first k singular vectors.
    """
    return q[:,:k].dot(s[:k,:k])


def plot_svd(X_new, features, p, f_num):
    """
    Plot transformed data and features on to the first two singular vectors
    
    Parameters
    ----------
    X_new : array
        Transformed data
    featurs : sequence of str
        Feature names
    p : array
        P matrix
    """
    fig, ax = plt.subplots(1, 2, subplot_kw=dict(aspect='equal'), 
                           gridspec_kw=dict(wspace=0.4), dpi=150)
    ax[0].scatter(X_new[:,0], X_new[:,1])
    ax[0].set_xlabel('SV1', fontsize=8)
    ax[0].set_ylabel('SV2', fontsize=8)

    for feature, vec in zip(features, p):
        ax[1].arrow(0, 0, vec[0], vec[1], width=0.01, ec='none', fc='r')
        ax[1].text(vec[0], vec[1], feature, ha='center', color='r', fontsize=5)
    ax[1].set_xlim(-1, 1)
    ax[1].set_ylim(-1, 1)
    ax[1].set_xlabel('SV1', fontsize=8)
    ax[1].set_ylabel('SV2', fontsize=8)
    plt.suptitle('Figure '+str(f_num)+'. Projection of features on'
                 ' SV1 and SV2',fontsize=10)

    
def plot_variance_per_n(tfidf, n_components=429,threshold_variance=0.80,
                        plot=True, f_num=None):
    """Return the optimal n_components `k` and the model `lsa`
    after performing the truncated SVD to the tfidf matrix.
    
    If plot is True, then plot the variance explained
    per n_components of the singular values SVs.
    
    Parameters
    ---------
    tfidf : array-like or matrix
        The data to be used in the analysis.
    n_components : int
        The number of components that will be used in the svd.
    threshold_variance : int
        The threshold variance that will be set in the analysis.
    plot : boolean
        If plot is True, then plot the variance explained
        per n_components of the singular values SVs.
    f_num : int
        Figure number to be annotated in the figure title.
    """
    
    lsa = TruncatedSVD(n_components=n_components)
    doc_topic = lsa.fit_transform(tfidf)
    variance_explained_ratio = lsa.explained_variance_ratio_
    df = pd.DataFrame(variance_explained_ratio.cumsum())
    k = df[df[0] > threshold_variance].index[0]
    
    if plot:
        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(range(1, len(variance_explained_ratio)+1), 
                variance_explained_ratio,'-', label='individual')
        ax.set_xlim(0, len(variance_explained_ratio)+1)
        ax.set_xlabel('SV', fontsize=14)
        ax.set_ylabel('variance explained', fontsize=14)
        ax = ax.twinx()
        ax.plot(range(1, len(variance_explained_ratio)+1), 
                variance_explained_ratio.cumsum(), 'r-', label='cumulative')
        ax.axhline(threshold_variance, ls='--', color='g')
        ax.axvline(k, ls='--', color='g')
        ax.set_ylabel('cumulative variance explained')
        ax.set_title('Figure '+str(f_num)+'. Variance explained for n SVs',
                     fontsize=16);
    return k, lsa, doc_topic


def plot_SV_vs_features(doc_topic, tfidf, model, feature_names, f_num=20):
    """Return topic vector `VT` and plot the projected features
    onto SV1 and SV2."""
    
    q = doc_topic / model.singular_values_
    sigma = np.diag(model.singular_values_)
    VT = model.components_
    U = model.transform(tfidf) / model.singular_values_
    X_new = project_svd(q, sigma, k=131)
    plot_svd(X_new, feature_names, VT, f_num)
    return VT


def get_matrix_tfidf(df):
    """Apply TFIDF to the data into and return the tfidf matrix and
    dataframe format."""
    
    corpus = df['cleaned_review_headline'].values
    TFIDF = TfidfVectorizer(min_df=20,lowercase=True)
    tfidf = TFIDF.fit_transform(corpus)
    feature_names = TFIDF.get_feature_names()
    df1 = pd.DataFrame(tfidf.toarray(), columns=TFIDF.get_feature_names())
    return tfidf, df1, feature_names


def plot_topics(Vt,feature_names, f_num=2):
    """Plot the top 10 topics using LSA model.Use only
    the first 6 SVs."""
    
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18,8))
    plt.suptitle('Figure '+str(f_num)+'. Top 10 Topics correlated with the'
                 ' first 6 SVs', fontsize=18)

    for i, ax in enumerate(axs.reshape(-1)): 
        order = np.argsort(np.abs(Vt[:, i]))[-10:]
        ax.barh([feature_names[o] for o in order], Vt[order, i])
        ax.set_title(f'SV{i+1}', fontsize=15)
        if i in [0,3]:
            ax.set_ylabel(str('Topic'), fontsize=15)
        ax.set_xlabel(str('Weight'), fontsize=15)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])


def display_topics(model, feature_names, no_top_words, topic_names=None,
                   n_topic=10):
    """Display the top 10 topics using the `model`."""
    
    for ix, topic in enumerate(model.components_[:n_topic]):
        if not topic_names or not topic_names[ix]:
            print("\nTopic ", ix)   
        else:
            print("\nTopic: '",topic_names[ix],"'")
        print(", ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))
        
