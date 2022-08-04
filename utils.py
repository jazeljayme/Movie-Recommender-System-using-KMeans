from wordcloud import WordCloud
from PIL import Image, ImageFont
import nltk
import spacy
import en_core_web_sm
import re
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize.toktok import ToktokTokenizer
import warnings
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity, sigmoid_kernel
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import euclidean_distances, linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, TruncatedSVD, PCA
from scipy import interpolate
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
import unicodedata

import pandas as pd
pd.set_option('display.max_colwidth', 0)


warnings.filterwarnings('ignore')

tokenizer = ToktokTokenizer()
nlp = en_core_web_sm.load()
nltk.download('stopwords', quiet=True)
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.append('nt')


def get_data():
    """Return a dataframe contaning the dataset"""

    df = pd.read_csv('final_cleaned.csv')
    df = df.drop(['Unnamed: 0'], axis=1)
    return df


def get_df_dummy(df):
    """Return a dataframe containing the cleaned movie genre"""

    y = df.genre.apply(lambda x: x.split(',')[0])
    genre = y.str.replace(r'\s+', '')
    df_dummy = pd.DataFrame()
    df_dummy['genre'] = genre
    return df_dummy


def process_doc(doc):
    """Return the processed text that will be used in the
    analysis."""

    doc = str(doc)
    doc = remove_accented_chars(doc)
    doc = remove_stopwords(doc)
    doc = remove_special_characters(doc)
    doc = lemmatize_text(doc)
    return doc


def remove_accented_chars(text):
    """Convert and standardize text into ASCII characters. It made
    sure characters which look identical actually are identical.
    Lastly, convert text into small letters.
    """

    text = unicodedata.normalize(
        'NFKD',
        text).encode(
        'ascii',
        'ignore').decode(
            'utf-8',
        'ignore')
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
    """Return the filtered text where special caharacters are removed."""

    text = re.sub(r'[\r|\n|\[|\]]+', '', text)
    text = text.replace('\\\\', '')
    # remove symbols
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


def add_overview_genre(df, df_dummy):
    df['cleaned_overview'] = df['about'].apply(lambda x: utils.process_doc(x))
    df['genre'] = df_dummy['genre'].str.lower()
    return df


def combined_features(row):
    return row['cleaned_overview'] + " " + row['genre']


def tfidf_data(df):
    TFIDF = TfidfVectorizer(min_df=10, lowercase=True)
    tfidf = TFIDF.fit_transform(df["combined_features"])
    feature_names = TFIDF.get_feature_names()
    df1 = pd.DataFrame(tfidf.toarray(), columns=TFIDF.get_feature_names())
    df1.head(2)
    return df1, tfidf


def add_combined_features(combined_features, df):
    df["combined_features"] = df.apply(combined_features, axis=1)
    df["combined_features"].head(2)
    return df


def get_nmf_rmse(df1):
    """Return RMSE or the root mean squared error and the best k for topic
    modelling"""

    rmse_all = []
    num_topics = df1.shape[1]
    for k in range(1, num_topics, 5):
        A = df1.copy()
        model = NMF(n_components=k, init='random', random_state=0)
        W = model.fit_transform(A)
        H = model.components_
        # get the reconstructed A with dimensions k
        A_k = W.dot(H)
        rmse_frob = mean_squared_error(A, A_k, squared=False)
        # getting reconstruction error (RMSE)
        rmse_all.append(rmse_frob)

    x = np.arange(1, num_topics, 5)
    rmse = rmse_all

    # interpolate missing values or "gaps"
    xnew = np.arange(1, num_topics)
    f = interpolate.interp1d(x, rmse, fill_value="extrapolate")
    rmse_new = f(xnew)

    len(rmse_new)

    # setting the threshold
    frob_norm_A = np.linalg.norm(A, 'fro')
    threshold = frob_norm_A * 0.001

    rmse_bool = rmse_new < threshold
    best_k = np.where(rmse_bool)[0][0] + 1

    return rmse_all, rmse_new, threshold, best_k, num_topics


def plot_interpolated_curve(rmse_all, num_topics, best_k):
    """Plot the interpolated reconstruction error curve"""

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, num_topics, 5), rmse_all, 'o-', color='#632547')

    plt.axvline(best_k, ls='--', color='g')
    plt.title('Optimal n_components for topic modeling of Movie '
              'Overview ', fontsize=14)
    plt.ylabel('Reconstruction Error', fontsize=12)
    plt.xlabel('n_components\n', fontsize=12)
    print(f"Min rec_err: {min(rmse_all)} at k: {(best_k)}", "\n")
    plt.show()


def nmf(best_k, tfidf, n_topics):
    """Reduce the dimension of data to the best n components"""

    n_topics = best_k
    nmf_ = NMF(n_components=n_topics, max_iter=100).fit(tfidf)
    nmf = nmf_.transform(tfidf)
    return nmf


def init_cosine_sim(tfidf):
    """Initiate cosine similarity"""

    cosine_sim = cosine_similarity(tfidf)
    return cosine_sim


def get_top_recommended_movies(df, title, k, cosine_sim):
    """Return the top k movie if found in the database, else
    tell user to input another movie.
    """
    movies = {}
    title = title.lower()
    if title in df.name.str.lower().unique():

        index = df[df.name.str.lower() == title].index[0]
        similar_movies_cos = list(enumerate(cosine_sim[index]))
        sorted_similar_movies_cos = sorted(similar_movies_cos,
                                           key=lambda x: x[1],
                                           reverse=True)[1:k + 1]

        for movie_id, score in sorted_similar_movies_cos:
            movies[df.iloc[movie_id]['name']] = df.iloc[movie_id]['about']
        recom_df = pd.DataFrame.from_dict(movies, orient='index')\
            .reset_index()
        recom_df.columns = ['Movie Name', 'About']
        return recom_df
    else:
        return "Movie not found in the database."


def get_random_model_movies(df, title, k, random_state=10):
    """Return  of the movie if found in the database, else
    tell user to input another movie.
    """
    movies = {}
    title = title.lower()
    if title in df.name.str.lower().unique():

        index = df[df.name.str.lower() == title].index[0]
        np.random.seed(random_state)
        choices = np.random.choice(df.index, size=k)

        for movie in choices:
            movies[df.iloc[movie]['name']] = df.iloc[movie]['about']
        recom_df = pd.DataFrame.from_dict(movies, orient='index')\
            .reset_index()
        recom_df.columns = ['Movie Name', 'About']
        return recom_df
    else:
        return "Movie not found in the database."


def cluster_range(X, clusterer, k_start, k_stop, actual=None):
    """For a given range of k, perform a given clustering method and
    return relevant results
    """
    ys = []
    centers = []
    inertias = []
    chs = []
    scs = []

    for k in range(k_start, k_stop + 1):
        clusterer_k = clone(clusterer)
        clusterer_k.set_params(n_clusters=k)
        clusterer_k.fit(X)
        y = clusterer_k.predict(X)
        ys.append(y)
        centers.append(clusterer_k.cluster_centers_)
        inertias.append(clusterer_k.inertia_)
        chs.append(calinski_harabasz_score(X, y))
        scs.append(silhouette_score(X, y))

    cluster_dict = {'ys': ys,
                    'centers': centers,
                    'inertias': inertias,
                    'chs': chs,
                    'scs': scs
                    }

    return cluster_dict


def plot_internal(chs, scs):
    """Plot internal validation values"""

    fig, ax = plt.subplots(figsize=(15, 8))
    ks = np.arange(2, len(chs) + 2)
    ax.plot(ks, chs, '-o', label='CH', color='#632547')
    ax.set_xlabel('$k$')
    ax.set_ylabel('Calinski Harabasz score')
    ax.yaxis.set_label_position("left")
    lines, labels = ax.get_legend_handles_labels()
    ax2 = ax.twinx()
    ax2.plot(ks, scs, '-ro', label='Silhouette coefficient')
    ax2.set_ylabel('Silhouette score')
    ax2.yaxis.set_label_position("right")

    plt.xlabel('Number of Clusters')
    plt.title('Internal Validation Values', size=18)
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines + lines2, labels + labels2, bbox_to_anchor=(1.1, 1))


def plot_inertia(cluster_dict):
    """Plot Clusters Inertia"""

    plt.subplots(figsize=(15, 8))
    plt.plot(np.arange(2, len(cluster_dict['inertias']) + 2),
             cluster_dict['inertias'], 'o-', label='inertia',
             color='#632547')
    plt.title('Inertia Validation Values', size=18)
    plt.xlabel('Number of Clusters', size=12)
    plt.ylabel('Inertia', size=12)
    plt.legend()


def plot_lsa(df1):
    """Plot the clusters of truncated data using LSA"""

    lsa = TruncatedSVD(n_components=2, random_state=1337)
    X_new = lsa.fit_transform(df1.to_numpy())
    kmeans_ng = KMeans(n_clusters=7, random_state=1337)
    y_predict = kmeans_ng.fit_predict(X_new)
    plt.figure(figsize=(10, 10))
    plt.scatter(X_new[:, 0], X_new[:, 1], c=y_predict, cmap='rocket')
    plt.title('First Two Singular Vectors Clustering', size=18)
    plt.xlabel('SV1', fontsize=12)
    plt.ylabel('SV2', fontsize=12)
    return y_predict


def similar_color_func(word=None, font_size=None,
                       position=None, orientation=None,
                       font_path=None, random_state=None):
    """Set the color palette for the Wordcloud texts"""

    h = 327  # 0 - 360c
    s = 46  # 0 - 100
    l = random_state.randint(30, 70)  # 0 - 100
    return "hsl({}, {}%, {}%)".format(h, s, l)


def wordcloud_cluster(df, y_predict, similar_color_func):
    """Plot the Wordcloud per cluster"""

    df['cluster_label'] = y_predict
    wiki = df[['name', 'cluster_label', 'votes']].\
        sort_values(by='cluster_label')
    for k in range(0, 7):

        df = wiki[wiki['cluster_label'] == k].sort_values(
            'votes', ascending=False)[:150]

        titles = df['name'].values
        d = dict(zip(titles, [1] * len(titles)))

        wordcloud = WordCloud(max_font_size=40, max_words=100,
                              min_font_size=6,
                              color_func=similar_color_func
                              ).generate_from_frequencies(d)
        # show
        plt.figure(figsize=(15, 10))
        plt.title('Cluster ' + str(k + 1), fontsize=18)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")

        plt.tight_layout()
        name_fig = 'cluster' + str(k) + '.png'
        plt.savefig(name_fig, dpi=300)


def get_index_of_top_recommended_movies(df, title, k, cosine_sim):
    """Return the labels of recommended movie if found in the database."""

    title = title.lower()
    if title in df.name.str.lower().unique():

        index = df[df.name.str.lower() == title].index[0]
        similar_movies_cos = list(enumerate(cosine_sim[index]))
        sorted_similar_movies_cos = sorted(similar_movies_cos,
                                           key=lambda x: x[1],
                                           reverse=True)[1:k + 1]
        movies_label = []
        for movie_idx, score in sorted_similar_movies_cos:
            movies_label.append(movie_idx)

        return movies_label


def get_index_of_random_movies(df, title, k, random_state=10):
    """Return  of the movie if found in the database, else
    tell user to input another movie.
    """
    title = title.lower()
    movies_label = []
    if title in df.name.str.lower().unique():

        index = df[df.name.str.lower() == title].index[0]
        np.random.seed(random_state)
        choices = np.random.choice(df.name, size=k)

        for movie in choices:
            random_label = df[df.name == movie].index[0]
            movies_label.append(random_label)
        return movies_label


def get_confusion(actual, results, all_labels):
    """Return the confusion matrix as a pandas DataFrame

    Accept the label of the correct class,
    the returned results as indices to the
    objects and all labels, and return the
    confusion matrix as a pandas DataFrame
    """
    df = pd.DataFrame(all_labels)
    ret_df = df[df.index.isin(results)]  # relevant
    not_df = df[~df.index.isin(results)]  # not relevant

    TP = sum(ret_df[0] == actual)
    FP = len(ret_df) - TP
    FN = sum(not_df[0] == actual)
    TN = len(not_df) - FN

    return pd.DataFrame([[TP, FP], [FN, TN]],
                        columns=['relevant', 'irrelevant'],
                        index=['relevant', 'irrelevant'])


def precision(confusion):
    """Accept a confusion matrix and returns the precision"""

    tp = confusion.relevant[0]
    fp = confusion.irrelevant[0]
    if tp + fp == 0:
        return 1
    return tp / (tp + fp)


def recall(confusion):
    """Accept a confusion matrix and returns the recall"""
    tp = confusion.relevant[0]
    fn = confusion.relevant[1]
    return tp / (tp + fn)


def summary_of_performance(actual, all_predicted_label_index, y_predict):
    """Return the precision and recall of the query."""

    confusion_movie = get_confusion(actual, all_predicted_label_index,
                                    y_predict)
    precision_score = precision(confusion_movie)
    recall_score = recall(confusion_movie)

    return precision_score, recall_score


def compute_model_performance(df_orig, k, cosine_sim):
    """Return the precision of all of the queries for the random model
    and the recommender model."""

    all_titles = df_orig.name.values
    y_predict = df_orig.label.values

    # store models precision scores
    model_precision_scores_all = []
    random_precision_scores_all = []

    for seed, title in enumerate(all_titles):

        movie_label = df_orig[df_orig['name'] == title]['label'].values[0]
        movies_recom_index = get_index_of_top_recommended_movies(
            df_orig, title, k, cosine_sim)
        movies_random_index = get_index_of_random_movies(df_orig, title,
                                                         k, random_state=seed)

        model_prec, model_rec = summary_of_performance(
            movie_label, movies_recom_index, y_predict)

        random_prec, random_rec = summary_of_performance(
            movie_label, movies_random_index, y_predict)

        model_precision_scores_all.append(model_prec)
        random_precision_scores_all.append(random_prec)
    return model_precision_scores_all, random_precision_scores_all


def plot_top_dir_votes(df):
    """Plot the directors with the highest votes"""

    director_vote = pd.DataFrame(df.groupby('director')['votes'].mean()
                                 .sort_values(ascending=False))
    plt.figure(figsize=(15, 8))
    plt.bar(director_vote[:10].index, director_vote[:10]['votes'],
            color='#632547')
    plt.title('Directors with the highest voted movies', size='18')
    plt.xticks(rotation=45)
    plt.xlabel('Director', size=12)
    plt.ylabel('Number of votes in Million', size=12)
    plt.plot()


def plot_low_dir_votes(df):
    """Plot the directors with the lowest votes"""

    director_vote = pd.DataFrame(df.groupby('director')['votes'].mean()
                                 .sort_values(ascending=False))
    plt.figure(figsize=(15, 8))
    plt.bar(director_vote[-10:].index, director_vote[-10:]['votes'],
            color='#632547')
    plt.title('Directors with the highest voted movies', size='18')
    plt.xticks(rotation=45)
    plt.xlabel('Director', size=12)
    plt.ylabel('Number of votes in Million', size=12)
    plt.plot()


def plot_genre(df):
    """Plot the top 10 genre with the highest number of movies"""

    df_genre = pd.DataFrame(df['genre'].value_counts().sort_values()
                            [-10:])
    plt.figure(figsize=(15, 8))
    plt.barh(df_genre.index, df_genre['genre'],
             color='#632547')
    plt.title('Genre with top most movies', size='18')
    plt.xticks(rotation=45)
    plt.xlabel('Numbr of moviews', size=12)
    plt.ylabel('Genre', size=12)
    plt.plot()


def plot_top_dir_rate(df):
    """Plot the directors with the highest rating"""

    director_rates = pd.DataFrame(df.groupby('director')['rating'].mean()
                                  .sort_values(ascending=False))
    plt.figure(figsize=(15, 8))
    plt.bar(director_rates[:10].index, director_rates[:10]['rating'],
            color='#632547')
    plt.title('Directors with the highest rated movies', size='18')
    plt.xticks(rotation=90)
    plt.xlabel('Director', size=12)
    plt.ylabel('Rates', size=12)
    plt.plot()


def plot_movies_per_yr(df):
    """Plot the number of movies per year"""

    df_year = pd.DataFrame(df.groupby('year').count()['id'])
    plt.figure(figsize=(15, 8))
    plt.plot(df_year.index, df_year['id'],
             color='#632547', marker='o')
    plt.title('Number of movies per year', size='18')
    plt.xlabel('Year', size=12)
    plt.plot()


def wordcloud_feature(df):
    """Plot the wordcloud for the About column of the dataset"""

    text = ' '.join([word for review in df['about']
                     for word in review.split()])
    wordcloud_about = WordCloud(background_color="white",
                                colormap='rocket', max_words=50,
                                collocations=False).generate(text)
    plt.figure(figsize=(15, 4), dpi=200)
    plt.title('Word Cloud of about column', fontsize=7)
    plt.imshow(wordcloud_about, interpolation='bilinear')
    plt.axis("off")
    plt.show()


def plot_corr(df):
    """Plot the correlation of the dataset"""

    mask = np.zeros_like(df.corr())
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(10, 7))
    ax = sns.heatmap(df.corr(), mask=mask, vmax=.3, square=True,
                     cmap='rocket')
