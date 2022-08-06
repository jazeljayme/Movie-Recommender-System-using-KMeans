# The Secrets of Binge Watching: A Movie Recommender System

<p align="justify">“What is the secret of binge-watching? and “what to watch next?” are related questions thay may have a single answer: a movie recommender system. A simple movie recommender offers generalized recommendations to users based on movie popularity and/or genre. This method does not provide personalized recommendations based on the user’s profile and history.</p>

<p align="justify">This project aims to provide a glimpse into the computer program that powers and contributes to the success of streaming giants like Netflix. Instead of popularity, movie synopses and genre were employed in this model, and the goal is to generate meaningful recommendations to users for movies that may interest them. The researchers believe that developing a simple cosine similarity-based movie recommender system may give local media companies and aspiring streaming service providers a headstart. It can also benefit the consumers by improving decision-making process and quality, thereby eliminating the question “what to watch next?”.</p>

<p align="justify">The recommender system was created using an IMDb scraped dataset, with the features 1) about and 2) genre as the main focus. The raw data were cleaned and preprocessed before converting the relevant features into a Term Frequency and Inverse Document Frequency (TF-IDF) matrix. This sparse matrix representation of token counts served as a basis for the two key techniques implemented in this project: 1) model creation using cosine similarity and 2) cluster analysis using k-means. The latter required dimensionality reduction, therefore Latent Semantic Analysis was performed to the TF-IDF matrix. The output clusters of similar movies were primarily used as pseudo-ground truth labels in evaluating the performance of the movie recommender system.</p>

<p align="justify">The following are the key findings from calculating the model performance:</p>

<ol>
<li>The most effective feature combination was the movie synopses paired with genres, which generated the highest precision score.</li>
<li>The seven output clusters produced by the k-means clustering algorithm that served as pseudo-ground truth labels were successful in evaluating the model's performance, when combined with the results of the model.</li>
<li>The recommender system outperforms random prediction by a factor of two. The former had an average precision score of 37.16%, whereas the latter had only 17.20%.</li>
</ol>

<p align="justify">This movie recommender system is simple, but its performance can be further enhanced. Improvements can be achieved by: 1) acquiring and utilizing larger dataset, as the model only used roughly 1,400 data points; 2) adding user demographic data as features, and; 3) including TV series data as episodic programming is also thought to be the key driver of binge-watching.</p>

<p align="justify">The researchers recognize the value that binge-watching provides people during this challenging period brought about by COVID-19, but do not in anyway advocate for it. It is recommended to deep dive into the negative attributes of binge-watching to unravel the multiple issues surrounding it, as well as highlighting the distinction between compulsive and recreational binge-watching.</p>

Open <a href="https://github.com/jazeljayme/Movie-Recommender-System-using-KMeans/blob/master/FinalProject_TechPaper.ipynb">FinalProject_TechPaper.ipynb</a> to view the full report.
