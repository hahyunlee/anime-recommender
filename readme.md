# Anime Recommender

![anime image](/img/anime_home.jpg)

## Background.
Anime refers to Japanese animation, which for decades has been produced by and for the local Japanese audience, but is now
a growing form of entertainment across the globe. Anime can be in the form of films, TV shows, and books.

The goal of this project is to explore concepts of recommendation systems while providing users a tool to find 
new anime to further enjoy. This writeup will highlight the process of creating two recommendation systems: 
- K Nearest Neighbors
- Alternating Least Squares

### Source and References.
Anime dataset was provided by "CooperUnion" from the 
![Kaggle](https://www.kaggle.com/CooperUnion/anime-recommendations-database).

Coding and implementation referenced from Kevin Liao's blog post on 
![Movie Recommendations](https://towardsdatascience.com/prototyping-a-recommender-system-step-by-step-part-1-knn-item-based-collaborative-filtering-637969614ea).

---
## Recommendation Systems.

The following is a brief review of the concepts of recommendation systems in order to contextualize the full process
in deciding which recommenders to create.

An anime recommendation system could be used in a growing anime entertainment platform such as Crunchyroll:

![crunchyroll homepage](/img/crunchyroll.png)


### Content-Based Filtering vs. Collaborative Filtering.
- ***Content-Based Filtering*** - Matching users to items they have personally liked.
- ***Collaborative Filtering*** - Making recommendations based on similar user activity.

In order to decide on which filtering to use, we must explore the available data. In the case of this project,
the data does not provide enough description on each user; therefore, we elect with **collaborative filtering** for 
anime recommendations.

The advantages of collaborative filtering is we do not require comprehensive description of user/items but make
recommendations based on user-item interactions.

The disadvantages to be wary of when using collaborative filtering are:
- Generally, recommendations will have high bias (but variance is low)
- Recommender inclined to favor popular items with lots of positive interaction
- Difficult to recommend new releases

### Memory-Based Approach vs. Model-Based Approach

#### ***Memory-Based Approach***
Find similar users or similar items using similarity methods.

  - The utility matrix (user to items matrix) is memorized
    and recommendations are made by querying the given user with the rest of the utility matrix.

KNN method will be implemented to represent the memory-based approach, finding the nearest neighbors of 
input anime in examination. 

The main advantage and disadvantage for the memory-based approach:
- **Advantage**: easy creation and explainability of results
- **Disadvantage**: performance reduces when data is sparse

#### ***Model-Based Approach***
Decomposing huge and sparse user-item interaction matrix into a product of two smaller and dense matrices 
(user-factor matrix containing user representations and factor-item matrix containing items representation).
This is also known as ***matrix factorization***.

![matrix factorization](/img/matrix_factorization.jpg)

The model-based approach will compute a rating for a user and the content by factorizing the interaction
matrix.

The main advantage and disadvantage for the model-based approach:
- **Advantage**: dimensionality reduction deals with sparse data
- **Disadvantage**: inference is not traceable because of hidden latent factors

There are various factorization techniques in the model-based approach (UVD, SVD, NMF, and ALS to name a few.)

##### Inspiration to implement ALS (Alternating Least Squares) method
- ALS is one of the more successful approaches to recommendation systems.
- ALS runs gradient descent in parallel across multiple partitions of underlying training data
from a cluster of machines making ALS scalable.
- Leverages SparkML and minimized two loss functions alternatively.

*This project will implement Collaborative Filtering Item-Based KNN Recommender and a
Collaborative Filtering Item-Based ALS Recommender.*

---
## The Process.

### KNN Method Process.
Please refer to `notebooks/knn_recommender.ipynb` notebook to view more in-depth code and expected outputs.

1) Load anime and ratings data
2) Prepare sparse matrix from for nearest neighbors model fitting.
3) Match input string with the most similar title in the database with fuzzy matching (python library)
4) Fit model with cosine similarity
   - Cosine similarity used to find orientation since anime ratings data are all in the same inner product space
5) Find k nearest neighbors and order anime based on highest similarity score

### ALS Method Process.
Please refer to `notebooks/als_recommender.ipynb` notebook to view more in-depth code and expected outputs.

1) Load anime and ratings data.
2) Tune model to find the optimal rank and regularization parameters to tune the best model.
3) Match the input anime title with most similar anime in the dataset.
4) Create a new user and append positive ratings to the title matches and append the new ratings to the
   ratings dataset.
5) Create inference data with the newly added user and the anime not matched with the input anime to predict ratings
for all the other anime.
6) Fit the model with the new ratings data and make predictions.
7) Return the titles of the highest predicted rankings for this particular user.

---
## Results.

After implementation this is a sample output of recommendations for a user who wants to find
recommendations after watching the anime "Hunter x Hunter".

### K-Nearest Neighbors

User input:
```shell
python src/knn_recommender.py --anime_title 'Hunter X' --top_n 10
```
Output:
```shell
Found possible matches in database: ['Hunter x Hunter', 'DNA Hunter', 'Thunder', 'Bio Hunter', 'City Hunter 3', 'City Hunter 2', 'City Hunter']

1: Eve no Jikan, with distance of 0.7127892208162444
2: Kamisama Hajimemashita: Kako-hen, with distance of 0.7109567951351417
3: Gintama: Jump Festa 2015 Special, with distance of 0.7082093334930141
4: Jigoku Shoujo Futakomori, with distance of 0.7011218352668558
5: City Hunter: The Secret Service, with distance of 0.7007536942494121
6: Seikai no Monshou, with distance of 0.6956590474848182
7: Yojouhan Shinwa Taikei, with distance of 0.694564311301947
8: Ghost in the Shell: Stand Alone Complex 2nd GIG, with distance of 0.6865257098308966
9: Koihime†Musou, with distance of 0.5977584997428402
10: Pokemon Diamond &amp; Pearl: Arceus Choukoku no Jikuu e, with distance of 0.41149490398141675
```

### ALS

User input:
```shell
python src/als_recommender.py --anime_title 'Hunter X' --top_n 10
```
Output:
```shell
Recommendation system start to make inference ...                               
You have input anime: Hunter X                                                  
Found possible matches in our database: ['Hunter x Hunter (2011)', 'Hunter x Hunter', 'Hunter x Hunter OVA', 'Hunter x Hunter: Greed Island Final', 'Hunter x Hunter: Greed Island', 'Irregular Hunter X: The Day of Sigma', 'Hunter x Hunter Movie: Phantom Rouge', 'Hunter x Hunter Pilot', 'Hunter x Hunter Movie: The Last Mission']

It took my system 79.92s to make inference                                      
              
Recommendations for Hunter X:
1: Gintama: Yorinuki Gintama-san on Theater 2D
2: Gintama: Shinyaku Benizakura-hen
3: Yuusha Keisatsu J-Decker
4: Time Travel Tondekeman!
5: 30-sai no Hoken Taiiku Specials
6: Mirai no Watashi
7: HORIZON feat. Hatsune Miku
8: Shokupan Mimi
9: Tarou-san no Kisha
10: Mononoke Dance
```

---
## Evaluation.

One way to evaluate or ALS recommendations is to run the `tune_ALS` method that recursively tunes our ALS model
based on rank, regularization parameter, and computes the RMSE for all models.

The following computes the RMSE for all models tuning rank and regularization ranging from 5 to 10 and 0.1 to 0.5,
respectively.

```
5 latent factors and regularization = 0.1: validation RMSE is 2.0776982112905453
5 latent factors and regularization = 0.2: validation RMSE is 2.0749591117108217
5 latent factors and regularization = 0.3: validation RMSE is 2.0961722198129173
5 latent factors and regularization = 0.4: validation RMSE is 2.1367131855578982
5 latent factors and regularization = 0.5: validation RMSE is 2.1887355460098923
6 latent factors and regularization = 0.1: validation RMSE is 2.0785174797064765
6 latent factors and regularization = 0.2: validation RMSE is 2.073323400418736
6 latent factors and regularization = 0.34: validation RMSE is 2.095198273667546
6 latent factors and regularization = 0.4: validation RMSE is 2.137606408015988
6 latent factors and regularization = 0.5: validation RMSE is 2.1891873357248675
7 latent factors and regularization = 0.1: validation RMSE is 2.0705172637251166
7 latent factors and regularization = 0.2: validation RMSE is 2.0623287078832218
7 latent factors and regularization = 0.3: validation RMSE is 2.0874786284318603
7 latent factors and regularization = 0.4: validation RMSE is 2.1334575837816785
7 latent factors and regularization = 0.5: validation RMSE is 2.188604749852555
8 latent factors and regularization = 0.1: validation RMSE is 2.0763121998353427
8 latent factors and regularization = 0.2: validation RMSE is 2.062904237292997
8 latent factors and regularization = 0.3: validation RMSE is 2.086468321991962
8 latent factors and regularization = 0.4: validation RMSE is 2.133761286144122
8 latent factors and regularization = 0.5: validation RMSE is 2.1881650657097693
9 latent factors and regularization = 0.1: validation RMSE is 2.075441733985669
9 latent factors and regularization = 0.2: validation RMSE is 2.0605128420543157
9 latent factors and regularization = 0.3: validation RMSE is 2.0851424297666963
9 latent factors and regularization = 0.4: validation RMSE is 2.134360666526805
9 latent factors and regularization = 0.5: validation RMSE is 2.188560923446981
10 latent factors and regularization = 0.1: validation RMSE is 2.0736247425759786
10 latent factors and regularization = 0.2: validation RMSE is 2.053184480026961
10 latent factors and regularization = 0.3: validation RMSE is 2.07956737980178
10 latent factors and regularization = 0.4: validation RMSE is 2.1311310426385304
10 latent factors and regularization = 0.5: validation RMSE is 2.187214563668069

The best model has 10 latent factors and regularization = 0.2
```
After tuning the model and finding the best model metric, we can set our model parameters to run our predictions.

```Python
recommender.set_model_params(maxIter=10,regParam=0.2,rank=10)
```

Ultimately, the best evaluation will require human evaluation. True evaluator of a good recommender is positive
feedback via review of recommendation itself or using A/B testing to assign a metric to good recommendations.

---
## Future Work.

1) Explore recommendations based on genre.
2) Implement A/B testing with other users to evaluate models.
3) Create a front-end tool for easy production use.

---