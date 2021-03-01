# Anime Recommender

![anime image](/img/anime_home.png)

## Background.
Anime refers to Japanese animation, which for decades has been produced by and for Japanese audience, but is now
a growing form of entertainment across the globe. Anime can be in the form of films, TV shows, and books (aka manga).

The goal of this project is to explore concepts of recommendation systems while providing users, cultured or new,
a tool to find new anime to enjoy. This writeup will highlight the process of creating two recommendation systems: 
- K Nearest Neighbors
- Alternating Least Squares

Anime dataset was provided by:

This project implements KNN and ALS recommendation systems and this is a snapshot of what the project accomplishes with
anime title input from the user that 

Coding process and implementation referenced from Kevin Liao's blog post on Movie Recommendations.

### K Nearest Neighbors

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

---
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
# ENTER RECOMMENDATION SYSTEM NOTES HERE TALKING ABOUT KNN AND ALS.

---

## Exploring the Data




Conclusion: KNN and ALS for collaborative filtering.


---

## The Process
### KNN Method Process
1) Load anime and ratings data
2) Prepare sparse matrix from for nearest neighbors model fitting.
3) Match input string with the most similar title in the database with fuzzy matching (python library)
4) Fit model with cosine similarity
5) Find k nearest neighbors and order anime based on highest similarity score


### ALS Method Process
1) Load anime and ratings data.
2) If necessary, tune model to find the optimal rank and regularization parameters to tune the best model.
3) Based on the user input's anime title, this method will match the input with most similar anime based on title.
4) This method will create a new user with a blank slate and append positive ratings to the title matches and
append the ratings to the main ratings data.
5) Create inference data with the newly added user and the anime not matched with the input anime to predict ratings
for all the other anime.
6) Fit the model with the new ratings data and make predictions.
7) Return the titles of the highest predicted rankings for this particular user.



---

## Evaluation

tuning model and returning best rmse

ultimately for recommendation based work only real evaluation comes from user sentiment or a/b testing to determine
if recommendations are actually working with positive feedback

---


## Future Work

