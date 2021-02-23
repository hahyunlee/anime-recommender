import pandas as pd
import os
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from fuzzywuzzy import fuzz

class KnnRecommender:
    """
        Class implements k Nearest Neighbors clustering
    """
    def __init__(self, anime_path, ratings_path):
        self.anime_df = self._load_file(anime_path)
        self.ratings_df = self._load_file(ratings_path)
        # self.model = NearestNeighbors()
        self.model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)

    def _load_file(self, file_path):
        """
        Load csv file into pandas DF
        """
        return pd.read_csv(file_path)

    def _prepare_data(self):
        ratings_df = self.ratings_df.groupby(['user_id', 'anime_id']).mean().reset_index()
        user_anime_mat = ratings_df.pivot(
            index='anime_id',
            columns='user_id',
            values='rating'
        )
        user_anime_mat.fillna(0,inplace=True)
        return csr_matrix(user_anime_mat.values)

    def _create_hashmap(self):
        anime_titles=self.anime_df[['anime_id','name']]
        anime_titles=anime_titles.set_index('anime_id')

        return {
            anime: i for i, anime in
            enumerate(list(anime_titles.name))
        }

    def _fuzzy_match(self,hash_dict,fav_anime):
        match_tuple = []

        for title, idx in hash_dict.items():
            ratio = fuzz.ratio(title.lower(), fav_anime.lower())
            if ratio >= 60:
                match_tuple.append((title, idx, ratio))

        match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]

        if not match_tuple:
            print('No match found.')
        else:
            print('Found possible matches in database: {0}'.format([x[0] for x in match_tuple]))
            return match_tuple[0][1]


    def make_recommendations(self,fav_anime,n_recommendations):
        """
        return top n anime recommendations based on user's input anime

        Parameters
        ----------
        fav_anime: str, name of user input anime
        n_recommendations: int, top n recommendations

        Return
        ------
        list of top n similar anime recommendations
        """

        hash_dict = self._create_hashmap()
        idx = self._fuzzy_match(hash_dict,fav_anime)
        sparse_mat = self._prepare_data()

        model = self.model.fit(sparse_mat)
        distance,indices = model.kneighbors(sparse_mat[idx],n_neighbors=n_recommendations+1)

        raw_recommends = sorted(
            list(
                zip(indices.squeeze().tolist(), distance.squeeze().tolist())
            ),
            key=lambda x: x[1])[:0:-1]

        reverse_hashmap = {v: k for k, v in hash_dict.items()}

        for i, (idx, dist) in enumerate(raw_recommends):
            print('{0}: {1}, with distance '
                  'of {2}'.format(i + 1, reverse_hashmap[idx], dist))


if __name__ == '__main__':
    # get args
    # args = parse_args()
    # data_path = args.proj_path
    # anime_filename = args.anime_filename
    # ratings_filename = args.ratings_filename
    # anime_title = args.anime_title
    # top_n = args.top_n
    data_path = '../data'
    anime_filename = 'anime.csv'
    ratings_filename = 'rating.csv'

    # initial recommender system
    recommender = KnnRecommender(
        os.path.join(data_path, anime_filename),
        os.path.join(data_path, ratings_filename))

    anime = 'Hunter X'
    top_n = 10
    recommender.make_recommendations(anime,top_n)








    # make recommendations
    # recommender.make_recommendations(anime_title, top_n)