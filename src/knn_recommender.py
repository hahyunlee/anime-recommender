import pandas as pd
import os
import argparse
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
        self.model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)

    def _load_file(self, file_path):
        """
        Load csv file into pandas DF
        """
        return pd.read_csv(file_path)

    def _prepare_data(self):
        """
        Create csr matrix with ratings data
        """
        ratings_df = self.ratings_df.groupby(['user_id', 'anime_id']).mean().reset_index()
        user_anime_mat = ratings_df.pivot(
            index='anime_id',
            columns='user_id',
            values='rating'
        )
        user_anime_mat.fillna(0,inplace=True)
        return csr_matrix(user_anime_mat.values)

    def _create_hashmap(self):
        """
        Match anime ids to their respective string titles
        """
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
            # tunable ratio for match strength
            if ratio >= 60:
                match_tuple.append((title, idx, ratio))

        match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]

        if not match_tuple:
            print('No match found for anime title: {0}'.format(fav_anime))
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
        # return best match for input anime title
        idx = self._fuzzy_match(hash_dict,fav_anime)
        sparse_mat = self._prepare_data()

        model = self.model.fit(sparse_mat)

        # finds k nearest neighors from anime title from sparse matrix
        # sparse_mat[idx] are all ratings from users for that specific title
        # returns distances for k nearest and the indices (similar anime)
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

def parse_args():
    parser = argparse.ArgumentParser(
        prog="Anime Recommender",
        description="Run KNN Anime Recommender")
    parser.add_argument('--proj_path', nargs='?', default='../recommend-anime/data',
                        help='input data path')
    parser.add_argument('--anime_filename', nargs='?', default='anime.csv',
                        help='provide anime filename')
    parser.add_argument('--ratings_filename', nargs='?', default='rating.csv',
                        help='provide ratings filename')
    parser.add_argument('--anime_title', nargs='?', default='',
                        help='provide your favorite anime title')
    parser.add_argument('--top_n', type=int, default=10,
                        help='top n anime recommendations')
    return parser.parse_args()

if __name__ == '__main__':
    # get args
    args = parse_args()
    data_path = args.proj_path
    anime_filename = args.anime_filename
    ratings_filename = args.ratings_filename
    anime_title = args.anime_title
    top_n = args.top_n

    recommender = KnnRecommender(
        os.path.join(data_path, anime_filename),
        os.path.join(data_path, ratings_filename))

    recommender.make_recommendations(anime_title,top_n)
