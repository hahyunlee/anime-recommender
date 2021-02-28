import os
import argparse
import time
import gc
import math

# spark imports
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col, lower
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS


class AlsRecommender:
    """
    Class implements alternating least squares from Spark
    """
    def __init__(self, spark_session, path_anime, path_ratings):
        self.spark = spark_session
        self.sc = spark_session.sparkContext

        self.anime_df = self._load_file(path_anime).select(['anime_id', 'name'])
        self.ratings_df = self._load_file(path_ratings).select(['user_id', 'anime_id', 'rating'])

        self.model = ALS(
            userCol='user_id',
            itemCol='anime_id',
            ratingCol='rating',
            coldStartStrategy="drop")


    def _load_file(self, filepath):
        """
        Load csv file into Spark DF
        """
        return self.spark.read.load(filepath, format='csv',
                                    header=True, inferSchema=True)


    def tune_model(self, maxIter, regParams, ranks, split_ratio=(0.6, 0.2, 0.2)):
        """
        Hyperparameter tuning for ALS model
        Parameters
        ----------
        maxIter: int, max number of learning iterations
        regParams: list of float, regularization parameter
        ranks: list of float, number of latent factors
        split_ratio: tuple, (train, validation, test)
        """
        # split data
        train, val, test = self.ratings_df.randomSplit(split_ratio)
        # holdout tuning
        self.model = tune_ALS(self.model, train, val,
                              maxIter, regParams, ranks)
        # test model
        predictions = self.model.transform(test).na.drop()
        evaluator = RegressionEvaluator(metricName="rmse",
                                        labelCol="rating",
                                        predictionCol="prediction")
        rmse = evaluator.evaluate(predictions)

        print('The out-of-sample RMSE from the best tuned model is:', rmse)

        # garbage clean up
        del train, val, test, predictions, evaluator
        gc.collect()


    def set_model_params(self, maxIter, regParam, rank):
        """
        Set model params for pyspark.ml.recommendation.ALS
        Parameters
        ----------
        maxIter: int, max number of learning iterations
        regParams: float, regularization parameter
        ranks: float, number of latent factors
        """
        self.model = ALS(userCol='user_id', itemCol='anime_id', rank=rank, maxIter=maxIter, regParam=regParam)

    def _wildcard_matching(self, fav_anime):
        """
        return the closest matches via SQL regex.
        If no match found, return None
        Parameters
        ----------
        fav_anime: str, name of user input anime
        Return
        ------
        list of indices of the matching animes
        """
        print('You have input anime:', fav_anime)
        matches_df = self.anime_df \
            .filter(
            lower(
                col('name')
            ).like('%{}%'.format(fav_anime.lower()))
        ) \
            .select('anime_id', 'name')

        if not len(matches_df.take(1)):
            print('No match is found.')
        else:
            anime_ids = matches_df.rdd.map(lambda r: r[0]).collect()
            names = matches_df.rdd.map(lambda r: r[1]).collect()
            print('Found possible matches in our database: '
                  '{0}\n'.format([x for x in names]))
            return anime_ids

    def _append_ratings(self, user_id, anime_ids):
        """
        append a user's anime ratings to ratings_df
        Parameter
        ---------
        userId: int, userId of a user
        anime_ids: int, anime_ids of user's favorite animes
        """
        # create new user rdd
        user_rdd = self.sc.parallelize(
            [(user_id, anime_id, 5.0) for anime_id in anime_ids])
        # transform to user rows
        user_rows = user_rdd.map(
            lambda x: Row(
                user_id=int(x[0]),
                anime_id=int(x[1]),
                rating=float(x[2])
            )
        )
        # transform rows to spark DF
        user_df = self.spark.createDataFrame(user_rows) \
            .select(self.ratings_df.columns)
        # append to ratings_df
        self.ratings_df = self.ratings_df.union(user_df)


    def _create_inference_data(self, user_id, anime_ids):
        """
        create a user with all animes except ones were rated for inferencing
        """
        # filter animes
        other_anime_ids = self.anime_df \
            .filter(~col('anime_id').isin(anime_ids)) \
            .select(['anime_id']) \
            .rdd.map(lambda r: r[0]) \
            .collect()
        # create inference rdd
        inference_raw = self.sc.parallelize(
            [(user_id, anime_id) for anime_id in other_anime_ids]
        ).map(
            lambda x: Row(
                user_id=int(x[0]),
                anime_id=int(x[1]),
            )
        )
        # transform to inference DF
        inference_df = self.spark.createDataFrame(inference_raw) \
            .select(['user_id', 'anime_id'])
        return inference_df


    def _make_inference(self, model, fav_anime, n_recommendations):
        """
        return top n anime recommendations based on user's input anime
        Parameters
        ----------
        model: spark ALS model
        fav_anime: str, name of user input anime
        n_recommendations: int, top n recommendations
        Return
        ------
        list of top n similar anime recommendations
        """
        # create a new user id
        user_id = self.ratings_df.agg({"user_id": "max"}).collect()[0][0] + 1

        # get anime_ids of favorite animes
        anime_ids = self._wildcard_matching(fav_anime)

        # append new user with his/her ratings into data
        self._append_ratings(user_id, anime_ids)
        # matrix factorization
        model = model.fit(self.ratings_df)
        # get data for inferencing
        inference_df = self._create_inference_data(user_id, anime_ids)
        # make inference

        results = model.transform(inference_df).select(['anime_id', 'prediction']).na.drop()

        return results \
            .orderBy('prediction', ascending=False) \
            .rdd.map(lambda r: (r[0], r[1])) \
            .take(n_recommendations)


    def make_recommendations(self, fav_anime, n_recommendations):
        """
        make top n anime recommendations
        Parameters
        ----------
        fav_anime: str, name of user input anime
        n_recommendations: int, top n recommendations
        """
        # make inference and get raw recommendations
        print('Recommendation system start to make inference ...')
        t0 = time.time()
        raw_recommends = \
            self._make_inference(self.model, fav_anime, n_recommendations)

        anime_ids = [r[0] for r in raw_recommends]
        scores = [r[1] for r in raw_recommends]

        print('It took my system {:.2f}s to make inference \n\
              '.format(time.time() - t0))
        # get anime titles
        anime_titles = self.anime_df \
            .filter(col('anime_id').isin(anime_ids)) \
            .select('name') \
            .rdd.map(lambda r: r[0]) \
            .collect()
        # print recommendations
        print('Recommendations for {}:'.format(fav_anime))
        for i in range(len(anime_titles)):
            print('{0}: {1}, with rating '
                  'of {2}'.format(i + 1, anime_titles[i], scores[i]))


def tune_ALS(train_data, validation_data, maxIter, regParams, ranks):
    """
    grid search function to select the best model based on RMSE of
    validation data
    Parameters
    ----------
    model: spark ML model, ALS
    train_data: spark DF with columns ['userId', 'anime_id', 'rating']
    validation_data: spark DF with columns ['userId', 'anime_id', 'rating']
    maxIter: int, max number of learning iterations
    regParams: list of float, one dimension of hyper-param tuning grid
    ranks: list of float, one dimension of hyper-param tuning grid
    Return
    ------
    The best fitted ALS model with lowest RMSE score on validation data
    """
    # initial
    min_error = math.inf
    best_rank = -1
    best_regularization = 0
    best_model = None

    for rank in ranks:
        for reg in regParams:
            # get ALS model
            # als = model.setMaxIter(maxIter).setRank(rank).setRegParam(reg)
            als = ALS(userCol='user_id', itemCol='anime_id', rank=rank, maxIter=maxIter, regParam=reg)
            # train ALS model
            model = als.fit(train_data)
            # evaluate the model by computing the RMSE on the validation data
            predictions = model.transform(validation_data)
            # drop na predictions
            predictions = predictions.na.drop()
            evaluator = RegressionEvaluator(metricName="rmse",
                                            labelCol="rating",
                                            predictionCol="prediction")
            rmse = evaluator.evaluate(predictions)
            print('{} latent factors and regularization = {}: validation RMSE is {}'.format(rank, reg, rmse))
            if rmse < min_error:
                min_error = rmse
                best_rank = rank
                best_regularization = reg
                best_model = model

    print('\n The best model has {} latent factors and regularization = {}'.format(best_rank, best_regularization))

    return best_model


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Anime Recommender",
        description="Run ALS Anime Recommender")
    parser.add_argument('--proj_path', nargs='?', default='../data',
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
    # initialize spark session
    spark = SparkSession \
        .builder \
        .appName("anime recommender") \
        .getOrCreate()

    # initialize recommender system
    recommender = AlsRecommender(
        spark,
        os.path.join(data_path, anime_filename),
        os.path.join(data_path, ratings_filename))

    # set params
    recommender.set_model_params(10, 0.05, 20)
    # make recommendations
    recommender.make_recommendations(anime_title, top_n)
    # stop
    spark.stop()
