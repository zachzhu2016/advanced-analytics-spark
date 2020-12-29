import os, random
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, DataFrame, SparkSession 
from pyspark.mllib.recommendation import ALS, Rating
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

DATA_FOLDER= "../../course/AAS_CH3/profiledata_06-May-2005"

def build_ratings(user_artist_data, artist_alias_broadcast):
    def build_artist_rating(line):
        (userid, artistid, count) = map(lambda x: int(x), line.split(' '))
        try:
            existing_artist_id = artist_alias_broadcast.value[artistid]
        except KeyError:
            existing_artist_id = artistid
        return Rating(userid, existing_artist_id, count)

    return user_artist_data.map(lambda line: build_artist_rating(line))

def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def build_artist_rdd(artist_data):
    return artist_data \
           .map(lambda x: x.split("\t", 1)) \
           .filter(lambda artist: artist[0] and is_int(artist[0])) \
           .map(lambda artist: (int(artist[0]), artist[1].strip()))


def build_artist_alias_dict(artist_alias_data):
    return artist_alias_data \
           .map(lambda line: line.split('\t')) \
           .filter(lambda artist: artist[0] and is_int(artist[0])) \
           .map(lambda artist: (int(artist[0]), int(artist[1]))) \
           .collectAsMap()

def build_user_artist_rdd(user_artist_data):
    return user_artist_data \
           .map(lambda x: x.split(' '))

def model(ratings, user_artist_rdd):
    model = ALS.trainImplicit(ratings=ratings, rank=10, iterations=5, lambda_=0.01, alpha=1.0)
    ratings.unpersist()

    sample_userid = 2093760
    recommendations = model.recommendProducts(sample_userid, 5)
    recommended_product_ids = map(lambda rec: rec.product, recommendations)
    existing_products = user_artist_rdd \
                        .filter(lambda x: int(x[0]) == sample_userid) \
                        .map(lambda x: int(x[1])).collect()

    existing_artists = artist_rdd.filter(lambda artist: artist[0] in existing_products).collect()
    recommended_artists = artist_rdd.filter(lambda artist: artist[0] in recommended_product_ids).collect()

def area_under_curve(positive_data, product_ids_broadcast, predict_function):
    def build_negative_group(positive_group):
        product_ids = product_ids_broadcast.value
        userid = positive_group[0]
        positive_products = set(positive_group[1])
        negative_products = []
        i = 0
        while i < len(product_ids) and len(negative_products) < len(positive_products):
            random_i = random.randrange(0, len(product_ids))
            product_id = product_ids[random_i]
            if product_id not in positive_products:
                negative_products.append(product_id)
            i += 1
        return (userid, negative_products)

    positive_products = positive_data.map(lambda x: (x.user, x.product))
    grouped_positive_products = positive_products.groupByKey()
    grouped_negative_products = grouped_positive_products.map(lambda positive_group: build_negative_group(positive_group))
    negative_products = grouped_negative_products.flatMapValues(lambda negative_group: negative_group);

    positive_predictions = predict_function(positive_products).groupBy(lambda x: x.user)
    negative_predictions = predict_function(negative_products).groupBy(lambda x: x.user)
    joined_ratings =  positive_predictions.join(negative_predictions).values()

    def probability_of_true_positive(ratings):
        positive_ratings = ratings[0]
        negative_ratings = ratings[1]
        correct = float(0)
        total = float(0)
        for positive in positive_ratings:
            for negative in negative_ratings:
                if positive.rating > negative.rating:
                    correct += 1
                total += 1
        return correct / total

    return joined_ratings.map(probability_of_true_positive).mean()


def predict_most_listened(sc, train_data):
    def predict_function(all_data):
        listen_count = train_data.map(lambda r: (r.product, r.rating)).reduceByKey(lambda a, b: a + b).collectAsMap()
        listen_count_broadcast = sc.broadcast(listen_count)
        return all_data.map(lambda data: Rating(data[0], data[1], listen_count_broadcast.value.get(data[1], 0.0)))

    return predict_function

def evaluate_recommender(sc, user_artist_rdd, artist_alias_broadcast, ratings):
    most_listened_aucs = []
    k_fold_data = ratings.randomSplit(weights=[0.1]*10)
    for k, cv_data in enumerate(k_fold_data):
        train_data = sc.union(list(filter(lambda rdd: rdd != cv_data, k_fold_data)))
        cv_data.cache()
        train_data.cache()

        product_ids = ratings.map(lambda rating: rating.product).distinct().collect()
        product_ids_broadcast = sc.broadcast(product_ids)
        predict_function = predict_most_listened(sc, train_data)
        most_listened_auc = area_under_curve(cv_data, product_ids_broadcast, predict_function)
        most_listened_aucs.append(most_listened_auc)

        print(f'k: {k}')
        print(f'Most Listened AUC: {most_listened_auc}')
        print('--------------')
        
        train_data.unpersist()
        cv_data.unpersist()
        
        """
        Example from AAS

        evaluations = []
        for rank in [10, 50]:
            for lambda_val in [1.0, 0.001]:
                for alpha in [1.0, 40.0]:
                    model = ALS.trainImplicit(ratings=train_data, rank=rank, iterations=10, lambda_=lambda_val, alpha=alpha)
                    auc = area_under_curve(cv_data, product_ids_broadcast, model.predict)
                    unpersist(model)
                    evaluations.append(((rank, lambda_val, alpha), auc))

        sorted(evaluations, key=itemgetter(1), reverse=True)
        print(evaluations)
        """
    
    most_listened_auc_avg = sum(most_listened_aucs) / len(most_listened_aucs)

if __name__ == '__main__':
    conf = SparkConf()
    conf.set("spark.driver.memory", "6g")
    conf.set("spark.executer.memory", "6g")
    conf.setMaster("local[*]")
    sc = SparkContext(appName="recommender", conf=conf)
    spark = SparkSession(sc)

    user_artist_schema = StructType([ \
                         StructField("userid", IntegerType(), True), \
                         StructField("artistid", IntegerType(), True), \
                         StructField("playcount", IntegerType(), True)]) 

    artist_schema = StructType([ \
                         StructField("artistid", IntegerType(), True), \
                         StructField("artist_name", StringType(), True)])

    artist_alias_schema = StructType([ \
                         StructField("badid", IntegerType(), True), \
                         StructField("goodid", IntegerType(), True)])
    
    user_artist_data = sc.textFile(f'{DATA_FOLDER}/user_artist_data.txt')
    artist_data = sc.textFile(f'{DATA_FOLDER}/artist_data.txt')
    artist_alias_data = sc.textFile(f'{DATA_FOLDER}/artist_alias.txt')
    # user_artist_df = spark.read.csv(f'{DATA_FOLDER}/user_artist_data.txt', header=False, schema=user_artist_schema, sep=' ').cache()
    # artist_df = spark.read.csv(f'{DATA_FOLDER}/artist_data.txt', header=False, schema=artist_schema, sep='\t').cache()
    # artist_alias_df = spark.read.csv(f'{DATA_FOLDER}/artist_alias.txt', header=False, schema=artist_alias_schema, sep='\t').cache()

    user_artist_rdd = build_user_artist_rdd(user_artist_data)
    artist_rdd = build_artist_rdd(artist_data).cache()
    artist_alias_dict = build_artist_alias_dict(artist_alias_data)
    artist_alias_broadcast = sc.broadcast(artist_alias_dict)
    ratings = build_ratings(user_artist_data, artist_alias_broadcast).cache()

    # model(ratings, user_artist_rdd)
    # evaluate_recommender(sc, user_artist_rdd, artist_alias_broadcast, ratings)




