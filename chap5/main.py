import os
from pyspark import SparkContext
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col
from pyspark.streaming import StreamingContext
from pyspark.ml.feature import HashingTF, Tokenizer, FeatureHasher
from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.linalg import DenseVector

sc = SparkContext("local[*]", "kmeans")
spark = SparkSession(sc)
spark.sparkContext.setLogLevel("ERROR")

def get_model():
    data = spark.read.csv("AAS_CH5/kddcup.data")
    hasher = FeatureHasher(inputCols=data.columns[:-1], outputCol='features')
    featurized = hasher.transform(data)
    traindata = featurized.select(col("_c41").alias("label"), col("features"))
    
    kmeans = KMeans().setK(150).setSeed(1)
    model = None
    if os.path.exists("./kmeans.model"):
        model = KMeansModel.load("./kmeans.model")
    else: 
        model = kmeans.fit(traindata)

    return model

if __name__ == '__main__':
    model = get_model()
    ssc = StreamingContext(sc, 1)
    testschema = spark.read.csv("AAS_CH5/kddcup.testdata").schema
    
    lines = spark.readStream \
            .format("socket") \
            .option("host", "localhost") \
            .option("port", 9999) \
            .load()

    hasher = FeatureHasher(inputCols=lines.columns, outputCol='features')
    featurized = hasher.transform(lines)

    query = featurized.writeStream \
            .format("console") \
            .start()
    
    ssc.awaitTermination() 

