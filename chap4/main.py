import os, random 
from operator import itemgetter
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors as millibVectors
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.tree import RandomForest, DecisionTree

from pyspark.ml.linalg import Vectors as mlVectors
from pyspark.ml.classification import *
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler


DATA_FOLDER= "../../course/AAS_CH4/"

def simple_decision_tree(train_data, cv_data):
    model = DecisionTree.trainClassifier(train_data, numClasses=7, categoricalFeaturesInfo={}, impurity="gini", maxDepth=4, maxBins=100)
    metrics = get_metrics(model, cv_data)

    print("metrics.confusionMatrix()")
    print(metrics.confusionMatrix())
    print("metrics.precision()")
    print(metrics.precision())

    for category in range(0, 7):
        print((metrics.precision(category), metrics.recall(category)))

def get_metrics(model, data):
    labels = data.map(lambda d: d.label)
    features = data.map(lambda d: d.features)
    predictions = model.predict(features)
    predictions_and_labels = predictions.zip(labels)
    return MulticlassMetrics(predictions_and_labels)

def random_classifier(train_data, cv_data):
    train_prior_probabilities = class_probabilities(train_data)
    cv_prior_probabilities = class_probabilities(cv_data)
    zip_probabilities = zip(train_prior_probabilities, cv_prior_probabilities)
    product_probabilities = map(lambda x: x[0]*x[1], zip_probabilities)
    accuracy = sum(product_probabilities)
    print("accuracy")
    print(accuracy)

def class_probabilities(data):
    counts_by_category = data.map(lambda x: x.label).countByValue().items()
    count_sorted = sorted(counts_by_category, key=itemgetter(0), reverse=False)
    count = map(lambda x: x[1], count_sorted)
    return map(lambda x: float(x)/sum(count), count)

def evaluate(train_data, cv_data, test_data):
    evaluations = []
    for impurity in ["gini", "entropy"]:
        for depth in [1, 20]:
            for bins in [10, 300]:
                model = DecisionTree.trainClassifier(train_data, numClasses=7, categoricalFeaturesInfo={}, impurity=impurity, maxDepth=depth, maxBins=bins)
                accuracy = get_metrics(model, cv_data).precision()
                evaluations.append(((impurity, depth, bins), accuracy))

    sorted(evaluations, key=itemgetter(1), reverse=True)
    for val in evaluations:
        print(val)

    model = DecisionTree.trainClassifier(train_data.union(cv_data),numClasses=7, categoricalFeaturesInfo={}, impurity="entropy", maxDepth=20, maxBins=300)
    print("get_metrics(model, test_data).precision()")
    print(get_metrics(model, test_data).precision())
    print("get_metrics(model, train_data.union(cv_data)).precision()")
    print(get_metrics(model, train_data.union(cv_data)).precision())

def evaluate_categorical(rawdata):
    data = rawdata.map(decode_onehot)
    train_data, cv_data, test_data = data.randomSplit(weights=[0.8, 0.1, 0.1])

    train_data.cache()
    cv_data.cache()
    test_data.cache()

    evaluations = []
    for impurity in ["gini", "entropy"]:
        for depth in [10, 20, 30]:
            for bins in [40, 300]:
                model = DecisionTree.trainClassifier(train_data, numClasses=7, categoricalFeaturesInfo={10:4, 11: 40}, impurity=impurity, maxDepth=depth, maxBins=bins)
                train_accuracy = get_metrics(model, train_data).precision()
                cv_accuracy = get_metrics(model, cv_data).precision()
                evaluations.append(((impurity, depth, bins), (train_accuracy, cv_accuracy)))

    sorted(evaluations, key=itemgetter(1,1), reverse=True)
    for val in evaluations:
        print(val)

    model = DecisionTree.trainClassifier(train_data, numClasses=7, categoricalFeaturesInfo={10: 4, 11: 40}, impurity="entropy", maxDepth=30, maxBins=300)
    print(get_metrics(model, test_data).precision())

    train_data.unpersist()
    cv_data.unpersist()
    test_data.unpersist()

def evaluate_forest(rawdata):
    data = rawdata.map(decode_onehot)
    train_data, cv_data = data.randomSplit(weights=[0.9, 0.1])

    train_data.cache()
    cv_data.cache()

    forest = RandomForest.trainClassifier(train_data, numClasses=7, categoricalFeaturesInfo={10: 4, 11: 40}, numTrees=20, featureSubsetStrategy="auto", impurity="entropy", maxDepth=30, maxBins=300)

    metrics = get_metrics(forest, cv_data)

    print(metrics.precision())

    input = "2709,125,28,67,23,3224,253,207,61,6094,0,29"
    vector = Vectors.dense(list(map(lambda x: float(x), input.split(","))))
    print(forest.predict(vector))

    train_data.unpersist()
    cv_data.unpersist()

def decode_onehot(line):
    values = list(map(lambda x: float(x), line.split(",")))
    wilderness = float(values[10:14].index(1.0))
    soil = float(values[14:54].index(1.0))
    feature_vec = Vectors.dense(values[0:10] + [wilderness] + [soil])
    label = values.pop() - 1
    return LabeledPoint(label, feature_vec)

def preprocess(line):
    values = list(map(lambda x: float(x), line.split(",")))
    last_el = values.pop()
    feature_vec = Vectors.dense(values)
    label = last_el - 1
    return LabeledPoint(label, feature_vec)
    
def decision_tree():
    raw_covtype = sc.textFile(f'{DATA_FOLDER}/covtype.data')
    data = raw_covtype.map(preprocess)

    train_data, cv_data, test_data = data.randomSplit(weights=[0.8, 0.1, 0.1])
    train_data.cache()
    cv_data.cache()
    test_data.cache()

    simple_decision_tree(train_data, cv_data)
    random_classifier(train_data, cv_data)
    evaluate(train_data, cv_data, test_data)
    evaluate_categorical(raw_covtype)
    evaluate_forest(raw_covtype)

def logistic_regression():
    data = spark.read.format("csv").load(f'{DATA_FOLDER}/covtype.data', inferSchema=True)
    data = spark.createDataFrame([(features, label) for features, label in\
                                  zip([mlVectors.dense([int(x) for x in row]) for row in data.select(data.columns[:-1]).collect()],\
                                      [row[0] for row in data.select(data.columns[-1]).collect()])\
                                 ],\
                                 ["features", "label"])
    train_data, test_data = data.randomSplit(weights=[0.6, 0.4])
    lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
    lr_model = lr.fit(train_data)
    print("Coefficients: \n" + str(lr_model.coefficientMatrix))
    print("Intercept: " + str(lr_model.interceptVector))

if __name__ == '__main__':
    conf = SparkConf()
    conf.set("spark.driver.memory", "6g")
    conf.set("spark.executer.memory", "6g")
    conf.setMaster("local[*]")
    sc = SparkContext(appName="pfc", conf=conf)
    spark = SparkSession(sc)

    """ Decision Tree """
    #decision_tree()

    """ Logistic Regression """ 
    #logistic_regression()
    
