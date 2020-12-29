# part 1

from pyspark.ml.image import ImageSchema
from pyspark.sql.functions import lit
img_dir = '../deep-learning-images/'

tulips_df = ImageSchema.readImages(img_dir + "/tulips").withColumn("label", lit(1)).limit(10)
daisy_df = ImageSchema.readImages(img_dir + "/daisy").withColumn("label", lit(0)).limit(10)
tulips_train, tulips_test = tulips_df.randomSplit([0.6, 0.4])
daisy_train, daisy_test = daisy_df.randomSplit([0.6, 0.4])
train_df = tulips_train.unionAll(daisy_train)
test_df = tulips_test.unionAll(daisy_test)

# part 2

from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from sparkdl import DeepImageFeaturizer
featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features", modelName="InceptionV3")
lr = LogisticRegression(maxIter=1, regParam=0.05, elasticNetParam=0.3, labelCol="label")
p = Pipeline(stages=[featurizer, lr]) 
p_model = p.fit(train_df)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
tested_df = p_model.transform(test_df)
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Test set accuracy = " + str(evaluator.evaluate(tested_df.select("prediction", "label"))))

# part 3

from pyspark.sql.functions import udf, col
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import expr
# a simple UDF to convert the value to a double
def _p1(v):
    return float(v.array[1])

p1 = udf(_p1, DoubleType())
df = tested_df.withColumn("p_1", p1(tested_df.probability))
wrong_df = df.orderBy(expr("abs(p_1 - label)"), ascending=False)
wrong_df.select("filePath", "p_1", "label").limit(10).show()

# part 4 

from sparkdl import DeepImagePredictor
predictor = DeepImagePredictor(inputCol="image", outputCol="predicted_labels", modelName="InceptionV3", decodePredictions=True, topK=10)
predictions_df = predictor.transform(tulips_df)

# part 5 (My Own Extension) 

from keras.applications import InceptionV3, MobileNet
from sparkdl.udf.keras_image_model import registerKerasImageUDF

# preprocessing
def keras_load_img(fpath):
    from keras.preprocessing.image import load_img, img_to_array
    import numpy as np
    from pyspark.sql import Row
    img = load_img(fpath, target_size=(299, 299))
    return img_to_array(img).astype(np.uint8)

train_df.createOrReplaceTempView("train_df")

k_model = InceptionV3(weights="imagenet")
m_model = MobileNet(weights="imagenet")

registerKerasImageUDF("k_model_udf", k_model, keras_load_img)
registerKerasImageUDF("m_model_udf", m_model, keras_load_img)

k_df = spark.sql("select label, k_model_udf(image) as prediction from train_df")
m_df = spark.sql("select label, m_model_udf(image) as prediction from train_df")

