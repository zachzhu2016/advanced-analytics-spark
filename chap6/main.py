import math
import json
import newspaper
import stanza
import string
from pyspark import SparkContext
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import Row
from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, IDF, CountVectorizer, StopWordsRemover
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import RowMatrix

sc = SparkContext("local[*]", "LSA")
spark = SparkSession(sc)
spark.sparkContext.setLogLevel("ERROR")

DATA_PATH = 'articles.json'

def download():
    """
    download about 1000 CNN news articles, save it as 'articles.json'
    """
    articles = []
    cnn_paper = newspaper.build('http://cnn.com', language='en',  memoize_articles=False)
    for article in cnn_paper.articles: 
        try:
            article.download()
            article.parse()
            articles.append({'title': article.title, 'text': article.text})
        except:
            continue 
    # save articles 
    with open(DATA_PATH, 'w') as f:
        json.dump(articles, f, indent=4)
    
def tfidf_matrix():
    """
    compute tfidf matrix based on 'articles.json'
    """

    # reads 50 articles for testing purposes
    df = spark.read.option("multiline", "true").json(DATA_PATH).limit(50)
    df.cache()

    # initialize Stanford Stanza NLP pipeline that lemmatizes our texts
    nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma')

    def lemmatize(row):
        """
        closure lemmatize function 
        """
        lemmatized = {'title': row['title'], 'words': []}
        try:
            unlemmatized = nlp(row['text'])
        except:
            return lemmatized
        # remove punctuations as well 
        lemmatized['words'] = [word.lemma for sentence in unlemmatized.sentences for word in sentence.words if word.lemma not in string.punctuation]
        return lemmatized

    df = df.rdd.map(lambda row: Row(**lemmatize(row))).toDF()
    # mapping to be used to get document titles with index
    doc_map = {i: str(row.title) for i, row in enumerate(df.select(['title']).collect())}

    # remove stopwords 
    remover = StopWordsRemover(inputCol="words", outputCol="filteredWords")
    df = remover.transform(df)

    # Count the frequencies of words with stopwords removed
    # HashingTF is an alternative but it does not support reverse mapping 
    cv = CountVectorizer(inputCol="filteredWords", outputCol="rawFeatures", vocabSize=20).fit(df)
    df = cv.transform(df)

    # mapping to be used to get word with index
    term_map = dict(enumerate(cv.vocabulary))

    # inverse document frequency for improving the quality of word weights
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    df = idf.fit(df).transform(df)
    df.cache()

    # TF-IDF Matrix
    matrix = RowMatrix(df.select(['features']).rdd.map(lambda v: Vectors.dense(v.features.toArray())))

    return matrix, doc_map, term_map

def topterms_in_topconcepts(svd, num_concepts, num_terms):
    """
    rank the top terms in each concept
    """
    topterms = []
    matrix = svd.V.toArray()
    for i in range(num_concepts):
        offset = i * svd.V.numRows
        # weights of terms under concept i
        term_weights = list(enumerate([row[i] for row in matrix]))
        term_weights.sort(key=lambda t: t[1], reverse=True)
        # add top terms under concept i to topterms
        topterms.extend([term_map[t[0]] for t in term_weights[:num_terms]])
    return topterms 

def topdocs_in_topconcepts(svd, num_concepts, num_docs):
    """
    rank the top documents in each concept
    """
    topdocs = []
    for i in range(num_concepts):
        # weights of documents under concept i
        doc_weights = svd.U.rows.map(lambda row: row.toArray()[i]).zipWithUniqueId()
        # add top terms under concept i to topterms
        topdocs.extend([doc_map[t[1]] for t in doc_weights.top(num_docs)])
    return topdocs

class LSAQueryEngine: 
    def normalize_rows(matrix: RowMatrix): 
        def normalize_vec(vec):
            array = vec.toArray()
            length = math.sqrt(sum(array.map(lambda x: x * x)))
            return Vectors.dense(array.map(lambda x: x / length))
        return RowMatrix(matrix.rows.map(lambda vec: normalize_vec(vec)))

matrix, doc_map, term_map = tfidf_matrix()
"""
num_concepts = 4
svd = matrix.computeSVD(num_concepts, computeU=True)
topterms = topterms_in_topconcepts(svd, num_concepts, 10)
topterms = [topterms[i:i + num_concepts] for i in range(0, len(topterms), num_concepts)]
topdocs = topdocs_in_topconcepts(svd, num_concepts, 10)
topdocs = [topdocs[i:i + num_concepts] for i in range(0, len(topdocs), num_concepts)]
for terms, docs in zip(topterms, topdocs):
    terms_str = ', '.join(terms)
    docs_str = ', '.join(docs)
    print(f'Concept terms: {terms_str}') 
    print(f'Concept docs: {docs_str}') 
    print()
"""

