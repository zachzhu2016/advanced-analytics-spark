from itertools import combinations
import xml.etree.ElementTree as ET
import hashlib
import math
import pyspark
from pyspark import SparkContext
from pyspark.sql import Row, SparkSession, DataFrame
from pyspark.sql.types import * 
from pyspark.sql.functions import coalesce, col, lit, sum, when 
from graphframes import * 
from graphframes.lib import Pregel

def parse_xml(xmlstring): 
    root = ET.fromstring(xmlstring)
    return root.getchildren()

def parse_topics(citation):
    topics = [descriptor.text for descriptor in citation.findall('./MeshHeadingList/MeshHeading/DescriptorName[@MajorTopicYN="Y"]')]
    return topics

def hash_topic(topic):
    return hashlib.md5(topic.encode()).hexdigest()[:8]

def chisquare(YY, YB, YA, T):
    NB, NA = T - YB, T - YA
    YN, NY = YA - YY, YB - YY
    NN = T - NY - YN -YY
    inner = abs(YY * NN - YN * NY) - T / 2.0
    return T * math.pow(inner, 2) / (YA * NA * YB * NB)

def merge_maps(m1, m2):
    keys = list(m1.keys()) + list(m2.keys())
    return {k: min(m1.get(k, float('inf')), m2.get(k, float('inf'))) for k in keys}

def update(vid, state, msg):
    return merge_maps(state, msg)

def check_increment(a, b, bid):
    aplus = {k: v + 1 for k, v in a.items()}
    if b != merge_maps(a, b):
        return iter((bid, aplus))
    else:
        return iter()

def iterate(edge):
    pass

if __name__ == '__main__':
    #sc = SparkContext("local[*]", "kmeans")
    #spark = SparkSession(sc)
    #spark.sparkContext.setLogLevel("ERROR")
    SparkContext.setSystemProperty('spark.executor.memory', '8g')

    
    raw_citationset = sc.wholeTextFiles("../../course/AAS_CH7/medline_data/medsamp2016a.xml.gz")
    citations = raw_citationset.flatMap(lambda t: parse_xml(t[1]))
    topics = citations.map(parse_topics)
    topics.cache()

    
    topics_df = topics.flatMap(lambda topic: topic).map(lambda topic: Row(topic)).toDF("topic:string")
    topics_df.cache()
    topics_df.createOrReplaceTempView("topics")

    
    topic_dist_df = spark.sql("select topic, count(*) cnt from topics group by topic order by cnt desc")
    topic_dist_df.createOrReplaceTempView("topic_dist")
    spark.sql("select cnt, count(*) dist from topic_dist group by cnt order by dist desc")

    
    topic_pairs_df = topics.flatMap(lambda topic: combinations(sorted(topic), 2)).map(lambda pair: Row(*pair)).toDF(["topic_a", "topic_b"])
    topic_pairs_df.createOrReplaceTempView("topic_pairs")
    cooccurs = spark.sql("select topic_a, topic_b, count(topic_a, topic_b) cnt from topic_pairs group by topic_a, topic_b")

    
    vertices = topics_df.rdd.map(lambda row: (hash_topic(row.topic), row.topic)).toDF(["id", "topic"])
    edges = cooccurs.rdd.map(lambda row: (*sorted((hash_topic(row.topic_a), hash_topic(row.topic_b))), row.cnt)).toDF(["src", "dst", "cnt"])

    
    vertices.createOrReplaceTempView("vertices")
    edges.createOrReplaceTempView("edges")
    spark.sql("select count(distinct topic) unique_topics from vertices")
    spark.sql("select count(distinct id) unique_hashids from vertices")

    
    g = GraphFrame(vertices.drop_duplicates(), edges)
    g.cache()

    
    sc.setCheckpointDir("/tmp/graphframes-connected-components")
    components = None
    strong_components = None

    
    comp_schema = StructType([StructField("id", StringType(), True),\
                         StructField("topic", StringType(), True),\
                         StructField("component", StringType(), True)])
    if not os.path.exists("./dfs/components"):
        components = g.connectedComponents() 
        components.write.csv("./dfs/components")
    else:
        components = spark.read.schema(comp_schema).csv("./dfs/components")

    
    if not os.path.exists("./dfs/strong_components"):
        strong_components = g.strongConnectedComponents(maxIter=10) 
        strong_components.write.csv("./dfs/strong_components")
    else:
        strong_components = spark.read.schema(comp_schema).csv("./dfs/strong_components")

    
    components.createOrReplaceTempView("components")
    spark.sql("select count(distinct component) num_components from components")
    spark.sql("select component, count(*) cnt from components group by component order by cnt desc")

    
    degrees = g.degrees.cache()
    g.vertices.join(degrees, ['id'], how='inner').sort(['degree'], ascending=False)

    
    T = citations.count()
    topic_count_vertices = spark.sql("select id, count(*) cnt from vertices group by id order by cnt desc")
    topic_count_graph = GraphFrame(topic_count_vertices, g.edges)
    chisquare_edges = topic_count_graph.triplets.rdd.map(lambda row: (row.src[0], row.dst[0], chisquare(row.src[1], row.edge[2], row.dst[1], T))).toDF(["src", "dst", "chisquare"])
    chisquare_graph = GraphFrame(topic_count_vertices, chisquare_edges)


    interesting = GraphFrame(topic_count_vertices, chisquare_edges.filter("chisquare > 19.5"))
    interesting.edges.count()
    if not os.path.exists("./dfs/interesting_components"):
        interesting_components = g.connectedComponents() 
        interesting_components.write.csv("./dfs/interesting_components")
    else:
        interesting_components = spark.read.csv("./dfs/interesting_components")
    interesting_components.createOrReplaceTempView("interesting_components")
    spark.sql("select component, count(*) num_components from interesting_components group by component order by num_components desc")
    interesting_degrees = interesting.degrees.cache()
    g.vertices.join(interesting_degrees, ['id'], how='inner').sort(['degree'], ascending=False)


    triangles = g.triangleCount()\
                 .select(['id', 'topic', col('count').alias('triangles')]).cache()
    max_triangles = degrees.withColumn("max_triangles", degrees.degree * (degrees.degree - 1) / 2).cache()
    cluster_cofs = triangles.join(max_triangles, on=['id'], how='inner')\
                            .withColumn('ratio', triangles.triangles / max_triangles.max_triangles)
    cluster_cofs.agg({'ratio': 'avg'})

    
    fraction = 0.005
    replacement = False
    sampled_ids = interesting.vertices.rdd.map(lambda v: v.id).sample(replacement, fraction, 1).collect()
    map_vertices_rdd = interesting.vertices.rdd.map(lambda v: (v.id, {v.id: 0}) if v.id in sampled_ids else (v.id, {}))
    map_vertices_schema = StructType([StructField("id", StringType(), True),StructField("map", MapType(StringType(), IntegerType()), True)])
    map_vertices = map_vertices_rdd.toDF(map_vertices_schema)
    map_graph = GraphFrame(map_vertices, g.edges)

    """
    alpha = 0.15 
    ranks = map_graph.pregel \ 
         .setMaxIter(5) \ 
         .withVertexColumn("rank", lit(1.0 / map_graph.vertices.count()), \ 
             coalesce(Pregel.msg(), lit(0.0)) * lit(1.0 - alpha) + lit(alpha / map_graph.vertices.count())) \ 
         .sendMsgToDst(Pregel.src("rank") / Pregel.src("outDegree")) \ 
         .aggMsgs(sum(Pregel.msg())) \ 
         .run()         
    """
