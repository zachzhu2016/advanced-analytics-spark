import pyspark
from pyspark import SparkContext, SparkConf
from pyspark.sql import Row, SparkSession, DataFrame
from pyspark.sql.types import *
import os, json
from itertools import islice
from datetime import datetime
from arcgis.geometry import Point, Polyline, Polygon, Geometry, GeometryFactory, SpatialReference

# JARFILE = "./ch08-geotime-2.0.0-jar-with-dependencies.jar"
GEOJSON = "./nyc-boroughs.geojson"
TRIPDATA = "./trip_data_1.csv"
WKID = 4326

def window(it, n=2):
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def parserow(row):
    license = row[1]
    pickup_datetime, dropoff_datetime = None, None
    pickup_loc, dropoff_loc = None, None

    parsefailed = False
    try:
        pickup_datetime = datetime.strptime(row[5], '%Y-%m-%d %H:%M:%S')
        dropoff_datetime = datetime.strptime(row[6], '%Y-%m-%d %H:%M:%S')
    except:
        pickup_datetime = datetime(1970, 1, 1)
        dropoff_datetime = datetime(1970, 1, 1)
        parsefailed = True

    try:
        #pickup_loc = Point({'x': float(row[10]), 'y': float(row[11]), 'spatialReference': {'wkid': WKID}})
        #dropoff_loc = Point({'x': float(row[12]), 'y': float(row[13]), 'spatialReference': {'wkid': WKID}})
        pickup_loc = {'x': float(row[10]), 'y': float(row[11])}
        dropoff_loc = {'x': float(row[12]), 'y': float(row[13])}
    except:
        #pickup_loc = Point({'x': 0.0, 'y': 0.0, 'spatialReference': {'wkid': WKID}})
        #dropoff_loc = Point({'x': 0.0, 'y': 0.0, 'spatialReference': {'wkid': WKID}})
        pickup_loc = {'x': 0.0, 'y': 0.0}
        dropoff_loc = {'x': 0.0, 'y': 0.0}
        parsefailed = True

    trip = {'license': license,
            'pickup_datetime': pickup_datetime,
            'dropoff_datetime': dropoff_datetime,
            'pickup_loc': pickup_loc,
            'dropoff_loc': dropoff_loc,
            'parsefailed': parsefailed}

    return Row(**trip)

def parsegeo(filepath):
    with open(filepath) as fp:
        geojson = json.load(fp)
        for feature in geojson['features']:
            geometry = Polygon(feature['geometry'])
            feature['geometry'] = geometry
        return geojson

def hours(row):
    interval = row.dropoff_datetime - row.pickup_datetime
    mins = interval.total_seconds() / 60.0
    hours = mins / 60.0
    return Row(**row.asDict(), duration_mins=mins, duration_hours=hours)

def borough(row):
    for feature in sorted_features:
        dropoff_point = Point({**row.dropoff_loc, 'spatialReference': {'wkid': WKID}})
        if feature['geometry'].contains(dropoff_point):
            return Row(**row.asDict(), borough=feature['properties']['borough'])
    return Row(**row.asDict(), borough="NA")
    
def borough_duration(trip1, trip2):
    duration = trip2.pickup_datetime - trip1.dropoff_datetime
    duration_mins = duration.total_seconds() / 60.0
    return (trip1.borough, duration_mins)

def split_shifts(trip1, trip2):
    duration = trip2.pickup_datetime - trip1.dropoff_datetime
    duration_hours = duration.total_seconds() / 60.0 / 60.0
    return duration_hours >= 4

SparkContext.setSystemProperty('spark.executor.memory', '8g')
sr = SpatialReference(WKID)

# preprocessing
taxi_raw = spark.read.option("header", "true").csv(TRIPDATA)
taxi_sample = taxi_raw.sample(withReplacement=False, fraction=0.01)
taxi_parsed = taxi_sample.rdd.map(parserow).toDF()
taxi_parsed.cache()

# divide 
taxi_good = taxi_parsed.rdd.filter(lambda row: row.parsefailed == False).toDF()
taxi_bad = taxi_parsed.rdd.filter(lambda row: row.parsefailed == True).toDF()

# clean
taxi_clean = taxi_good.rdd.map(hours).filter(lambda row: row.duration_mins > 1 and row.duration_hours < 3).toDF()
taxi_parsed.unpersist()
taxi_clean.cache()

# sort features
geojson = parsegeo(GEOJSON)
features = geojson['features']
sorted_features = sorted(features, key=lambda feature: (int(feature['properties']['boroughCode']), -feature['geometry'].area))
sc.broadcast(sorted_features)

# borough distribution
taxi_done = taxi_clean.rdd.map(borough).toDF()
taxi_done = taxi_done.filter(taxi_done.borough != 'NA')
# taxi_done.groupBy('borough').count().show() # this takes very long

# remove zero points
zero = {'x': 0.0, 'y': 0.0}
taxi_done = taxi_done.rdd.filter(lambda row: row.pickup_loc != zero and row.dropoff_loc != zero).toDF()
taxi_clean.unpersist()
taxi_done.cache()

# sessionization
sessions = taxi_done.repartition('license').sortWithinPartitions('license', 'pickup_datetime')
taxi_done.unpersist()
sessions.cache()
sessions.show()

sessions.rdd.mapPartitions(lambda trips: window(trips))\
        .filter(lambda pair: len(pair) == 2 and pair[0].license == pair[1].license)\
        .filter(lambda pair: split_shifts(pair[0], pair[1]))\
        .map(lambda pair: borough_duration(pair[0], pair[1]))\
        .take(5)

# my own extension 
from arcgis.gis import GIS
gis = GIS()
nycmap = gis.map("New York City")

def parse_points(row):
    pickup_point = Point({**row.pickup_loc, 'spatialReference': {'wkid': WKID}})
    dropoff_point = Point({**row.dropoff_loc, 'spatialReference': {'wkid': WKID}})
    return pickup_point, dropoff_point
points = taxi_done.rdd.map(parse_points).collect()

for feature in sorted_features:
    nycmap.draw(feature['geometry'])

pickup_pt_sym = {
        "type": "esriSMS",
        "style": "esriSMSDiamond",
        "color": [0, 255, 0],        
        "size": 2,
        "angle": 0,
        "xoffset": 0,
        "yoffset": 0,
        }
dropoff_pt_sym = {
        "type": "esriSMS",
        "style": "esriSMSDiamond",
        "color": [255,140,0,255],        
        "size": 2,
        "angle": 0,
        "xoffset": 0,
        "yoffset": 0,
        }

for p in points[:300]:
    nycmap.draw(p[0], symbol=pickup_pt_sym)
    nycmap.draw(p[1], symbol=dropoff_pt_sym)
