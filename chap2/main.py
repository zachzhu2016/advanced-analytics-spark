from typing import NamedTuple
from collections import namedtuple
from pyspark import SparkContext, SparkConf
from pyspark.statcounter import StatCounter
from helper import NAStatCounter
import math

def is_header(line: str) -> bool:
    return '"' in line

def parse(line: str) -> NamedTuple:
    arr = line.split(',')
    id1 = int(arr[0])
    id2 = int(arr[1])
    scores = [float('nan') if x == '?' else float(x) for x in arr[2:11]]
    matched = True if arr[11] == 'TRUE' else False
    MatchData = namedtuple("MatchData", ["id1", "id2", "scores", "matched"])
    return MatchData(id1, id2, scores, matched)

def process_partition(partition):
    nas = list(map(lambda d: NAStatCounter(d), next(partition, [])))
    res = list()
    for arr in partition:
        for n, d in zip(nas, arr):
            res.append(n.add(d))
    yield res

def stats_with_missing(rdd):
    nastats = rdd.mapPartitions(process_partition)
    return nastats.reduce(lambda n1, n2: [a.merge(b) for a, b in zip(n1, n2)])

def calc_weight(mds, frequencies):
    
    """ 
    My Extension:
    The u probability is the probability that an identifier in two non-matching records will agree purely by chance
    The m probability is the probability that an identifier in matching pairs will agree
    """

    frequencies_headers = [s.strip('"') for s in frequencies.first().split(',')]
    frequencies_data = frequencies.filter(lambda line: not is_header(line))
    frequencies_parsed = frequencies_data.first().split(',')
    match = dict()
    nonmatch = dict()

    for i, (header, freq) in enumerate(zip(frequencies_headers, frequencies_parsed)):
        d = mds.filter(lambda md: md.matched).map(lambda md: md.scores[i] == 1).countByValue()
        m = float(d[True] / (d[True] + d[False]))
        u = float(freq)
        match[header] = {'m': m, 'u': u, 'ratio': m / u, 'weight': math.log(m / u) / math.log(2)}
        nonmatch[header] = {'m': 1 - m, 'u': 1 - u, 'ratio': (1 - m) / (1 - u), 'weight': math.log((1 - m) / (1 - u)) / math.log(2)}

    return (match, nonmatch)

if __name__ == '__main__': 
    conf = SparkConf()
    conf.setMaster('local[*]')
    conf.set('spark.driver.memory', '4g')
    conf.set('spark.executor.memory', '4g')

    sc = SparkContext(conf=conf)
    base = "./data"
    
    block1 = sc.textFile(f'{base}/block_1.csv')
    block1_header = block1.first()
    block1_data = block1.filter(lambda line: not is_header(line))

    frequencies = sc.textFile(f'{base}/frequencies.csv')

    mds = block1_data.map(parse)
    mds.cache()
    match, nonmatch = calc_weight(mds, frequencies)

    """

    Below are the code blocks from AAS.

    grouped = mds.groupBy(lambda md: md.matched) 
    # grouped.mapValues(lambda x: len(x)).foreach(print)
    matched_counts = mds.map(lambda md: md.matched).countByValue()
    match_counts_list = list(matched_counts.items())
    match_counts_list.sort(lambda tup: tup[0])
    match_counts_list.sort(lambda tup: tup[1])

    stats = [mds.filter(lambda md: not math.isnan(md.scores[i])).map(lambda md: md.scores[i]).stats() for i in range(9)]

    nasRDD = mds.map(lambda md: list(map(lambda d: NAStatCounter(d), md.scores)))
    reduced = nasRDD.reduce(lambda n1, n2: [a.merge(b) for a, b in zip(n1, n2)])
    
    statsm = stats_with_missing(mds.filter(lambda md: md.matched).map(lambda md: md.scores))
    statsn = stats_with_missing(mds.filter(lambda md: not md.matched).map(lambda md: md.scores)) 

    """
