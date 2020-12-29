from pyspark.statcounter import StatCounter
import math

class NAStatCounter(StatCounter):
    
    def __init__(self, x: float):
        self.stats = StatCounter()
        self.missing = 0
        self.add(x)

    def add(self, x: float) -> 'NAStatCounter':
        if math.isnan(x):
            self.missing += 1
        else:
            self.stats.merge(x) 
        return self

    def merge(self, other: 'NAStatCounter') -> 'NAStatCounter':
        self.stats.mergeStats(other.stats)
        self.missing += other.missing
        return self

    @staticmethod
    def apply(x: float) -> 'NAStatCounter':
        return NAStatCounter().add(x)

    def __repr__(self) -> str:
        return f'stats: {self.stats} NaN: {self.missing}' 
