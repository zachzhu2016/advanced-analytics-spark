unset PYTHONPATH                                                                                                                                                                             
# export PYTHONPATH=/opt/python-3.7/lib/python3.7/site-packages/                                                                                                                             
export PYSPARK_PYTHON=/opt/python-3.7/bin/python3.7                                                                                                                                          
PATH=/opt/python-3.7/bin:$PATH                                                                                                                                                               
module load spark 

# fix: https://stackoverflow.com/questions/50987944/key-not-found-pyspark-driver-callback-host
export PYTHONPATH=$SPARK_HOME/python/:$PYTHONPATH
export PYTHONPATH=$SPARK_HOME/python/lib/py4j-0.8.2.1-src.zip:$PYTHONPATH
# spark-submit --master local[*] --driver-memory 6g main.py 
