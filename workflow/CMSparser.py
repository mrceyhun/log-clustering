import os
import re
import sys
import time
import logging
import json
from datetime import datetime as dt
from datetime import timedelta
from subprocess import Popen, PIPE
from pyspark import SparkContext, StorageLevel
from pyspark.sql import Row
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql.types import DoubleType, IntegerType, StructType, StructField, StringType, BooleanType, LongType

def glob_files(sc, url,verbose):
    """
    Return a list of files. It uses the jvm gateway. 
    This function should be prefered to files when using glob expressions.
    """
    URI = sc._gateway.jvm.java.net.URI
    Path = sc._gateway.jvm.org.apache.hadoop.fs.Path
    FileSystem = sc._gateway.jvm.org.apache.hadoop.fs.FileSystem
    fs = FileSystem.get(URI("hdfs:///"), sc._jsc.hadoopConfiguration())
    l = fs.globStatus(Path(url))
    return [f.getPath().toString() for f in l]

def spark_context(appname='cms', yarn=None, verbose=False, python_files=[]):
    # define spark context, it's main object which allow
    # to communicate with spark
    if  python_files:
        return SparkContext(appName=appname, pyFiles=python_files)
    else:
        return SparkContext(appName=appname)

def spark_session(appName="log-parser"):
    sc = SparkContext(appName="log-parser")
    return SparkSession.builder.config(conf=sc._conf).getOrCreate()

def fts_tables(spark,
        hdir='hdfs:///project/monitoring/archive/fts/raw/complete',
        date=None, verbose=False):
    """
    Parse fts HDFS records
    The fts record consist of data and metadata parts where data part squashed
    into single string all requested parameters.
    :returns: a dictionary with fts Spark DataFrame
    """
    if  not date:
        # by default we read yesterdate data
        date = time.strftime("%Y/%m/%d", time.gmtime(time.time()-60*60*24))

    hpath = '%s/%s' % (hdir, date)
    _schema = StructType([
    StructField('metadata', StructType([StructField("timestamp",LongType(), nullable=False)])),
    StructField('data', StructType([StructField("t__error_message", StringType(), nullable=False)])),
    ])
    # create new spark DataFrame
    fts_df = spark.read.json(hpath, _schema)
    fts_df.createOrReplaceTempView('fts_df')
    fts_df = spark.sql("select data.t__error_message as error_message from fts_df where data.t__error_message <> ''") # extract data part of JSON records
    return fts_df

def main():    
    sc = spark_session()
    tables = fts_tables(sc,date="2020/01/01")
    tables.toPandas().drop_duplicates("error_message").to_csv("errors.txt",index=False, header=True)
    tables.show()

if __name__ == "__main__":
    main()
    
