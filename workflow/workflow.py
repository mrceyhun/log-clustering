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
from pyspark.sql.functions import col, lit, regexp_replace, trim, lower, concat, count
import numpy as np
import pandas as pd
from clusterlogs import pipeline
import nltk
import uuid

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

def fts_tables(spark,schema,
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
    # create new spark DataFrame
    fts_df = spark.read.json(hpath, schema)
    return fts_df

def main():
    _schema = StructType([
        StructField('metadata', StructType([StructField('timestamp',LongType(), nullable=True)])),
        StructField('data', StructType([
            StructField('t__error_message', StringType(), nullable=True),
            StructField('src_hostname', StringType(), nullable=True),
            StructField('dst_hostname', StringType(), nullable=True)
        ])),
    ])    
    sc = spark_session()
    fts_df = fts_tables(sc,date="2020/01/05",schema=_schema).select(
        col('metadata.timestamp').alias('timestamp'),
        col('data.src_hostname').alias('src_hostname'),
        col('data.dst_hostname').alias('dst_hostname'),
        col('data.t__error_message').alias('error_message')
    ).where('error_message <> ""')
    fts_df.show()
    df = fts_df.toPandas()
    cluster = pipeline.Chain(df, target='error_message', mode='create')
    cluster.process()
    a=cluster.result.index
    for el in a:
        df.loc[cluster.result.loc[el,'indices'],'cluster_id'] = uuid.uuid4()
        df.loc[cluster.result.loc[el,'indices'],'cluster_pattern'] = cluster.result.loc[el,'pattern']
    res = df[['timestamp','cluster_id','cluster_pattern','model','src_hostname','dst_hostname','error_message']]
    res.to_json(r'results.json',orient='records',default_handler=str)
if __name__ == "__main__":
    main()
    
