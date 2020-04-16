import os
import re
import sys
import time
import logging
import json
import site
#sys.path.insert(0,site.getusersitepackages())
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
from CMSMonitoring.StompAMQ import StompAMQ

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

def df_to_batches(data, samples=1000):
    """
    """
    #data = data.to_dict('records')
    leng = len(data)
    for i in range(0, leng, samples):
        yield data[i:i + samples].to_dict('records')
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
    df.loc[:,'cluster_id']=int(1)
    df.loc[:,'model']='Levenshtein'
    a=cluster.result.index
    for el in a:
        df.loc[cluster.result.loc[el,'indices'],'cluster_id'] = uuid.uuid4()
        df.loc[cluster.result.loc[el,'indices'],'cluster_pattern'] = cluster.result.loc[el,'pattern']
    res = df[['timestamp','cluster_id','cluster_pattern','model','src_hostname','dst_hostname','error_message']]
    #res.to_json(r'results.json',orient='records',default_handler=str)
    username = ""
    password = ""
    producer = "cms-training"
    topic = "/topic/cms.training"
    host = "cms-test-mb.cern.ch"
    port = 61323
    cert = "/eos/user/n/ntuckus/SWAN_projects/training/globus/robot-training-cert.pem"
    ckey = "/eos/user/n/ntuckus/SWAN_projects/training/globus/robot-training-key.pem"
    stomp_amq = StompAMQ(username, password, producer, topic, key=ckey, cert=cert, validation_schema=None, host_and_ports=[(host, port)])
    for d in df_to_batches(res,10000):
        messages = []
        for msg in d:
            notif,_,_ = stomp_amq.make_notification(msg, "training_document", metadata={"version":"129"}, dataSubfield=None)#, ts=msg['timestamp'])
            messages.append(notif)
        stomp_amq.send(messages)
        time.sleep(0.1)
        print(len(messages))
if __name__ == "__main__":
    main()
    
