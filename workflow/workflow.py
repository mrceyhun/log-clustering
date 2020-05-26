#!/usr/bin/env python3

# system modules
import os
import re
import sys
import time
import json
import site
import logging
import argparse
try:
    from pyspark import SparkContext, StorageLevel
    from pyspark.sql import Row
    from pyspark.sql import SparkSession
    from pyspark.sql import DataFrame
    from pyspark.sql.types import DoubleType, IntegerType, StructType, StructField, StringType, BooleanType, LongType
    from pyspark.sql.functions import col, lit, regexp_replace, trim, lower, concat, count
except ImportError:
    pass
import numpy as np
import pandas as pd
from clusterlogs import pipeline
import nltk
import uuid
from CMSMonitoring.StompAMQ import StompAMQ


class OptionParser():
    def __init__(self):
        "User based option parser"
        self.parser = argparse.ArgumentParser(prog='PROG')
        self.parser.add_argument("--creds", action="store",
            dest="creds", default="", help="Stomp AMQ credentials file, if provided the data will be send to MONIT")
        self.parser.add_argument("--fout", action="store",
            dest="fout", default="", help="Write results into file")
        self.parser.add_argument("--date", action="store",
            dest="date", default="", help="date for scraping FTS logs in YYYY/MM/DD format")
        self.parser.add_argument("--verbose", action="store_true",
            dest="verbose", default=False, help="verbose output")

def spark_context(appname='cms', yarn=None, verbose=False, python_files=[]):
    # define spark context, it's main object which allow
    # to communicate with spark
    if  python_files:
        return SparkContext(appName=appname, pyFiles=python_files)
    else:
        return SparkContext(appName=appname)
        

def spark_session(appName="log-parser"):
    """
    Function to create new spark session
    """
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
    Function that takes Pandas' dataframe and
    yields part of it as a list of jsons
    """
    leng = len(data)
    for i in range(0, leng, samples):
        yield data[i:i + samples].to_dict('records')
        

def run(creds, fout, date=None):
    _schema = StructType([
        StructField('metadata', StructType([StructField('timestamp',LongType(), nullable=True)])),
        StructField('data', StructType([
            StructField('t__error_message', StringType(), nullable=True),
            StructField('src_hostname', StringType(), nullable=True),
            StructField('dst_hostname', StringType(), nullable=True)
        ])),
    ]) #schema of the FTS data that is taken
    sc = spark_session()
    if not date:
        tstamp = time.time()-24*60*60 # one day ago
        date = time.strftime("%Y/%m/%d", time.gmtime(tstamp))
    fts_df = fts_tables(sc, date=date, schema=_schema).select(
        col('metadata.timestamp').alias('tstamp'),
        col('data.src_hostname').alias('src_hostname'),
        col('data.dst_hostname').alias('dst_hostname'),
        col('data.t__error_message').alias('error_message')
    ).where('error_message <> ""') #taking non-empty messages, if date is not given then the records from yesterday are taken

    fts_df.show()

    df = fts_df.toPandas() #the messages are converted to Pandas df

    mod_name = 'word2vec_'+time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time()))+'.model' #model is named according to the run date

    cluster = pipeline.Chain(df, target='error_message', mode='create', model_name=mod_name)
    cluster.process() #messages are clustered with 'clusterlogs' module

    df.loc[:,'cluster_id']=int(1)
    print(cluster.model_name)
    if cluster.clustering_type == 'SIMILARITY':
        df.loc[:,'model']='Levenshtein'
    else:
        df.loc[:,'model']=cluster.model_name #info about clustering model is added to the messages

    a=cluster.result.index
    for el in a:
        df.loc[cluster.result.loc[el,'indices'],'cluster_id'] = str(uuid.uuid4())
        df.loc[cluster.result.loc[el,'indices'],'cluster_pattern'] = cluster.result.loc[el,'pattern'] #info about the clusters is added to the error messages

    res = df[['tstamp','cluster_id','cluster_pattern','model','src_hostname','dst_hostname','error_message']]

    print("Number of messages: ",res.shape[0])
    if fout:
        nrows = df.shape[0]
        count = 0
        with open(fout, 'w') as ostream:
            ostream.write('[' + '\n')
            for d in df_to_batches(res, 10000):
                for r in d:
                    if nrows - count == 1: # last row to write
                        ostream.write(json.dumps(r)+'\n')
                    else:
                        ostream.write(json.dumps(r)+',\n')
                    count += 1
            ostream.write(']' + '\n')

    creds = credentials(creds)
    if creds:
        username = creds.get('username', '')
        password = creds.get('password', '')
        producer = creds.get('producer', 'cms-fts-logsanalysis')
        topic = creds.get('topic', '/topic/cms.fts.logsanalysis')
        host = creds.get('host', 'cms-mb.cern.ch')
        port = int(creds.get('port', 61313))
        cert = creds.get('cert', None)
        ckey = creds.get('ckey', None)
        stomp_amq = StompAMQ(username, password, producer, topic, key=ckey, cert=cert, validation_schema=None, host_and_ports=[(host, port)])
        for d in df_to_batches(res,10000):
            messages = []
            for msg in d:
                notif,_,_ = stomp_amq.make_notification(msg, "training_document", producer=producer, dataSubfield=None)
                messages.append(notif)
            stomp_amq.send(messages)
            time.sleep(0.1) #messages are sent to AMQ queue in batches of 10000

        print("Message sending is finished")

def credentials(fname):
    if os.path.exists(fname):
        data = json.load(open(fname))
        return data
    return {}

def main():
    "Main function"
    optmgr  = OptionParser()
    opts = optmgr.parser.parse_args()
    run(opts.creds, opts.fout, opts.date)

if __name__ == "__main__":
    main()
