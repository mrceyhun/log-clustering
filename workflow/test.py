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


def main():
    #print(res.shape)
    #res.to_json(r'results.json',orient='records',default_handler=str)
    username = ""
    password = ""
    producer = "cms-fts-logsanalysis"
    topic = "/topic/cms.fts.logsanalysis"
    host = "cms-mb.cern.ch"
    port = 61323
    cert = "/afs/cern.ch/user/n/ntuckus/.globus/usercert.pem"#"/afs/cern.ch/user/n/ntuckus/public/log-clustering/workflow/certs/usercert.pem"
    ckey = "/afs/cern.ch/user/n/ntuckus/.globus/userkey.pem"#"/afs/cern.ch/user/n/ntuckus/public/log-clustering/workflow/certs/userkey.pem"
    stomp_amq = StompAMQ(username, password, producer, topic, key=ckey, cert=cert, validation_schema=None, host_and_ports=[(host, port)])
    msg={'a':'x','b':'y'}#df = pd.DataFrame(np.array([[1, 2, 3]]]),columns=['a', 'b', 'c'])
    notif,_,_ = stomp_amq.make_notification(msg, "training_document", metadata={"version":"655"}, dataSubfield=None)#, ts=msg['timestamp'])
    stomp_amq.send(notif)

if __name__ == "__main__":
    main()
    
