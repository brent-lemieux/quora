import pyspark as ps

import numpy as np

import json
import os

# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
# from pyspark.mllib.clustering import KMeans
# from collections import Counter
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# import string
#
# from pyspark.ml.classification import NaiveBayes
# from pyspark.ml.feature import CountVectorizer, IDF
# from pyspark.sql.functions import udf
# from pyspark.sql.types import ArrayType, StringType
# from pyspark.sql import Row
#
# PUNCTUATION = set(string.punctuation)
# STOPWORDS = set(stopwords.words('english'))

spark = ps.sql.SparkSession.builder \
            .appName("quora-model") \
            .getOrCreate()

train = spark.read.csv('s3n://quora-bwl/train')
