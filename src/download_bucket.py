import boto
import boto.s3.connection

import os
import json
import pickle

try:
    access_key = os.environ["AWS_ACCESS_KEY_ID"]
    secret_key = os.environ["AWS_SECRET_ACCESS_KEY"]

except:
    import json
    with open('/home/ubuntu/amazon.json') as key_file:
        keys = json.load(key_file)
        access_key = keys["AWS_ACCESS_KEY_ID"]
        secret_key = keys["AWS_SECRET_ACCESS_KEY"]

bucket_name = "quora-bwl"
conn = boto.connect_s3(access_key, secret_key, host='s3.amazonaws.com')
bucket = conn.get_bucket(bucket_name)


key = bucket.get_key('train.csv')
print (key.ongoing_restore)
file_name = str(key.name).split('/')[-1]
try:
    key.get_contents_to_filename(file_name)
    print ('Passed --- ', file_name)
except:
    print ('Failed --- ', file_name)
