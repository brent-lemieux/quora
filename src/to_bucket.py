import boto
from boto.s3.key import Key
import os

import json

with open('/home/ubuntu/amazon.json') as key_file:
    keys = json.load(key_file)
    access_key = keys["AWS_ACCESS_KEY_ID"]
    access_secret_key = keys["AWS_SECRET_ACCESS_KEY"]

def to_bucket(f):
    file_name = '/home/ubuntu/{}'.format(f)
    print file_name
    bucket_name = "tasty-tweets"
    fil = open(file_name)
    conn = boto.connect_s3(access_key, access_secret_key)
    bucket = conn.get_bucket(bucket_name)
    #Get the Key object of the bucket
    k = Key(bucket)
    #Crete a new key with id as the name of the file
    k.key = file_name
    #Upload the file
    result = k.set_contents_from_file(fil)
    #result contains the size of the file uploaded
    os.remove(file_name)

if __name__ == '__main__':
    to_bucket('embeddings.csv')
