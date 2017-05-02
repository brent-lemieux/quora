import boto3



def download_files(f_list, bucket):
    s3 = boto3.resource('s3')
    for f in f_list:
        s3.meta.client.download_file(bucket, f, '/home/ubuntu/data/{}'.format(f))
    print('Complete...')


if __name__ == '__main__':
    download_files(['train_cleaned.csv', 'test_cleaned.csv'], 'quora-bwl')
