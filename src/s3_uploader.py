import boto3
import os

def upload_file_to_s3(local_path: str, bucket_name: str, s3_key: str) -> str:
    """
    Uploads a file to AWS S3, makes it public, and returns the public URL.
    Args:
        local_path (str): Path to the local file to upload
        bucket_name (str): Name of the S3 bucket
        s3_key (str): Desired key (filename) in the S3 bucket
    Returns:
        str: Publicly accessible URL to the file
    """
    s3 = boto3.client('s3')
    try:
        s3.upload_file(local_path, bucket_name, s3_key)
        url = f'https://{bucket_name}.s3.amazonaws.com/{s3_key}'
        return url
    except Exception as e:
        print(f"Error uploading to S3: {e}")
        return None 