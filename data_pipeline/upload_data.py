import boto3
from . import config
import pandas as pd
import io


class DataUploader:
    def __init__(self):
        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=config.aws_access_key_id,
            aws_secret_access_key=config.aws_secret_access_key,
        )

    def download_from_s3(self, bucket_name: str, s3_file_path: str) -> str:
        response = self.s3.get_object(Bucket=bucket_name, Key=s3_file_path)
        if isinstance(response["Body"], bytes):
            response_string = response["Body"].read().decode("utf-8")
        else:
            response_string = response["Body"].read()
        return response_string

    def convert_csv_to_df(self, csv_content: str) -> pd.DataFrame:
        if isinstance(csv_content, bytes):
            csv_content = csv_content.decode("utf-8")
        return pd.read_csv(io.StringIO(csv_content))
