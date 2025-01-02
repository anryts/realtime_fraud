import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
import json

class SparkToInfluxDB:
    def __init__(self, influxdb_url, bucket, org, token):
        self.influxdb_url = influxdb_url
        self.bucket = bucket
        self.org = org
        self.token = token

    def write_to_influxdb(self):
        """
        Convert Spark DataFrame to Pandas and write to InfluxDB.
        :param df: Spark DataFrame to be written to InfluxDB
        """
        print("Writing data to InfluxDB")
        # Convert DataFrame to Pandas (for simplicity)
        client = influxdb_client.InfluxDBClient(url=self.influxdb_url, token=self.token)
        write_api = client.write_api(write_options=SYNCHRONOUS)
        print("Writing data to InfluxDB")
        p = influxdb_client.Point("credit_card_fraud").tag("batch_id", 1)
        print("Creating InfluxDB point")
        # Write data to InfluxDB
        write_api.write(self.bucket, self.org, p)
        print("Data written to InfluxDB")
       
