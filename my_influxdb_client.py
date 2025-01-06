from influxdb_client import Point, InfluxDBClient, WriteOptions
from influxdb_client.client.write_api import SYNCHRONOUS, ASYNCHRONOUS
import pandas as pd

class InfluxDbClient: 
    def __init__(self, host, port, token, org, database):
        self.host = host
        self.port = port
        self.org = org
        self.token = token
        self.database = database

        self.client = InfluxDBClient(url=f"http://{self.host}:{self.port}", 
                                     token=f"{self.token}",
                                     org=self.org)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)


    def write_to_influxdb(self, data: pd.DataFrame, approach_name: str) -> None:
        """Write data to InfluxDB."""
        points = []
        # Current time
        current_time = pd.Timestamp.now()

        # Time exactly 3 days ago
        current_time = current_time - pd.Timedelta(days=3)
        for _, row in data.iterrows():
            adjust_time = current_time + pd.Timedelta(seconds=row.Time)
            point = Point("fraud_detection") \
                .tag("approach", approach_name) \
                .field("Amount", float(row.Amount)) \
                .field("Class", float(row.Class)) \
                .field("is_anomaly", float(row.is_anomaly)) \
                .time(adjust_time)
            points.append(point)
        
        self._write_data_to_influxdb(points)
        print("Data written to InfluxDB.")


    def _write_data_to_influxdb(self, points: list) -> None:
        """Write data to InfluxDB."""
        self.write_api.write(bucket=self.database, record=points)
