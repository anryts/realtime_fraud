from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, MinMaxScaler, StringIndexer
from fraud_detection_lib import FraudDetectionLib
from my_influxdb_client import InfluxDbClient

# Create a SparkSession
spark = SparkSession.builder \
    .appName("RealTimeAnomaliesDetection") \
    .master("spark://localhost:7077") \
    .config('spark.executor.memory', '15g') \
    .config('spark.driver.memory', '15g') \
    .config('spark.sql.shuffle.partitions', '200') \
    .getOrCreate()

# Load the data
file_path = "file:///opt/spark/data/creditcard.csv"
data = spark.read.csv(file_path, header=True, inferSchema=True)
fraud_detection_lib = FraudDetectionLib()
fraud_detection_lib.calculate_iqr_bounds(data, ["Amount", "V11", "V4", "V2"])
fraud_detection_lib.train_classification_model(data)
fraud_detection_lib.train_isolation_forest(data)

#Initialise the InfluxDB client
influxdb_client = InfluxDbClient(host='localhost', 
                                 port=8086, 
                                 token='my-token',
                                 org='my_org',
                                 database='fraud_detection')

# Stream data
stream_data = spark.readStream \
    .format("csv") \
    .option("header", "true") \
    .option("maxFilesPerTrigger", 1) \
    .schema(data.schema) \
    .load("file:///opt/spark/data/stream_data")

def process_batch(df, epoch_id):
    print(f"current batch: {epoch_id}")
    if df.count() < 1: 
        return
    # Prepare the data
    df = df.dropna()
    features = df.columns[:-1]
    assember = VectorAssembler(inputCols=features, outputCol="features")
    scaler = MinMaxScaler(inputCol="features", outputCol="scaled_features")
    #indexer = StringIndexer(inputCol="Class", outputCol="Class")
    assembled_data = assember.transform(df)
    prepared_data = scaler.fit(assembled_data).transform(assembled_data)
    #prepared_data = indexer.fit(prepared_data).transform(prepared_data)

    # Make inferences with diff approaches

    # IQR
    iqr_anomalies = fraud_detection_lib.iqr_detection(df, ["Amount", "V11", "V4", "V2"])
    influxdb_client.write_to_influxdb(iqr_anomalies, "IQR")

    # Random Forest
    rf_anomalies = fraud_detection_lib.random_forest_detection(prepared_data)
    influxdb_client.write_to_influxdb(rf_anomalies, "RandomForest")

    # Isolation Forest from sci-kit learn
    iso_forest_anomalies = fraud_detection_lib.isolate_forest_detection(df)
    influxdb_client.write_to_influxdb(iso_forest_anomalies, "IsolationForest")



query = stream_data.writeStream \
    .foreachBatch(process_batch) \
    .outputMode("append") \
    .trigger(processingTime="5 seconds") \
    .start()

query.awaitTermination()

