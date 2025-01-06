from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, MinMaxScaler, StringIndexer
from fraud_detection_lib import FraudDetectionLib
from pyspark.sql.functions import col
from my_influxdb_client import InfluxDbClient
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Create a SparkSession
spark = SparkSession.builder \
    .appName("RealTimeAnomaliesDetection") \
    .master("spark://localhost:7077") \
    .config('spark.executor.memory', '15g') \
    .config('spark.driver.memory', '15g') \
    .config('spark.sql.shuffle.partitions', '200') \
    .getOrCreate()

#TODO: just for test
global counter
counter = 0

# Load the data
file_path = "file:///opt/spark/data/creditcard.csv"
data = spark.read.csv(file_path, header=True, inferSchema=True)
fraud_detection_lib = FraudDetectionLib()
fraud_detection_lib.calculate_iqr_bounds(data, ["Amount", "V11", "V4", "V2"])
#fraud_detection_lib.train_classification_model_with_data_drift(data)
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
    global counter
    if df.count() < 1: 
        return
    # Prepare the data
    df = df.dropna()
    features = [col for col in df.columns if col not in ['Class', 'Time']]
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
    iso_forest_anomalies = fraud_detection_lib.isolate_forest_detection(prepared_data)
    influxdb_client.write_to_influxdb(iso_forest_anomalies, "IsolationForest")

    # Combined IQR and RandomForest
    #combined_anomalies = fraud_detection_lib.iqr_plus_random_forest_detection(df, ["Amount", "V11", "V4", "V2"])


def process_batch_with_data_drift(df, epoch_id):
    '''
    A method to process the batch with data drift detection
    ONLY FOR TEST
    '''
    # Simulate data drift after 20 batches
    if epoch_id > 20:
        drift_koef = 1.7  # Define your drift coefficient
        df = df.withColumn("Amount", col("Amount") * drift_koef)
        df = df.withColumn("V11", col("V11") * drift_koef)
        df = df.withColumn("V4", col("V4") * drift_koef)
        df = df.withColumn("V2", col("V2") * drift_koef)
    
    process_batch(df, epoch_id)

query = stream_data.writeStream \
    .foreachBatch(process_batch) \
    .outputMode("append") \
    .start()

query.awaitTermination()

