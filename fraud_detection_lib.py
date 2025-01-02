from pyspark.ml.regression import RandomForestRegressionModel
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql.functions import col
from pyspark.sql import DataFrame
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np

class FraudDetectionLib: 
    """
    A class to detect anomalies in a dataset.
    Each method implements a different anomaly detection technique
    and returns a pandas.Series with 1 for anomaly and 0 otherwise.
    """
    def __init__(self):
        # Load pre-trained model
        # Feel free to comment not needed models. to save your CPU from burning
        #self.classification_model = RandomForestRegressionModel.load("/opt/spark/data/saved_models/random_forest_model_save")
        self.iqr_bounds = {}
        self.classification_model = None
        self.isolation_forest_model = None
    
    def calculate_iqr_bounds(self, data: DataFrame, features: list) -> None:
        """
        Calculate and store IQR bounds for each feature.
        
        :param data: Input Spark DataFrame.
        :param features: List of feature names to analyze for outliers.
        """
        for feature in features:
            Q1 = data.approxQuantile(feature, [0.25], 0)[0]
            Q3 = data.approxQuantile(feature, [0.75], 0)[0]
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.iqr_bounds[feature] = (lower_bound, upper_bound)

    def iqr_detection(self, data: DataFrame, features: list) -> pd.DataFrame:
        """
        Detect anomalies using the Interquartile Range (IQR) method.
       
        :param data: Input Spark DataFrame.
        :param features: List of feature names to analyze for outliers.
        :return: Pandas DataFrame with an additional 'is_anomaly' column.
        """
        outlier_conditions = []
        for feature, (lower_bound, upper_bound) in self.iqr_bounds.items():
            condition = (data[feature] < lower_bound) | (data[feature] > upper_bound)
            outlier_conditions.append(condition)
        
        # Combine conditions for all features
        combined_outliers = outlier_conditions[0]
        for condition in outlier_conditions[1:]:
            combined_outliers |= condition

        # Add 'is_anomaly' column to indicate outliers
        df_with_anomalies = data.withColumn("is_anomaly", combined_outliers.cast("int"))

        # Convert to Pandas DataFrame
        result_df = df_with_anomalies.toPandas() # problem, shape is 1000, 32 need to convert to just 1 column

        return result_df
    
    def zscore_detection(data, feature, threshold = 3):
        """
        Detect anomalies using the Z-score method.
        Returns:
            anomalies (pandas.Series): 1 for anomaly, 0 otherwise
        """
        mean = data[feature].mean()
        std = data[feature].std()
        
        zscores = (data[feature] - mean) / std
        anomalies = zscores.abs() > threshold
        return anomalies.astype(int)
    
    def oversample_data(self, data: DataFrame) -> DataFrame:
        fraud_count = data.where(col("Class") == 1).count()
        non_fraud_df = data.where(col("Class") == 0)
        non_fraud_count = non_fraud_df.count()
        n_samples_to_add = non_fraud_count - fraud_count
        
        # Randomly sample the fraud transactions to add duplicates
        fraud_oversampled_df = data.where(col("Class") == 1).sample(withReplacement=True, 
                                                                    fraction=n_samples_to_add / fraud_count,
                                                                    seed=3445)

        # Combine the oversampled fraud data with the non-fraud data
        return non_fraud_df.union(fraud_oversampled_df)

    def prepare_data(self, data: DataFrame) -> DataFrame:
        # Define the feature columns (excluding 'Class' and 'time')
        balanced_df = data #self.oversample_data(data)
        feature_columns = [col for col in balanced_df.columns if col not in ['Class', 'Time']]

        # VectorAssembler to combine features into a single feature vector
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features") 
        assembled_data = assembler.transform(balanced_df)
        # MinMaxScaler to scale the features
        scaler = MinMaxScaler(inputCol="features", outputCol="scaled_features") 
        scaled_data = scaler.fit(assembled_data).transform(assembled_data)
        return scaled_data      

    def train_classification_model(self, data: DataFrame):
        # Split the data into training and test sets
        data = self.prepare_data(data)
        train_data, test_data = data.randomSplit([0.8, 0.2], seed=3445)
        # Train a Random Forest classifier
        rf = RandomForestClassifier(featuresCol="scaled_features", labelCol="Class", numTrees=30)
        self.classification_model = rf.fit(train_data)

        # Evaluate the model
        evaluator = BinaryClassificationEvaluator(labelCol="Class")
        predictions_rf = self.classification_model.transform(test_data)
        accuracy_rf = evaluator.evaluate(predictions_rf)
        print(f"Random forest accuracy: {accuracy_rf}")

        # Show the confusion matrix
        predictions_rf.groupBy("Class", "prediction").count().show()
        

    def random_forest_detection(self, data: DataFrame) -> pd.DataFrame:
        """
        Detect anomalies using a Random Forest classifier.
        Returns:
            anomalies (pandas.Series): 1 for anomaly, 0 otherwise
        """
        # Prepare the data
        predictions = self.classification_model.transform(data)
        
        # Convert to Pandas DataFrame and add 'is_anomaly' column
        result_df = predictions.toPandas()
        result_df['is_anomaly'] = result_df['prediction'].astype(int)
        return result_df
    
    def train_isolation_forest(self, data: DataFrame) -> None:
        features = data.columns[:-1]
        assembler = VectorAssembler(inputCols=features, outputCol="features")
        assembled_data = assembler.transform(data)

        scaler = MinMaxScaler(inputCol="features", outputCol="scaled_features")
        scaler_model = scaler.fit(assembled_data)
        normalized_features = scaler_model.transform(assembled_data)

        normalized_features_pd = normalized_features.select("scaled_features").toPandas()
        normalized_features_np = np.array(normalized_features_pd["scaled_features"].tolist())

        iso_forest = IsolationForest(contamination=0.01, random_state=3345)
        iso_forest.fit(normalized_features_np)
        predictions = iso_forest.predict(normalized_features_np)

        # Convert predictions to binary: 0 for inliers, 1 for outliers
        balanced_df_pd = data.toPandas()
        balanced_df_pd['is_anomaly'] = (predictions == -1).astype(int)

        # Simple validation: count of Class == 1 and is_anomaly == 1
        validation_count = balanced_df_pd[(balanced_df_pd['Class'] == 1) & (balanced_df_pd['is_anomaly'] == 1)].shape[0]
        print(f"Validation count (Class == 1 and is_anomaly == 1): {validation_count}")

    def isolate_forest_detection(self, data: DataFrame) -> pd.DataFrame:
        """
        Detect anomalies using the Isolation Forest method.
        Returns:
            anomalies (pandas.Series): 1 for anomaly, 0 otherwise
        """
        
        features = data.columns[:-1]
        assembler = VectorAssembler(inputCols=features, outputCol="features")
        assembled_data = assembler.transform(data)

        scaler = MinMaxScaler(inputCol="features", outputCol="scaled_features")
        scaler_model = scaler.fit(assembled_data)
        normalized_features = scaler_model.transform(assembled_data)

        normalized_features_pd = normalized_features.select("scaled_features").toPandas()
        normalized_features_np = np.array(normalized_features_pd["scaled_features"].tolist())

        anomalies = self.isolation_forest_model.predict(normalized_features_np)
        result_df = data.toPandas()
        result_df['is_anomaly'] = (anomalies == -1).astype(int)
        #print results by is anomaly and class
        print(f"Anomalies: {result_df['is_anomaly'].value_counts()}")
        print(f"Class: {result_df['Class'].value_counts()}")
        return result_df