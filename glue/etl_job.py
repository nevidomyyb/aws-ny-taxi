"""
Job parameters (--key value):
  --JOB_NAME
  --raw_path             s3://<seu-bucket>/raw/yellow/          (entrada parquet)
  --processed_path       s3://<seu-bucket>/processed/yellow/    (saída parquet)
  --high_tip_threshold   0.20                                   (limiar do label)
"""


import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import functions as F

args = getResolvedOptions(
  sys.argv,
  ["JOB_NAME", "raw_path", "processed_path", "high_tip_threshold"]
)

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

HIGH_TIP_THRESHOLD = float(args["high_tip_threshold"])

df = spark.read.parquet(args['raw_path'])

# Limpeza

df = (
  df.filter(F.col("payment_type") == 1) # Filtrando apenas cartão de crédito, no dataset gorjetas em dinheiro não são registradas.
  .filter(F.col("fare_amount") > 0)
  .filter(F.col("fare_amount") < 500)
  .filter(F.col("trip_distance") > 0)
  .filter(F.col("trip_distance") < 100)
  .filter(F.col("tip_amount") >= 0)
  .filter(F.col("passenger_count").between(1, 6))
  .dropna(subset=[
      "tpep_pickup_datetime", "tpep_dropoff_datetime",
      "fare_amount", "trip_distance", "tip_amount",
      "PULocationID", "DOLocationID",
  ])
)

# Feature engineering

df = df.withColum(
  "trip_duration_min",
  (F.unix_timestamp("tpep_dropoff_datetime") - F.unix_timestamp("tpep_pickup_datetime")) / 60.0,
)

df = df.filter(F.col("trip_duration_min").between(1, 120))

df = (
    df
    .withColumn("pickup_hour", F.hour("tpep_pickup_datetime"))
    .withColumn("pickup_dayofweek", F.dayofweek("tpep_pickup_datetime"))  # 1=Dom ... 7=Sáb
    .withColumn(
        "is_weekend",
        F.when(F.col("pickup_dayofweek").isin(1, 7), 1).otherwise(0),
    )
    .withColumn("tip_pct", F.col("tip_amount") / F.col("fare_amount"))
)

df = df.filter(F.col("tip_pct") < 1.0)

df = df.withColumn(
    "high_tip",
    F.when(F.col("tip_pct") > HIGH_TIP_THRESHOLD, 1).otherwise(0),
)

# identificador único + event_time (ISO-8601) exigidos pelo Feature Store
df = df.withColumn("ride_id", F.sha2(
    F.concat_ws("_",
        F.col("tpep_pickup_datetime").cast("string"),
        F.col("PULocationID").cast("string"),
        F.col("DOLocationID").cast("string"),
        F.monotonically_increasing_id().cast("string"),
    ), 256))
df = df.withColumn(
    "event_time",
    F.date_format(F.col("tpep_pickup_datetime"), "yyyy-MM-dd'T'HH:mm:ss'Z'"),
)

final_cols = [
    "ride_id",
    "event_time",
    # features
    "trip_distance",
    "trip_duration_min",
    "passenger_count",
    "pickup_hour",
    "pickup_dayofweek",
    "is_weekend",
    "PULocationID",
    "DOLocationID",
    "fare_amount",
    # label
    "high_tip",
]
df_out = df.select(*final_cols)

df_out = (
    df_out
    .withColumn("trip_distance", F.col("trip_distance").cast("double"))
    .withColumn("trip_duration_min", F.col("trip_duration_min").cast("double"))
    .withColumn("fare_amount", F.col("fare_amount").cast("double"))
    .withColumn("passenger_count", F.col("passenger_count").cast("int"))
    .withColumn("pickup_hour", F.col("pickup_hour").cast("int"))
    .withColumn("pickup_dayofweek", F.col("pickup_dayofweek").cast("int"))
    .withColumn("is_weekend", F.col("is_weekend").cast("int"))
    .withColumn("PULocationID", F.col("PULocationID").cast("int"))
    .withColumn("DOLocationID", F.col("DOLocationID").cast("int"))
    .withColumn("high_tip", F.col("high_tip").cast("int"))
)

(
    df_out
    .coalesce(4)  # poucos arquivos -> leitura mais simples nos notebooks
    .write
    .mode("overwrite")
    .parquet(args["processed_path"])
)
 
job.commit()
