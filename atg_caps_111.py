# Databricks notebook source
# MAGIC %sql
# MAGIC USE CATALOG atg_catalog4;
# MAGIC  
# MAGIC CREATE SCHEMA IF NOT EXISTS bronze;
# MAGIC CREATE SCHEMA IF NOT EXISTS silver;
# MAGIC CREATE SCHEMA IF NOT EXISTS gold;

# COMMAND ----------

storage_account = "atgstorage5"
container = "atgcontainer5"
file_path = "retail_iot_data.csv"
 
# Credentials - Replace with actual values (or keep them in a secret scope)
client_id = "db02ce5c-145d-4e09-979f-446e84fd614c"
tenant_id = "5950e39e-81d1-45a1-8618-fe39a39b0448"
client_secret = "ktT8Q~CTWWRSHdAZX0lYNOlCm1aIas6nrNu_wbnZ"
 
# ---------------------------
# Configurations for ABFS access (no mount needed in UC)
# ---------------------------
spark.conf.set(f"fs.azure.account.auth.type.{storage_account}.dfs.core.windows.net", "OAuth")
spark.conf.set(f"fs.azure.account.oauth.provider.type.{storage_account}.dfs.core.windows.net",
               "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider")
spark.conf.set(f"fs.azure.account.oauth2.client.id.{storage_account}.dfs.core.windows.net", client_id)
spark.conf.set(f"fs.azure.account.oauth2.client.secret.{storage_account}.dfs.core.windows.net", client_secret)
spark.conf.set(f"fs.azure.account.oauth2.client.endpoint.{storage_account}.dfs.core.windows.net",
               f"https://login.microsoftonline.com/{tenant_id}/oauth2/token")
 
# ---------------------------
# Build ABFS path directly (no /mnt needed)
# ---------------------------
abfs_path = f"abfss://{container}@{storage_account}.dfs.core.windows.net/{file_path}"
 
# ---------------------------
# Try reading the file to verify access
# ---------------------------
# Read raw file from ADLS
df_bronze = (
    spark.read
        .option("header", "true")
        .option("inferSchema", "true")   # or provide schema for better performance
        .csv(abfs_path)
)
 
print(f"✅ Loaded {df_bronze.count()} rows from {abfs_path}")
display(df_bronze.limit(5))

# COMMAND ----------

catalog = "atg_catalog4"
schema = "bronze"
table_name = "bronze"
 
# Write into UC Bronze Delta table
df_bronze.write.format("delta").mode("overwrite").saveAsTable(
    f"{catalog}.{schema}.{table_name}"
)
 
print(f"✅ Bronze Delta table created: {catalog}.{schema}.{table_name}")

# COMMAND ----------

import pyspark.sql.functions as F
bronze_df = spark.table("atg_catalog4.bronze.bronze")
silver_df = (bronze_df
    .dropDuplicates(["OrderID"])
    .withColumn("OrderTime", F.to_timestamp("OrderTime"))
    .withColumn("Region", F.upper(F.col("Region")))
    .withColumn("AnomalyType", F.coalesce(F.col("AnomalyType"), F.lit("None")))
    .withColumn("IsTempAnomaly",
                F.when((F.col("Temperature") < 0) | (F.col("Temperature") > 35), 1).otherwise(0))
)
silver_df.write.format("delta").mode("overwrite").saveAsTable(
    "atg_catalog4.silver.retail_cleaned"
)
print("✅ Silver table created: atg_catalog4.silver.retail_cleaned")

# COMMAND ----------

import pyspark.sql.functions as F

# Load silver
sdf = spark.table("atg_catalog4.silver.retail_cleaned")

# Monthly sales per region
monthly_sales = (sdf.groupBy("Region", F.date_trunc("month","OrderTime").alias("Month"))
                   .agg(F.sum("SalesAmount").alias("TotalSales")))

monthly_sales.write.format("delta").mode("overwrite").saveAsTable(
    "atg_catalog4.gold.monthly_sales"
)


# COMMAND ----------

 
import pyspark.sql.functions as F
 
sdf = spark.table("atg_catalog4.silver.retail_cleaned")
 
monthly_sales = (sdf
    .groupBy("Region", F.date_trunc("month", "OrderTime").alias("Month"))
    .agg(F.sum("SalesAmount").alias("TotalSales"),
         F.sum("Quantity").alias("UnitsSold"))
)
 
monthly_sales.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable(
    "atg_catalog4.gold.monthly_sales"
)

# COMMAND ----------

#gold layer kpi2 - anamoly trend per store
 
anomaly_trend = (sdf
    .groupBy("StoreID", F.to_date("OrderTime").alias("Day"))
    .agg(F.sum("AnomalyFlag").alias("SystemAnomalies"),
         F.sum("IsTempAnomaly").alias("TempAnomalies"),
         F.sum("SalesAmount").alias("Sales"))
)
 
anomaly_trend.write.format("delta").mode("overwrite").saveAsTable(
    "atg_catalog4.gold.anomaly_trend"
)

# COMMAND ----------

#gold layer kpi3 - impact of device failure
 
impact = (sdf
    .withColumn("AnomalyDetected", (F.col("AnomalyFlag") + F.col("IsTempAnomaly")) > 0)
    .groupBy("AnomalyDetected")
    .agg(F.sum("SalesAmount").alias("TotalSales"),
         F.countDistinct("OrderID").alias("Orders"))
)
 
impact.write.format("delta").mode("overwrite").saveAsTable(
    "atg_catalog4.gold.anomaly_impact"
)

# COMMAND ----------

#gold layer kpi4 - store into tiers
 
store_perf = (sdf
    .groupBy("StoreID")
    .agg(F.sum("SalesAmount").alias("TotalSales"))
    .withColumn("Tier",
        F.when(F.col("TotalSales") > 50000, "High")
         .when(F.col("TotalSales") > 20000, "Medium")
         .otherwise("Low"))
)
 
store_perf.write.format("delta").mode("overwrite").saveAsTable(
    "atg_catalog4.gold.store_performance"
)

# COMMAND ----------

#gold layer kpi5 - weighted average of sales vs anomalies
 
store_sales_anoms = (sdf
    .groupBy("StoreID")
    .agg(F.sum("SalesAmount").alias("TotalSales"),
         F.sum("AnomalyFlag").alias("AnomalyCount"))
    .filter("AnomalyCount > 0")  # only consider stores with anomalies
)
# Compute weighted average: (Sales * AnomalyCount) / Sum(AnomalyCount)
weighted_avg = (store_sales_anoms
    .select((F.sum(F.col("TotalSales") * F.col("AnomalyCount")) /
             F.sum("AnomalyCount")).alias("WeightedAvgSales"))
)
weighted_avg.write.format("delta").mode("overwrite").saveAsTable(
    "atg_catalog4.gold.weighted_avg_sales_vs_anomalies"
)

# COMMAND ----------

# MAGIC %sql
# MAGIC OPTIMIZE atg_catalog4.gold.monthly_sales ZORDER BY (Month);

# COMMAND ----------

# MAGIC  
# MAGIC %sql
# MAGIC OPTIMIZE atg_catalog4.gold.anomaly_trend ZORDER BY (StoreID, Day);

# COMMAND ----------

# MAGIC %sql
# MAGIC OPTIMIZE atg_catalog4.gold.anomaly_impact ZORDER BY (AnomalyDetected);

# COMMAND ----------

# MAGIC %sql
# MAGIC OPTIMIZE atg_catalog4.gold.weighted_avg_sales_vs_anomalies;

# COMMAND ----------

# MAGIC %sql
# MAGIC OPTIMIZE atg_catalog4.gold.monthly_sales;
# MAGIC OPTIMIZE atg_catalog4.gold.anomaly_trend;
# MAGIC OPTIMIZE atg_catalog4.gold.anomaly_impact;
# MAGIC OPTIMIZE atg_catalog4.gold.store_performance;
# MAGIC OPTIMIZE atg_catalog4.gold.weighted_avg_sales_vs_anomalies;

# COMMAND ----------

event_hub_connection_string = "Endpoint=sb://atgeventhub.servicebus.windows.net/;SharedAccessKeyName=atgpolicy;SharedAccessKey=9Yu5yyTMtiqlphRhsftua5FS7gnz/IPLt+AEhEEhzNk=;EntityPath=atgevent"
eh_conf = {
  'eventhubs.connectionString': sc._jvm.org.apache.spark.eventhubs.EventHubsUtils.encrypt(event_hub_connection_string)
}
