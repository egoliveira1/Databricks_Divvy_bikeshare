# Databricks notebook source
# MAGIC %md
# MAGIC ### 1. Imports

# COMMAND ----------

import pyspark
import numpy                as np
import pandas               as pd
import seaborn              as sns 
import matplotlib.pyplot    as plt
import matplotlib.ticker    as mtick
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import (StringType, BooleanType, IntegerType, FloatType, DateType, TimestampNTZType, TimestampType)
from pyspark.sql.functions import *

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. Creating a Spark Session

# COMMAND ----------

spark = SparkSession \
    .builder \
    .appName("Divvy") \
    .getOrCreate()

# COMMAND ----------

spark.sparkContext.getConf().getAll()

# COMMAND ----------

spark

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3. Reading and writing data

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.1 Reading Data

# COMMAND ----------

riders_df = spark.read.format("csv") \
    .option("inferschema", "false") \
    .option("header", "false") \
    .option("sep", ",") \
    .load("/FileStore/tables/Divvy/riders.csv")

display(riders_df)

# COMMAND ----------

payments_df = spark.read.format("csv") \
    .option("inferschema", "false") \
    .option("header", "false") \
    .option("sep", ",") \
    .load("/FileStore/tables/Divvy/payments.csv")

display(payments_df)

# COMMAND ----------

stations_df = spark.read.format("csv") \
    .option("inferschema", "false") \
    .option("header", "false") \
    .option("sep", ",") \
    .load("/FileStore/tables/Divvy/stations.csv")

display(stations_df)

# COMMAND ----------

trips_df = spark.read.format("csv") \
    .option("inferschema", "false") \
    .option("header", "false") \
    .option("sep", ",") \
    .load("/FileStore/tables/Divvy/trips.csv")

display(trips_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.2 Writing Data

# COMMAND ----------

riders_df.write.format("delta") \
    .mode("overwrite") \
    .save("/FileStore/delta/Divvy_riders_bronze")

# COMMAND ----------

payments_df.write.format("delta") \
    .mode("overwrite") \
    .save("/FileStore/delta/Divvy_payments_bronze")

# COMMAND ----------

stations_df.write.format("delta") \
    .mode("overwrite") \
    .save("/FileStore/delta/Divvy_stations_bronze")

# COMMAND ----------

trips_df.write.format("delta") \
    .mode("overwrite") \
    .save("/FileStore/delta/Divvy_trips_bronze")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Transformation steps

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.1 Riders table

# COMMAND ----------

riders_df = spark.read.format("delta").load("/FileStore/delta/Divvy_riders_bronze")

# COMMAND ----------

# Renaming columns
riders_new_columns = ["rider_id", "first_name", "last_name", "rider_address", "birthday_date", "account_start_date", "account_end_date", "is_member"]
riders_df = riders_df.toDF(*riders_new_columns)
riders_df.show(3)

# COMMAND ----------

# Changing data types
riders_columns_type_map = {
    "rider_id" : StringType(),
    "first_name" : StringType(),
    "last_name" : StringType(),
    "rider_address" : StringType(),
    "birthday_date" : DateType(),
    "account_start_date" : DateType(),
    "account_end_date" : DateType(),
    "is_member" : BooleanType()
}

for column in riders_columns_type_map:
  riders_df = riders_df.withColumn(column,col(column).cast(riders_columns_type_map[column]))

riders_df.printSchema()

# COMMAND ----------

# Creating table and loading data
riders_df.write.format("delta") \
    .mode("overwrite") \
    .save("/FileStore/delta/Divvy_riders_gold")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.2 Payments table

# COMMAND ----------

# Loading bronze table
payments_df = spark.read.format("delta").load("/FileStore/delta/Divvy_payments_bronze")

# COMMAND ----------

# Renaming columns
payments_new_columns = ["payment_id", "payment_date", "payment_amount", "rider_id"]
payments_df = payments_df.toDF(*payments_new_columns)
payments_df.show(3)

# COMMAND ----------

# Changing data types
payments_columns_type_map = {
    "payment_id" : StringType(),
    "payment_date" : DateType(),
    "payment_amount" : FloatType(), 
    "rider_id" : StringType()
}

for column in payments_columns_type_map:
  payments_df = payments_df.withColumn(column,col(column).cast(payments_columns_type_map[column]))

payments_df.printSchema()

# COMMAND ----------

# Creating table and loading data
payments_df.write.format("delta") \
    .mode("overwrite") \
    .save("/FileStore/delta/Divvy_payments_gold")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.3 Stations table

# COMMAND ----------

# Loading bronze table
stations_df = spark.read.format("delta").load("/FileStore/delta/Divvy_stations_bronze")

# COMMAND ----------

# Renaming columns
stations_new_columns = ["station_id", "station_name", "station_lat", "station_long"]
stations_df = stations_df.toDF(*stations_new_columns)
stations_df.show(3)

# COMMAND ----------

# Changing data types
stations_columns_type_map = {
    "station_id" : StringType(), 
    "station_name" : StringType(), 
    "station_lat" : FloatType(), 
    "station_long" : FloatType()
}

for column in stations_columns_type_map:
  stations_df = stations_df.withColumn(column,col(column).cast(stations_columns_type_map[column]))

stations_df.printSchema()

# COMMAND ----------

# Creating table and loading data
stations_df.write.format("delta") \
    .mode("overwrite") \
    .save("/FileStore/delta/Divvy_stations_gold")

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4.4 Trips table

# COMMAND ----------

# Loading bronze table
trips_df = spark.read.format("delta").load("/FileStore/delta/Divvy_trips_bronze")

# COMMAND ----------

# Renaming columns
trips_new_columns = ["trip_id", "rideable_type", "started_at", "ended_at", "start_station_id", "end_station_id", "rider_id"]
trips_df = trips_df.toDF(*trips_new_columns)
trips_df.show(3)

# COMMAND ----------

# Changing data types
trips_columns_type_map = {
    "trip_id" : StringType(), 
    "rideable_type" : StringType(), 
    "started_at" : TimestampType(), 
    "ended_at" : TimestampType(), 
    "start_station_id" : StringType(), 
    "end_station_id" : StringType(), 
    "rider_id" : StringType()
}

for column in trips_columns_type_map:
  trips_df = trips_df.withColumn(column,col(column).cast(trips_columns_type_map[column]))

trips_df.printSchema()

# COMMAND ----------

# Creating new columns - Trip duration (minutes)
trip_duration = col("ended_at").cast("long") - col("started_at").cast("long")
trips_df = trips_df.withColumn("trip_duration_min", round(trip_duration/60))

# Creating new columns - Rider age
df_birthdays = riders_df["rider_id", "birthday_date"]
df_birthdays = df_birthdays.withColumnRenamed("rider_id", "rider_id_to_age")

trips_df = trips_df.join(df_birthdays, trips_df.rider_id == df_birthdays.rider_id_to_age, 'inner')
trips_df = trips_df.withColumn("current_date", current_timestamp())
trips_df = trips_df.withColumn("rider_age", floor(datediff(col("current_date"),col("birthday_date"))/365))
trips_df = trips_df.withColumn("rider_age",col("rider_age").cast(IntegerType()))
trips_df = trips_df.drop("current_date", "birthday_date", "rider_id_to_age")

trips_df.show(5)

# COMMAND ----------

# Creating table and loading data
trips_df.write.format("delta") \
    .mode("overwrite") \
    .save("/FileStore/delta/Divvy_trips_gold")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5. Creating tables and loading data

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.1 Fact and dimension tables

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE dim_riders 
# MAGIC USING DELTA LOCATION '/FileStore/delta/Divvy_riders_gold'

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE dim_payments 
# MAGIC USING DELTA LOCATION '/FileStore/delta/Divvy_payments_gold'

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE dim_stations 
# MAGIC USING DELTA LOCATION '/FileStore/delta/Divvy_stations_gold'

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE dim_trips 
# MAGIC USING DELTA LOCATION '/FileStore/delta/Divvy_trips_gold'

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5.2 Date table

# COMMAND ----------

beginDate = "2013-02-01"
endDate = "2022-02-01"

(
  spark.sql(f"select explode(sequence(to_date('{beginDate}'), to_date('{endDate}'), interval 1 day)) as calendarDate")
    .createOrReplaceTempView('dates')
)

# COMMAND ----------

# MAGIC %sql
# MAGIC select
# MAGIC   year(calendarDate) * 10000 + month(calendarDate) * 100 + day(calendarDate) as dateInt,
# MAGIC   CalendarDate,
# MAGIC   year(calendarDate) AS CalendarYear,
# MAGIC   date_format(calendarDate, 'MMMM') as CalendarMonth,
# MAGIC   month(calendarDate) as MonthOfYear,
# MAGIC   date_format(calendarDate, 'EEEE') as CalendarDay,
# MAGIC   dayofweek(calendarDate) AS DayOfWeek,
# MAGIC   weekday(calendarDate) + 1 as DayOfWeekStartMonday,
# MAGIC   case
# MAGIC     when weekday(calendarDate) < 5 then 'Y'
# MAGIC     else 'N'
# MAGIC   end as IsWeekDay,
# MAGIC   dayofmonth(calendarDate) as DayOfMonth,
# MAGIC   case
# MAGIC     when calendarDate = last_day(calendarDate) then 'Y'
# MAGIC     else 'N'
# MAGIC   end as IsLastDayOfMonth,
# MAGIC   dayofyear(calendarDate) as DayOfYear,
# MAGIC   weekofyear(calendarDate) as WeekOfYearIso,
# MAGIC   quarter(calendarDate) as QuarterOfYear,
# MAGIC   /* Use fiscal periods needed by organization fiscal calendar */
# MAGIC   case
# MAGIC     when month(calendarDate) >= 10 then year(calendarDate) + 1
# MAGIC     else year(calendarDate)
# MAGIC   end as FiscalYearOctToSep,
# MAGIC   (month(calendarDate) + 2) % 12 + 1 AS FiscalMonthOctToSep,
# MAGIC   case
# MAGIC     when month(calendarDate) >= 7 then year(calendarDate) + 1
# MAGIC     else year(calendarDate)
# MAGIC   end as FiscalYearJulToJun,
# MAGIC   (month(calendarDate) + 5) % 12 + 1 AS FiscalMonthJulToJun
# MAGIC from
# MAGIC   dates
# MAGIC order by
# MAGIC   calendarDate

# COMMAND ----------

dates_df = _sqldf

# COMMAND ----------

dates_df.write.format("delta") \
    .mode("overwrite") \
    .save("/FileStore/delta/Divvy_dates_bronze")

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE dim_dates 
# MAGIC USING DELTA LOCATION '/FileStore/delta/Divvy_dates_bronze'
