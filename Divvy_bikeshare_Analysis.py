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
# MAGIC ### 3. Analisys and results

# COMMAND ----------

# Loading gold tables
riders_df = spark.read.table("dim_riders").toPandas()
payments_df = spark.read.table("dim_payments").toPandas()
stations_df = spark.read.table("dim_stations").toPandas()
trips_df = spark.read.table("dim_trips").toPandas()
dates_df = spark.read.table("dim_dates").toPandas()


# COMMAND ----------

# MAGIC %md
# MAGIC #### 3.1 How much time is spent per ride

# COMMAND ----------

# Creating a key to allow merge trips and date tables
trips_df["key_date"] = trips_df["started_at"]
trips_df["key_date"] = pd.to_datetime(trips_df["key_date"], unit='s').dt.date

# Adding day of week and hour of the day in the trips table
trips_df_weekday = pd.merge(trips_df, dates_df[["CalendarDate", "DayOfWeek", "CalendarDay"]], how = "inner", left_on = "key_date", right_on = "CalendarDate")
trips_df_weekday["start_hour"] = trips_df_weekday["started_at"].dt.hour
trips_df_weekday = trips_df_weekday.drop(columns = "key_date")
trips_df_weekday.head(3)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Time spent per day of week

# COMMAND ----------

# Calculating the total time spent by day of week
trips_total_time = trips_df_weekday[["trip_duration_min", "DayOfWeek", "CalendarDay"]].groupby("DayOfWeek").sum("trip_duration_min")

# Counting the number of rides per day of week
trips_total_count = trips_df_weekday[["trip_id", "DayOfWeek"]].groupby("DayOfWeek").count()
trips_total_count.rename(columns = {"trip_id":"number_of_trips"}, inplace = True)

# Calculating the averare time spent by day of week
trips_avg_time = pd.merge(trips_total_time, trips_total_count, on = "DayOfWeek")
trips_avg_time["trips_avg_time_min"] = trips_avg_time["trip_duration_min"] / trips_avg_time["number_of_trips"]
trips_avg_time["trips_avg_time_min"] = trips_avg_time["trips_avg_time_min"].round(2)
trips_avg_time = pd.merge(trips_avg_time, trips_df_weekday[["DayOfWeek", "CalendarDay"]], how = "inner", on = "DayOfWeek").drop_duplicates().reset_index()
trips_avg_time = trips_avg_time.drop(columns = "index")
trips_avg_time

# COMMAND ----------

fig = plt.subplots(figsize = (8,3))
plot = sns.barplot(y = "trips_avg_time_min", x = "CalendarDay", data = trips_avg_time, palette="Blues_d")
plot.set_title("Average time spent per day of the week")
plot.bar_label(plot.containers[0], fontsize=8, padding = 5)
plot.set_xlabel("Day of week", fontsize = 10)
plot.set_ylabel("Avg. time spent", fontsize = 10)
plt.xticks(fontsize = 8)
plt.yticks(fontsize = 8)
plot.set_ylim(0, 35);

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Time spent by hour of the day

# COMMAND ----------

# Calculating the total time spent by hour of the day
trips_total_hours = trips_df_weekday[["trip_duration_min", "start_hour"]].groupby("start_hour").sum("trip_duration_min")

# Counting the number of rides per hour of the day
trips_total_count_hours = trips_df_weekday[["trip_id", "start_hour"]].groupby("start_hour").count()
trips_total_count_hours.rename(columns = {"trip_id":"number_of_trips"}, inplace = True)

# Calculating the averare time spent per hour of the day
trips_avg_time_hour = pd.merge(trips_total_hours, trips_total_count_hours, on = "start_hour")
trips_avg_time_hour["Avg_time_spent"] = trips_avg_time_hour["trip_duration_min"] / trips_avg_time_hour["number_of_trips"]
trips_avg_time_hour = trips_avg_time_hour.reset_index()
trips_avg_time_hour["Avg_time_spent"] = trips_avg_time_hour["Avg_time_spent"].round(2)
trips_avg_time_hour

# COMMAND ----------

fig = plt.subplots(figsize = (18,3))
plot = sns.barplot(y = "Avg_time_spent", x = "start_hour", data = trips_avg_time_hour, palette="Blues_d")
plot.set_title("Average time spent per hour of the day")
plot.bar_label(plot.containers[0], fontsize=8, padding = 5)
plot.set_xlabel("Hour of the day", fontsize = 10)
plot.set_ylabel("Avg. time spent", fontsize = 10)
plt.xticks(fontsize = 8)
plt.yticks(fontsize = 8)
plot.set_ylim(0, 45);

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Time spent by start station

# COMMAND ----------

# Calculating the total time spent by start station
trips_station_hours = trips_df_weekday[["trip_duration_min", "start_station_id"]].groupby("start_station_id").sum("trip_duration_min").sort_values("trip_duration_min", ascending=False).reset_index()
trips_station_hours["trip_duration_hour"] = (trips_station_hours["trip_duration_min"]/60).round(2)
trips_station_hours = trips_station_hours.head(10)
trips_station_hours

# COMMAND ----------

fig = plt.subplots(figsize = (18,3))
plot = sns.barplot(y = "trip_duration_hour", x = "start_station_id", data = trips_station_hours, palette="Blues_d", order = trips_station_hours.sort_values("trip_duration_hour", ascending = False).start_station_id)
plot.set_title("Time spent per start station - Top 10")
plot.bar_label(plot.containers[0], fontsize=8, padding = 5)
plot.set_xlabel("Start station", fontsize = 10)
plot.set_ylabel("Total time spent", fontsize = 10)
plt.xticks(fontsize = 8)
plt.yticks(fontsize = 8)
plot.set_ylim(0, 60000);

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Time spent by rider age

# COMMAND ----------

# Calculating the total time spent by rider age
trips_station_age = trips_df_weekday[["trip_duration_min", "rider_age"]].groupby("rider_age").sum("trip_duration_min").sort_values("trip_duration_min", ascending=False).reset_index()
trips_station_age["trip_duration_hour"] = (trips_station_age["trip_duration_min"]/60).round(2)
trips_station_age

# COMMAND ----------

fig = plt.subplots(figsize = (18,3))
plot = sns.barplot(y = "trip_duration_hour", x = "rider_age", data = trips_station_age, palette="Blues_d", order = trips_station_age.sort_values("rider_age", ascending = True).rider_age)
plot.set_title("Time spent per rider age - Top 10")
plot.bar_label(plot.containers[0], label_type = 'edge', rotation = 90, fontsize = 7, padding = 5)
plot.set_xlabel("Rider age", fontsize = 10)
plot.set_ylabel("Total time spent", fontsize = 10)
plt.xticks(fontsize = 8)
plt.yticks(fontsize = 8)
plot.set_ylim(0, 85000);

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Time spent by relation type - Member or casual rider 

# COMMAND ----------

# Adding relation type (member or casual) in the trips table
trips_df_relation = pd.merge(trips_df_weekday, riders_df[["rider_id", "is_member"]], how = "inner", on = "rider_id")
trips_df_relation["relation_type"] = np.where(trips_df_relation["is_member"] == True, "Member", "Casual")

# Calculating the total time spent by relation type (member or casual rider)
trips_time_relation = trips_df_relation[["trip_duration_min", "relation_type"]].groupby("relation_type").sum("trip_duration_min").sort_values("trip_duration_min", ascending=False).reset_index()
trips_time_relation["trip_duration_hour"] = (trips_time_relation["trip_duration_min"]/60).round() 
trips_time_relation

# COMMAND ----------

fig = plt.subplots(figsize = (4,4))
data = trips_time_relation["trip_duration_hour"]
keys = trips_time_relation["relation_type"]
plt.pie(data, labels=keys, autopct='%.2f%%')
plt.title("Time spent by relation type", fontsize = 12)
plt.show() 

# COMMAND ----------

# MAGIC %md
# MAGIC #### 6.2 How much money is spent

# COMMAND ----------

# Calculating rider age at account start
riders_df_money = riders_df
riders_df_money["age_at_account_start"] = (riders_df_money.account_start_date - riders_df_money.birthday_date).dt.days/365.25
riders_df_money["age_at_account_start"] = riders_df_money["age_at_account_start"].apply(np.floor)

# Adding date and rider information to payments table 
payments_by_time = pd.merge(payments_df, dates_df[["CalendarDate", "CalendarYear", "CalendarMonth", "MonthOfYear", "QuarterOfYear"]], how = "inner", left_on = "payment_date", right_on = "CalendarDate").sort_values("payment_date").reset_index(drop = True)
payments_by_time = pd.merge(payments_by_time, riders_df_money[["rider_id", "age_at_account_start"]], how = "inner", on = "rider_id").sort_values("payment_date").reset_index(drop = True)
payments_by_time

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Money spent by month

# COMMAND ----------

money_by_month = payments_by_time[["payment_amount", "MonthOfYear"]].groupby("MonthOfYear").sum("payment_amount").sort_values("MonthOfYear").reset_index()
money_by_month = pd.merge(money_by_month, payments_by_time[["CalendarMonth", "MonthOfYear"]].drop_duplicates(), how = "inner", on = "MonthOfYear")
money_by_month["payment_amount"] = money_by_month["payment_amount"].round(2)
money_by_month

# COMMAND ----------

fig = plt.subplots(figsize = (18,3))
plot = sns.barplot(y = "payment_amount", x = "CalendarMonth", data = money_by_month, palette="Blues_d")
plot.set_title("Money spent by month")

plot.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: '{:,.0f}'.format(x)))

for container in plot.containers:
    plot.bar_label(container, label_type='edge', fmt='%.2f', fontsize = 8, padding = 5)

plot.set_xlabel("Month", fontsize = 10)
plot.set_ylabel("Money spent", fontsize = 10)
plt.xticks(fontsize = 8)
plt.yticks(fontsize = 8)
plot.set_ylim(0, 2500000);

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Money spent by quarter

# COMMAND ----------

money_by_quarter = payments_by_time[["payment_amount", "QuarterOfYear"]].groupby("QuarterOfYear").sum("payment_amount").sort_values("QuarterOfYear").reset_index()
money_by_quarter["payment_amount"] = money_by_quarter["payment_amount"].round(2)
money_by_quarter

# COMMAND ----------

fig = plt.subplots(figsize = (9,3))
plot = sns.barplot(y = "payment_amount", x = "QuarterOfYear", data = money_by_quarter, palette="Blues_d")
plot.set_title("Money spent by quarter")

plot.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: '{:,.0f}'.format(x)))

for container in plot.containers:
    plot.bar_label(container, label_type='edge', fmt='%.2f', fontsize = 8, padding = 5)

plot.set_xlabel("Quarter", fontsize = 10)
plot.set_ylabel("Money spent", fontsize = 10)
plt.xticks(fontsize = 8)
plt.yticks(fontsize = 8)
plot.set_ylim(0, 6500000);

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Money spent by year

# COMMAND ----------

money_by_year = payments_by_time[["payment_amount", "CalendarYear"]].groupby("CalendarYear").sum("payment_amount").sort_values("CalendarYear").reset_index()
money_by_year["payment_amount"] = money_by_year["payment_amount"].astype("float").round(2)
money_by_year

# COMMAND ----------

fig, ax = plt.subplots(figsize = (12,3))
plot = sns.barplot(y = "payment_amount", x = "CalendarYear", data = money_by_year, palette="Blues_d")
plot.set_title("Money spent by year")
plot.set_xlabel("Year", fontsize = 10)
plot.set_ylabel("Money spent", fontsize = 10)

plot.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: '{:,.0f}'.format(x)))

for container in plot.containers:
    plot.bar_label(container, label_type='edge', fmt='%.2f', fontsize = 8, padding = 5)

plt.xticks(fontsize = 8)
plt.yticks(fontsize = 8)
plot.set_ylim(0, 7000000);

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Money spent by rider age at start account

# COMMAND ----------

money_by_age = payments_by_time[["payment_amount", "age_at_account_start"]].groupby("age_at_account_start").sum("payment_amount").sort_values("payment_amount", ascending = False).reset_index()
money_by_age["payment_amount"] = money_by_age["payment_amount"].round()
money_by_age["age_at_account_start"] = money_by_age["age_at_account_start"].astype("int")
money_by_age

# COMMAND ----------

fig = plt.subplots(figsize = (20,3))
plot = sns.barplot(y = "payment_amount", x = "age_at_account_start", data = money_by_age, palette="Blues_d", order = money_by_age.sort_values("age_at_account_start", ascending = True).age_at_account_start)
plot.set_title("Money spent by rider age at start account")
plot.bar_label(plot.containers[0], label_type = 'edge', rotation = 90, fontsize = 7, padding = 5)
plot.set_xlabel("Rider age", fontsize = 10)
plot.set_ylabel("Total money spent", fontsize = 10)
plt.xticks(fontsize = 8)
plt.yticks(fontsize = 8)
plot.set_ylim(0, 900000);

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Money spent by member

# COMMAND ----------

# Adding year and month on payments table
payments_member = payments_df
payments_member["month"] = pd.to_datetime(payments_member["payment_date"]).dt.month
payments_member["year"] = pd.to_datetime(payments_member["payment_date"]).dt.year
payments_member = payments_member.sort_values("payment_date").reset_index(drop = True)
payments_member.head()

# COMMAND ----------

# Calculating money spent by month and creating a key to merge tables
payments_by_month = payments_member.groupby(["rider_id", "year", "month"]).sum("payment_amount").reset_index()
payments_by_month["paymentsbymonth_id"] = payments_by_month["rider_id"].astype(str) + payments_by_month["year"].astype(str) + payments_by_month["month"].astype(str)
payments_by_month

# COMMAND ----------

# Adding year and month on trips table
trips_member = trips_df
trips_member["month"] = pd.to_datetime(trips_member["key_date"]).dt.month
trips_member["year"] = pd.to_datetime(trips_member["key_date"]).dt.year
trips_member = trips_member.sort_values("key_date").reset_index(drop = True)
trips_member.head()

# COMMAND ----------

# Calculating number of trips by month and creating a key to merge tables
trips_by_month = trips_member[["trip_id", "rider_id", "year", "month"]].groupby(["rider_id", "year", "month"]).count().reset_index()
trips_by_month["tripsbymonth_id"] = trips_by_month["rider_id"].astype(str) + payments_by_month["year"].astype(str) + payments_by_month["month"].astype(str)
trips_by_month = trips_by_month.rename(columns = {"trip_id" : "n_of_trips"})
trips_by_month

# COMMAND ----------

# Calculating time spent by month and creating a key to merge tables
trips_duration_by_month = trips_member[["trip_duration_min", "rider_id", "year", "month"]].groupby(["rider_id", "year", "month"]).sum("trip_duration_min").reset_index()
trips_duration_by_month["tripsduration_id"] = trips_by_month["rider_id"].astype(str) + payments_by_month["year"].astype(str) + payments_by_month["month"].astype(str)
trips_duration_by_month = trips_duration_by_month.rename(columns = {"trip_duration_min" : "time_of_trips"})
trips_duration_by_month

# COMMAND ----------

# Merging tables and calculating the results
money_rides_by_member = pd.merge(payments_by_month, trips_by_month[["n_of_trips", "tripsbymonth_id"]], how = "inner", left_on = "paymentsbymonth_id", right_on = "tripsbymonth_id")
money_rides_by_member = pd.merge(money_rides_by_member, trips_duration_by_month[["time_of_trips", "tripsduration_id"]], how = "inner", left_on = "paymentsbymonth_id", right_on = "tripsduration_id")
money_rides_by_member = money_rides_by_member.drop(columns = ["paymentsbymonth_id", "tripsbymonth_id", "tripsduration_id"])
money_rides_by_member["money_per_trip"] = (money_rides_by_member["payment_amount"] / money_rides_by_member["n_of_trips"]).round(2)
money_rides_by_member["money_per_minute"] = (money_rides_by_member["payment_amount"] / money_rides_by_member["time_of_trips"]).round(2)
money_rides_by_member.head(10)
