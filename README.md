# Introduction

In this project, I'll apply what I've learned on Spark and data lakes to
build an ETL pipeline for a data lake hosted on S3. To complete the project,
I will load data from S3, process the data into analytics tables using Spark,
and load them back into S3. I'll deploy this Spark process on a cluster using AWS.

# Background

A music streaming startup, Sparkify, has grown their user base and song database
and want to move their processes and data onto the cloud. Their data resides in S3,
in a directory of JSON logs on user activity on the app, as well as a directory
with JSON metadata on the songs in their app.

As their data engineer, I'm tasked with building an ETL pipeline that extracts
their data from S3, stages them in Redshift, and transforms data into a set of
dimensional tables for their analytics team to continue finding insights in what
songs their users are listening to. I'll be able to test the database and ETL
pipeline by running queries given to me by the analytics team from Sparkify and
compare my results with their expected results.

# Datasets

I'll be working with two datasets that reside in S3. Here are the S3 links for each:

 - Song data: `s3://udacity-dend/song_data`
 - Log data: `s3://udacity-dend/log_data`

## Songs

The first dataset is a subset of real data from the [Million Song](http://millionsongdataset.com/) Dataset. Each file is in JSON format and contains metadata about a song and the artist of that song. The files are partitioned by the first three letters of each song's track ID. For example, here are filepaths to two files in this dataset.

```
song_data/A/B/C/TRABCEI128F424C983.json
song_data/A/A/B/TRAABJL12903CDCF1A.json
```

And below is an example of what a single song file, TRAABJL12903CDCF1A.json, looks like.

```
{
    "num_songs": 1,
    "artist_id": "ARJIE2Y1187B994AB7",
    "artist_latitude": null,
    "artist_longitude": null,
    "artist_location": "",
    "artist_name": "Line Renaud",
    "song_id": "SOUPIRU12A6D4FA1E1",
    "title": "Der Kleine Dompfaff",
    "duration": 152.92036,
    "year": 0
}
```

## Logs

The second dataset consists of log files in JSON format generated by this [event simulator](https://github.com/Interana/eventsim) based on the songs in the dataset above. These simulate activity logs from a music streaming app based on specified configurations.

The log files in the dataset I'll be working with are partitioned by year and month. For example, here are filepaths to two files in this dataset.

```
log_data/2018/11/2018-11-12-events.json
log_data/2018/11/2018-11-13-events.json
```

And below is an example of what the data in a log file, `2018-11-12-events.json`, looks like.

![alt text](images/log-data.png)

# Development environment
This project is implemented using PySpark on AWS. We will spin up an EMR
cluster to be used as a development environment. There is a [SAM](https://aws.amazon.com/serverless/sam/) template,
which makes it easy to deploy the necessary resources using the concept of
infrastracture as code.

```shell
cd infra
sam deploy --guided
```

> Please note that the template is not fully productionised and you might encounter
deployment issues since `VPC` and `Subnet` CIDR blocks are hardcoded. Also the
cluster is deployed on a public subnet, which is generally not a good practice.  

# Project structure

1. `etl.py` reads data from S3, processes that data using Spark, and writes them back to S3.
2. `dl.cfg` contains AWS credentials.
3. `README.md` provides discussion on your process and decisions.
3. `infra` contains SAM application for deploying EMR.

# Schema for Song Play Analysis

Using the song and event datasets, I've created a star schema optimized for
queries on song play analysis. This includes the following tables.

## Fact Table

- `songplays` - records in event data associated with song plays i.e. records with *page* `NextSong`. This fact table has the following attributes: *start_time*, *user_id*, *level*, *song_id*, *artist_id*, *session_id*, *location*, and *user_agent*.

## Dimension Tables

- `users` - users in the app. This dimension table has the following attributes: *user_id*, *first_name*, *last_name*, *gender*, and *level*.

- `songs` - songs in music database. This dimension table has the following attributes: *song_id*, *title*, *artist_id*, *year*, and *duration*.

- `artists` - artists in music database. This dimension table has the following attributes: *artist_id*, *name*, *location*, *lattitude*, and *longitude*.

- `time` - timestamps of records in `songplays` broken down into specific units: *start_time*, *hour*, *day*, *week*, *month*, *year*, and *weekday*.

