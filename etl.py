import os
import configparser
from datetime import datetime

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import SparkSession


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    """
    Processes song data by extracting song and artist dimension data.

    Args:
        spark: spark context
        input_data: S3 URI prefix where raw data resides
        output_data: S3 URI prefix where facts and dimensions need to be stored
    """
    
    # get filepath to song data file
    # e.g. song_data/A/B/C/TRABCEI128F424C983.json
    song_data = f'{input_data}/song_data/*/*/*/*'
    
    # read song data file
    df = spark.read.option('multiline', 'true').json(song_data)
    df.createOrReplaceTempView("raw_songs")
    
    # extract columns to create songs table 
    songs_table = spark.sql("""
        SELECT 
            song_id,
            title,
            artist_id,
            year,
            duration
        FROM raw_songs
    """)
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.write.partitionBy('year', 'artist_id').mode('overwrite').parquet(f'{output_data}/songs.parquet')

    # extract columns to create artists table
    artists_table = spark.sql("""
        SELECT
            s.artist_id,
            s.artist_name,
            s.artist_location,
            s.artist_latitude,
            s.artist_longitude
        FROM ( 
          SELECT 
              artist_id,
              artist_name,
              artist_location,
              artist_latitude,
              artist_longitude,
              RANK() OVER(PARTITION BY artist_id ORDER BY year DESC) RANK
          FROM raw_songs
        ) AS s
        WHERE s.rank = 1
    """)
    
    # write artists table to parquet files
    artists_table.write.mode('overwrite').parquet(f'{output_data}/artists.parquet')

def process_log_data(spark, input_data, output_data):
    """
    Processes log data by extracting user and time dimensions as well song play facts.

    Args:
        spark: spark context
        input_data: S3 URI prefix where raw data resides
        output_data: S3 URI prefix where facts and dimensions need to be stored
    """

    # get filepath to log data file
    # e.g. log_data/2018/11/2018-11-12-events.json
    log_data = f'{input_data}/log_data/*/*/*'
        
    # read log data file
    df = spark.read.option('multiline', 'false').json(log_data)
    df.createOrReplaceTempView("raw_logs")
    
    # filter by actions for song plays
    df = spark.sql("""
        SELECT * FROM raw_logs WHERE page = 'NextSong'
    """)
    df.createOrReplaceTempView("raw_logs")
    
    # extract columns for users table    
    users_table = spark.sql("""
        SELECT 
            u.user_id,
            u.first_name,
            u.last_name,
            u.gender,
            u.level
        FROM (
          SELECT
              userId AS user_id,
              firstname as first_name,
              lastname as last_name,
              gender,
              level,
              RANK() OVER(PARTITION BY userId ORDER BY ts DESC) rank
          FROM raw_logs
          WHERE userId IS NOT NULL
        ) AS u
        WHERE u.rank = 1
    """)
    
    # write users table to parquet files
    users_table.write.mode('overwrite').parquet(f'{output_data}/users.parquet')

    # create timestamp column from original timestamp column
    get_timestamp = F.udf(lambda x: datetime.utcfromtimestamp(x/1000) if x else x, T.TimestampType())
    df = df.withColumn('ts_parsed', get_timestamp(col("ts")))
    df.createOrReplaceTempView("raw_logs")

    # extract columns to create time table
    time_table = spark.sql("""
        SELECT
            t.start_time,
            EXTRACT(hour FROM t.start_time) AS hour,
            EXTRACT(day FROM t.start_time) AS day,
            EXTRACT(week FROM t.start_time) AS week,
            EXTRACT(month FROM t.start_time) AS month,
            EXTRACT(year FROM t.start_time) AS year,
            EXTRACT(dayofweek FROM t.start_time) AS weekday
        FROM
        (
          SELECT 
              DISTINCT(ts_parsed) AS start_time
          FROM raw_logs
        ) AS t
    """)

    # write time table to parquet files partitioned by year and month
    time_table.write.partitionBy('year', 'month').mode('overwrite').parquet(f'{output_data}/time.parquet')

    
    # read in song and artis data to use for songplays table
    song_df = spark.read.parquet(f'{output_data}/songs.parquet')
    song_df.createOrReplaceTempView("songs")
    artist_df = spark.read.parquet(f'{output_data}/artists.parquet')
    artist_df.createOrReplaceTempView("artists")
    
    # extract columns from joined song and log datasets to create songplays table 
    songplays_table = spark.sql("""
        SELECT 
            sp.start_time,
            sp.user_id,
            sp.level,
            sp.song_id,
            sp.artist_id,
            sp.session_id,
            sp.location,
            sp.user_agent,
            EXTRACT(month FROM sp.start_time) AS month,
            EXTRACT(year FROM sp.start_time) AS year
        FROM (
          SELECT
              se.ts_parsed as start_time,
              se.userId as user_id,
              se.level,
              ds.song_id,
              da.artist_id,
              se.sessionId as session_id,
              se.location,
              se.userAgent as user_agent
          FROM raw_logs se
          LEFT JOIN (SELECT DISTINCT artist_id, artist_name FROM artists) da ON TRIM(da.artist_name) = TRIM(se.artist)
          LEFT JOIN (SELECT DISTINCT song_id, title FROM songs) ds ON TRIM(ds.title) = TRIM(se.song)
        ) AS sp
    """)
    
    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.partitionBy('year', 'month').mode('overwrite').parquet(f'{output_data}/songplays.parquet')


def main():
    """
    Entrypoint of the script. Sets up spark session, processes song data and then
    log data.
    """

    spark = create_spark_session()

    input_data = config.get('IO', 'INPUT_DATA')
    output_data = config.get('IO', 'OUTPUT_DATA')

    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
