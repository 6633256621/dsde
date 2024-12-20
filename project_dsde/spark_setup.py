# spark_setup.py
from pyspark.sql import SparkSession

# Initialize Spark session
def create_spark_session(app_name="University Research Analysis", master="local"):
    spark = SparkSession.builder \
        .appName(app_name) \
        .master(master) \
        .getOrCreate()
    return spark

# Load data into Spark DataFrames and return a dictionary of DataFrames
def load_data(spark, file_paths):
    dataframes = {}
    for name, path in file_paths.items():
        dataframes[name] = spark.read.csv(path, header=True, inferSchema=True)
        # Register as a temporary view
        dataframes[name].createOrReplaceTempView(name)
    return dataframes

# File paths for each dataset
file_paths = {
    "author_affiliations": "data/author_affiliations.csv",
    "affiliations": "data/affiliations.csv",
    "subject_areas": "data/subject_areas.csv",
    "keywords": "data/keywords.csv",
    "publications": "data/publications.csv",
    "authors": "data/authors.csv",
    "embeddings": "data/embeddings.csv",
    "clustering": "data/clustering.csv",
    "abstracts": "data/abstracts.csv",
    "geospatial_clustering_data": "data/geospatial_clustering_data.csv",
    "geospatial_data_by_publication": "data/geospatial_data_by_publication.csv",
    "scopus_affiliation_data": "data/scopus_affiliation_data.csv"
}

if __name__ == "__main__":
    spark = create_spark_session()
    dataframes = load_data(spark, file_paths)
    print("Spark session initialized and data loaded.")
