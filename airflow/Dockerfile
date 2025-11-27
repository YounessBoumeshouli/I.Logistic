FROM apache/airflow:2.9.0

# 1. Switch to root to install Java (Spark needs Java to run)
USER root
RUN apt-get update && \
    apt-get install -y default-jdk && \
    apt-get clean

# Set JAVA_HOME environment variable so Spark can find Java
ENV JAVA_HOME=/usr/lib/jvm/default-java

# 2. Switch back to airflow user to install PySpark
USER airflow
RUN pip install pyspark