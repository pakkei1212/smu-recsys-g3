# Dockerfile: PySpark-based Chunk Merger (no Jupyter, local file support)

FROM openjdk:11-slim

# Install Python and pip
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy all project files
COPY . /app

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Spark + Java environment
ENV JAVA_HOME=/usr/local/openjdk-11
ENV PATH="$JAVA_HOME/bin:$PATH"
ENV PYSPARK_PYTHON=python
ENV PYSPARK_DRIVER_PYTHON=python
