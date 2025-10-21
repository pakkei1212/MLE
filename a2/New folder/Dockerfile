# Use the official Apache Airflow image (adjust the version as needed)
FROM apache/airflow:2.6.1

# Switch to root to install additional packages
USER root

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install Java (OpenJDK 17 headless), procps (for 'ps') and bash
# Use bash for the rest of the Dockerfile and enable pipefail
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# ensure we’re using bash with pipefail
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      openjdk-17-jre-headless \
      ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME without using which/readlink if you prefer the known Debian path
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt ./

# Switch to the airflow user before installing Python dependencies
USER airflow

# Install Python dependencies using requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create a volume mount point for notebooks
VOLUME /app
