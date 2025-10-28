# Pin both Airflow and Python version explicitly
FROM apache/airflow:2.9.3-python3.10

USER root
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Robust apt install with retries; JRE is usually enough (use -jdk- if you need javac)
RUN set -euxo pipefail; \
    for i in {1..3}; do \
      apt-get update && break || sleep 5; \
    done; \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        openjdk-17-jre-headless \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME without using which/readlink if you prefer the known Debian path
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# Ensure Spark’s scripts run with bash instead of dash
# RUN ln -sf /bin/bash /bin/sh
    
# Set JAVA_HOME to the directory expected by Spark
#RUN JHOME="$(dirname "$(dirname "$(readlink -f /usr/bin/javac)")")" && \
#    ln -sfn "$JHOME" /usr/lib/jvm/java-home && \
#    echo "Resolved JHOME = $JHOME" && \
#    ls -ld /usr/lib/jvm/java-home && \
#    # Must have java and javac (JDK, not just JRE)
#    test -x "$JHOME/bin/java" && test -x "$JHOME/bin/javac" && \
#    "$JHOME/bin/java" -version && "$JHOME/bin/javac" -version
#ENV JAVA_HOME=/usr/lib/jvm/java-home

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt ./

# ---- Python deps (airflow user) ----
USER airflow

# Remove any preinstalled pyarrow that might be from the base image
RUN pip uninstall -y pyarrow || true

# Upgrade pip tooling, then install *fresh* wheels that are compatible with NumPy 2
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r /app/requirements.txt 

# Build-time import sanity check to fail fast if ABI mismatch persists
RUN python - <<'PY'
import numpy as np, pandas as pd, sklearn, xgboost, scipy
print("NumPy:", np.__version__)
print("pandas:", pd.__version__)
print("sklearn:", sklearn.__version__)
print("xgboost:", xgboost.__version__)
print("scipy:", scipy.__version__)
PY

# Expose the default JupyterLab port
EXPOSE 8888

# Create a volume mount point for notebooks
VOLUME /app

# Enable JupyterLab via environment variable
ENV JUPYTER_ENABLE_LAB=yes

# Set up the command to run JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/app"]
