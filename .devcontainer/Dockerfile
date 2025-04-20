FROM ubuntu:jammy-20240627.1

# Set Non-Interactive Mode
ARG DEBIAN_FRONTEND=noninteractive

# Set Timezone
ARG TZ="America/Chicago"
RUN echo "${TZ}" > /etc/localtime && \
    echo "${TZ}" > /etc/timezone

# Install Required Ubuntu Packages
RUN apt-get update && apt-get install --no-install-recommends -y python3 \
    python3-pip \
    python-is-python3 \
    python3-venv \
    python3-dev \
    build-essential \
    nano \
    git \
    ssh

# Install Python Packages
RUN pip install matplotlib \
    numpy \
    scikit-learn \
    pandas

# Set Labels
LABEL authors="WAW"
LABEL maintainer="willweidler@mst.edu"
LABEL org.opencontainers.image.source=https://github.com/wiidler/CS5402-GPA-Prediction-Project
LABEL org.opencontainers.image.licenses=GPL-3.0-only
LABEL org.opencontainers.image.version="v0.0.1"
LABEL org.opencontainers.image.description="Docker Image for CS5402 GPA Prediction Project"


