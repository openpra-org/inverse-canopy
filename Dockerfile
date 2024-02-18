# Use the official Python 3.11 image as a base image
FROM python:3.11

ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory in the container
WORKDIR /app

SHELL ["/bin/bash", "-c"]
RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt update && \
    apt install -y --no-install-recommends build-essential &&\
    rm -rf /var/lib/apt/lists/*

 # Copy the current directory contents into the container at /app
COPY . /app

# Upgrade pip and install package dependencies
RUN pip install --upgrade pip \
    && pip install twine wheel setuptools

# Install the package along with the development dependencies
RUN pip install -e .[dev]