FROM rust:latest

RUN apt-get update && apt install git -y --no-install-recommends

COPY . /app

WORKDIR /app
