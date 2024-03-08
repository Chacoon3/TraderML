# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.3.9.2rc1
FROM python:3.9-slim-bookworm as base
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY ./requirements.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -r requirements.txt


FROM base AS production

ENV APP_PORT=8000
ENV PYTHONDONTWRITEBYTECODE=0
ENV PYTHONUNBUFFERED=0
ENV APP_WORKERS=4
ENV APP_ML_DEVICE="cpu"

# Copy the source code into the container.
COPY . /app/

USER root

EXPOSE $APP_PORT

CMD uvicorn --host 0.0.0.0 --port $APP_PORT --workers $APP_WORKERS main:app