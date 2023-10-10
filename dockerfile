FROM python:3.10-slim-buster

RUN apt-get update \
    && apt-get install -y libgeos-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

COPY requirements.txt /app/
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt 

COPY harmonic-regression.py /app/
CMD ["python", "-u", "harmonic-regression.py"]