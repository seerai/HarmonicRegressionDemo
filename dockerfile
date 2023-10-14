FROM python:3.10-slim-buster as builder
WORKDIR /app
COPY requirements.txt /app/
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt --prefix=/install 

FROM python:3.10-slim-buster
COPY --from=builder /install /usr/local
WORKDIR /app
COPY harmonic-regression.py /app/
CMD ["python", "-u", "harmonic-regression.py"]