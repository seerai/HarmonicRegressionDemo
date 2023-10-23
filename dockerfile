FROM python:3.10-slim-buster as builder
WORKDIR /app
COPY requirements.txt /app/
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt --prefix=/install 
# COPY tesseract-python-sdk /app/tesseract-python-sdk
# RUN python -m pip install /app/tesseract-python-sdk --prefix=/install

FROM python:3.10-slim-buster
COPY --from=builder /install /usr/local
WORKDIR /app
COPY harmonic_regression.py /app/
CMD ["python", "-u", "harmonic_regression.py"]