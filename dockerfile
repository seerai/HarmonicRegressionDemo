# build and install all needed python libraries
FROM python:3.10-slim-buster as builder
WORKDIR /app
COPY requirements.txt /app/
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt --prefix=/install 

# Multi-stage build that grabs the installed libraries and copies them to a new image.
# This is to keep the final image as small as possible as python images tend to be large.
FROM python:3.10-slim-buster
COPY --from=builder /install /usr/local
WORKDIR /app
COPY harmonic_regression.py /app/
CMD ["python", "-u", "harmonic_regression.py"]