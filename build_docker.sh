# build and validate the docker image
docker build -t us-central1-docker.pkg.dev/double-catfish-291717/seerai-docker/images/har-reg:v0.0.7 -f dockerfile .
tesseract-sdk validate us-central1-docker.pkg.dev/double-catfish-291717/seerai-docker/images/har-reg:v0.0.7