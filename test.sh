#!/bin/sh
set -eu
docker build -f Dockerfile -t pytorch-lifestream-tests .
docker run --rm -it -v "${PWD}/ptls:/app/ptls" -v "${PWD}/ptls_tests:/app/ptls_tests" pytorch-lifestream-tests
