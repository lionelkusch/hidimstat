#!/bin/bash

set -x -e

wget $GITHUB_ARTIFACT_URL
unzip DocHTML.zip -d DocHTML