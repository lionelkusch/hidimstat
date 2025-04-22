#!/bin/bash

set -x -e

wget $GITHUB_ARTIFACT_URL
mkdir -p doc_conf/_build/html
unzip doc*.zip -d doc_conf/_build/html