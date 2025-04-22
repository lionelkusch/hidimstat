#!/bin/bash

set -x -e

wget $GITHUB_ARTIFACT_URL
unzip doc*.zip 