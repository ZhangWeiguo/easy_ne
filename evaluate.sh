#!/bin/bash
set -e
set -x
python classify.py Cora deepwalk 0.8
python map.py Cora deepwalk