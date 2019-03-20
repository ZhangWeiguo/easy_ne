#!/bin/bash
# You must rewrite the config file: /config/${data_name}.ini
# You can use it with the data name as your cmd parameter
# And the data name must be same as the config file name
set -e
set -x
cd train_deepwalk
python reconstruct.py Cora
cd ..