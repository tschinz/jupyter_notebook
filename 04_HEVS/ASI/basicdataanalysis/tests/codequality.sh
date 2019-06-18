#!/bin/bash

# A simple script to run flake8 on all iPython Notebooks (jupater Notebooks) in this repo
#  1- Convert all iPython Nothbook to plain .py files
#  2- Run flake8 on each .py file


# Start of test
echo "Control Jyputer notebooks code quality: ..."

# Check there are some ipynb in this repo
IPYNBLIST=$(find ./.. -not -path '*/\.ipynb_checkpoints*' -type f -name "*.ipynb")
# Exit cleanly if no ipynb are found
if test -z "$IPYNBLIST";
then
    echo "No iPython Notebook file found!!";
    exit 0;
fi

# create a tmp subdirectory if it doesn't exist
mkdir -p tmp

# Convert each ipynb in a python source file (Make sure filenames with spaces do not cause issues)
OIFS="$IFS"
IFS=$'
'
for IPYNBFULLPATH in $(find ./.. -not -path '*/\.ipynb_checkpoints*' -type f -name "*.ipynb")
do
    echo "  --> Converting $IPYNBFULLPATH ..."
    IPYNBBASENAME=$(basename "$IPYNBFULLPATH")
    jupyter nbconvert --output-dir='./tmp' --to script $IPYNBFULLPATH
done
IFS="$OIFS"

# Lint all .py files created
flake8 ./tmp

