#!/bin/bash

#================================================================================
# jupyterExports.bash - Generate Jupyter Notebook exports
#
base_directory="$(dirname "$(readlink -f "$0")")"
pushd $base_directory

SEPARATOR='--------------------------------------------------------------------------------'
INDENT='  '

echo "$SEPARATOR"
echo "-- ${0##*/} Started!"
echo ""

#-------------------------------------------------------------------------------
# Define Constants
#
# Define output format
to=pdf
# set to=html

# Define if input cells should also be integrated
input=0

# define outut directory
dir="$to"
dir+="_export"

mkdir -p $dir

# Some strage stuff for spaces to work
IFS='
'
set -f
for file in $(find $base_directory -type f -name '*.ipynb' ! -path '*.ipynb_checkpoints*'); do
  filename=$(basename -- "$file")
  extension="${filename##*.}"
  filename_withoutext="${filename%.*}"
  path=$(dirname "${file}")
  pushd $path
  echo "  * found file $file";
  if [[ "$input" -eq 1 ]]; then
    jupyter nbconvert --to $to $filename --output $base_directory/$dir/$filename_withoutext
  else
    jupyter nbconvert --to $to --no-input $filename --output $base_directory/$dir/$filename_withoutext
  fi
  popd
done

#-------------------------------------------------------------------------------
# Remove notebook generated images now integrated in the pdf
#
find $base_directory/$dir -type f -name '*.png' | xargs -r rm -v
find $base_directory/$dir -type f -name '*.svg' | xargs -r rm -v
find $base_directory/$dir -type f -name '*.jpg' | xargs -r rm -v

#-------------------------------------------------------------------------------
# Exit
#
echo ""
echo "-- ${0##*/} Finished!"
echo "$SEPARATOR"
popd

wait
