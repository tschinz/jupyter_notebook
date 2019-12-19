#!/bin/bash

#================================================================================
# svg-generate-images.sh - Create pdf from svg within ./../images folder
#
base_directory="$(dirname "$(readlink -f "$0")")"
pushd $base_directory
base_directory="$base_directory/../images"
pushd $base_directory

SEPARATOR='--------------------------------------------------------------------------------'
INDENT='  '

echo "$SEPARATOR"
echo "-- ${0##*/} Started!"
echo ""

#-------------------------------------------------------------------------------
# Remove generated and cache files
#
file_searches=(".svg")
file_output=".pdf"

echo "Search files in $base_directory"
for i in "${file_searches[@]}"
do
    for file in $(find $base_directory -type f -name *$i ! -path '*.ipynb_checkpoints*');
    do
        filename=$(basename -- "$file")
        extension="${filename##*.}"
        filename_withoutext="${filename%.*}"
        path=$(dirname "${file}")
        filename_out="$filename_withoutext$file_output"
        pushd $path
        if [[ $file_output == '.png' ]]; then
          cmd="inkscape -D -z --file=$filename --export-png=$filename_out"
        fi
        if [[ $file_output == '.pdf' ]]; then
          cmd="inkscape -D -z --file=$filename --export-pdf=$filename_out"
        fi
        echo $cmd
        eval $cmd
        popd

    done
done

#-------------------------------------------------------------------------------
# Exit
#
echo ""
echo "-- ${0##*/} Finished!"
echo "$SEPARATOR"
popd
popd