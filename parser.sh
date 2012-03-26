#!/bin/bash

function get_filename() {
  echo $1 | rev | cut -d. -f2- | rev;
}
corpus=GMIAS_CMU

# if no flag set, do both
if [[ $1 != "--parse" ]]; then
  for f in `ls $corpus/*.xls`; do 
    python xls2csv.0.4.py -q -p '\' -i $f -o `get_filename $f`.csv;
  done;
fi
if [[ $1 != "--csv" ]]; then
  for f in `ls $corpus/*.csv`; do
    # luminoso-ified version??
    # <utterance>#<speaker>,<topic>,<speech act>
    cat $f | awk -F'\' '{print $6 "#" $3 "," $4 "," $5;}' > `get_filename $f`.parsed;
  done;
fi

