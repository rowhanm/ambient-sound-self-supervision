#!/bin/bash

# fetch_clip(videoID, startTime, endTime)
fetch_clip() {
  echo "Fetching $1 ($2 to $3)..."
  outname="$1_$2"
  echo $outname
  youtube-dl https://youtube.com/watch?v=$1 --postprocessor-args "-ss $2 -to $3" \
    --output "$outname.%(ext)s"
}

grep -E '^[^#]' | while read line
do
  fetch_clip $(echo "$line" | sed -E 's/, / /g')
done
