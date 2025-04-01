#!/bin/bash

# Create destination directory
mkdir -p slakh2100/{train,validation,test}

# Process bass data
echo "Processing bass data..."
for dir in bass_22050/{train,validation,test}/*; do
  if [ -d "$dir" ]; then
    track_name=$(basename "$dir")
    mkdir -p "slakh2100/${dir#*/}/$track_name"
    ln -f "$dir/mix.wav" "slakh2100/${dir#*/}/$track_name/bass.wav"
    echo "Processed $track_name (bass)"
  fi
done

# Process drums data
echo "Processing drums data..."
for dir in drums_22050/{train,validation,test}/*; do
  if [ -d "$dir" ]; then
    track_name=$(basename "$dir")
    mkdir -p "slakh2100/${dir#*/}/$track_name"
    ln -f "$dir/mix.wav" "slakh2100/${dir#*/}/$track_name/drums.wav"
    echo "Processed $track_name (drums)"
  fi
done

# Process guitar data
echo "Processing guitar data..."
for dir in guitar_22050/{train,validation,test}/*; do
  if [ -d "$dir" ]; then
    track_name=$(basename "$dir")
    mkdir -p "slakh2100/${dir#*/}/$track_name"
    ln -f "$dir/mix.wav" "slakh2100/${dir#*/}/$track_name/guitar.wav"
    echo "Processed $track_name (guitar)"
  fi
done

# Process piano data
echo "Processing piano data..."
for dir in piano_22050/{train,validation,test}/*; do
  if [ -d "$dir" ]; then
    track_name=$(basename "$dir")
    mkdir -p "slakh2100/${dir#*/}/$track_name"
    ln -f "$dir/mix.wav" "slakh2100/${dir#*/}/$track_name/piano.wav"
    echo "Processed $track_name (piano)"
  fi
done

echo "Conversion completed! Data is now available in slakh2100/ directory" 