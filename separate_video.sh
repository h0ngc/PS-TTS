#!/bin/bash

# 입력으로 동영상 경로를 받음
input_video_path="$1"
output_audio_path="$2"
output_video_path="$3"

# 경로들이 모두 입력되었는지 확인
if [ -z "$input_video_path" ] || [ -z "$output_audio_path" ] || [ -z "$output_video_path" ]; then
  echo "how to use: bash separate_video.sh <input_video_path> <output_audio_path> <output_video_path>"
  exit 1
fi

python separate_vd.py -i "$input_video_path" -o "$output_audio_path" -v "$output_video_path"
echo "separate_vd done"