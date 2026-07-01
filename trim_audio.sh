#!/bin/bash

if [ -z "$1" ]; then
    echo "用法: $0 <输入音频文件>"
    exit 1
fi

INPUT="$1"
OUTPUT="${INPUT%.*}_30s.mp3"

if [ ! -f "$INPUT" ]; then
    echo "错误: 文件不存在: $INPUT"
    exit 1
fi

ffmpeg -y -i "$INPUT" \
    -af "silenceremove=start_periods=1:start_silence=0.1:start_threshold=-30dB" \
    -t 30 \
    -acodec libmp3lame -ab 192k \
    "$OUTPUT"

echo "输出: $OUTPUT"
