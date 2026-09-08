#!/bin/bash

SOURCE_DIR=$1
TARGET_DIR=$2

# 检查源目录和目标目录是否存在
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Source directory does not exist: $SOURCE_DIR"
    exit 1
fi

if [ ! -d "$TARGET_DIR" ]; then
    mkdir -p "$TARGET_DIR"
fi

# 遍历源目录中的文件
find "$SOURCE_DIR" -type f | while read -r SOURCE_FILE; do
    TARGET_FILE="$TARGET_DIR/${SOURCE_FILE#$SOURCE_DIR/}"

    # 确保目标目录存在
    mkdir -p "$(dirname "$TARGET_FILE")"

    # 如果目标文件不存在或时间戳较旧，则拷贝
    if [ ! -f "$TARGET_FILE" ] || [ "$SOURCE_FILE" -nt "$TARGET_FILE" ]; then
        cp "$SOURCE_FILE" "$TARGET_FILE"
        echo "Copied: $SOURCE_FILE -> $TARGET_FILE"
    fi
done