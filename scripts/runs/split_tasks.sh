#!/bin/bash -l

# Usage: ./split_tasks.sh <file_number> <lines_per_file>
# Example: ./split_tasks.sh 0 10
#   -> splits tasks_0.txt into tasks_0X01.txt, tasks_0X02.txt, ...

if [ $# -ne 2 ]; then
    echo "Usage: $0 <file_number> <lines_per_file>"
    exit 1
fi

file_number=$1
lines_per_file=$2
input_file="tasks_${file_number}.txt"

if [ ! -f "$input_file" ]; then
    echo "Error: $input_file not found"
    exit 1
fi

awk -v n="$lines_per_file" -v fnum="$file_number" '{
    file = sprintf("tasks_%dX%02d.txt", fnum, int((NR-1)/n)+1)
    print > file
}' "$input_file"

echo "✅ Split $input_file into chunks of $lines_per_file lines"
