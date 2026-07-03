#!/bin/bash
# Split a text file into train and validation sets.
# Usage: bash split_train_val.sh <input_file> <num_train_lines> [--shuffle]
#
# Example:
#   bash split_train_val.sh h5list_showeroriginreco_ncpi0filter.txt 50000
#   bash split_train_val.sh h5list_showeroriginreco_ncpi0filter.txt 50000 --shuffle

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <input_file> <num_train_lines> [--shuffle]"
    echo "  Splits <input_file> (M lines) into two files:"
    echo "    <basename>_train.txt  (N lines)"
    echo "    <basename>_val.txt    (M-N lines)"
    exit 1
fi

INPUT_FILE="$1"
N="$2"
SHUFFLE="${3:-}"

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: file '$INPUT_FILE' not found."
    exit 1
fi

M=$(wc -l < "$INPUT_FILE")

if [ "$N" -gt "$M" ]; then
    echo "Error: requested $N train lines but file only has $M lines."
    exit 1
fi

DIR=$(dirname "$INPUT_FILE")
BASE=$(basename "$INPUT_FILE" .txt)
TRAIN_FILE="${DIR}/${BASE}_train.txt"
VAL_FILE="${DIR}/${BASE}_val.txt"

if [ "$SHUFFLE" = "--shuffle" ]; then
    echo "Shuffling input before splitting..."
    TMPFILE=$(mktemp)
    shuf "$INPUT_FILE" > "$TMPFILE"
    head -n "$N" "$TMPFILE" > "$TRAIN_FILE"
    tail -n +"$((N + 1))" "$TMPFILE" > "$VAL_FILE"
    rm "$TMPFILE"
else
    head -n "$N" "$INPUT_FILE" > "$TRAIN_FILE"
    tail -n +"$((N + 1))" "$INPUT_FILE" > "$VAL_FILE"
fi

TRAIN_COUNT=$(wc -l < "$TRAIN_FILE")
VAL_COUNT=$(wc -l < "$VAL_FILE")

echo "Done."
echo "  Train: $TRAIN_FILE ($TRAIN_COUNT lines)"
echo "  Val:   $VAL_FILE ($VAL_COUNT lines)"
