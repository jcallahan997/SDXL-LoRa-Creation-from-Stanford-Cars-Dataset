#!/bin/bash

# Merge Stanford Cars train and test datasets
# Handles duplicate filenames by prefixing test images with "test_"

set -e

# Directories
TRAIN_DIR="./train"
TEST_DIR="./test"
COMBINED_DIR="./combined"

echo "=== Merging Stanford Cars Train + Test Datasets ==="
echo ""

# Check if directories exist
if [ ! -d "$TRAIN_DIR" ]; then
    echo "ERROR: Train directory not found: $TRAIN_DIR"
    exit 1
fi

if [ ! -d "$TEST_DIR" ]; then
    echo "ERROR: Test directory not found: $TEST_DIR"
    exit 1
fi

# Create combined directory
mkdir -p "$COMBINED_DIR"

echo "Step 1: Copying training images..."
cp -r "$TRAIN_DIR"/* "$COMBINED_DIR"/
train_count=$(find "$COMBINED_DIR" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | wc -l)
echo "  Copied $train_count training images"
echo ""

echo "Step 2: Merging test images..."
test_copied=0

# Iterate through test class directories
for class_dir in "$TEST_DIR"/*/ ; do
    if [ -d "$class_dir" ]; then
        class_name=$(basename "$class_dir")

        # Create class directory if it doesn't exist
        mkdir -p "$COMBINED_DIR/$class_name"

        # Copy each test image with "test_" prefix
        for file in "$class_dir"*; do
            if [ -f "$file" ]; then
                filename=$(basename "$file")

                # Check if file already exists (same name in train)
                if [ -f "$COMBINED_DIR/$class_name/$filename" ]; then
                    # Rename test file with prefix
                    cp "$file" "$COMBINED_DIR/$class_name/test_$filename"
                else
                    # No conflict, copy as-is
                    cp "$file" "$COMBINED_DIR/$class_name/$filename"
                fi

                ((test_copied++))
            fi
        done
    fi
done

echo "  Copied $test_copied test images"
echo ""

# Count final results
total_images=$(find "$COMBINED_DIR" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | wc -l)
total_classes=$(find "$COMBINED_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)

echo "=== Merge Complete! ==="
echo ""
echo "Output directory: $COMBINED_DIR"
echo "Total classes: $total_classes"
echo "Total images: $total_images"
echo "Average per class: $((total_images / total_classes))"
echo ""
echo "Next step:"
echo "  python3 pipeline_full.py \\"
echo "    --input-dir ./combined \\"
echo "    --output-base ./processed \\"
echo "    --vision ollama"
echo ""
