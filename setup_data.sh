#!/bin/bash
# Script to set up data file from Windows

WINDOWS_PATH="/mnt/c/Users/Rayen/Desktop/data_enriched.csv"
LOCAL_PATH="data.csv"

echo "Looking for data file at: $WINDOWS_PATH"

if [ -f "$WINDOWS_PATH" ]; then
    echo "✅ Found Windows data file!"
    echo "Copying to local directory..."
    cp "$WINDOWS_PATH" "$LOCAL_PATH"
    
    # Check if copy was successful
    if [ -f "$LOCAL_PATH" ]; then
        echo "✅ Data file copied successfully!"
        echo "File size: $(du -h $LOCAL_PATH | cut -f1)"
        echo "First few lines:"
        head -5 "$LOCAL_PATH"
    else
        echo "❌ Failed to copy data file"
    fi
else
    echo "❌ Data file not found at Windows path"
    echo "Please check the path: $WINDOWS_PATH"
    echo ""
    echo "Alternative: Place your data_enriched.csv in the current directory"
    echo "Or run: cp /path/to/your/data_enriched.csv ./data.csv"
fi
