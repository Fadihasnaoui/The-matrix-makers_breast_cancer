#!/bin/bash
# Script to copy data from Windows to WSL

echo "📂 Copying data from Windows to WSL..."

WINDOWS_DATA="/mnt/c/Users/Rayen/Desktop/data_enriched.csv"
LOCAL_DATA="data.csv"

# Check if Windows file exists
if [ -f "$WINDOWS_DATA" ]; then
    echo "✅ Found Windows data file: $WINDOWS_DATA"
    echo "📊 File size: $(du -h "$WINDOWS_DATA" | cut -f1)"
    
    # Copy the file
    cp "$WINDOWS_DATA" "$LOCAL_DATA"
    
    if [ -f "$LOCAL_DATA" ]; then
        echo "✅ Successfully copied to: $LOCAL_DATA"
        echo "📋 First 3 lines of data:"
        head -3 "$LOCAL_DATA"
        echo ""
        echo "📊 Data file info:"
        wc -l "$LOCAL_DATA"
        echo "🎉 Data ready for training!"
    else
        echo "❌ Failed to copy data file"
    fi
else
    echo "❌ Windows data file not found at: $WINDOWS_DATA"
    echo ""
    echo "💡 Try these alternatives:"
    echo "1. Check the path is correct: $WINDOWS_DATA"
    echo "2. Manually copy your data_enriched.csv to this directory"
    echo "3. Run: cp /mnt/c/path/to/your/data_enriched.csv ./data.csv"
fi
