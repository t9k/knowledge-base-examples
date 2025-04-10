mkdir -p /workspace/blue /workspace/green
            
# Check if flag file exists
if [ -f /workspace/flag.txt ]; then
    CURRENT=$(cat /workspace/flag.txt)
    # Toggle between blue and green
    if [ "$CURRENT" = "blue" ]; then
    echo "green" > /workspace/flag.txt
    echo "green"
    else
    echo "blue" > /workspace/flag.txt
    echo "blue"
    fi
else
    # Default to blue for first run
    echo "blue" > /workspace/flag.txt
    echo "blue"
fi
