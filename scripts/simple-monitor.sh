#!/bin/bash
echo "📊 Simple API Monitor"
echo "===================="

while true; do
    clear
    
    echo "🕒 $(date)"
    echo ""
    
    # 1. Check API Health directly
    echo "✅ API Health Check:"
    if curl -s http://localhost:8000/health > /dev/null; then
        echo "   Status: UP ✓"
        
        # Get health details
        HEALTH_JSON=$(curl -s http://localhost:8000/health)
        PREDICTIONS=$(echo "$HEALTH_JSON" | grep -o '"predictions_made":[0-9]*' | cut -d: -f2)
        RETRAINS=$(echo "$HEALTH_JSON" | grep -o '"retrains_done":[0-9]*' | cut -d: -f2)
        UPTIME=$(echo "$HEALTH_JSON" | grep -o '"uptime_seconds":[0-9]*' | cut -d: -f2)
        
        HOURS=$((UPTIME / 3600))
        MINUTES=$(( (UPTIME % 3600) / 60 ))
        
        echo "   Predictions: ${PREDICTIONS:-0}"
        echo "   Retrains: ${RETRAINS:-0}"
        echo "   Uptime: ${HOURS}h ${MINUTES}m"
    else
        echo "   Status: DOWN ✗"
    fi
    
    echo ""
    
    # 2. Check model info
    echo "🤖 Model Status:"
    if curl -s http://localhost:8000/model/info > /dev/null; then
        MODEL_JSON=$(curl -s http://localhost:8000/model/info)
        if echo "$MODEL_JSON" | grep -q '"status":"loaded"'; then
            ACCURACY=$(echo "$MODEL_JSON" | grep -o '"accuracy":[0-9.]*' | cut -d: -f2)
            echo "   Loaded: YES"
            echo "   Accuracy: $(echo "$ACCURACY * 100" | bc -l | cut -c1-5)%"
        else
            echo "   Loaded: NO"
        fi
    else
        echo "   Cannot check"
    fi
    
    echo ""
    
    # 3. Quick metrics from /metrics endpoint
    echo "📈 Quick Metrics:"
    METRICS=$(curl -s http://localhost:8000/metrics 2>/dev/null | head -50)
    
    REQUESTS_TOTAL=$(echo "$METRICS" | grep 'http_requests_total{' | wc -l)
    echo "   Active endpoints: $REQUESTS_TOTAL"
    
    MEMORY_MB=$(echo "$METRICS" | grep 'process_resident_memory_bytes' | head -1 | awk '{print $2}' | \
        python3 -c "import sys; print(f'{int(sys.stdin.read())/1024/1024:.1f}')" 2>/dev/null || echo "N/A")
    echo "   Memory: ${MEMORY_MB}MB"
    
    echo ""
    echo "🔗 Endpoints:"
    echo "   Dashboard: http://localhost:8000"
    echo "   Prometheus: http://localhost:9090"
    echo "   API Docs: http://localhost:8000/docs"
    
    echo ""
    echo "🔄 Refreshing in 15 seconds..."
    echo "Press Ctrl+C to stop"
    
    sleep 15
done
