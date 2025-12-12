#!/bin/bash
echo "🤖 Automated Monitoring Dashboard"
echo "================================="

while true; do
    clear
    
    # Get current time
    echo "🕒 $(date)"
    echo ""
    
    # Query Prometheus for metrics
    echo "📊 API PERFORMANCE"
    echo "-----------------"
    
    # Total requests
    TOTAL_REQUESTS=$(curl -s "http://localhost:9090/api/v1/query?query=http_requests_total" | \
        python3 -c "import sys,json; data=json.load(sys.stdin); \
        print(sum(float(r['value'][1]) for r in data['data']['result']))")
    echo "Total Requests: $TOTAL_REQUESTS"
    
    # Request rate (last 5 minutes)
    RATE=$(curl -s "http://localhost:9090/api/v1/query?query=rate(http_requests_total[5m])" | \
        python3 -c "import sys,json; data=json.load(sys.stdin); \
        print(sum(float(r['value'][1]) for r in data['data']['result']) if data['data']['result'] else 0)")
    echo "Request Rate: ${RATE:.2f}/s"
    
    # Predictions made
    PREDICTIONS=$(curl -s "http://localhost:9090/api/v1/query?query=http_requests_total{handler=\"/predict\"}" | \
        python3 -c "import sys,json; data=json.load(sys.stdin); \
        print(data['data']['result'][0]['value'][1] if data['data']['result'] else 0)")
    echo "Predictions Made: $PREDICTIONS"
    
    # Memory usage
    MEMORY=$(curl -s "http://localhost:9090/api/v1/query?query=process_resident_memory_bytes" | \
        python3 -c "import sys,json; data=json.load(sys.stdin); \
        print(int(float(data['data']['result'][0]['value'][1])/1024/1024) if data['data']['result'] else 0)")
    echo "Memory Usage: ${MEMORY}MB"
    
    # API Health
    HEALTH=$(curl -s "http://localhost:8000/health" | \
        python3 -c "import sys,json; data=json.load(sys.stdin); print(data.get('status', 'unknown'))")
    echo "API Status: $HEALTH"
    
    # Model accuracy
    ACCURACY=$(curl -s "http://localhost:8000/model/info" | \
        python3 -c "import sys,json; data=json.load(sys.stdin); \
        print(f\"{float(data.get('accuracy', 0))*100:.1f}%\")" 2>/dev/null || echo "N/A")
    echo "Model Accuracy: $ACCURACY"
    
    echo ""
    echo "🔄 Refreshing in 10 seconds..."
    echo "Press Ctrl+C to stop"
    
    sleep 10
done
