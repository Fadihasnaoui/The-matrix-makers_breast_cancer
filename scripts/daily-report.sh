#!/bin/bash
# Daily metrics report script
DATE=$(date +%Y-%m-%d)
REPORT_FILE="metrics-report-$DATE.txt"

echo "📈 Daily Metrics Report - $DATE" > $REPORT_FILE
echo "================================" >> $REPORT_FILE
echo "" >> $REPORT_FILE

# Get metrics from Prometheus
echo "📊 API Usage Summary:" >> $REPORT_FILE
echo "---------------------" >> $REPORT_FILE

# Total requests today
TOTAL_REQUESTS=$(curl -s "http://localhost:9090/api/v1/query?query=increase(http_requests_total[24h])" | \
    python3 -c "import sys,json; data=json.load(sys.stdin); \
    total = sum(float(r['value'][1]) for r in data['data']['result']) if data['data']['result'] else 0; \
    print(f'{total:.0f}')")
echo "Total Requests (24h): $TOTAL_REQUESTS" >> $REPORT_FILE

# Predictions today
PREDICTIONS=$(curl -s "http://localhost:9090/api/v1/query?query=increase(http_requests_total{handler=\"/predict\"}[24h])" | \
    python3 -c "import sys,json; data=json.load(sys.stdin); \
    pred = float(data['data']['result'][0]['value'][1]) if data['data']['result'] else 0; \
    print(f'{pred:.0f}')")
echo "Predictions Made: $PREDICTIONS" >> $REPORT_FILE

# Average response time
AVG_RESPONSE=$(curl -s "http://localhost:9090/api/v1/query?query=rate(http_request_duration_seconds_sum[24h])/rate(http_request_duration_seconds_count[24h])" | \
    python3 -c "import sys,json; data=json.load(sys.stdin); \
    avg = float(data['data']['result'][0]['value'][1])*1000 if data['data']['result'] else 0; \
    print(f'{avg:.1f}')")
echo "Avg Response Time: ${AVG_RESPONSE}ms" >> $REPORT_FILE

# Error rate
ERROR_RATE=$(curl -s "http://localhost:9090/api/v1/query?query=rate(http_requests_total{status=~\"5..\"}[24h])/rate(http_requests_total[24h])*100" | \
    python3 -c "import sys,json; data=json.load(sys.stdin); \
    rate = float(data['data']['result'][0]['value'][1]) if data['data']['result'] else 0; \
    print(f'{rate:.2f}')")
echo "Error Rate: ${ERROR_RATE}%" >> $REPORT_FILE

echo "" >> $REPORT_FILE
echo "💻 System Metrics:" >> $REPORT_FILE
echo "-----------------" >> $REPORT_FILE

# Peak memory usage
PEAK_MEMORY=$(curl -s "http://localhost:9090/api/v1/query_range?query=process_resident_memory_bytes/1024/1024&start=$(date -d '24 hours ago' +%s)&end=$(date +%s)&step=3600" | \
    python3 -c "import sys,json; data=json.load(sys.stdin); \
    values = [float(v[1]) for r in data['data']['result'] for v in r['values']] if data['data']['result'] else [0]; \
    print(f'{max(values):.0f}')")
echo "Peak Memory Usage: ${PEAK_MEMORY}MB" >> $REPORT_FILE

# Uptime
UPTIME=$(curl -s "http://localhost:8000/health" | \
    python3 -c "import sys,json; data=json.load(sys.stdin); \
    uptime = int(data.get('uptime_seconds', 0)); \
    hours = uptime // 3600; minutes = (uptime % 3600) // 60; \
    print(f'{hours}h {minutes}m')")
echo "API Uptime: $UPTIME" >> $REPORT_FILE

echo "" >> $REPORT_FILE
echo "✅ Report saved to: $REPORT_FILE"
