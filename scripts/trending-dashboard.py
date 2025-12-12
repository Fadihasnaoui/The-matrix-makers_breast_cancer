#!/usr/bin/env python3
"""
Performance Trending Dashboard
"""
import requests
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64

def get_metric_history(metric_name, hours=24):
    """Get metric history from Prometheus"""
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours)
    
    params = {
        'query': metric_name,
        'start': start_time.timestamp(),
        'end': end_time.timestamp(),
        'step': '300'  # 5 minute intervals
    }
    
    try:
        response = requests.get('http://localhost:9090/api/v1/query_range', params=params, timeout=10)
        data = response.json()
        
        if data['status'] == 'success' and data['data']['result']:
            result = data['data']['result'][0]
            timestamps = [datetime.fromtimestamp(float(point[0])) for point in result['values']]
            values = [float(point[1]) for point in result['values']]
            return timestamps, values
    except Exception as e:
        print(f"Error getting {metric_name}: {e}")
    
    return [], []

def create_trending_html():
    """Create HTML with trending graphs"""
    
    # Get various metrics
    accuracy_times, accuracy_values = get_metric_history('api_model_accuracy', 24)
    request_times, request_values = get_metric_history('rate(http_requests_total[5m])', 24)
    memory_times, memory_values = get_metric_history('process_resident_memory_bytes / 1024 / 1024', 24)
    
    # Create plots
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # Accuracy trend
    if accuracy_values:
        axes[0].plot(accuracy_times, accuracy_values, 'g-', linewidth=2)
        axes[0].set_title('📈 Model Accuracy Over Time', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].fill_between(accuracy_times, accuracy_values, alpha=0.3, color='green')
    
    # Request rate trend
    if request_values:
        axes[1].plot(request_times, request_values, 'b-', linewidth=2)
        axes[1].set_title('📊 Request Rate (requests/second)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Requests/s', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].fill_between(request_times, request_values, alpha=0.3, color='blue')
    
    # Memory usage trend
    if memory_values:
        axes[2].plot(memory_times, memory_values, 'r-', linewidth=2)
        axes[2].set_title('💾 Memory Usage Over Time', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('Memory (MB)', fontsize=12)
        axes[2].set_xlabel('Time', fontsize=12)
        axes[2].grid(True, alpha=0.3)
        axes[2].fill_between(memory_times, memory_values, alpha=0.3, color='red')
    
    plt.tight_layout()
    
    # Save plot to base64 for HTML
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    # Create HTML
    html = f"""
    <html>
    <head>
        <title>Performance Trending Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                      color: white; padding: 30px; border-radius: 10px; margin-bottom: 20px; }}
            .stats {{ display: flex; justify-content: space-between; margin: 20px 0; }}
            .stat-card {{ background: white; padding: 20px; border-radius: 10px; 
                         box-shadow: 0 4px 6px rgba(0,0,0,0.1); width: 30%; text-align: center; }}
            .stat-value {{ font-size: 32px; font-weight: bold; margin: 10px 0; }}
            .trend-chart {{ background: white; padding: 20px; border-radius: 10px; 
                           box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📈 Performance Trending Dashboard</h1>
            <p>Last 24 hours of metrics</p>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <h3>Current Accuracy</h3>
                <div class="stat-value">{accuracy_values[-1] if accuracy_values else 'N/A'}</div>
                <p>Latest model accuracy</p>
            </div>
            
            <div class="stat-card">
                <h3>Avg Request Rate</h3>
                <div class="stat-value">{np.mean(request_values) if request_values else '0:.2f'}/s</div>
                <p>Average requests per second</p>
            </div>
            
            <div class="stat-card">
                <h3>Peak Memory</h3>
                <div class="stat-value">{max(memory_values) if memory_values else '0:.1f'} MB</div>
                <p>Highest memory usage</p>
            </div>
        </div>
        
        <div class="trend-chart">
            <h2>📊 Performance Trends</h2>
            <img src="data:image/png;base64,{image_base64}" style="width:100%;">
        </div>
        
        <div style="text-align: center; color: #666; margin-top: 30px;">
            <p>Breast Cancer Detection API - Automated Performance Monitoring</p>
            <p>Refresh page to update charts | Data updates every 5 minutes</p>
        </div>
    </body>
    </html>
    """
    
    return html

if __name__ == "__main__":
    html = create_trending_html()
    
    # Save to file
    with open('trending-dashboard.html', 'w') as f:
        f.write(html)
    
    print("✅ Trending dashboard created: trending-dashboard.html")
    print("📊 Open in browser to view performance trends")
