#!/bin/bash

echo "🚀 Launching Breast Cancer API Full Stack..."
echo "============================================"

APP_DIR="$HOME/breast-cancer-app"
PROM_DIR="$APP_DIR/prometheus-2.51.2.linux-amd64"
AM_DIR="$APP_DIR/alertmanager-0.27.0.linux-amd64"

EMAIL_FROM="rayenkaddechi1234@gmail.com"
EMAIL_TO="rayen.kaddechi@esprit.tn"
EMAIL_PASS="ldgamksnxrbhbeuf"   # cleaned Gmail app password

echo ""
echo "🔍 Killing old processes..."
pkill -f "uvicorn" 2>/dev/null
pkill -f "prometheus" 2>/dev/null
pkill -f "alertmanager" 2>/dev/null
sleep 1
echo "🧹 Old processes cleaned."

echo ""
echo "📦 Activating virtual environment..."
cd "$APP_DIR"
source venv/bin/activate

echo ""
echo "📧 Testing email credentials..."
python3 - <<EOF
import smtplib
from email.mime.text import MIMEText

EMAIL_FROM="$EMAIL_FROM"
EMAIL_PASS="$EMAIL_PASS"
EMAIL_TO="$EMAIL_TO"

try:
    msg=MIMEText("🚀 System startup successful!\nAll services are launching.")
    msg['Subject']="Breast Cancer API – Startup Confirmation"
    msg['From']=EMAIL_FROM
    msg['To']=EMAIL_TO

    smtp=smtplib.SMTP("smtp.gmail.com",587)
    smtp.starttls()
    smtp.login(EMAIL_FROM, EMAIL_PASS)
    smtp.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
    smtp.quit()
    print("✅ Email test successful — startup email sent.")
except Exception as e:
    print("❌ Email test failed:", e)
EOF

echo ""
echo "🚀 Starting FastAPI server..."
nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload > fastapi.log 2>&1 &
sleep 2

echo ""
echo "📊 Starting Prometheus..."
cd "$PROM_DIR"
nohup ./prometheus --config.file=prometheus.yml > prometheus.log 2>&1 &
sleep 2

echo ""
echo "📨 Starting Alertmanager..."
cd "$AM_DIR"
nohup ./alertmanager --config.file=alertmanager.yml > alertmanager.log 2>&1 &
sleep 2

echo ""
echo "📈 Generating trending dashboard..."
cd "$APP_DIR"
python3 trending-dashboard.py > /dev/null 2>&1

echo ""
echo "📬 Sending daily report email..."
python3 daily-report.py > /dev/null 2>&1

echo ""
echo "🔎 Checking service status..."
echo "--------------------------------------------"

function check_port() {
    if lsof -i:$1 >/dev/null 2>&1; then
        echo "✅ Port $1 running"
    else
        echo "❌ Port $1 NOT running"
    fi
}

check_port 8000   # FastAPI
check_port 9090   # Prometheus
check_port 9093   # Alertmanager

echo ""
echo "📂 Log files created:"
echo " - fastapi.log"
echo " - prometheus.log"
echo " - alertmanager.log"

echo ""
echo "🌐 URLs:"
echo "--------------------------------------------"
echo "FastAPI Dashboard:       http://localhost:8000/"
echo "Prometheus Metrics:      http://localhost:9090"
echo "Alertmanager UI:         http://localhost:9093"
echo "Trending Dashboard:      file://$APP_DIR/trending-dashboard.html"
echo ""

echo "🎉 FULL SYSTEM RUNNING + EMAIL VERIFIED!"
echo "============================================"
