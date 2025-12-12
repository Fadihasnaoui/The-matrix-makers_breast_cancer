#!/bin/bash
echo "⏰ Setting up scheduled emails..."
echo ""

# Add daily report at 8:00 AM
(crontab -l 2>/dev/null | grep -v "daily-report.py"; echo "0 8 * * * cd /home/rayen/breast-cancer-app && /home/rayen/breast-cancer-app/venv/bin/python3 /home/rayen/breast-cancer-app/daily-report.py") | crontab -

echo "✅ Daily email scheduled for 8:00 AM"
echo ""
echo "📋 Current scheduled jobs:"
echo "=========================="
crontab -l
echo ""
echo "📧 Emails will be sent to: rayen.kaddechi@esprit.tn"
echo "⏰ Time: Every day at 8:00 AM"
