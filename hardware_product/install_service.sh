#!/bin/bash
# ============================================================
# 芯宝 - 安装 systemd 服务 (开机自启)
# 使用: sudo bash install_service.sh
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVICE_FILE="/etc/systemd/system/xinbao.service"

cat > /tmp/xinbao.service << EOF
[Unit]
Description=芯宝 Xinbao - 桌面陪伴机器人
After=network.target

[Service]
Type=simple
User=${SUDO_USER:-root}
WorkingDirectory=${SCRIPT_DIR}
ExecStart=${SCRIPT_DIR}/venv/bin/python3 ${SCRIPT_DIR}/app/main.py
Restart=on-failure
RestartSec=10
StandardOutput=append:${SCRIPT_DIR}/xinbao.log
StandardError=append:${SCRIPT_DIR}/xinbao.log

[Install]
WantedBy=multi-user.target
EOF

cp /tmp/xinbao.service "$SERVICE_FILE"
systemctl daemon-reload
systemctl enable xinbao.service
systemctl start xinbao.service

echo "✅ systemd 服务已安装并启动!"
echo "   查看状态: sudo systemctl status xinbao"
echo "   查看日志: tail -f ${SCRIPT_DIR}/xinbao.log"
echo "   停止服务: sudo systemctl stop xinbao"
echo "   禁用自启: sudo systemctl disable xinbao"
