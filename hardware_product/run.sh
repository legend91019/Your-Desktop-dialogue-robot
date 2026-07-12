#!/bin/bash
# ============================================================
# 芯宝 - 启动脚本 (后台运行)
# 使用: bash run.sh start|stop|restart|status
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$SCRIPT_DIR/xinbao.pid"
LOG_FILE="$SCRIPT_DIR/xinbao.log"

start() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "芯宝已经在运行中 (PID: $(cat "$PID_FILE"))"
        return 1
    fi

    cd "$SCRIPT_DIR"
    source venv/bin/activate
    echo "🚀 启动芯宝..."
    nohup python3 app/main.py > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    echo "✅ 芯宝已启动 (PID: $!, 日志: $LOG_FILE)"
    echo "🌐 访问 http://$(hostname -I | awk '{print $1}'):5000"
}

stop() {
    if [ ! -f "$PID_FILE" ]; then
        echo "芯宝未在运行"
        return 1
    fi
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        kill "$PID"
        rm -f "$PID_FILE"
        echo "✅ 芯宝已停止"
    else
        rm -f "$PID_FILE"
        echo "芯宝未在运行 (残留 PID 已清理)"
    fi
}

status() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "✅ 芯宝运行中 (PID: $(cat "$PID_FILE"))"
    else
        echo "❌ 芯宝未运行"
    fi
}

case "${1:-start}" in
    start)   start ;;
    stop)    stop ;;
    restart) stop; sleep 1; start ;;
    status)  status ;;
    *)       echo "用法: $0 start|stop|restart|status" ;;
esac
