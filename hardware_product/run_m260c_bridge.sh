#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$SCRIPT_DIR/m260c_bridge.pid"
LOG_FILE="$SCRIPT_DIR/m260c_bridge.log"

start() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "M260C bridge already running (PID: $(cat "$PID_FILE"))"
        return 0
    fi

    cd "$SCRIPT_DIR"
    nohup python3 m260c_wakeup_bridge.py "$@" > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    echo "M260C bridge started (PID: $!, log: $LOG_FILE)"
}

stop() {
    if [ ! -f "$PID_FILE" ]; then
        echo "M260C bridge is not running"
        return 0
    fi
    PID="$(cat "$PID_FILE")"
    if kill -0 "$PID" 2>/dev/null; then
        kill "$PID"
    fi
    rm -f "$PID_FILE"
    echo "M260C bridge stopped"
}

status() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "M260C bridge running (PID: $(cat "$PID_FILE"))"
    else
        echo "M260C bridge not running"
    fi
}

case "${1:-start}" in
    start)
        shift
        start "$@"
        ;;
    stop)
        stop
        ;;
    restart)
        shift
        stop
        sleep 1
        start "$@"
        ;;
    status)
        status
        ;;
    log)
        tail -n 80 "$LOG_FILE"
        ;;
    *)
        echo "Usage: bash run_m260c_bridge.sh start|stop|restart|status|log [bridge args]"
        ;;
esac
