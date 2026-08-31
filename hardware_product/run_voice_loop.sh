#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$SCRIPT_DIR/voice_loop.pid"
LOG_FILE="$SCRIPT_DIR/voice_loop.log"

cd "$SCRIPT_DIR"

start() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "voice loop already running (PID: $(cat "$PID_FILE"))"
        return 0
    fi
    nohup python3 m260c_voice_loop.py "$@" > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    echo "voice loop started (PID: $!, log: $LOG_FILE)"
}

stop() {
    if [ ! -f "$PID_FILE" ]; then
        echo "voice loop is not running"
        return 0
    fi
    PID="$(cat "$PID_FILE")"
    if kill -0 "$PID" 2>/dev/null; then
        kill "$PID"
    fi
    rm -f "$PID_FILE"
    echo "voice loop stopped"
}

case "${1:-test}" in
    test)
        shift
        python3 m260c_voice_loop.py "$@"
        ;;
    once)
        shift
        python3 m260c_voice_loop.py --once "$@"
        ;;
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
        if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            echo "voice loop running (PID: $(cat "$PID_FILE"))"
        else
            echo "voice loop not running"
        fi
        ;;
    log)
        tail -n 100 "$LOG_FILE"
        ;;
    *)
        echo "Usage: bash run_voice_loop.sh test|once|start|stop|restart|status|log [args]"
        ;;
esac
