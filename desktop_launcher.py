import sys
import queue
import threading
import time
import urllib.request


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5000


def build_window_url(host=DEFAULT_HOST, port=DEFAULT_PORT):
    return f"http://{host}:{port}/"


def missing_pywebview_message():
    return (
        "缺少桌面窗口依赖 pywebview。请先运行：pip install pywebview，"
        "之后再执行 python desktop_launcher.py。"
    )


def wait_for_server(url, timeout_seconds=120, startup_errors=None):
    deadline = time.time() + timeout_seconds
    last_error = None
    while time.time() < deadline:
        if startup_errors is not None:
            try:
                raise startup_errors.get_nowait()
            except queue.Empty:
                pass
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status < 500:
                    return True
        except Exception as exc:
            last_error = exc
            time.sleep(0.5)
    raise RuntimeError(f"后端启动超时，最后一次错误：{last_error}")


def run_backend(host=DEFAULT_HOST, port=DEFAULT_PORT, startup_errors=None):
    import BackEnd.simple as backend
    from waitress import serve

    try:
        backend.classifier, backend.retrieve_answer = backend.init_model()
        serve(backend.app, host=host, port=port, threads=4)
    except Exception as exc:
        if startup_errors is not None:
            startup_errors.put(exc)
        raise


def open_desktop_window(url):
    try:
        import webview
    except ImportError as exc:
        raise RuntimeError(missing_pywebview_message()) from exc

    webview.create_window("芯宝 Xinbao", url, width=1280, height=860)
    webview.start()


def main():
    host = DEFAULT_HOST
    port = DEFAULT_PORT
    url = build_window_url(host, port)
    startup_errors = queue.Queue()

    server_thread = threading.Thread(
        target=run_backend,
        kwargs={"host": host, "port": port, "startup_errors": startup_errors},
        daemon=True,
    )
    server_thread.start()
    wait_for_server(url, startup_errors=startup_errors)
    open_desktop_window(url)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)
