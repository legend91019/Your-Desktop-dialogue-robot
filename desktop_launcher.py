import sys
import queue
import subprocess
import time
import urllib.request
from pathlib import Path

from runtime_paths import project_root, user_data_root
from startup_checks import find_free_port, run_all


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5000


def build_window_url(host=DEFAULT_HOST, port=DEFAULT_PORT):
    return f"http://{host}:{port}/"


def missing_pywebview_message():
    return (
        "缺少桌面窗口依赖 pywebview，请重新安装芯宝。开发者可运行：pip install pywebview"
    )


def wait_for_server(url, timeout_seconds=120, startup_errors=None, backend_process=None, backend_log=None):
    deadline = time.time() + timeout_seconds
    last_error = None
    while time.time() < deadline:
        if backend_process is not None:
            exit_code = backend_process.poll()
            if exit_code is not None:
                log_hint = f"，日志：{backend_log}" if backend_log else ""
                raise RuntimeError(f"后端进程异常退出，退出码 {exit_code}{log_hint}")
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


def start_backend_process(host, port):
    command = build_backend_command(host, port)
    log_path = user_data_root() / "logs" / "backend.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8")
    try:
        return subprocess.Popen(
            command,
            cwd=str(project_root()),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
    finally:
        log_handle.close()


def build_backend_command(host, port):
    return [sys.executable, "-m", "BackEnd.simple", "--host", host, "--port", str(port)]


def open_desktop_window(url):
    try:
        import webview
    except ImportError as exc:
        raise RuntimeError(missing_pywebview_message()) from exc

    webview.create_window("芯宝 Xinbao", url, width=1280, height=860)
    webview.start()


def main():
    host = DEFAULT_HOST
    port = find_free_port(host)
    url = build_window_url(host, port)
    startup_errors = queue.Queue()

    failed_checks = [result for result in run_all(project_root()) if not result.ok]
    if failed_checks:
        details = "\n".join(f"[{item.code}] {item.message}" for item in failed_checks)
        raise RuntimeError(f"芯宝启动检查未通过：\n{details}")

    backend_process = start_backend_process(host, port)
    try:
        wait_for_server(
            url,
            startup_errors=startup_errors,
            backend_process=backend_process,
            backend_log=user_data_root() / "logs" / "backend.log",
        )
        open_desktop_window(url)
        return 0
    finally:
        if backend_process.poll() is None:
            backend_process.terminate()
            try:
                backend_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                backend_process.kill()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)
