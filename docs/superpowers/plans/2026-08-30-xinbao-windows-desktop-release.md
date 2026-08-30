# 芯宝 Windows 桌面安装版实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `IntelliChat-Platform-upload` 交付为 Windows 10/11 安装程序，用户无需安装 Python、Conda、pip 或手工脚本即可启动芯宝。

**Architecture:** 构建机使用官方 CPython 3.11 创建并验证项目私有 `.venv`，把该运行时、代码和本地模型封装进 one-folder 目录，再由 Inno Setup 生成安装程序。启动器负责 GPU/模型检查、动态端口、后端生命周期和 pywebview 窗口；可变配置与记忆写入 `%APPDATA%\Xinbao`。

**Tech Stack:** Python 3.11, Flask, Waitress, pywebview, ChromaDB, sentence-transformers, CUDA PyTorch, PowerShell, Inno Setup, unittest。

## Global Constraints

- 目标平台：Windows 10/11。
- 硬件要求：NVIDIA GPU，建议显存不低于 4 GB；系统安装支持 CUDA 12.x 的 NVIDIA 驱动。
- 用户不安装 Conda、Python、pip，不运行依赖或模型下载脚本。
- DeepSeek API Key 由用户在前端设置页自行填写；联网仅用于 DeepSeek 和 edge-tts。
- `hardware_product`、跨平台支持、Index-TTS 默认集成、Qt 重写、云端账号和自动更新不在本计划范围内。
- 安装目录默认 `C:\Program Files\Xinbao\`；用户数据目录 `%APPDATA%\Xinbao\`。
- 不把 API Key 写入日志；发布配置模板必须为空密钥。
- 保留当前 Flask/HTML 业务接口，前端不得写死 `127.0.0.1:5000`。

---

## 文件地图

- Create: `runtime_paths.py`：统一解析安装目录、资源目录和 `%APPDATA%\Xinbao` 可写目录。
- Create: `startup_checks.py`：GPU、模型清单和端口检查，返回结构化诊断结果。
- Modify: `BackEnd/simple.py`：从运行时路径加载配置、知识库和 ChromaDB，增加健康检查接口。
- Modify: `desktop_launcher.py`：启动前检查、动态端口、子进程回收、pywebview 地址传递。
- Modify: `FrontEnd/robot.html`：所有 API/SSE/音频地址改为相对路径。
- Create: `tools/build_runtime.ps1`：在干净构建机创建 CPython `.venv`、安装依赖、准备模型和校验清单。
- Create: `packaging/Xinbao.iss`：Inno Setup 安装、快捷方式、升级和卸载配置。
- Create: `tools/build_installer.ps1`：调用运行时构建、生成 one-folder 产物、调用 ISCC、输出 SHA-256。
- Modify: `config.example.json`：仅保留发布默认值和空 API Key。
- Create/Modify: `tests/test_runtime_paths.py`, `tests/test_startup_checks.py`, `tests/test_desktop_launcher.py`, `tests/test_frontend_urls.py`, `tests/test_release_build.py`。
- Modify: `README.md`、Create: `docs/USER_GUIDE.md`, `docs/RELEASE_CHECKLIST.md`。

### Task 1: 运行时路径与用户数据目录

**Files:**
- Create: `runtime_paths.py`
- Modify: `BackEnd/simple.py`（配置、知识库、ChromaDB、上传和音频目录初始化处）
- Modify: `BackEnd/memory_admin.py`（默认数据库路径参数）
- Test: `tests/test_runtime_paths.py`

**Interfaces:**
- `runtime_paths.project_root() -> Path`
- `runtime_paths.user_data_root() -> Path`
- `runtime_paths.config_path() -> Path`
- `runtime_paths.resolve_resource(value: str | Path) -> Path`
- `runtime_paths.ensure_user_dirs() -> dict[str, Path]`

- [ ] **Step 1: Write failing tests**

```python
def test_user_data_root_uses_appdata(monkeypatch, tmp_path):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    import runtime_paths
    assert runtime_paths.user_data_root() == tmp_path / "Xinbao"

def test_relative_resource_resolves_from_project_root():
    import runtime_paths
    assert runtime_paths.resolve_resource("knowledge.md").name == "knowledge.md"
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m unittest tests.test_runtime_paths -v`  
Expected: FAIL because `runtime_paths` does not exist.

- [ ] **Step 3: Implement minimal path module**

Implement the five functions above. `user_data_root()` must use `%APPDATA%` and fall back to `Path.home() / "AppData" / "Roaming"` only when `APPDATA` is absent. `resolve_resource()` must reject path traversal outside `project_root()` for relative resources. `ensure_user_dirs()` must create `chroma_db`, `uploads`, `static/audio`, and `logs`.

- [ ] **Step 4: Rewire backend paths**

Replace `BackEnd/simple.py` relative `config.json`, `uploads`, `static`, and ChromaDB path construction with `runtime_paths` calls. Copy `config.example.json` to the user config path on first start; never mutate the installed template.

- [ ] **Step 5: Run focused tests**

Run: `python -m unittest tests.test_runtime_paths tests.test_memory_admin -v`  
Expected: PASS.

- [ ] **Step 6: Commit**

```text
git add runtime_paths.py BackEnd/simple.py BackEnd/memory_admin.py tests/test_runtime_paths.py
git commit -m "feat: isolate Xinbao runtime and user data paths"
```

### Task 2: 启动器、健康检查和 GPU 诊断

**Files:**
- Create: `startup_checks.py`
- Modify: `BackEnd/simple.py`
- Modify: `desktop_launcher.py`
- Test: `tests/test_startup_checks.py`, `tests/test_desktop_launcher.py`

**Interfaces:**
- `startup_checks.check_gpu(min_vram_mb: int = 4096) -> CheckResult`
- `startup_checks.check_models(root: Path) -> CheckResult`
- `startup_checks.find_free_port(host: str = "127.0.0.1") -> int`
- `startup_checks.run_all(project_root: Path) -> list[CheckResult]`
- `desktop_launcher.build_window_url(host: str, port: int) -> str`
- `desktop_launcher.wait_for_server(url: str, timeout_seconds: int, startup_errors: queue.Queue) -> bool`

- [ ] **Step 1: Write failing tests**

```python
def test_find_free_port_returns_bindable_port():
    port = startup_checks.find_free_port()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))

def test_model_check_reports_missing_embedding(tmp_path):
    result = startup_checks.check_models(tmp_path)
    assert result.ok is False
    assert "embedding" in result.message
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m unittest tests.test_startup_checks -v`  
Expected: FAIL because `startup_checks` and `CheckResult` do not exist.

- [ ] **Step 3: Implement structured checks**

Create a dataclass `CheckResult(ok: bool, code: str, message: str, details: dict)`. Use `torch.cuda.is_available()` and `torch.cuda.get_device_properties(0).total_memory` for GPU checks, catching import/runtime errors and returning diagnostic results instead of raising. Model checks read `models/manifest.json` and verify required files plus SHA-256. `find_free_port()` binds port 0 and returns the assigned port.

- [ ] **Step 4: Add backend health endpoint**

Add `GET /api/health` returning HTTP 200 only after `CONFIG`, classifier and retriever initialization is complete; return JSON fields `status`, `version`, `gpu`, and `models` without secrets.

- [ ] **Step 5: Update launcher lifecycle**

Keep backend in a child `multiprocessing.Process` or `subprocess.Popen` with an explicit `--host 127.0.0.1 --port <free-port>`. Poll `/api/health`, pass the URL to pywebview, and terminate/join the child in a `finally` block. Convert failed checks into a Chinese message-box/console error with code and remediation.

- [ ] **Step 6: Run focused tests**

Run: `python -m unittest tests.test_startup_checks tests.test_desktop_launcher -v`  
Expected: PASS.

- [ ] **Step 7: Commit**

```text
git add startup_checks.py BackEnd/simple.py desktop_launcher.py tests/test_startup_checks.py tests/test_desktop_launcher.py
git commit -m "feat: add desktop startup diagnostics and health checks"
```

### Task 3: 前端 API 地址解耦与配置提示

**Files:**
- Modify: `FrontEnd/robot.html`
- Test: `tests/test_frontend_urls.py`

- [ ] **Step 1: Write failing static-analysis tests**

```python
def test_frontend_has_no_hardcoded_backend_origin():
    html = Path("FrontEnd/robot.html").read_text(encoding="utf-8")
    assert "http://127.0.0.1:5000" not in html

def test_frontend_uses_relative_api_paths():
    html = Path("FrontEnd/robot.html").read_text(encoding="utf-8")
    assert "fetch('/api/chat'" in html or "fetch(`/api/chat`" in html
```

- [ ] **Step 2: Run and verify failure**

Run: `python -m unittest tests.test_frontend_urls -v`  
Expected: FAIL because current HTML contains absolute port-5000 URLs.

- [ ] **Step 3: Replace URL construction**

Change memories, settings, chat/SSE and audio playback to `/api/...` and `/static/...` paths. Keep cache-busting query parameters. Do not alter response parsing or visual behavior.

- [ ] **Step 4: Add missing-key startup state**

On page load, call `/api/settings/status` (or extend an existing settings endpoint) and visibly focus the settings panel when `deepseek_api_key` is empty. The endpoint returns only `has_api_key: bool`; never return the key.

- [ ] **Step 5: Run tests**

Run: `python -m unittest tests.test_frontend_urls tests.test_frontend_audio_fallback -v`  
Expected: PASS.

- [ ] **Step 6: Commit**

```text
git add FrontEnd/robot.html tests/test_frontend_urls.py
git commit -m "feat: make frontend use dynamic local backend origin"
```

### Task 4: 构建 CPython + venv 运行时和模型清单

**Files:**
- Create: `tools/build_runtime.ps1`
- Create: `tools/create_model_manifest.py`
- Create: `models/manifest.json`（仅在构建产物中生成，不提交模型本体）
- Modify: `requirements.txt`
- Test: `tests/test_release_build.py`

- [ ] **Step 1: Write build contract tests**

```python
def test_build_script_uses_python_311_and_venv():
    script = Path("tools/build_runtime.ps1").read_text(encoding="utf-8")
    assert "3.11" in script and "-m venv" in script

def test_manifest_contains_required_model_keys(tmp_path):
    manifest = create_manifest(tmp_path, [tmp_path / "embedding"])
    assert manifest["files"]
```

- [ ] **Step 2: Run and verify failure**

Run: `python -m unittest tests.test_release_build -v`  
Expected: FAIL because build scripts and manifest generator do not exist.

- [ ] **Step 3: Implement deterministic runtime build**

`build_runtime.ps1` must: verify `python --version` is 3.11.x, create `.venv`, upgrade pip/setuptools/wheel, install the exact requirements and CUDA 12.8 wheels, copy source excluding tests/docs/hardware/temporary files, and fail on any command error. It must never read or package the developer `pytorch_env`.

- [ ] **Step 4: Generate and verify model manifest**

`create_model_manifest.py` walks `models/embedding`, `models/reranker`, and `assets/classifier/route_classifier.joblib`, records relative path, byte size, and SHA-256, and writes `models/manifest.json`. The runtime check in Task 2 consumes this schema.

- [ ] **Step 5: Run build-contract tests**

Run: `python -m unittest tests.test_release_build -v`  
Expected: PASS. On a build machine, also run `powershell -ExecutionPolicy Bypass -File tools/build_runtime.ps1 -Clean` and verify `.venv\Scripts\python.exe -c "import torch, chromadb, webview"` succeeds.

- [ ] **Step 6: Commit**

```text
git add tools/build_runtime.ps1 tools/create_model_manifest.py requirements.txt tests/test_release_build.py
git commit -m "build: create reproducible CPython venv runtime"
```

### Task 5: Inno Setup 安装器、升级和卸载

**Files:**
- Create: `packaging/Xinbao.iss`
- Create: `tools/build_installer.ps1`
- Create: `packaging/assets/xinbao.ico`
- Modify: `start_xinbao_desktop.bat`（仅保留开发者兼容入口）
- Test: `tests/test_installer_config.py`

- [ ] **Step 1: Write installer contract tests**

```python
def test_iss_creates_shortcuts_and_uninstall_entry():
    text = Path("packaging/Xinbao.iss").read_text(encoding="utf-8")
    assert "[Icons]" in text and "[UninstallDelete]" in text

def test_iss_does_not_install_hardware_product():
    text = Path("packaging/Xinbao.iss").read_text(encoding="utf-8")
    assert "hardware_product" not in text
```

- [ ] **Step 2: Run and verify failure**

Run: `python -m unittest tests.test_installer_config -v`  
Expected: FAIL because the Inno Setup script does not exist.

- [ ] **Step 3: Implement Inno Setup script**

Define `DefaultDirName={autopf}\Xinbao`, install the one-folder payload from `dist\Xinbao`, create desktop/start-menu shortcuts to `Xinbao.exe`, stop an existing `Xinbao.exe` before upgrade, preserve `{userappdata}\Xinbao`, and delete only installed program files on uninstall. Set `PrivilegesRequired=admin` and include version metadata.

- [ ] **Step 4: Implement installer build wrapper**

`build_installer.ps1` must clean `dist`, invoke `build_runtime.ps1`, copy the application payload, call `ISCC.exe packaging\Xinbao.iss`, and write `Xinbao-Setup-v<version>.sha256`. Fail if ISCC is missing or required payload files are absent.

- [ ] **Step 5: Run tests and local installer build**

Run: `python -m unittest tests.test_installer_config -v`  
Expected: PASS. On the packaging machine run `powershell -ExecutionPolicy Bypass -File tools/build_installer.ps1 -Version 1.0.0` and verify the `.exe` and `.sha256` files exist.

- [ ] **Step 6: Commit**

```text
git add packaging tools/build_installer.ps1 start_xinbao_desktop.bat tests/test_installer_config.py
git commit -m "build: add Xinbao Inno Setup installer"
```

### Task 6: 发布文档、完整测试和干净机器验收

**Files:**
- Modify: `README.md`
- Create: `docs/USER_GUIDE.md`
- Create: `docs/RELEASE_CHECKLIST.md`
- Modify: `docs/superpowers/specs/2026-08-30-xinbao-windows-desktop-release-design.md` only if implementation decisions materially differ
- Test: full `tests/` suite and manual acceptance records

- [ ] **Step 1: Write release checklist before manual testing**

Include exact system requirements, NVIDIA driver check, install/upgrade/uninstall cases, no-key behavior, DeepSeek network failure, edge-tts failure, port conflict, model checksum failure, abnormal close and process cleanup.

- [ ] **Step 2: Run complete automated suite**

Run: `.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v`  
Expected: all tests pass; record the command and result in the checklist.

- [ ] **Step 3: Test clean Windows machine matrix**

Install the generated installer on a Windows 10/11 machine with a supported NVIDIA driver and no Python/Conda. Verify first launch, settings save, streaming chat, RAG, audio, restart persistence and upgrade persistence. Repeat once offline and once without NVIDIA hardware to verify diagnostics.

- [ ] **Step 4: Scan release payload**

Verify no `config.json` containing a key, `pytorch_env`, `.claude`, `hardware_product`, test fixtures, personal reports, `chroma_db` runtime data or generated audio is inside the installer payload. Verify model manifest hashes after installation.

- [ ] **Step 5: Update user documentation**

README must describe the one-file install flow, NVIDIA requirement, API Key setup, network requirement for chat/voice, data location, upgrade behavior and troubleshooting. Do not instruct ordinary users to run `install.bat` or `download_models.bat`.

- [ ] **Step 6: Commit release documentation**

```text
git add README.md docs/USER_GUIDE.md docs/RELEASE_CHECKLIST.md
git commit -m "docs: document Xinbao desktop installation release"
```

## Plan Self-Review

- Spec coverage: runtime isolation (Task 1), startup/GPU/model checks (Task 2), relative frontend URLs (Task 3), CPython + venv packaging (Task 4), Inno Setup install/upgrade/uninstall (Task 5), testing and release docs (Task 6).
- Placeholder scan: no unresolved placeholder token or unspecified implementation step is used; every task includes files, interfaces, commands and expected outcomes.
- Type consistency: `CheckResult`, `find_free_port`, `run_all`, `runtime_paths` functions and manifest schema are defined before consumers.
- Scope: hardware code is excluded, while all ordinary-user desktop release requirements remain covered.
