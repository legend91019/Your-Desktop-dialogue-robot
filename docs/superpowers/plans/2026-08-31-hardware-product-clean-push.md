# Hardware Product Clean Repository Push Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a clean, source-only `hardware_product` implementation for Orange Pi AI Pro 20T without model weights, runtime caches, personal memories, secrets, or a GitHub Release.

**Architecture:** Sync only the verified board runtime additions from the local board checkout into the upload repository. Keep deployment and model-download scripts plus enclosure source/assets, sanitize tracked configuration and dynamic routing data, and strengthen ignore rules for board-generated artifacts.

**Tech Stack:** Python, Bash, JSON, Markdown, Git.

## Global Constraints

- Keep `hardware_product` in the repository but out of GitHub Release assets.
- Do not commit model weights, virtual environments, caches, logs, audio, databases, or generated user data.
- Keep `setup_pi.sh` and `requirements_pi.txt` so users can create the board environment and download models themselves.
- Keep DeepSeek and Baidu credentials empty or environment-based.
- Keep `hardware_product` hardware-only; do not alter the desktop release payload.

### Task 1: Sync Board Runtime Sources

**Files:**
- Create: `hardware_product/m260c_voice_loop.py`
- Create: `hardware_product/run_voice_loop.sh`
- Create: `hardware_product/README_ON_BOARD.txt`
- Modify: `hardware_product/app/main.py`

- [ ] Copy the verified board voice-loop, launcher, and on-board instructions from the local board checkout.
- [ ] Carry over the tested TTS text sanitization change in `app/main.py` without copying dynamic test data.

### Task 2: Sanitize Public Board Configuration

**Files:**
- Modify: `hardware_product/config.json`
- Modify: `hardware_product/dynamic_keywords.txt`

- [ ] Keep only neutral default user settings and an empty `deepseek_api_key`.
- [ ] Remove names, conversation-derived keywords, and other personal test residue; retain only stable product routing terms.

### Task 3: Exclude Generated Board Artifacts

**Files:**
- Modify: `.gitignore`

- [ ] Ignore board `__pycache__`, `venv`, runtime data, audio, logs, databases, local ASR env files, and downloaded `hardware_product/models/`.
- [ ] Preserve model download scripts and documentation in source control.

### Task 4: Verify and Push Repository

- [ ] Compile all board Python sources.
- [ ] Scan tracked and untracked board files for credential-like values and personal test data.
- [ ] Confirm no model/cache/runtime artifacts or oversized files are staged.
- [ ] Review `git diff --check`, commit the clean board version, and push the commit to the repository branch.
- [ ] Do not create or modify a GitHub Release.
