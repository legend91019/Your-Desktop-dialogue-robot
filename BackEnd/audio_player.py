import platform
import subprocess
from pathlib import Path


def build_windows_player_script(audio_path):
    safe_path = str(Path(audio_path)).replace("'", "''")
    return (
        "$ErrorActionPreference = 'Stop'; "
        "$player = New-Object -ComObject WMPlayer.OCX; "
        f"$player.URL = '{safe_path}'; "
        "$player.settings.volume = 100; "
        "$player.controls.play(); "
        "$startDeadline = (Get-Date).AddSeconds(8); "
        "while ($player.playState -ne 3 -and (Get-Date) -lt $startDeadline) { "
        "Start-Sleep -Milliseconds 100 "
        "}; "
        "if ($player.playState -ne 3) { throw '播放启动超时' }; "
        "while ($player.playState -ne 1) { Start-Sleep -Milliseconds 100 }"
    )


def play_audio_file(audio_path, enabled=True):
    if not enabled:
        return {"attempted": False, "played": False, "verified": False, "method": "disabled"}

    if platform.system() != "Windows":
        return {"attempted": False, "played": False, "verified": False, "method": "unsupported"}

    script = build_windows_player_script(audio_path)
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.Popen(
        [
            "powershell",
            "-NoProfile",
            "-STA",
            "-ExecutionPolicy",
            "Bypass",
            "-WindowStyle",
            "Hidden",
            "-Command",
            script,
        ],
        creationflags=creationflags,
    )
    return {
        "attempted": True,
        "played": False,
        "verified": False,
        "method": "windows_media_player",
    }
