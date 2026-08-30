# 芯宝 Windows 桌面版使用说明

## 系统要求

- Windows 10 或 Windows 11，64 位
- NVIDIA GPU，建议显存 4 GB 或以上
- 支持 CUDA 12.x 的 NVIDIA 驱动
- 首次配置 DeepSeek API Key 时需要联网
- 使用语音功能时需要联网访问 edge-tts

## 下载、安装与启动

完整安装器通过 GitHub Release 说明中的外部下载地址提供。由于安装器包含 CUDA PyTorch 和本地模型，文件约 3.55 GiB，不能作为单个 GitHub Release 附件。

1. 从项目维护者提供的外部地址下载 `Xinbao-Setup-v<version>.exe`。
2. 按 Release 中的 `.sha256` 文件校验下载结果。
3. 双击安装器并完成安装。
4. 双击桌面上的“芯宝 Xinbao”。
5. 首次打开时，在设置面板填写自己的 DeepSeek API Key，并保存主人信息。

用户不需要安装 Python、Conda、pip，也不需要运行命令或下载模型。

## 数据位置

配置、API Key、长期记忆、上传文件、生成音频和日志保存在：

```text
%APPDATA%\Xinbao\
```

升级安装不会覆盖这些数据。卸载默认只删除程序文件；删除用户数据前请使用明确的清除操作并确认。

## 常见问题

- **提示没有 NVIDIA GPU**：芯宝首版只支持 NVIDIA CUDA 环境。
- **提示驱动或显存不足**：更新 NVIDIA 驱动，或使用显存至少 4 GB 的显卡。
- **对话失败**：检查 API Key、网络连接和 DeepSeek 服务状态。
- **没有声音但文字正常**：edge-tts 需要联网；文字对话不受影响。
- **模型校验失败**：重新下载安装包并校验 SHA-256。
