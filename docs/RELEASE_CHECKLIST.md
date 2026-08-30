# 芯宝桌面版发布检查清单

## 构建前

- [ ] 使用干净 Windows 构建机和 CPython 3.11.x
- [ ] 不读取或打包开发机 `pytorch_env`
- [ ] requirements、CUDA PyTorch 和模型版本已锁定
- [ ] `config.example.json` 不含真实 API Key
- [ ] 发布 payload 不含 `hardware_product`、个人报告、测试缓存、`chroma_db` 和运行音频
- [ ] `models/manifest.json` 已生成并包含 SHA-256

## 自动化验证

- [ ] `.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v`
- [ ] `py_compile` 覆盖启动器、后端、路径和构建脚本
- [ ] 安装器配置测试通过
- [ ] 发布目录包含 `Xinbao.exe`、`.venv`、`python`、模型、前端和后端

## 干净机器验收

- [ ] Windows 10/11 + 合格 NVIDIA 驱动：安装、启动、设置保存、流式聊天、RAG、语音
- [ ] 断网：应用可打开并显示可理解的网络错误，文字界面不崩溃
- [ ] 无 NVIDIA GPU：启动前显示明确不支持提示
- [ ] 驱动过低或显存不足：显示诊断和处理建议
- [ ] 端口被占用：自动选择其他回环端口
- [ ] 升级：API Key、设置和 ChromaDB 记忆保留
- [ ] 卸载：程序删除，用户数据仅在明确操作后删除
- [ ] 异常关闭：后端进程被回收，下一次可以正常启动

## 发布物

- [ ] `Xinbao-Setup-v<version>.exe`
- [ ] `.sha256` 校验文件
- [ ] `README.md` 和 `docs/USER_GUIDE.md`
- [ ] 发布说明包含系统要求、网络边界和已知限制
