# Xinbao v1.0.7 Release Notes

v1.0.7 是芯宝 Windows 桌面安装版的首个品牌化发布版本。

## 用户可见变化

- 提供内置 CPython 3.11、项目私有运行环境、CUDA PyTorch、本地 RAG 模型和路由分类器的桌面安装器。
- 启动器在打开窗口前检查 NVIDIA GPU、显存、模型清单和后端健康状态。
- 配置、长期记忆、上传文件、语音缓存和日志写入 `%APPDATA%\Xinbao\`，升级不会覆盖用户数据。
- README、使用说明和安装器使用统一的芯宝 logo。
- DeepSeek API Key 继续由用户在前端设置，发布包不包含密钥。

## 下载说明

完整安装器约 3.55 GiB，超过 GitHub Release 单个资产的 2 GiB 限制。GitHub Release 仅承载源代码、发布说明、logo 和 SHA-256 校验信息；完整安装器由维护者通过夸克网盘提供。

我用夸克网盘给你分享了「Xinbao」，点击链接或复制整段内容，打开「夸克APP」即可获取。

- 口令：`/~c1e73aaEi4~:/`
- 下载链接：[https://pan.quark.cn/s/4a4913e45811?pwd=tpCL](https://pan.quark.cn/s/4a4913e45811?pwd=tpCL)
- 提取码：`tpCL`
- 安装文件：`Xinbao-Setup-v1.0.7.exe`

安装器 SHA-256（发布构建）：

```text
08cd6fa7b9259ec0aaf0181a5f830acae998b527d9f62e777254f43c79862d69  Xinbao-Setup-v1.0.7.exe
```

## 系统边界

- Windows 10/11 64 位
- NVIDIA GPU，建议显存不低于 4 GB
- 支持 CUDA 12.x 的 NVIDIA 驱动
- DeepSeek 对话和 edge-tts 语音需要联网
- `hardware_product` 硬件版本不在本次桌面发布范围内
