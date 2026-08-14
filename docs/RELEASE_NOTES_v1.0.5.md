# Xinbao v1.0.5 Release Notes

这是面向默认 `edge-tts` release 路线的安装体验修复版，重点移除普通用户的分类器训练步骤。

## 修复内容

- release 内置轻量路由分类器：

```text
assets/classifier/route_classifier.joblib
```

- `BackEnd/simple.py` 启动时加载该轻量分类器，不再加载 `models/classifier` 下约 390MB 的 BERT 分类器权重。
- `download_models.bat` 和兼容入口 `models_ready.bat` 不再运行 `train_classifier.py`，只下载基础 RAG 模型并检查内置分类器是否存在。
- 新增 `tools/build_route_classifier.py`，给开发者从 `classifier_corpus.csv` 重新生成轻量分类器使用；普通用户不需要运行。
- 旧版 `train_classifier.py` 保留，但定位改为开发者工具，不进入 release 三步启动流程。

## 说明

这次改动解决的是“用户不该自己训练分类器”。不过当前 RAG/embedding/reranker 仍基于 `sentence-transformers`，它们还会间接使用 PyTorch。因此这版不会彻底删除 PyTorch 依赖；彻底去 PyTorch 需要后续把 RAG embedding/reranker 也换成更轻的方案。
