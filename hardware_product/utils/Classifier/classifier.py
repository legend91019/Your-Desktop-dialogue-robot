import os
import random
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification


class TextClassifier:
    """基于BERT的文本分类器 (ARM CPU 优化版)"""

    def __init__(self, model_path, num_labels=2, device=None):
        self.model_path = model_path
        self.num_labels = num_labels
        # 香橙派 AIPro 使用 CPU 推理 (Ascend NPU 需要额外 CANN 适配)
        self.device = device or torch.device("cpu")
        self.tokenizer = None
        self.model = None

    def load_model(self):
        try:
            required_files = ["config.json", "model.safetensors", "tokenizer.json"]
            missing_files = [
                f for f in required_files
                if not os.path.exists(os.path.join(self.model_path, f))
            ]

            if missing_files:
                print(f"模型文件缺失：{missing_files}，开始下载 bert-base-chinese...")
                os.makedirs(self.model_path, exist_ok=True)

                self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
                self.model = BertForSequenceClassification.from_pretrained(
                    "bert-base-chinese", num_labels=self.num_labels
                )

                self.tokenizer.save_pretrained(self.model_path)
                self.model.save_pretrained(self.model_path, safe_serialization=True)
                print(f"模型已保存至：{self.model_path}")
            else:
                self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
                self.model = BertForSequenceClassification.from_pretrained(
                    self.model_path, num_labels=self.num_labels, use_safetensors=True
                )

            self.model.to(self.device)
            print(f"✅ 模型加载成功 from {self.model_path} (device: {self.device})")
            return True

        except Exception as e:
            print(f"❌ 模型加载失败: {str(e)}")
            return False

    def train(self, questions, labels, batch_size=4, learning_rate=2e-5, epochs=5, val_size=0.2, augmenter=None, augment_times=3):
        if self.model is None or self.tokenizer is None:
            print("❌ 请先加载模型")
            return False

        train_q, val_q, train_lbl, val_lbl = train_test_split(
            questions, labels, test_size=val_size, random_state=42
        )

        from .data_utils import prepare_data

        train_inputs, train_labels = prepare_data(
            self.tokenizer, train_q, train_lbl, augmenter, augment_times
        )
        val_inputs, val_labels = prepare_data(
            self.tokenizer, val_q, val_lbl, None, 0
        )

        train_loader = DataLoader(
            TensorDataset(train_inputs["input_ids"], train_inputs["attention_mask"], train_labels),
            batch_size=batch_size, shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(val_inputs["input_ids"], val_inputs["attention_mask"], val_labels),
            batch_size=batch_size,
        )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        print("\n🚀 开始训练...")
        self._train_model(self.model, train_loader, optimizer, epochs)

        accuracy = self._evaluate(self.model, val_loader)
        print(f"\n🎯 验证集准确率: {accuracy*100:.2f}%")

        return True

    def predict(self, texts, apply_post_processing=True):
        if self.model is None or self.tokenizer is None:
            print("❌ 请先加载模型")
            return []

        self.model.eval()
        predictions = []

        for text in texts:
            inputs = self.tokenizer(
                text, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                pred = torch.argmax(outputs.logits, dim=1).item()

            if apply_post_processing:
                pred = self._apply_post_processing(text, pred)

            predictions.append(pred)

        return predictions

    def save_model(self, save_path=None):
        if self.model is None or self.tokenizer is None:
            print("❌ 没有可保存的模型")
            return False

        save_path = save_path or self.model_path
        os.makedirs(save_path, exist_ok=True)

        try:
            self.tokenizer.save_pretrained(save_path)
            self.model.save_pretrained(save_path, safe_serialization=True)
            print(f"✅ 模型已保存至: {save_path}")
            return True
        except Exception as e:
            print(f"❌ 模型保存失败: {str(e)}")
            return False

    def _train_model(self, model, dataloader, optimizer, epochs=5):
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                batch = [item.to(self.device) for item in batch]
                optimizer.zero_grad()
                input_ids, attention_mask, labels = batch
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")

    def _evaluate(self, model, dataloader):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in dataloader:
                batch = [item.to(self.device) for item in batch]
                input_ids, attention_mask, labels = batch
                outputs = model(input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs.logits, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        return correct / total

    def _apply_post_processing(self, text, pred):
        if any(kw in text for kw in ["天气", "温度", "下雨", "气温", "最新情况", "最近", "目前"]):
            return 1
        return pred
