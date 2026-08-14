from pathlib import Path

import joblib


class RouteClassifier:
    """Lightweight backend router with the same predict() API as the old BERT classifier."""

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def predict(self, texts, apply_post_processing=True):
        raw_predictions = self.pipeline.predict(list(texts))
        predictions = [int(label) for label in raw_predictions]
        if apply_post_processing:
            predictions = [
                self._apply_post_processing(text, pred)
                for text, pred in zip(texts, predictions)
            ]
        return predictions

    def _apply_post_processing(self, text, pred):
        direct_chat_phrases = [
            "今天天气真好",
            "下雨了好烦",
            "晴天适合出门",
            "下雨天适合睡觉",
        ]
        if any(phrase in text for phrase in direct_chat_phrases):
            return 0

        route_keywords = [
            "还记得",
            "记得我",
            "查看",
            "查询",
            "提醒",
            "闹钟",
            "待办",
            "课表",
            "打开",
            "关闭",
            "调大",
            "调小",
            "调高",
            "调低",
            "删除",
            "修改",
            "更新",
            "导出",
            "连接",
            "断开",
            "重启",
            "设备",
            "记忆记录",
            "历史记忆",
        ]
        if any(keyword in text for keyword in route_keywords):
            return 1

        return pred


def load_route_classifier(artifact_path):
    artifact = Path(artifact_path)
    if not artifact.exists():
        raise FileNotFoundError(
            f"Route classifier artifact was not found: {artifact}. "
            "Release packages should include assets/classifier/route_classifier.joblib."
        )
    return RouteClassifier(joblib.load(artifact))
