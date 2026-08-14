import unittest
from pathlib import Path

from utils.Classifier.route_classifier import load_route_classifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RouteClassifierTest(unittest.TestCase):
    def test_loads_packaged_classifier_and_predicts_backend_route_labels(self):
        classifier = load_route_classifier(PROJECT_ROOT / "assets" / "classifier" / "route_classifier.joblib")

        predictions = classifier.predict([
            "陪我聊聊天",
            "查看我的记忆记录",
            "帮我设置七点闹钟",
        ])

        self.assertEqual(predictions, [0, 1, 1])

    def test_predict_returns_int_labels_for_single_sentence_list(self):
        classifier = load_route_classifier(PROJECT_ROOT / "assets" / "classifier" / "route_classifier.joblib")

        predictions = classifier.predict(["你还记得我喜欢吃什么吗"])

        self.assertEqual(len(predictions), 1)
        self.assertIsInstance(predictions[0], int)


if __name__ == "__main__":
    unittest.main()
