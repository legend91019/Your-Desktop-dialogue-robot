import unittest
from pathlib import Path
from unittest.mock import patch

from BackEnd.tts_engine import (
    PROJECT_ROOT,
    generate_tts_audio,
    get_default_indextts_cfg_path,
    get_default_indextts_model_dir,
    get_tts_extension,
    _post_indextts_service,
    limit_tts_text,
    reset_indextts_cache,
    resolve_project_path,
    sanitize_tts_text,
)


class TTSEngineTest(unittest.TestCase):
    def tearDown(self):
        reset_indextts_cache()

    def test_selects_wav_for_indextts(self):
        self.assertEqual(get_tts_extension({"engine": "indextts"}), ".wav")

    def test_selects_mp3_for_edge_tts(self):
        self.assertEqual(get_tts_extension({"engine": "edge_tts"}), ".mp3")
        self.assertEqual(get_tts_extension({}), ".mp3")

    def test_sanitize_tts_text_removes_symbols_emoticons_and_parenthetical_text(self):
        text = "⚡[系统]主人早上好呀 QwQ (^_^) **今天也加油~** #tag"

        result = sanitize_tts_text(text)

        self.assertEqual(result, "主人早上好呀 今天也加油")
        self.assertNotIn("⚡", result)
        self.assertNotIn("^", result)
        self.assertNotIn("QwQ", result)

    def test_limit_tts_text_keeps_first_sentence_and_max_chars(self):
        text = "第一句应该被读出来。第二句先不要读。第三句更不要读。"

        result = limit_tts_text(text, max_chars=20, max_sentences=1)

        self.assertEqual(result, "第一句应该被读出来。")

    def test_default_indextts_paths_live_under_project_models(self):
        self.assertEqual(
            get_default_indextts_model_dir(),
            str(PROJECT_ROOT / "models" / "IndexTTS" / "checkpoints"),
        )
        self.assertEqual(
            get_default_indextts_cfg_path(),
            str(PROJECT_ROOT / "models" / "IndexTTS" / "checkpoints" / "config.yaml"),
        )

    def test_resolves_relative_model_paths_against_project_root(self):
        self.assertEqual(
            resolve_project_path("models\\IndexTTS\\checkpoints"),
            str(PROJECT_ROOT / "models" / "IndexTTS" / "checkpoints"),
        )

    @patch("BackEnd.tts_engine._run_edge_tts")
    @patch("BackEnd.tts_engine._post_indextts_service")
    def test_falls_back_to_edge_tts_when_indextts_service_fails(self, post_service, edge_tts_runner):
        post_service.side_effect = RuntimeError("service down")
        edge_tts_runner.side_effect = lambda text, output_path, config, fallback=False: {
            "engine": "edge_tts",
            "fallback": fallback,
            "output_path": output_path,
        }

        result = generate_tts_audio(
            "主人，我在呢。",
            "D:/tmp/reply.wav",
            {
                "engine": "indextts",
                "fallback_engine": "edge_tts",
                "indextts_service_url": "http://127.0.0.1:7862/tts",
                "indextts_model_dir": "models\\IndexTTS\\checkpoints",
                "indextts_cfg_path": "models\\IndexTTS\\checkpoints\\config.yaml",
                "indextts_speaker_audio": "models\\IndexTTS\\xinbao_voice.wav",
            },
        )

        self.assertEqual(result["engine"], "edge_tts")
        self.assertTrue(result["fallback"])
        self.assertEqual(Path(result["output_path"]), Path("D:/tmp/reply.mp3"))
        edge_tts_runner.assert_called_once()

    @patch("BackEnd.tts_engine._post_indextts_service")
    def test_indextts_calls_local_service_with_project_paths(self, post_service):
        post_service.return_value = {"output_path": "D:/tmp/first.wav"}

        result = generate_tts_audio(
            "第一句。",
            "D:/tmp/first.wav",
            {
                "engine": "indextts",
                "indextts_service_url": "http://127.0.0.1:7862/tts",
                "indextts_model_dir": "models\\IndexTTS\\checkpoints",
                "indextts_cfg_path": "models\\IndexTTS\\checkpoints\\config.yaml",
                "indextts_speaker_audio": "models\\IndexTTS\\xinbao_voice.wav",
                "indextts_use_fp16": False,
                "indextts_use_cuda_kernel": False,
                "indextts_use_deepspeed": False,
            },
        )

        self.assertEqual(result["engine"], "indextts")
        post_service.assert_called_once()
        self.assertEqual(post_service.call_args.args[0], "http://127.0.0.1:7862/tts")
        payload = post_service.call_args.args[1]
        self.assertEqual(payload["text"], "第一句。")
        self.assertEqual(Path(payload["output_path"]), Path("D:/tmp/first.wav"))
        self.assertEqual(Path(payload["model_dir"]), PROJECT_ROOT / "models" / "IndexTTS" / "checkpoints")
        self.assertEqual(Path(payload["cfg_path"]), PROJECT_ROOT / "models" / "IndexTTS" / "checkpoints" / "config.yaml")
        self.assertEqual(Path(payload["speaker_audio"]), PROJECT_ROOT / "models" / "IndexTTS" / "xinbao_voice.wav")
        self.assertFalse(payload["use_fp16"])
        self.assertFalse(payload["use_cuda_kernel"])
        self.assertFalse(payload["use_deepspeed"])

    @patch("BackEnd.tts_engine._import_requests")
    def test_indextts_service_error_includes_response_body(self, import_requests):
        response = import_requests.return_value.post.return_value
        response.ok = False
        response.status_code = 500
        response.reason = "Internal Server Error"
        response.text = '{"error": "speaker audio not found"}'

        with self.assertRaisesRegex(RuntimeError, "speaker audio not found"):
            _post_indextts_service("http://127.0.0.1:7862/tts", {"text": "hi"}, 3)


if __name__ == "__main__":
    unittest.main()
