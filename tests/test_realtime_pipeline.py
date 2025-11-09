"""
Minimal smoke test for the realtime pipeline using the standard library.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

from ml_backend.realtime.config import PipelineConfig, ThingSpeakConfig
from ml_backend.realtime.model_service import ModelService
from ml_backend.realtime.pipeline import RealtimePipeline


class FakeClient:
    def __init__(self):
        self.calls = 0

    def fetch_latest(self, results=1):
        self.calls += 1
        return {
            "channel": {"id": 999, "name": "Test"},
            "feeds": [
                {
                    "entry_id": self.calls,
                    "created_at": "2024-10-05T10:00:00Z",
                    "field1": "24.0",
                    "field2": "7.2",
                    "field3": "0.7",
                    "field4": "4.0",
                    "field5": "",
                }
            ],
        }


@unittest.skipUnless(Path("ml_backend/common/do_imputer.pkl").exists(), "DO imputer artifact required.")
class RealtimePipelineTest(unittest.TestCase):
    def test_pipeline_with_fake_client(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "latest.json"
            os.environ["REALTIME_OUTPUT_PATH"] = str(output_path)

            config = PipelineConfig(thingspeak=ThingSpeakConfig(channel_id="TEST"))
            config.validate()

            pipeline = RealtimePipeline(
                config=config,
                model_service=ModelService(),
                thingspeak_client=FakeClient(),
            )

            result = pipeline.run_once()
            self.assertIsNotNone(result)
            self.assertTrue(output_path.exists())

            data = json.loads(output_path.read_text())
            self.assertIn("wqi", data)
            self.assertIn("contamination", data)
            self.assertIn("disease", data)


if __name__ == "__main__":
    unittest.main()

