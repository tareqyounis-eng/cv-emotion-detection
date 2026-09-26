import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("torchvision") is not None


@unittest.skipUnless(HAS_TORCH, "Optional training dependencies are not installed")
class TrainingTests(unittest.TestCase):
    def test_stratified_split_is_reproducible_and_disjoint(self):
        from emotion_app.training import split_indices
        targets = [0] * 11 + [1] * 4 + [2] * 2
        train, val = split_indices(targets)
        self.assertEqual((train, val), split_indices(targets))
        self.assertFalse(set(train) & set(val))
        self.assertEqual(sorted(train + val), list(range(len(targets))))
        self.assertEqual(set(np.array(targets)[val]), {0, 1, 2})

    def test_validation_transform_has_no_random_augmentation(self):
        from emotion_app.training import CLASSES, make_datasets
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            for label in CLASSES:
                target = Path(folder) / "train" / label
                target.mkdir(parents=True)
                for i in range(3):
                    pixels = np.random.default_rng(i).integers(0, 255, (48, 48), dtype=np.uint8)
                    Image.fromarray(pixels).save(target / f"{i}.png")
            train, val, _ = make_datasets(folder)
            self.assertIsNot(train.dataset, val.dataset)
            np.testing.assert_array_equal(val[0][0].numpy(), val[0][0].numpy())
            self.assertFalse((Path(folder) / "test").exists())

    def test_checkpoint_class_order_and_smoke_guard(self):
        from emotion_app.training import CLASSES, CheckpointModel, EmotionCNN, read_checkpoint, torch
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            data = {"architecture": "emotion-cnn-v1", "classes": list(CLASSES),
                    "preprocessing": "gray48-minus1-plus1", "state_dict": EmotionCNN().state_dict(), "smoke_test": True}
            torch.save(data, path)
            model, _ = read_checkpoint(path)
            with torch.inference_mode():
                self.assertEqual(tuple(model(torch.zeros(1, 1, 48, 48)).shape), (1, 7))
            with self.assertRaisesRegex(ValueError, "smoke-test"):
                CheckpointModel(path)
            data["classes"] = list(reversed(CLASSES))
            torch.save(data, path)
            with self.assertRaisesRegex(ValueError, "class order"):
                read_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
