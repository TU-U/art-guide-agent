import unittest
from unittest.mock import Mock, patch

from vl_rag_system_v1.services.vlm_image_store import consume_latest_image


class ConsumeLatestImageTests(unittest.TestCase):
    def test_returns_none_when_directory_has_no_images(self):
        with patch("vl_rag_system_v1.services.vlm_image_store.find_latest_image", return_value=None):
            image_bytes, image_path = consume_latest_image(Mock())
        self.assertIsNone(image_bytes)
        self.assertIsNone(image_path)

    def test_consumes_only_the_newest_image_once(self):
        image_dir = Mock()
        original_path = Mock()
        original_path.stem = "latest"
        original_path.suffix = ".png"
        claimed_path = Mock()
        claimed_path.read_bytes.return_value = b"newer"
        original_path.with_name.return_value = claimed_path

        with patch(
            "vl_rag_system_v1.services.vlm_image_store.find_latest_image",
            return_value=original_path,
        ):
            image_bytes, image_path = consume_latest_image(image_dir)

        self.assertEqual(image_bytes, b"newer")
        self.assertIs(image_path, original_path)
        original_path.replace.assert_called_once_with(claimed_path)
        claimed_path.read_bytes.assert_called_once_with()
        claimed_path.unlink.assert_called_once_with(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
