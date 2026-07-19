import json

import cv2
import numpy as np
import torch

import pir_sbfr.coco_inference as coco_inference
from pir_sbfr.data.degradations import DegradationCondition
from pir_sbfr.inference import acquisition_tensors, image_tensor_rgb, letterbox_rgb, restore_boxes


def test_letterbox_round_trip_boxes():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    prepared, info = letterbox_rgb(image, 640)
    assert prepared.shape == (640, 640, 3)
    original = torch.tensor([[10.0, 20.0, 100.0, 80.0]])
    network = original.clone()
    network[:, [0, 2]] = network[:, [0, 2]] * info.ratio + info.pad_x
    network[:, [1, 3]] = network[:, [1, 3]] * info.ratio + info.pad_y
    torch.testing.assert_close(restore_boxes(network, info), original)


def test_float_images_and_explicit_metadata_preserve_semantics():
    image = np.full((10, 20, 3), 0.5, dtype=np.float32)
    prepared, _ = letterbox_rgb(image, 40)
    assert prepared.dtype == np.float32
    assert float(prepared.max()) <= 1.0
    tensor = image_tensor_rgb(prepared, torch.device("cpu"))
    assert float(tensor.max()) <= 1.0
    values, mask = acquisition_tensors(1, torch.device("cpu"), metadata=(2.0, 0.3, 20.0))
    torch.testing.assert_close(values, torch.tensor([[2.0, 0.3, 20.0]]))
    torch.testing.assert_close(mask, torch.ones(1, 3))


def test_controlled_coco_inference_restores_non_square_boxes(tmp_path, monkeypatch):
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image_path = tmp_path / "scene.jpg"
    assert cv2.imwrite(str(image_path), image)
    annotation_path = tmp_path / "gt.json"
    annotation_path.write_text(
        json.dumps(
            {
                "images": [{"id": 7, "file_name": "scene.jpg", "width": 200, "height": 100}],
                "annotations": [],
                "categories": [{"id": 1, "name": "object"}],
            }
        ),
        encoding="utf-8",
    )

    def fake_infer_images(model, images, metadata, availability, **kwargs):
        del model, metadata, availability, kwargs
        assert images[0].shape == (640, 640, 3)
        # Original xyxy [10, 10, 100, 90] after gain=3.2 and top pad=160.
        detection = torch.tensor([[32.0, 192.0, 320.0, 448.0, 0.9, 0.0]])
        aux = {
            "weights": torch.tensor([[0.2, 0.3, 0.5]]),
            "rho_phy": torch.ones(1, 3),
            "scale_estimate": torch.tensor([[0.2, 0.3, 0.5]]),
        }
        return [detection], aux

    monkeypatch.setattr(coco_inference, "infer_images", fake_infer_images)
    predictions, _ = coco_inference.run_coco_inference(
        weights=None,
        annotations=annotation_path,
        images_root=tmp_path,
        output=tmp_path / "predictions.json",
        condition=DegradationCondition(),
        model=object(),
    )
    np.testing.assert_allclose(predictions[0]["bbox"], [10.0, 10.0, 90.0, 80.0], atol=1e-5)
