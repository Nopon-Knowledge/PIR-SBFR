import pytest

from pir_sbfr.data.dior import DiorImage, DiorObject, _coco_document
from pir_sbfr.evaluation import evaluate_coco, remap_paired_coco_sample


def _ground_truth(box):
    return {
        "info": {},
        "licenses": [],
        "images": [{"id": 1, "file_name": "one.jpg", "width": 640, "height": 640}],
        "categories": [{"id": 3, "name": "object"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 3,
                "bbox": list(box),
                "area": float(box[2] * box[3]),
                "iscrowd": 0,
            }
        ],
    }


def _prediction(box):
    return [{"image_id": 1, "category_id": 3, "bbox": list(box), "score": 0.99}]


def test_paper_area_protocols_accept_perfect_predictions():
    dior_metrics = evaluate_coco(_ground_truth((4, 5, 16, 16)), _prediction((4, 5, 16, 16)), "dior")
    assert dior_metrics["AP"] == pytest.approx(1.0)
    assert dior_metrics["APS"] == pytest.approx(1.0)

    aitod_metrics = evaluate_coco(
        _ground_truth((4, 5, 4, 4)),
        _prediction((4, 5, 4, 4)),
        "aitod",
        max_detections=1500,
        dior_input_size=None,
    )
    assert aitod_metrics["AP"] == pytest.approx(1.0)
    assert aitod_metrics["APVT"] == pytest.approx(1.0)


def test_paired_bootstrap_remaps_repeated_images():
    ground_truth = _ground_truth((4, 5, 16, 16))
    predictions = _prediction((4, 5, 16, 16))
    sample = remap_paired_coco_sample(ground_truth, predictions, predictions, [1, 1])
    assert [image["id"] for image in sample.ground_truth["images"]] == [1, 2]
    assert [annotation["image_id"] for annotation in sample.ground_truth["annotations"]] == [1, 2]
    assert [prediction["image_id"] for prediction in sample.predictions_a] == [1, 2]


def test_coco_crowd_flag_ignores_difficult_ground_truth():
    ground_truth = _ground_truth((4, 5, 16, 16))
    ground_truth["annotations"][0]["difficult"] = 1
    ground_truth["annotations"][0]["ignore"] = 1
    ground_truth["annotations"][0]["iscrowd"] = 1
    metrics = evaluate_coco(ground_truth, [], "dior")
    assert metrics["AP"] == -1.0


def test_dior_converter_maps_difficult_to_coco_ignore(tmp_path):
    image = DiorImage(
        image_id="one",
        source=tmp_path / "unused.jpg",
        output_name="one.jpg",
        width=640,
        height=640,
        objects=(DiorObject(category_index=0, bbox=(4.0, 5.0, 16.0, 16.0), difficult=1, truncated=0),),
    )
    annotation = _coco_document("test", [image])["annotations"][0]
    assert annotation["difficult"] == 1
    assert annotation["ignore"] == 1
    assert annotation["iscrowd"] == 1
