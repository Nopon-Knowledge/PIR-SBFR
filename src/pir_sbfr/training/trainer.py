"""Ultralytics trainer extended with paper paired-view optimization."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import RANK, yaml_save

from pir_sbfr.data.degradations import PairedDegradationGenerator
from pir_sbfr.models.detector import PIRSBFRModel


def paper_train_overrides(data: str, seed: int = 2023, **extra) -> Dict:
    """Return the reported schedule plus explicitly pinned framework defaults."""
    project_root = Path(__file__).resolve().parents[3]
    overrides = {
        "model": str(project_root / "configs" / "pir_sbfr_model.yaml"),
        "data": str(data),
        "task": "detect",
        "epochs": 200,
        "batch": 16,
        "imgsz": 640,
        "optimizer": "SGD",
        "lr0": 0.005,
        # The PDF gives only the initial LR. These are pinned Ultralytics
        # 8.3.0 defaults so future package defaults cannot change this run.
        "lrf": 0.01,
        "cos_lr": False,
        "momentum": 0.937,
        "weight_decay": 5e-4,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        # Native YOLO11 detection-loss gains are unreported by the PDF.
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "mosaic": 1.0,
        "close_mosaic": 20,
        "hsv_h": 0.015,
        "hsv_s": 0.70,
        "hsv_v": 0.40,
        "scale": 0.50,
        "translate": 0.10,
        "fliplr": 0.50,
        "flipud": 0.0,
        "degrees": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "pretrained": False,
        "patience": 0,  # fixed 200 epochs; disable early termination
        "seed": int(seed),
        "deterministic": True,
        "nbs": 16,  # one optimizer update per 16-source batch, as stated
    }
    overrides.update(extra)
    return overrides


class PIRTrainer(DetectionTrainer):
    """DetectionTrainer that forms clean/degraded pairs after shared augmentation."""

    def __init__(self, *args, pir_config: Optional[Dict] = None, **kwargs) -> None:
        self.pir_config = deepcopy(pir_config or {})
        super().__init__(*args, **kwargs)
        degradation = self.pir_config.get("degradation", {})
        self.paired_training = bool(degradation.get("enabled", True))
        self.pair_generator = PairedDegradationGenerator(
            seed=int(self.args.seed),
            metadata_dropout=float(degradation.get("metadata_dropout", 0.25)),
            blur_range=degradation.get("blur_range", (0.15, 0.45)),
            snr_range=degradation.get("snr_range", (10.0, 28.0)),
            mode_probabilities=degradation.get("mode_probabilities", (0.35, 0.35, 0.30)),
        )
        if RANK in {-1, 0}:
            yaml_save(self.save_dir / "pir_config.yaml", self.pir_config)

    def preprocess_batch(self, batch):
        """Apply standard shared augmentation first, then create the paired view."""
        batch = super().preprocess_batch(batch)
        if not self.paired_training:
            return batch
        paths = batch.get("im_file")
        if paths is None:
            paths = [f"batch-{index}" for index in range(batch["img"].shape[0])]
        paired = self.pair_generator(batch["img"], [str(path) for path in paths], int(self.epoch))
        batch["img_degraded"] = paired.degraded
        batch["metadata"] = paired.metadata_clean
        batch["metadata_degraded"] = paired.metadata_degraded
        batch["availability"] = paired.availability
        batch["availability_degraded"] = paired.availability
        batch["degradation_mode"] = paired.modes
        return batch

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Construct PIR-SBFR rather than parsing the placeholder model YAML."""
        model_cfg = self.pir_config.get("model", {})
        model = PIRSBFRModel(
            nc=self.data["nc"],
            scale_mode=model_cfg.get("scale_mode", "dior"),
            router_config=model_cfg.get("router", {}),
            fach_experts=int(model_cfg.get("fach_experts", 3)),
            lambda_scale=float(model_cfg.get("lambda_scale", 0.1)),
            lambda_consistency=float(model_cfg.get("lambda_consistency", 0.1)),
            verbose=bool(verbose and RANK == -1),
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        validator = super().get_validator()
        self.loss_names = ("box_loss", "cls_loss", "dfl_loss", "scale_loss", "cons_loss")
        return validator
