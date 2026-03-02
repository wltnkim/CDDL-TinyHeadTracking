# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
CDDL (Cross-Domain Detection Loss) Trainer.

Teacher(frozen) + Student(trainable) distillation trainer implementing the CDDL
method from "Using Cross-Domain Detection Loss to Infer Multi-Scale Information
for Improved Tiny Head Tracking" (FG2025, arXiv 2505.22677).
"""

from copy import copy

import torch

from ultralytics.models import yolo
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import CDDLDetectionModel, attempt_load_one_weight
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK


class CDDLTrainer(DetectionTrainer):
    """Teacher(frozen) + Student(trainable) CDDL trainer.

    Extends DetectionTrainer with:
    - Teacher model loading and freezing
    - Teacher forward pass injection into batch
    - CDDLDetectionModel as student model
    - Optional MultiScaleInput module

    Usage:
        ```python
        from ultralytics import YOLO
        from ultralytics.models.yolo.detect.train_cddl import CDDLTrainer

        model = YOLO('yolov8s-p2.yaml')
        model.train(
            data='CrowdHuman.yaml',
            trainer=CDDLTrainer,
            cddl_teacher_weights='runs/detect/yolov8l-p2/weights/best.pt',
            cddl_multi_scale=False,
            epochs=100,
        )
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize CDDLTrainer, extract teacher_weights from overrides."""
        overrides = overrides or {}
        self.teacher_weights = overrides.pop('cddl_teacher_weights', '')
        self.use_multi_scale = overrides.pop('cddl_multi_scale', False)
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        self.teacher_model = None

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a CDDLDetectionModel (student), with optional MSI integrated."""
        model = CDDLDetectionModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        # Integrate MSI into model so it is automatically included in EMA,
        # optimizer, and model save. Must be created in get_model() so that
        # super()._setup_train()'s build_optimizer/ModelEMA picks it up.
        if self.use_multi_scale:
            from ultralytics.nn.modules.block import MultiScaleInput
            model.ms_module = MultiScaleInput(base_size=self.args.imgsz)
            LOGGER.info(f"MultiScaleInput integrated into model "
                        f"({sum(p.numel() for p in model.ms_module.parameters()):,} params)")
        return model

    def setup_model(self):
        """Load student model and teacher model."""
        ckpt = super().setup_model()
        self._load_teacher()
        return ckpt

    def _load_teacher(self):
        """Load and freeze teacher model."""
        if not self.teacher_weights:
            raise ValueError("CDDLTrainer requires 'cddl_teacher_weights' to be set. "
                             "Provide the path to a pretrained teacher .pt file.")

        LOGGER.info(f"Loading CDDL teacher from: {self.teacher_weights}")
        teacher_model, _ = attempt_load_one_weight(self.teacher_weights)
        teacher_model = teacher_model.to(self.device).float()

        # Freeze all teacher parameters
        for p in teacher_model.parameters():
            p.requires_grad = False
        teacher_model.eval()

        self.teacher_model = teacher_model

        # Validate teacher-student compatibility
        t_detect = self.teacher_model.model[-1]
        s_detect = self.model.model[-1] if hasattr(self.model, 'model') else None
        if s_detect is not None:
            assert t_detect.nc == s_detect.nc, \
                f"Teacher nc={t_detect.nc} != Student nc={s_detect.nc}"
            assert t_detect.reg_max == s_detect.reg_max, \
                f"Teacher reg_max={t_detect.reg_max} != Student reg_max={s_detect.reg_max}"
            assert t_detect.nl == s_detect.nl, \
                f"Teacher nl={t_detect.nl} != Student nl={s_detect.nl}"
        LOGGER.info(f"CDDL teacher loaded: nc={t_detect.nc}, reg_max={t_detect.reg_max}, "
                    f"nl={t_detect.nl}, params={sum(p.numel() for p in self.teacher_model.parameters()):,}")

    def _setup_train(self, world_size):
        """Setup training, load teacher if needed.

        MSI is integrated into the model in get_model(), so it is automatically
        included in super()._setup_train()'s build_optimizer and ModelEMA.
        """
        super()._setup_train(world_size)

        # Ensure teacher is loaded (handles case where model.train() bypasses setup_model)
        if self.teacher_model is None:
            self._load_teacher()

    def preprocess_batch(self, batch):
        """Preprocess batch: Asymmetric MSI -- Teacher uses original, Student applies MSI.

        Asymmetric design:
        - Teacher: forward on original image (x) -- strong model needs no augmentation
        - Student: model._predict_once() -> MSI(x) -> backbone (automatic, ms_module integrated)
        - MSI provides multi-scale information to student to help reach teacher-level performance
        - CDDL gradient backpropagates through MSI to learn useful multi-scale transforms
        """
        batch['img'] = batch['img'].to(self.device, non_blocking=True).float() / 255

        # Teacher forward: always uses original image (no MSI)
        # Student applies MSI automatically in model._predict_once
        with torch.no_grad():
            teacher_detect = self.teacher_model.model[-1]
            teacher_detect.training = True
            teacher_preds = self.teacher_model._predict_once(batch['img'])
            teacher_detect.training = False
        batch['teacher_preds'] = teacher_preds

        return batch

    def get_validator(self):
        """Returns validator with standard + KD loss names."""
        self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss', 'kd_box', 'kd_cls', 'kd_dfl'
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def label_loss_items(self, loss_items=None, prefix='train'):
        """Return a loss dict with labelled training loss items tensor.

        Training returns 6 items (det 3 + kd 3), validation returns 3 items (det only).
        """
        keys = [f'{prefix}/{x}' for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]
            # Validation returns only 3 items (standard det loss) vs training 6 items
            if len(loss_items) < len(keys):
                # Pad with 0 for missing KD losses during validation
                loss_items = loss_items + [0.0] * (len(keys) - len(loss_items))
            return dict(zip(keys, loss_items))
        else:
            return keys
