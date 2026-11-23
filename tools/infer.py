"""Reusable in-memory inference API for SurroundOcc real-camera frames."""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import load_checkpoint
from mmdet.datasets.pipelines import Compose
from mmdet3d.models import build_model


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
)

ImageInput = Union[str, Path, np.ndarray]


@dataclass(frozen=True)
class CameraCalibration:
    """Fixed camera calibration expressed in a shared LiDAR reference frame."""

    intrinsic: np.ndarray
    camera_to_lidar: np.ndarray


@dataclass
class OccupancyPrediction:
    """Dense voxel prediction and the geometry needed to interpret it."""

    occupancy: np.ndarray
    confidence: np.ndarray
    semantic: bool
    point_cloud_range: Tuple[float, float, float, float, float, float]
    class_names: Tuple[str, ...]

    def to_points(self, threshold=0.5):
        """Return occupied voxel centers as XYZ points and labels.

        For semantic models, every nonzero predicted class is occupied and
        threshold is ignored. For binary occupancy models, threshold applies
        to the occupancy probabilities.
        """
        if self.semantic:
            mask = self.occupancy != 0
            labels = self.occupancy[mask].astype(np.int32, copy=False)
        else:
            mask = self.occupancy >= threshold
            labels = np.ones(int(mask.sum()), dtype=np.int32)

        indices = np.argwhere(mask).astype(np.float32)
        lower = np.asarray(self.point_cloud_range[:3], dtype=np.float32)
        upper = np.asarray(self.point_cloud_range[3:], dtype=np.float32)
        dims = np.asarray(self.occupancy.shape, dtype=np.float32)
        points = lower + (indices + 0.5) * ((upper - lower) / dims)
        return points, labels


def _resolve_path(path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _validate_calibration(camera_name, calibration):
    if isinstance(calibration, CameraCalibration):
        intrinsic = calibration.intrinsic
        camera_to_lidar = calibration.camera_to_lidar
    else:
        intrinsic = calibration.get("intrinsic")
        camera_to_lidar = calibration.get("camera_to_lidar")
    if intrinsic is None or camera_to_lidar is None:
        raise ValueError(
            "{} calibration must define intrinsic and camera_to_lidar".format(
                camera_name))

    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    camera_to_lidar = np.asarray(camera_to_lidar, dtype=np.float64)
    if intrinsic.shape != (3, 3):
        raise ValueError(
            "{} intrinsic must be 3x3; got {}".format(
                camera_name, intrinsic.shape))
    if camera_to_lidar.shape != (4, 4):
        raise ValueError(
            "{} camera_to_lidar must be 4x4; got {}".format(
                camera_name, camera_to_lidar.shape))
    if not np.all(np.isfinite(intrinsic)) or not np.all(
            np.isfinite(camera_to_lidar)):
        raise ValueError(
            "{} calibration contains non-finite values".format(camera_name))
    if not np.allclose(camera_to_lidar[3], [0, 0, 0, 1]):
        raise ValueError(
            "{} camera_to_lidar must be a homogeneous transform".format(
                camera_name))
    if abs(np.linalg.det(camera_to_lidar[:3, :3])) < 1e-10:
        raise ValueError(
            "{} camera_to_lidar rotation is not invertible".format(
                camera_name))

    return CameraCalibration(
        intrinsic=intrinsic.copy(),
        camera_to_lidar=camera_to_lidar.copy())


def prepare_calibration(calibrations):
    """Validate fixed calibration and precompute model camera projections.

    `calibrations` maps each camera in CAMERA_ORDER to a dict with a 3x3
    `intrinsic` and a 4x4 `camera_to_lidar` transform. The latter maps camera
    coordinates into the common reference frame used by the model.
    """
    if set(calibrations) != set(CAMERA_ORDER):
        missing = sorted(set(CAMERA_ORDER) - set(calibrations))
        extra = sorted(set(calibrations) - set(CAMERA_ORDER))
        raise ValueError(
            "Provide exactly the six model cameras. Missing: {}; "
            "unexpected: {}".format(missing, extra))

    fixed_calibration = {}
    lidar_to_images = []
    camera_intrinsics = []
    lidar_to_cameras = []
    for camera_name in CAMERA_ORDER:
        calibration = _validate_calibration(
            camera_name, calibrations[camera_name])
        intrinsic4 = np.eye(4, dtype=np.float64)
        intrinsic4[:3, :3] = calibration.intrinsic
        lidar_to_camera = np.linalg.inv(calibration.camera_to_lidar)
        fixed_calibration[camera_name] = calibration
        camera_intrinsics.append(intrinsic4)
        lidar_to_cameras.append(lidar_to_camera)
        lidar_to_images.append(intrinsic4 @ lidar_to_camera)

    return {
        "calibrations": fixed_calibration,
        "cam_intrinsic": camera_intrinsics,
        "lidar2cam": lidar_to_cameras,
        "lidar2img": lidar_to_images,
    }


def load_model(config_path, checkpoint_path, device="cuda:0"):
    """Load the SurroundOcc config, plugin, checkpoint, and model once."""
    config_path = _resolve_path(config_path)
    checkpoint_path = _resolve_path(checkpoint_path)
    if not config_path.is_file():
        raise FileNotFoundError("Config does not exist: {}".format(config_path))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Checkpoint does not exist: {}".format(checkpoint_path))
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Start the container with `--gpus all`.")

    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("SurroundOcc inference requires a CUDA device")
    device_index = (
        torch.cuda.current_device() if device.index is None else device.index)
    if device_index >= torch.cuda.device_count():
        raise ValueError(
            "CUDA device {} is unavailable ({} device(s) found)".format(
                device_index, torch.cuda.device_count()))
    torch.cuda.set_device(device_index)

    cfg = Config.fromfile(str(config_path))
    if cfg.get("plugin", False):
        import projects.mmdet3d_plugin  # noqa: F401

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.is_vis = False
    cfg.data.test.test_mode = True
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, str(checkpoint_path), map_location="cpu")
    model = MMDataParallel(
        model.cuda(device_index), device_ids=[device_index])
    model.eval()
    return cfg, model


def _load_image(camera_name, image):
    if isinstance(image, (str, Path)):
        image_path = Path(image).expanduser()
        if not image_path.is_absolute():
            image_path = REPO_ROOT / image_path
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(
                "{} image could not be read: {}".format(
                    camera_name, image_path))
    if not isinstance(image, np.ndarray):
        raise TypeError(
            "{} image must be a filesystem path or BGR NumPy array".format(
                camera_name))
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(
            "{} image must be a uint8 BGR array with shape HxWx3".format(
                camera_name))
    return np.ascontiguousarray(image)


def build_preprocessing_pipeline(cfg):
    """Build the configured image transforms once for reuse across frames."""
    pipeline_config = list(cfg.data.test.pipeline)
    if not pipeline_config:
        raise ValueError("Inference config has an empty test pipeline")
    if pipeline_config[0].get("type") == "LoadMultiViewImageFromFiles":
        pipeline_config = pipeline_config[1:]
    return Compose(pipeline_config)


def prepare_frame(images, fixed_calibration, cfg, frame_id="frame",
                  pipeline=None):
    """Apply configured inference preprocessing to one in-memory camera set."""
    if set(images) != set(CAMERA_ORDER):
        missing = sorted(set(CAMERA_ORDER) - set(images))
        extra = sorted(set(images) - set(CAMERA_ORDER))
        raise ValueError(
            "Provide exactly the six model cameras. Missing: {}; "
            "unexpected: {}".format(missing, extra))
    if fixed_calibration.get("calibrations") is None:
        raise ValueError(
            "Call prepare_calibration() once before preparing frames")

    loaded_images = [
        _load_image(camera_name, images[camera_name])
        for camera_name in CAMERA_ORDER
    ]
    image_shapes = {image.shape for image in loaded_images}
    if len(image_shapes) != 1:
        raise ValueError(
            "All six views must have the same HxWxC shape; got {}".format(
                sorted(image_shapes)))

    height, width, channels = loaded_images[0].shape
    zero_mean = np.zeros(channels, dtype=np.float32)
    one_std = np.ones(channels, dtype=np.float32)
    results = dict(
        img=loaded_images,
        img_filename=list(CAMERA_ORDER),
        filename=list(CAMERA_ORDER),
        img_shape=(height, width, channels, len(CAMERA_ORDER)),
        ori_shape=(height, width, channels, len(CAMERA_ORDER)),
        pad_shape=(height, width, channels, len(CAMERA_ORDER)),
        scale_factor=1.0,
        img_norm_cfg=dict(
            mean=zero_mean, std=one_std, to_rgb=False),
        img_fields=["img"],
        bbox3d_fields=[],
        pts_mask_fields=[],
        pts_seg_fields=[],
        lidar2img=fixed_calibration["lidar2img"],
        cam_intrinsic=fixed_calibration["cam_intrinsic"],
        lidar2cam=fixed_calibration["lidar2cam"],
        occ_path="real/{}.npy".format(frame_id),
        occ_size=np.asarray(cfg.get("occ_size", [200, 200, 16])),
        pc_range=np.asarray(
            cfg.get("point_cloud_range", [-50, -50, -5, 50, 50, 3])),
    )

    if pipeline is None:
        pipeline = build_preprocessing_pipeline(cfg)
    sample = pipeline(results)
    if sample is None:
        raise RuntimeError(
            "The configured image pipeline discarded frame {}".format(
                frame_id))
    return sample


def infer_frame(model, cfg, fixed_calibration, images, frame_id="frame",
                pipeline=None):
    """Run inference for one synchronized frame and return voxel arrays.

    `images` maps camera names in CAMERA_ORDER to uint8 BGR arrays or image
    paths. Calibration is precomputed once with `prepare_calibration`.
    No input images or predictions are written to disk.
    """
    sample = prepare_frame(
        images, fixed_calibration, cfg, frame_id, pipeline=pipeline)
    batch = collate([sample], samples_per_gpu=1)
    with torch.no_grad():
        output = model(return_loss=False, rescale=True, **batch)

    predictions = output.get("occ_preds") if isinstance(output, dict) else None
    if predictions is None:
        raise RuntimeError(
            "The model did not return occupancy logits. Check that the "
            "detector supports inference without ground truth.")
    if isinstance(predictions, (list, tuple)):
        predictions = predictions[-1]
    if predictions.ndim != 5 or predictions.shape[0] != 1:
        raise RuntimeError(
            "Expected occupancy logits shaped [1, classes, X, Y, Z], got {}"
            .format(tuple(predictions.shape)))

    detector = model.module if hasattr(model, "module") else model
    point_range = tuple(float(value) for value in cfg.point_cloud_range)
    class_names = tuple(["empty"] + list(cfg.class_names))
    logits = predictions[0]
    if detector.use_semantic:
        probabilities = torch.softmax(logits, dim=0)
        confidence, occupancy = probabilities.max(dim=0)
        occupancy_array = occupancy.cpu().numpy().astype(np.uint8, copy=False)
        confidence_array = confidence.cpu().numpy().astype(np.float32, copy=False)
        semantic = True
    else:
        occupancy_probability = torch.sigmoid(logits[0])
        occupancy_array = occupancy_probability.cpu().numpy().astype(
            np.float32, copy=False)
        confidence_array = occupancy_array.copy()
        semantic = False

    return OccupancyPrediction(
        occupancy=occupancy_array,
        confidence=confidence_array,
        semantic=semantic,
        point_cloud_range=point_range,
        class_names=class_names,
    )


class SurroundOccRunner:
    """Load fixed model/calibration once and infer successive camera frames."""

    def __init__(self, checkpoint_path, calibrations, config_path=(
            "projects/configs/surroundocc/surroundocc_inference.py"),
            device="cuda:0"):
        self.cfg, self.model = load_model(
            config_path, checkpoint_path, device=device)
        self.fixed_calibration = prepare_calibration(calibrations)
        self.pipeline = build_preprocessing_pipeline(self.cfg)

    def infer(self, images, frame_id="frame"):
        return infer_frame(
            self.model,
            self.cfg,
            self.fixed_calibration,
            images,
            frame_id=frame_id,
            pipeline=self.pipeline,
        )


def _load_calibration_file(path):
    path = _resolve_path(path)
    if not path.is_file():
        raise FileNotFoundError(
            "Calibration JSON does not exist: {}".format(path))
    with path.open("r") as stream:
        calibration = json.load(stream)
    return calibration.get("cameras", calibration)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run SurroundOcc on one six-camera image set.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration", required=True,
                        help="JSON file mapping each camera to calibration.")
    parser.add_argument("--images", required=True,
                        help="Directory containing one image per camera.")
    parser.add_argument("--config", default=(
        "projects/configs/surroundocc/surroundocc_inference.py"))
    parser.add_argument("--frame-id", default="frame")
    parser.add_argument("--output", default="occupancy_prediction.npz")
    args = parser.parse_args()

    image_dir = _resolve_path(args.images)
    extensions = (".jpg", ".jpeg", ".png", ".bmp")
    images = {}
    for camera_name in CAMERA_ORDER:
        matches = [
            image_dir / (camera_name + extension)
            for extension in extensions
            if (image_dir / (camera_name + extension)).is_file()
        ]
        if len(matches) != 1:
            raise FileNotFoundError(
                "Expected one image named {} with one of {} in {}".format(
                    camera_name, extensions, image_dir))
        images[camera_name] = matches[0]

    runner = SurroundOccRunner(
        args.checkpoint,
        _load_calibration_file(args.calibration),
        config_path=args.config,
    )
    prediction = runner.infer(images, frame_id=args.frame_id)
    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(output_path),
        occupancy=prediction.occupancy,
        confidence=prediction.confidence,
        semantic=np.asarray(prediction.semantic),
        point_cloud_range=np.asarray(prediction.point_cloud_range),
        class_names=np.asarray(prediction.class_names),
    )
    print("Saved occupancy array: {}".format(output_path))
    print("Voxel grid shape: {}".format(prediction.occupancy.shape))
    print("Class names: {}".format(json.dumps(prediction.class_names)))


if __name__ == "__main__":
    main()
