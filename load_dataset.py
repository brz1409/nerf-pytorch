"""
Dataset loader for Vanilla/NeRF that auto-converts Agisoft Metashape cameras XML
to a Nerfstudio-style transforms.json and then loads it.

- Auto-detects cameras.xml (Metashape) and converts to transforms.json.
- Loads images, poses, and intrinsics from transforms.json.
- Keeps behavior close to Nerfstudio where reasonable, but simplified (no lens distortion handling, etc.).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import imageio.v2 as imageio
import torch
import xml.etree.ElementTree as ET
import argparse

logger = logging.getLogger(__name__)


# =========================
# Nerfstudio-style loader
# =========================

@dataclass
class LoadedTransforms:
    data_dir: Path
    transforms_path: Path
    meta: Dict[str, Any]
    image_filenames: List[Path]
    poses: np.ndarray  # (N,4,4)
    intrinsics: Dict[str, Any]


def _load_json(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_meta_path(data: Path) -> Tuple[Path, Path]:
    """Return (data_dir, transforms.json path). Accepts either a directory or a path to transforms.json."""
    if data.is_dir():
        tr = data / "transforms.json"
        if not tr.exists():
            raise FileNotFoundError(f"Directory '{data}' does not contain transforms.json")
        return data, tr
    else:
        if data.suffix.lower() != ".json":
            raise ValueError(f"When providing a file, it must be a .json file, got: {data}")
        return data.parent, data


def _resolve_path(data_dir: Path, rel_or_abs: str) -> Path:
    p = Path(rel_or_abs.replace("./", ""))
    if p.is_absolute():
        return p
    return (data_dir / p).resolve()


def _filter_by_split(meta: Dict[str, Any], data_dir: Path, image_paths: List[Path], split: str) -> List[int]:
    if split == "all":
        return list(range(len(image_paths)))

    split_key = f"{split}_filenames"
    if split_key not in meta:
        return list(range(len(image_paths)))

    split_filenames = {_resolve_path(data_dir, x) for x in meta[split_key]}
    indices = [i for i, p in enumerate(image_paths) if p in split_filenames]
    return indices


def load_transforms(data: Path, split: str = "all") -> LoadedTransforms:
    data_dir, transforms_path = _resolve_meta_path(data)
    meta = _load_json(transforms_path)

    frames = meta.get("frames", [])
    if not frames:
        raise ValueError(f"No frames found in {transforms_path}")

    # resolve paths
    image_filenames = [_resolve_path(data_dir, f["file_path"]) for f in frames if "file_path" in f]
    indices = _filter_by_split(meta, data_dir, image_filenames, split)
    image_filenames = [image_filenames[i] for i in indices]

    # poses
    poses = []
    for i in indices:
        tm = np.array(frames[i]["transform_matrix"], dtype=np.float32)
        if tm.shape != (4, 4):
            raise ValueError(f"Frame {i} has invalid transform_matrix shape: {tm.shape}")
        poses.append(tm)
    poses = np.stack(poses, axis=0) if poses else np.zeros((0, 4, 4), dtype=np.float32)

    # intrinsics (optional, but Nerfstudio puts constant values at the top-level)
    intrinsics: Dict[str, Any] = {}
    for k in ["fl_x", "fl_y", "cx", "cy", "h", "w", "camera_model", "distortion_params"]:
        if k in meta:
            intrinsics[k] = meta[k]

    # per-frame intrinsics (if present)
    per_frame_keys = ["fl_x", "fl_y", "cx", "cy", "h", "w", "k1", "k2", "k3", "k4", "p1", "p2"]
    per_frame_intrinsics: Dict[str, List[Any]] = {k: [] for k in per_frame_keys}
    any_per_frame = False
    for i in indices:
        f = frames[i]
        found_any = False
        for k in per_frame_keys:
            if k in f:
                per_frame_intrinsics[k].append(f[k])
                found_any = True
        any_per_frame = any_per_frame or found_any
    if any_per_frame:
        intrinsics["per_frame"] = {k: v for k, v in per_frame_intrinsics.items() if len(v) > 0}

    return LoadedTransforms(
        data_dir=data_dir,
        transforms_path=transforms_path,
        meta=meta,
        image_filenames=image_filenames,
        poses=poses,
        intrinsics=intrinsics,
    )


# =========================
# PCA alignment utilities
# =========================

def _pca_align_c2w(poses_4x4: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Align camera centers via PCA to the origin and canonical axes.

    - Centers camera positions at their mean.
    - Rotates world such that PCA axes map to (x, y, z), with sign flips so the
      average camera forward roughly points along -Z and average up along +Y.
    Returns transformed c2w matrices and metadata with the applied transform.
    """
    assert poses_4x4.ndim == 3 and poses_4x4.shape[1:] == (4, 4)
    centers = poses_4x4[:, :3, 3]
    mean = centers.mean(axis=0)
    centered = centers - mean
    # SVD-based PCA (rows are samples)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    E = Vt.T  # columns are principal directions
    R_pca = E.T  # rotate world into PCA basis
    # Choose signs to satisfy NeRF-ish convention
    fwd = -poses_4x4[:, :3, 2].mean(axis=0)
    upv = poses_4x4[:, :3, 1].mean(axis=0)
    f_pca = R_pca @ fwd
    u_pca = R_pca @ upv
    sz = -1.0 if f_pca[2] > 0 else 1.0  # want forward.z < 0
    sy = 1.0 if u_pca[1] >= 0 else -1.0  # want up.y >= 0
    # keep right-handed
    sx = 1.0 if np.linalg.det(R_pca) * sy * sz > 0 else -1.0
    Sgn = np.diag([sx, sy, sz])
    R_align = Sgn @ R_pca
    t_align = -R_align @ mean
    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = R_align.astype(np.float32)
    M[:3, 3] = t_align.astype(np.float32)
    poses_aligned = (M[None, ...] @ poses_4x4).astype(np.float32)

    bb_min = centers.min(axis=0).tolist()
    bb_max = centers.max(axis=0).tolist()
    centers_a = poses_aligned[:, :3, 3]
    bb_min_a = centers_a.min(axis=0).tolist()
    bb_max_a = centers_a.max(axis=0).tolist()
    meta = {
        "align_transform": M.tolist(),
        "pca_mean": mean.tolist(),
        "pca_eigvecs": E.tolist(),
        "sign_flips": [sx, sy, sz],
        "bbox_before": {"min": bb_min, "max": bb_max},
        "bbox_after": {"min": bb_min_a, "max": bb_max_a},
    }
    return poses_aligned, meta


# =========================
# Metashape XML -> transforms.json
# =========================

CAMERA_MODELS = {
    "perspective": "OPENCV",
    "fisheye": "OPENCV_FISHEYE",
    "equirectangular": "EQUIRECTANGULAR",
}


def _find_param(calib_xml: ET.Element, param_name: str) -> float:
    param = calib_xml.find(param_name)
    if param is not None and param.text is not None:
        return float(param.text)
    return 0.0


def _gather_images(images_dir: Path, verbose: bool = False) -> Dict[str, Path]:
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".exr", ".webp"}
    image_map: Dict[str, Path] = {}
    for p in images_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            stem = p.stem
            if stem in image_map and verbose:
                print(f"[warn] Duplicate image stem '{stem}' found: {p} (keeping first: {image_map[stem]})")
            image_map.setdefault(stem, p)
    if verbose:
        print(f"[info] Found {len(image_map)} images under {images_dir}")
    return image_map


def _relative_path(from_dir: Path, to_file: Path) -> str:
    try:
        return to_file.relative_to(from_dir).as_posix()
    except Exception:
        return str(to_file.resolve().as_posix())


def metashape_to_json(
    images_dir: Path,
    xml_filename: Path,
    output_dir: Path,
    copy_images: bool = False,
    ply_filename: Optional[Path] = None,  # accepted but unused here to avoid extra deps
    verbose: bool = False,
) -> List[str]:
    """Convert Agisoft Metashape Cameras XML to Nerfstudio-style transforms.json."""
    xml_tree = ET.parse(xml_filename)
    root = xml_tree.getroot()
    if len(root) == 0:
        raise ValueError("Unexpected Metashape XML format: root has no children.")
    chunk = root[0]

    sensors = chunk.find("sensors")
    if sensors is None:
        raise ValueError("No sensors found in Metashape XML")

    calibrated_sensors = [
        sensor for sensor in sensors.iter("sensor") if sensor.get("type") == "spherical" or sensor.find("calibration")
    ]
    if not calibrated_sensors:
        raise ValueError("No calibrated sensor found in Metashape XML")

    sensor_type_list = [s.get("type") for s in calibrated_sensors]
    if sensor_type_list.count(sensor_type_list[0]) != len(sensor_type_list):
        raise ValueError("All Metashape sensors do not have the same sensor type. Only one type is supported.")
    sensor_type = sensor_type_list[0]

    # Create output dict and record camera_model once (do not overwrite later)
    data: Dict[str, Any] = {}
    if sensor_type == "frame":
        data["camera_model"] = CAMERA_MODELS["perspective"]
    elif sensor_type == "fisheye":
        data["camera_model"] = CAMERA_MODELS["fisheye"]
    elif sensor_type == "spherical":
        data["camera_model"] = CAMERA_MODELS["equirectangular"]
    else:
        raise ValueError(f"Unsupported Metashape sensor type '{sensor_type}'")

    # Build sensor_dict: sensor_id -> intrinsics and image size
    sensor_dict: Dict[Optional[str], Dict[str, Any]] = {}
    for sensor in calibrated_sensors:
        s: Dict[str, Any] = {}
        resolution = sensor.find("resolution")
        if resolution is None:
            raise ValueError("Resolution not found in Metashape XML for a sensor")
        s["w"] = int(resolution.get("width"))  # type: ignore
        s["h"] = int(resolution.get("height"))  # type: ignore

        calib = sensor.find("calibration")
        if calib is None:
            if sensor_type != "spherical":
                raise ValueError("Missing calibration for non-spherical sensor")
            # Approx for equirectangular
            s["fl_x"] = s["w"] / 2.0
            s["fl_y"] = s["h"]
            s["cx"] = s["w"] / 2.0
            s["cy"] = s["h"] / 2.0
            s.update({"k1": 0.0, "k2": 0.0, "k3": 0.0, "k4": 0.0, "p1": 0.0, "p2": 0.0})
        else:
            f = calib.find("f")
            if f is None or f.text is None:
                raise ValueError("Focal length 'f' not found in Metashape XML calibration")
            s["fl_x"] = s["fl_y"] = float(f.text)
            # Metashape stores principal point offsets from image center
            s["cx"] = _find_param(calib, "cx") + s["w"] / 2.0
            s["cy"] = _find_param(calib, "cy") + s["h"] / 2.0
            s["k1"] = _find_param(calib, "k1")
            s["k2"] = _find_param(calib, "k2")
            s["k3"] = _find_param(calib, "k3")
            s["k4"] = _find_param(calib, "k4")
            s["p1"] = _find_param(calib, "p1")
            s["p2"] = _find_param(calib, "p2")

        sensor_dict[sensor.get("id")] = s

    # Optional components (global transforms applied by Metashape)
    components = chunk.find("components")
    component_dict: Dict[Optional[str], np.ndarray] = {}
    if components is not None:
        for component in components.iter("component"):
            transform = component.find("transform")
            if transform is not None:
                rotation = transform.find("rotation")
                r = np.eye(3) if rotation is None or rotation.text is None else np.array(
                    [float(x) for x in rotation.text.split()]
                ).reshape((3, 3))
                translation = transform.find("translation")
                t = np.zeros(3) if translation is None or translation.text is None else np.array(
                    [float(x) for x in translation.text.split()]
                )
                scale = transform.find("scale")
                s = 1.0 if scale is None or scale.text is None else float(scale.text)
                m = np.eye(4, dtype=float)
                m[:3, :3] = r
                m[:3, 3] = t / s
                component_dict[component.get("id")] = m

    images_map = _gather_images(images_dir, verbose=verbose)

    out_images_dir = output_dir / "images"
    if copy_images:
        out_images_dir.mkdir(parents=True, exist_ok=True)

    frames: List[Dict[str, Any]] = []
    cameras = chunk.find("cameras")
    if cameras is None:
        raise AssertionError("Cameras not found in Metashape XML")

    num_skipped = 0
    for camera in cameras.iter("camera"):
        frame: Dict[str, Any] = {}
        camera_label = camera.get("label")
        if camera_label is None:
            if verbose:
                print("[warn] Camera without label found, skipping.")
            num_skipped += 1
            continue

        # link image by exact match or stem match
        chosen_image_path: Optional[Path] = None
        if camera_label in images_map:
            chosen_image_path = images_map[camera_label]
        else:
            stem = camera_label.split(".")[0]
            chosen_image_path = images_map.get(stem, None)

        if chosen_image_path is None:
            if verbose:
                print(f"[warn] Missing image for '{camera_label}', skipping this frame.")
            num_skipped += 1
            continue

        if copy_images:
            dest = out_images_dir / chosen_image_path.name
            if not dest.exists():
                try:
                    import shutil
                    shutil.copy2(chosen_image_path, dest)
                except Exception:
                    pass
            frame["file_path"] = f"images/{dest.name}"
        else:
            frame["file_path"] = _relative_path(output_dir, chosen_image_path)

        sensor_id = camera.get("sensor_id")
        if sensor_id not in sensor_dict:
            if verbose:
                print(f"[warn] Missing sensor calibration for '{camera_label}', skipping.")
            num_skipped += 1
            continue
        frame.update(sensor_dict[sensor_id])

        cam_transform_xml = camera.find("transform")
        if cam_transform_xml is None or cam_transform_xml.text is None:
            if verbose:
                print(f"[warn] Missing transform for '{camera_label}', skipping.")
            num_skipped += 1
            continue

        transform = np.array([float(x) for x in cam_transform_xml.text.split()]).reshape((4, 4))
        component_id = camera.get("component_id")
        if component_id in component_dict:
            transform = component_dict[component_id] @ transform

        # Metashape (OpenCV): -Z forward, +X right, +Y up  → Nerfstudio/OpenGL camera-to-world
        transform = transform[[2, 0, 1, 3], :]
        transform[:, 1:3] *= -1

        frame["transform_matrix"] = transform.tolist()
        frames.append(frame)

    # Assemble output dict (preserve the camera_model set earlier)
    data["frames"] = frames

    # Add applied_transform like Nerfstudio (used for optional point cloud alignment etc.)
    applied_transform = np.eye(4)[:3, :]  # 3x4
    applied_transform = applied_transform[np.array([2, 0, 1]), :]
    data["applied_transform"] = applied_transform.tolist()

    # Identity transforms (placeholders for downstream)
    data["transform"] = {
        "orig_to_norm": np.eye(4, dtype=np.float32).tolist(),
        "norm_to_orig": np.eye(4, dtype=np.float32).tolist(),
    }

    # Put constant intrinsics at top-level if consistent across frames
    if frames:
        keys = ["w", "h", "fl_x", "fl_y", "cx", "cy"]
        consistent = {}
        for k in keys:
            vals = {f.get(k, None) for f in frames}
            vals = {v for v in vals if v is not None}
            if len(vals) == 1:
                consistent[k] = list(vals)[0]
        for k, v in consistent.items():
            data[k] = v

    # Save transforms.json
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "transforms.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    summaries: List[str] = []
    if num_skipped == 1:
        summaries.append("1 image was skipped (missing pose or image).")
    elif num_skipped > 1:
        summaries.append(f"{num_skipped} images were skipped (missing pose or image).")
    summaries.append(f"Final dataset has {len(data['frames'])} frames.")
    return summaries


# =========================
# Dataset pipeline
# =========================

def _stack_images(img_files: List[Path]) -> np.ndarray:
    """Load and stack images to RGB uint8 array of shape (N,H,W,3)."""
    imgs: List[np.ndarray] = []
    for p in img_files:
        im = imageio.imread(str(p))
        if im.ndim == 2:
            im = np.stack([im, im, im], axis=-1)
        if im.shape[-1] == 4:
            im = im[..., :3]
        imgs.append(im)
    if not imgs:
        raise FileNotFoundError("No images found for the frames listed in transforms.json")
    return np.stack(imgs, axis=0)


def _find_metashape_xml(basedir: Path) -> Optional[Path]:
    """Find a single Metashape Cameras XML in basedir. Prefer 'cameras.xml', else a unique *.xml with typical tags."""
    cand = basedir / "cameras.xml"
    if cand.exists():
        return cand
    xmls = [p for p in basedir.glob("*.xml")]
    good = []
    for p in xmls:
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            if "<sensors" in txt and "<cameras" in txt:
                good.append(p)
        except Exception:
            pass
    if len(good) == 1:
        return good[0]
    return None


def _maybe_convert_metashape_to_transforms(basedir: Path) -> None:
    """If transforms.json is missing and a Metashape XML exists, convert it."""
    tr = basedir / "transforms.json"
    if tr.exists():
        return
    xml = _find_metashape_xml(basedir)
    if xml is None:
        return
    summaries = metashape_to_json(
        images_dir=basedir,
        xml_filename=xml,
        output_dir=basedir,
        copy_images=False,
        ply_filename=None,
        verbose=True,
    )
    for s in summaries:
        print("[dataset] ", s)


def _intrinsics_from_meta(meta: Dict[str, Any], frames_meta: List[Dict[str, Any]]) -> Tuple[int, int, float]:
    """Return (H,W,focal). Use top-level keys if present, else first-frame fallback."""
    if all(k in meta for k in ("w", "h", "fl_x", "fl_y")):
        W = int(meta["w"])
        H = int(meta["h"])
        fx = float(meta["fl_x"])
        fy = float(meta["fl_y"])
        return H, W, 0.5 * (fx + fy)
    f0 = frames_meta[0]
    H = int(f0["h"])
    W = int(f0["w"])
    fx = float(f0["fl_x"])
    fy = float(f0["fl_y"])
    return H, W, 0.5 * (fx + fy)


def load_dataset(
    basedir: str,
    *,
    downsample: Optional[int] = None,
    save_downsampled: bool = False,  # kept for API compat; not used
):
    """Main entry point used by train.py.

    Returns:
        images: (N, H, W, 3) uint8
        poses:  (N, 3, 4) float32 (camera-to-world)
        hwf:    (H, W, focal)
        near:   float
        far:    float
        meta:   dict – includes minimal extras for downstream
    """
    basedir_p = Path(basedir)

    # 1) If no transforms.json, try to convert Metashape XML automatically
    _maybe_convert_metashape_to_transforms(basedir_p)

    # 2) Load Nerfstudio-style transforms.json
    loaded = load_transforms(basedir_p, split="all")
    frames = loaded.meta.get("frames", [])
    if not frames:
        raise ValueError(f"transforms.json contains no frames: {loaded.transforms_path}")

    # 3) Load images
    images = _stack_images(loaded.image_filenames)

    # 4) Extract and PCA-align poses
    poses_4x4 = loaded.poses.astype(np.float32)
    poses_4x4, align_meta = _pca_align_c2w(poses_4x4)
    poses_3x4 = poses_4x4[:, :3, :4]
    # Log bbox info
    bb = align_meta
    try:
        logger.info(
            "PCA align: bbox before min=%s max=%s | after min=%s max=%s",
            np.array(bb["bbox_before"]["min"]).round(3).tolist(),
            np.array(bb["bbox_before"]["max"]).round(3).tolist(),
            np.array(bb["bbox_after"]["min"]).round(3).tolist(),
            np.array(bb["bbox_after"]["max"]).round(3).tolist(),
        )
    except Exception:
        pass

    # 5) Intrinsics (Nerfstudio convention)
    H, W, focal = _intrinsics_from_meta(loaded.meta, frames)

    # 6) Optional integer downsample
    if downsample and downsample > 1:
        H_ds = H // downsample
        W_ds = W // downsample
        if H_ds < 1 or W_ds < 1:
            raise ValueError(f"Downsample factor {downsample} is too large for ({H},{W})")
        images = images[:, ::downsample, ::downsample]
        focal = float(focal) / float(downsample)
        H, W = H_ds, W_ds

    # 7) Near/Far heuristic based on aligned camera positions
    centers = poses_3x4[:, :3, 3]
    center = centers.mean(axis=0)
    rel = centers - center
    dists = np.linalg.norm(rel, axis=1)
    min_dist = float(np.min(dists)) if dists.size else 0.05
    max_dist = float(np.max(dists)) if dists.size else 2.0
    near = float(max(0.5 * min_dist, 1e-3))
    far = float(1.5 * max_dist + near)

    # 8) Minimal meta – include intrinsics if present at top-level
    meta_out: Dict[str, Any] = {
        "camera_model": loaded.meta.get("camera_model", "OPENCV"),
        "intrinsics": {k: loaded.meta.get(k) for k in ("fl_x", "fl_y", "cx", "cy", "w", "h") if k in loaded.meta},
        "applied_transform": loaded.meta.get("applied_transform"),
        "alignment": align_meta,
    }

    return images, poses_3x4, (H, W, float(focal)), float(near), float(far), meta_out


class SimpleDataset:
    """Simple Tensor wrapper used by train.py"""
    def __init__(self, images: np.ndarray, poses: np.ndarray, hwf: Tuple[int, int, float]):
        self.images = torch.from_numpy(images.astype(np.float32) / 255.0)  # (N,H,W,3) in [0,1]
        self.poses = torch.from_numpy(poses.astype(np.float32))            # (N,3,4)
        self.hwf = hwf

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, idx: int):
        return self.images[idx], self.poses[idx]


# Backward-compat stubs (train.py may import these); no-ops by default.
def transform_water_level(level: float, transform: Any) -> float:
    return float(level)


def transform_water_depth(depth: float, transform: Any) -> float:
    return float(depth)


def main():
    parser = argparse.ArgumentParser(description="Konvertiert Agisoft Metashape cameras.xml zu transforms.json im Nerfstudio-Format.")
    parser.add_argument("xml", type=str, help="Pfad zur cameras.xml (Agisoft Metashape)")
    parser.add_argument("output", type=str, help="Zielordner für transforms.json")
    parser.add_argument("--copy-images", action="store_true", help="Bilder in Zielordner kopieren")
    parser.add_argument("--verbose", action="store_true", help="Mehr Ausgaben anzeigen")
    args = parser.parse_args()

    xml_path = Path(args.xml)
    output_dir = Path(args.output)
    images_dir = xml_path.parent

    summaries = metashape_to_json(
        images_dir=images_dir,
        xml_filename=xml_path,
        output_dir=output_dir,
        copy_images=args.copy_images,
        ply_filename=None,
        verbose=args.verbose,
    )
    for s in summaries:
        print("[dataset]", s)
    print(f"transforms.json gespeichert unter: {output_dir / 'transforms.json'}")

if __name__ == "__main__":
    main()
