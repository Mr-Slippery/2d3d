"""Run all: uv run pytest   |   fast only: uv run pytest -m "not slow"."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import photo2mesh  # noqa: E402

FORMATS = ("glb", "obj", "ply", "stl")


def run_cli(*args):
    return subprocess.run(
        [sys.executable, str(ROOT / "photo2mesh.py"), *map(str, args)],
        capture_output=True, text=True, cwd=ROOT,
    )


def assert_outputs(out_dir: Path):
    for ext in FORMATS:
        assert (out_dir / f"model.{ext}").stat().st_size > 0, ext
    mesh = trimesh.load(out_dir / "model.glb", force="mesh")
    assert len(mesh.vertices) > 100
    assert max(mesh.extents) == pytest.approx(1.0, abs=1e-3)  # normalized to unit size
    assert trimesh.load(out_dir / "model.stl").is_watertight


def render_turntable(mesh, out_dir, n=48, size=900, focal=1100.0):
    """Splat a textured mesh's surface points into n views orbiting it."""
    out_dir.mkdir()
    pts, face_idx = trimesh.sample.sample_surface(mesh, 2_500_000)
    base = np.asarray(mesh.visual.face_colors)[face_idx][:, :3].astype(float)
    rng = np.random.default_rng(1)  # high-frequency texture so SfM finds features
    tex = np.sin(pts @ (rng.normal(size=(12, 3)) * 40).T + rng.uniform(0, 6.28, 12)).sum(1) / 12
    cols = np.clip(base * (0.55 + 1.2 * (tex[:, None] + 0.5)), 0, 255).astype(np.uint8)
    for i in range(n):
        el, az = np.radians(15 + 10 * np.sin(i * 0.7)), 2 * np.pi * i / n
        cam = 2.2 * np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])
        z = -cam / np.linalg.norm(cam)
        x = np.cross([0, 0, 1.0], z); x /= np.linalg.norm(x)
        pc = (pts - cam) @ np.stack([x, np.cross(z, x), z]).T
        keep = pc[:, 2] > 0.1
        pc, c = pc[keep], cols[keep]
        order = np.argsort(-pc[:, 2])
        u = (focal * pc[order, 0] / pc[order, 2] + size / 2).astype(int)
        v = (focal * pc[order, 1] / pc[order, 2] + size / 2).astype(int)
        img = np.full((size, size, 3), (200, 210, 220), np.uint8)
        for dx in (-1, 0, 1, 2):
            for dy in (-1, 0, 1, 2):
                img[np.clip(v + dy, 0, size - 1), np.clip(u + dx, 0, size - 1)] = c[order]
        Image.fromarray(img).save(out_dir / f"v{i:03d}.jpg")


def test_collect_images_and_cleanup(tmp_path):
    for name in ("a.jpg", "b.PNG", "notes.txt"):
        (tmp_path / name).write_bytes(b"x")
    found = photo2mesh.collect_images([str(tmp_path)])
    assert [p.name for p in found] == ["a.jpg", "b.PNG"]

    # cleanup drops stray components, caps faces and normalizes size
    big = trimesh.creation.icosphere(subdivisions=5, radius=3.0)
    stray = trimesh.creation.icosphere(subdivisions=1, radius=0.1)
    stray.apply_translation([20, 0, 0])
    cleaned = photo2mesh.cleanup(trimesh.util.concatenate([big, stray]), max_faces=5000)
    assert len(cleaned.faces) <= 5000
    assert max(cleaned.extents) == pytest.approx(1.0, abs=1e-6)
    assert np.allclose(cleaned.bounds.mean(axis=0), 0, atol=1e-6)


@pytest.mark.slow
def test_single_photo(tmp_path):
    res = run_cli(ROOT / "vendor/TripoSR/examples/robot.png", "-o", tmp_path)
    assert res.returncode == 0, res.stderr[-2000:]
    assert_outputs(tmp_path)


@pytest.mark.slow
def test_multi_photo_recovers_shape(tmp_path):
    # ground truth: textured ellipsoid with a bump, so silhouettes vary by view
    truth = trimesh.creation.icosphere(subdivisions=4)
    truth.apply_scale([0.5, 0.35, 0.7])
    truth.visual.face_colors = [180, 120, 90, 255]
    photos = tmp_path / "photos"
    render_turntable(truth, photos)
    out = tmp_path / "out"
    res = run_cli(photos, "-o", out)
    assert res.returncode == 0, res.stderr[-2000:]
    assert_outputs(out)

    # compare to ground truth after normalizing to the same unit size/centre
    truth.apply_translation(-truth.bounds.mean(axis=0))
    truth.apply_scale(1 / max(truth.extents))
    got = trimesh.load(out / "model.ply", force="mesh")
    # carved shape has the same axis proportions (sorted extents within 10%)
    assert np.allclose(np.sort(got.extents), np.sort(truth.extents), atol=0.1)
