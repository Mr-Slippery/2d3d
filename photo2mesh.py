#!/usr/bin/env python3
"""Create a 3D mesh from one or several photos of a sculpture.

One photo   -> TripoSR (neural single-image-to-3D; the back side is inferred).
Many photos -> structure-from-motion (pycolmap) + silhouette carving.

Usage:
    uv run photo2mesh.py photo.jpg -o out
    uv run photo2mesh.py photos_dir/ -o out
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MAX_SIDE = 1600  # photos are downscaled to this before processing


def log(msg: str) -> None:
    print(f"[photo2mesh] {msg}", flush=True)


# --------------------------------------------------------------------------
# Input handling and background removal
# --------------------------------------------------------------------------
def collect_images(inputs: list[str]) -> list[Path]:
    found: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            found += sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        elif p.is_file():
            found.append(p)
        else:
            found += sorted(Path().glob(item))
    found = [f for f in found if f.suffix.lower() in IMAGE_EXTS]
    if not found:
        sys.exit("error: no input images found")
    return found


def load_image(path: Path) -> Image.Image:
    from PIL import ImageOps

    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    scale = MAX_SIDE / max(img.size)
    if scale < 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    return img


def remove_background(imgs: list[Image.Image]) -> list[Image.Image]:
    """Return RGBA images whose alpha channel is the sculpture mask."""
    from rembg import new_session, remove

    log("removing background (first run downloads the U2Net model)")
    session = new_session("u2net")
    out = []
    for i, img in enumerate(imgs, 1):
        log(f"  background {i}/{len(imgs)}")
        out.append(remove(img, session=session).convert("RGBA"))
    return out


# --------------------------------------------------------------------------
# Single image: TripoSR
# --------------------------------------------------------------------------
def install_torchmcubes_shim() -> None:
    """TripoSR imports torchmcubes (needs compiling); provide a scikit-image stand-in."""
    import torch
    from skimage import measure

    def marching_cubes(vol, thresh):
        v = vol.detach().cpu().numpy()
        thresh = float(np.clip(thresh, v.min() + 1e-6, v.max() - 1e-6))
        verts, faces, _, _ = measure.marching_cubes(v, level=thresh)
        # torchmcubes returns vertices in (z, y, x) order; TripoSR flips them back
        verts = np.ascontiguousarray(verts[:, ::-1])
        return torch.from_numpy(verts).float(), torch.from_numpy(faces.astype(np.int64))

    mod = types.ModuleType("torchmcubes")
    mod.marching_cubes = marching_cubes
    sys.modules["torchmcubes"] = mod


def reconstruct_single(rgba: Image.Image, resolution: int, seed: int) -> trimesh.Trimesh:
    import torch

    sys.path.insert(0, str(ROOT / "vendor" / "TripoSR"))
    install_torchmcubes_shim()
    from tsr.system import TSR

    torch.manual_seed(seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    # crop to the sculpture, pad, and composite on neutral grey as TripoSR expects
    arr = np.asarray(rgba).astype(np.float32) / 255.0
    ys, xs = np.nonzero(arr[..., 3] > 0.5)
    if len(xs) == 0:
        sys.exit("error: background removal found no foreground; try --no-rembg")
    arr = arr[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    side = int(max(arr.shape[:2]) / 0.85)
    canvas = np.zeros((side, side, 4), np.float32)
    y0, x0 = (side - arr.shape[0]) // 2, (side - arr.shape[1]) // 2
    canvas[y0 : y0 + arr.shape[0], x0 : x0 + arr.shape[1]] = arr
    rgb = canvas[..., :3] * canvas[..., 3:] + 0.5 * (1 - canvas[..., 3:])
    image = Image.fromarray((rgb * 255).astype(np.uint8))

    log("loading TripoSR (first run downloads ~1.7 GB of weights)")
    model = TSR.from_pretrained(
        "stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt"
    )
    model.renderer.set_chunk_size(8192)
    model.to("cpu")

    log("running TripoSR on CPU (about a minute)")
    with torch.no_grad():
        codes = model([image], device="cpu")
        meshes = model.extract_mesh(codes, True, resolution=resolution)
    return meshes[0]


# --------------------------------------------------------------------------
# Multiple images: SfM + silhouette carving
# --------------------------------------------------------------------------
def reconstruct_multi(
    imgs: list[Image.Image], rgbas: list[Image.Image], workdir: Path, grid: int, sequential: bool
) -> trimesh.Trimesh:
    import pycolmap
    from scipy import ndimage
    from skimage import measure

    img_dir = workdir / "images"
    img_dir.mkdir()
    for i, img in enumerate(imgs):
        img.save(img_dir / f"{i:04d}.jpg", quality=95)
    db = workdir / "database.db"
    sparse = workdir / "sparse"
    sparse.mkdir()

    log("SfM: extracting features")
    pycolmap.extract_features(db, img_dir, camera_mode=pycolmap.CameraMode.SINGLE)
    log("SfM: matching")
    if sequential:
        pycolmap.match_sequential(db)
    else:
        pycolmap.match_exhaustive(db)
    log("SfM: incremental mapping")
    recs = pycolmap.incremental_mapping(db, img_dir, sparse)
    if not recs:
        sys.exit("error: SfM failed to reconstruct any camera poses (need overlapping, textured photos)")
    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    n_reg = rec.num_reg_images()
    log(f"SfM registered {n_reg}/{len(imgs)} images, {rec.num_points3D()} sparse points")
    if n_reg < 3:
        sys.exit("error: fewer than 3 images registered; take more photos with more overlap")

    # camera matrices and masks for registered views
    views = []
    for image in rec.images.values():
        if not image.has_pose:
            continue
        idx = int(Path(image.name).stem)
        cam = image.camera
        P = np.asarray(image.cam_from_world().matrix())  # 3x4
        K = np.asarray(cam.calibration_matrix())
        mask = np.asarray(rgbas[idx])[..., 3] > 127
        # shrink mask slightly to be conservative at noisy edges
        mask = ndimage.binary_erosion(mask, iterations=2)
        views.append((K @ P, mask, image.projection_center(), np.asarray(imgs[idx])))

    pts = np.array([p.xyz for p in rec.points3D.values()])
    lo, hi = np.percentile(pts, 3, axis=0), np.percentile(pts, 97, axis=0)
    pad = (hi - lo) * 0.35
    lo, hi = lo - pad, hi + pad
    log(f"carving a {grid}^3 voxel grid from {len(views)} silhouettes")
    axes = [np.linspace(lo[a], hi[a], grid) for a in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    vox = np.stack([gx, gy, gz, np.ones_like(gx)], -1).reshape(-1, 4)

    inside = np.zeros(len(vox), np.int16)
    seen = np.zeros(len(vox), np.int16)
    for M, mask, _, _ in views:
        proj = vox @ M.T
        z = proj[:, 2]
        u = np.round(proj[:, 0] / np.where(z == 0, 1, z)).astype(int)
        v = np.round(proj[:, 1] / np.where(z == 0, 1, z)).astype(int)
        h, w = mask.shape
        ok = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        seen += ok
        hit = np.zeros(len(vox), bool)
        hit[ok] = mask[v[ok], u[ok]]
        inside += hit
    tolerance = 1 if len(views) >= 10 else 0
    occ = ((seen >= 3) & (inside >= seen - tolerance)).reshape(grid, grid, grid)
    occ = ndimage.binary_opening(occ, iterations=1)
    labels, n = ndimage.label(occ)
    if n == 0:
        sys.exit("error: carving produced an empty volume; check masks / camera poses")
    sizes = ndimage.sum(occ, labels, range(1, n + 1))
    occ = labels == (1 + int(np.argmax(sizes)))
    field = ndimage.gaussian_filter(occ.astype(np.float32), 1.0)
    verts, faces, _, _ = measure.marching_cubes(field, 0.5)
    step = np.array([(hi[a] - lo[a]) / (grid - 1) for a in range(3)])
    verts = verts * step + lo
    mesh = trimesh.Trimesh(verts, faces, process=True)

    # colour each vertex from the camera that faces it best
    log("projecting colours onto the mesh")
    normals = mesh.vertex_normals
    best = np.full(len(verts), -2.0)
    colors = np.full((len(verts), 3), 128, np.uint8)
    vh = np.c_[verts, np.ones(len(verts))]
    for M, _, centre, rgb in views:
        proj = vh @ M.T
        z = proj[:, 2]
        u = np.round(proj[:, 0] / np.where(z == 0, 1, z)).astype(int)
        v = np.round(proj[:, 1] / np.where(z == 0, 1, z)).astype(int)
        h, w = rgb.shape[:2]
        ok = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        to_cam = centre - verts
        to_cam /= np.linalg.norm(to_cam, axis=1, keepdims=True) + 1e-9
        facing = np.einsum("ij,ij->i", normals, to_cam)
        better = ok & (facing > best)
        colors[better] = rgb[v[better], u[better]]
        best[better] = facing[better]
    mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=colors)
    return mesh


# --------------------------------------------------------------------------
# Cleanup and export
# --------------------------------------------------------------------------
def cleanup(mesh: trimesh.Trimesh, max_faces: int) -> trimesh.Trimesh:
    log("cleaning up mesh")
    colors = mesh.visual.vertex_colors if mesh.visual.kind == "vertex" else None
    mesh = trimesh.Trimesh(mesh.vertices, mesh.faces, vertex_colors=colors, process=True)
    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        mesh = max(parts, key=lambda m: len(m.faces))
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fill_holes(mesh)
    if len(mesh.faces) > max_faces:
        log(f"decimating {len(mesh.faces)} -> {max_faces} faces")
        import open3d as o3d

        o3m = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces)
        )
        if colors is not None:
            o3m.vertex_colors = o3d.utility.Vector3dVector(np.asarray(mesh.visual.vertex_colors)[:, :3] / 255.0)
        o3m = o3m.simplify_quadric_decimation(max_faces)
        vc = (np.asarray(o3m.vertex_colors) * 255).astype(np.uint8) if colors is not None else None
        mesh = trimesh.Trimesh(np.asarray(o3m.vertices), np.asarray(o3m.triangles), vertex_colors=vc, process=True)
    # centre on origin and scale to unit height of the largest dimension
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    mesh.apply_scale(1.0 / max(mesh.extents))
    return mesh


def close_holes(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Fan-fill every boundary loop with a centroid vertex (fill_holes misses larger ones)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    edges = mesh.edges  # directed, one row per face edge
    _, inv, counts = np.unique(mesh.edges_sorted, axis=0, return_inverse=True, return_counts=True)
    boundary = edges[counts[inv.reshape(-1)] == 1]
    if len(boundary) == 0:
        return mesh
    n = len(mesh.vertices)
    graph = coo_matrix((np.ones(len(boundary)), (boundary[:, 0], boundary[:, 1])), shape=(n, n))
    _, label = connected_components(graph, directed=False)
    verts, faces = [mesh.vertices], [mesh.faces]
    for lab in np.unique(label[boundary[:, 0]]):
        loop = boundary[label[boundary[:, 0]] == lab]
        centre = mesh.vertices[np.unique(loop)].mean(axis=0)
        idx = sum(len(v) for v in verts)
        verts.append(centre[None])
        faces.append(np.c_[loop[:, 1], loop[:, 0], np.full(len(loop), idx)])
    return trimesh.Trimesh(np.vstack(verts), np.vstack(faces), process=True)


def export(mesh: trimesh.Trimesh, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("glb", "obj", "ply"):
        path = out_dir / f"{name}.{ext}"
        mesh.export(path)
        log(f"wrote {path}")
    printable = trimesh.Trimesh(mesh.vertices, mesh.faces, process=True)
    trimesh.repair.fill_holes(printable)
    printable = close_holes(printable)
    trimesh.repair.fix_normals(printable)
    path = out_dir / f"{name}.stl"
    printable.export(path)
    log(f"wrote {path} (watertight: {printable.is_watertight})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="image files, directories or globs")
    ap.add_argument("-o", "--out", default="out", help="output directory (default: out)")
    ap.add_argument("-n", "--name", default="model", help="output file base name")
    ap.add_argument("--mode", choices=["auto", "single", "multi"], default="auto")
    ap.add_argument("--no-rembg", action="store_true", help="skip background removal (single mode only)")
    ap.add_argument("--faces", type=int, default=100_000, help="max faces after decimation")
    ap.add_argument("--resolution", type=int, default=256, help="TripoSR marching-cubes resolution")
    ap.add_argument("--grid", type=int, default=192, help="voxel grid size for multi-photo carving")
    ap.add_argument("--sequential", action="store_true", help="photos were taken in order (faster matching)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    paths = collect_images(args.inputs)
    mode = args.mode if args.mode != "auto" else ("single" if len(paths) == 1 else "multi")
    log(f"{len(paths)} image(s), mode: {mode}")
    imgs = [load_image(p) for p in paths]

    if mode == "single":
        if len(imgs) > 1:
            log(f"warning: using only the first image ({paths[0].name}) in single mode")
        rgba = imgs[0].convert("RGBA") if args.no_rembg else remove_background(imgs[:1])[0]
        mesh = reconstruct_single(rgba, args.resolution, args.seed)
        note = "note: the back of the sculpture is inferred by the model, not measured."
    else:
        if len(imgs) < 3:
            sys.exit("error: multi mode needs at least 3 photos")
        if len(imgs) < 15:
            log("warning: fewer than 15 photos; reconstruction may fail. 30-60 around the sculpture is ideal")
        rgbas = remove_background(imgs)
        with tempfile.TemporaryDirectory(prefix="photo2mesh_") as tmp:
            mesh = reconstruct_multi(imgs, rgbas, Path(tmp), args.grid, args.sequential)
        note = "note: silhouette carving cannot recover concavities (e.g. deep recesses)."

    mesh = cleanup(mesh, args.faces)
    export(mesh, Path(args.out), args.name)
    log(note)


if __name__ == "__main__":
    main()
