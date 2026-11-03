# photo2mesh

Turn photos of a sculpture into a 3D mesh (`.glb`, `.obj`, `.ply`, `.stl`). CPU only.

```bash
uv run photo2mesh.py photo.jpg -o out          # 1 photo  -> TripoSR (neural, back side is guessed)
uv run photo2mesh.py photos/ -o out            # many     -> SfM (pycolmap) + silhouette carving
uv run photo2mesh.py photos/ --sequential      # photos taken in order around the object (faster)
```

Options: `--mode auto|single|multi`, `--faces N` (default 100000), `--resolution` (TripoSR, 256), `--grid` (carving voxels, 192), `--name`, `--seed`.

First run downloads U2Net (~170 MB) and TripoSR weights (~1.7 GB). Single mode takes a few minutes on CPU.

## Tips for multi-photo mode
- 30-60 photos circling the sculpture, ~60% overlap, two or three heights, consistent lighting.
- Plain, uncluttered background (the sculpture is cut out automatically).
- Textured surfaces work best; glossy, white or transparent ones confuse SfM.
- Silhouette carving can't recover deep concavities, and the model is scaled to unit size (no real-world scale).

`vendor/TripoSR` is a clone of https://github.com/VAST-AI-Research/TripoSR. `torchmcubes` is replaced by a scikit-image shim, so nothing needs compiling.
