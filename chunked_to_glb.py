"""
Run chunked GLB conversion on saved reconstruct_scene outputs.

Loads to_glb_inputs.pt + chunk_inputs.pt produced by reconstruct_scene.py,
runs global remesh + per-chunk bake, and writes a multi-primitive scene.glb.

Pass --viewer to additionally write a lighter scene_viewer.glb. The full-quality
scene.glb is always generated and keeps every post-remesh face by default.

Usage:
    python chunked_to_glb.py \
        --inputs path/to/to_glb_inputs.pt \
        --chunk_inputs path/to/chunk_inputs.pt \
        --output_dir Xabl/chunked_v1
"""

from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

import torch

from inference.chunked_glb import chunked_to_glb


def load_pt(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, help="Path to to_glb_inputs.pt")
    parser.add_argument("--chunk_inputs", required=True, help="Path to chunk_inputs.pt")
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write scene.glb and, with --viewer, scene_viewer.glb.",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Also write scene_viewer.glb using viewer-friendly defaults: 100k faces "
        "per chunk, 1024px textures, and remesh resolution 1024. Explicit values "
        "override the viewer preset; scene.glb remains full-quality.",
    )
    parser.add_argument(
        "--texture_size",
        type=int,
        default=None,
        help="Texture resolution per chunk (default: 4096; with --viewer this overrides "
        "the viewer preset only).",
    )
    parser.add_argument(
        "--simplify_threshold",
        type=int,
        default=None,
        help="Maximum faces per chunk (default: unlimited; with --viewer this overrides "
        "the viewer preset only).",
    )
    parser.add_argument(
        "--skip_fill_holes",
        action="store_true",
        help="Skip the global cumesh.fill_holes step.",
    )
    parser.add_argument("--hole_perim", type=float, default=1.0)
    parser.add_argument(
        "--skip_remesh",
        action="store_true",
        help="Skip the global narrow-band DC remesh step.",
    )
    parser.add_argument(
        "--remesh_res",
        type=int,
        default=None,
        help="Global remesh resolution (default: automatic; with --viewer this overrides "
        "the viewer preset only).",
    )
    parser.add_argument("--remesh_band", type=float, default=1.0)
    parser.add_argument("--remesh_project", type=float, default=0.9)
    parser.add_argument(
        "--smooth_iters",
        type=int,
        default=0,
        help="Feature-preserving Taubin λ-μ smoothing iterations applied to the "
        "(possibly skip-remeshed) global mesh before chunk partition. 0 = disabled.",
    )
    parser.add_argument(
        "--smooth_lambda",
        type=float,
        default=0.5,
        help="Taubin λ (positive smoothing factor). Typical range 0.3–0.6.",
    )
    parser.add_argument(
        "--smooth_mu",
        type=float,
        default=-0.53,
        help="Taubin μ (negative anti-shrinkage factor). Should satisfy |μ| > |λ|.",
    )
    parser.add_argument(
        "--smooth_feature_angle",
        type=float,
        default=25.0,
        help="Dihedral angle threshold (deg). Vertices on edges above this — plus all "
        "boundary / non-manifold edges — are locked and not smoothed.",
    )
    parser.add_argument(
        "--dump_geometry_plys",
        action="store_true",
        help="Diagnostic mode: dump global_mesh.ply + per-chunk chunk_NNN_geom.ply into "
        "output_dir and skip op.to_glb (no decimation, xatlas, or texture bake). Lets you "
        "inspect the pre-bake geometry and iterate on smoothing/fill_holes/remesh in seconds.",
    )
    parser.add_argument(
        "--chunks_dir",
        default=None,
        help="Directory to save per-chunk GLBs as they're baked. Already-present chunks are "
        "loaded from disk on rerun (resumable). Defaults to a settings-specific cache. "
        "With --viewer, an explicit path applies to the viewer cache only. "
        "Pass --no_chunk_cache to disable.",
    )
    parser.add_argument(
        "--no_chunk_cache",
        action="store_true",
        help="Disable per-chunk GLB save / resume entirely (in-memory only).",
    )
    args = parser.parse_args()

    if args.texture_size is not None and args.texture_size <= 0:
        parser.error("--texture_size must be positive")
    if args.simplify_threshold is not None and args.simplify_threshold <= 0:
        parser.error("--simplify_threshold must be positive")
    if args.remesh_res is not None and args.remesh_res <= 0:
        parser.error("--remesh_res must be positive")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    to_glb_inputs = load_pt(Path(args.inputs))
    chunk_inputs = load_pt(Path(args.chunk_inputs))

    print(
        f"Loaded {args.inputs}: "
        f"vertices {tuple(to_glb_inputs['vertices'].shape)}, "
        f"faces {tuple(to_glb_inputs['faces'].shape)}, "
        f"attr_volume {tuple(to_glb_inputs['attr_volume'].shape)}, "
        f"coords {tuple(to_glb_inputs['coords'].shape)}"
    )
    print(
        f"Loaded {args.chunk_inputs}: "
        f"{len(chunk_inputs['chunk_indices'])} chunks, "
        f"chunk_size_world={chunk_inputs['chunk_size_world']:.4f}"
    )

    def run_conversion(
        preset: str,
        *,
        texture_size: int,
        simplify_threshold: int | None,
        remesh_res: int | None,
        chunks_save_dir: Path | None,
        out_file: Path | None,
        dump_geometry: bool = False,
    ) -> bool:
        threshold_text = (
            f"{simplify_threshold:,} faces/chunk"
            if simplify_threshold is not None
            else "unlimited (no decimation)"
        )
        print(
            f"GLB preset: {preset}, simplify_threshold={threshold_text}, "
            f"texture_size={texture_size}, remesh_res={remesh_res or 'auto'}"
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        t0 = time.perf_counter()
        try:
            scene = chunked_to_glb(
                vertices_world=to_glb_inputs["vertices"],
                faces=to_glb_inputs["faces"],
                attr_volume=to_glb_inputs["attr_volume"],
                coords=to_glb_inputs["coords"],
                attr_layout=to_glb_inputs["attr_layout"],
                aabb_world=to_glb_inputs["aabb"],
                voxel_size_world=to_glb_inputs["voxel_size"],
                chunk_centers_world=chunk_inputs["chunk_centers_world"],
                chunk_size_world=chunk_inputs["chunk_size_world"],
                chunk_indices=chunk_inputs["chunk_indices"],
                do_fill_holes=not args.skip_fill_holes,
                hole_perim=args.hole_perim,
                do_remesh=not args.skip_remesh,
                remesh_res=remesh_res,
                remesh_band=args.remesh_band,
                remesh_project=args.remesh_project,
                smooth_iters=args.smooth_iters,
                smooth_lambda=args.smooth_lambda,
                smooth_mu=args.smooth_mu,
                smooth_feature_angle=args.smooth_feature_angle,
                simplify_threshold=simplify_threshold,
                texture_size=texture_size,
                dump_geometry_dir=out_dir if dump_geometry else None,
                chunks_save_dir=chunks_save_dir,
                verbose=True,
            )
        except Exception:
            print(f"[chunked_to_glb] {preset} preset FAILED:")
            traceback.print_exc()
            return False

        elapsed = time.perf_counter() - t0
        if dump_geometry:
            print(f"\n[chunked_to_glb] dump-geometry mode: PLYs written to {out_dir}")
        else:
            assert out_file is not None
            print(f"\n[chunked_to_glb] writing {out_file} ...")
            scene.export(str(out_file))
        print(f"[chunked_to_glb] {preset} total time: {elapsed:.1f}s")
        if torch.cuda.is_available():
            peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"[chunked_to_glb] {preset} VRAM peak: {peak_gb:.2f} GB")
            torch.cuda.empty_cache()
        return True

    if args.dump_geometry_plys:
        dump_texture_size = args.texture_size or (1024 if args.viewer else 4096)
        dump_threshold = args.simplify_threshold if args.simplify_threshold is not None else (
            100_000 if args.viewer else None
        )
        dump_remesh_res = args.remesh_res if args.remesh_res is not None else (
            1024 if args.viewer else None
        )
        run_conversion(
            "viewer" if args.viewer else "full",
            texture_size=dump_texture_size,
            simplify_threshold=dump_threshold,
            remesh_res=dump_remesh_res,
            chunks_save_dir=None,
            out_file=None,
            dump_geometry=True,
        )
        return

    full_texture_size = 4096 if args.viewer else (args.texture_size or 4096)
    full_threshold = None if args.viewer else args.simplify_threshold
    full_remesh_res = None if args.viewer else args.remesh_res
    if args.no_chunk_cache:
        full_chunks_dir = None
    elif not args.viewer and args.chunks_dir is not None:
        full_chunks_dir = Path(args.chunks_dir)
    else:
        full_threshold_tag = full_threshold if full_threshold is not None else "none"
        full_remesh_tag = "skip" if args.skip_remesh else (full_remesh_res or "auto")
        full_chunks_dir = out_dir / (
            f"chunks_full_s{full_threshold_tag}_t{full_texture_size}_r{full_remesh_tag}"
        )

    if not run_conversion(
        "full",
        texture_size=full_texture_size,
        simplify_threshold=full_threshold,
        remesh_res=full_remesh_res,
        chunks_save_dir=full_chunks_dir,
        out_file=out_dir / "scene.glb",
    ):
        return

    if not args.viewer:
        return

    viewer_texture_size = args.texture_size or 1024
    viewer_threshold = args.simplify_threshold or 100_000
    viewer_remesh_res = args.remesh_res or 1024
    if args.no_chunk_cache:
        viewer_chunks_dir = None
    elif args.chunks_dir is not None:
        viewer_chunks_dir = Path(args.chunks_dir)
    else:
        viewer_chunks_dir = out_dir / (
            f"chunks_viewer_s{viewer_threshold}_t{viewer_texture_size}_r{viewer_remesh_res}"
        )
    run_conversion(
        "viewer",
        texture_size=viewer_texture_size,
        simplify_threshold=viewer_threshold,
        remesh_res=viewer_remesh_res,
        chunks_save_dir=viewer_chunks_dir,
        out_file=out_dir / "scene_viewer.glb",
    )


if __name__ == "__main__":
    main()
