from __future__ import annotations

import argparse
import colorsys
import json
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image
from scipy import sparse


@dataclass
class ProposalSet:
    masks: np.ndarray
    scores: np.ndarray
    predicted_iou: np.ndarray
    stability: np.ndarray
    boxes: np.ndarray


@dataclass
class Projection:
    superpoints: np.ndarray
    pixels_x: np.ndarray
    pixels_y: np.ndarray
    depth_weights: np.ndarray
    face_ids: np.ndarray
    depth: np.ndarray


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MV3DIS-inspired consistent instance masks for GenRecon ScanNet++ outputs."
    )
    parser.add_argument("output_dir", type=Path, help="A GenRecon output/scannetpp/<scene> directory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/sam2.1_hiera_small.pt"))
    parser.add_argument("--sam-config", default="configs/sam2.1/sam2.1_hiera_s.yaml")
    parser.add_argument("--work-resolution", type=int, default=512)
    parser.add_argument("--output-resolution", type=int, default=1024)
    parser.add_argument("--points-per-side", type=int, default=32)
    parser.add_argument("--points-per-batch", type=int, default=128)
    parser.add_argument("--pred-iou-threshold", type=float, default=0.75)
    parser.add_argument("--stability-threshold", type=float, default=0.88)
    parser.add_argument("--mask-nms-threshold", type=float, default=0.80)
    parser.add_argument("--max-masks-per-view", type=int, default=128)
    parser.add_argument("--min-mask-area", type=int, default=100)
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--depth-tolerance", type=float, default=0.05)
    parser.add_argument("--frame-visibility", type=float, default=0.30)
    parser.add_argument("--mask-visibility", type=float, default=0.90)
    parser.add_argument("--merge-threshold", type=float, default=0.70)
    parser.add_argument("--cluster-similarity", type=float, default=0.45)
    parser.add_argument("--min-cluster-views", type=int, default=1)
    parser.add_argument("--max-region-changes", type=int, default=2)
    parser.add_argument("--min-instance-vertices", type=int, default=30)
    parser.add_argument("--min-instance-support", type=int, default=20)
    parser.add_argument("--force-sam", action="store_true")
    parser.add_argument("--skip-ply", action="store_true")
    return parser.parse_args()


def _json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)


def _source_scene_dir(output_dir: Path) -> Path:
    args_path = output_dir / "args.json"
    with args_path.open("r", encoding="utf-8") as handle:
        saved_args = json.load(handle)
    for key in ("path", "scene_path", "input", "input_path"):
        value = saved_args.get(key)
        if value and Path(value).is_dir():
            return Path(value)
    raise KeyError(f"Could not find the ScanNet++ input directory in {args_path}")


def _load_cameras(output_dir: Path, resolution: int) -> tuple[list[dict], list[np.ndarray], list[np.ndarray]]:
    with (output_dir / "cameras.json").open("r", encoding="utf-8") as handle:
        cameras = json.load(handle)["scene"]
    with (output_dir / "chunk_transforms.json").open("r", encoding="utf-8") as handle:
        original_to_chunk0 = np.asarray(
            json.load(handle)["chunks"][0]["M_original_to_chunk"], dtype=np.float32
        )

    world_to_camera: list[np.ndarray] = []
    intrinsics: list[np.ndarray] = []
    for camera in cameras:
        world_to_camera.append(np.asarray(camera["extrinsics_c0"], dtype=np.float32) @ original_to_chunk0)
        intr = np.asarray(camera["intrinsics"], dtype=np.float32).copy()
        intr[0] *= resolution
        intr[1] *= resolution
        intrinsics.append(intr)
    return cameras, world_to_camera, intrinsics


def _resize_bool(mask: np.ndarray, resolution: int) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), (resolution, resolution), interpolation=cv2.INTER_NEAREST).astype(bool)


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    intersection = int(np.count_nonzero(a & b))
    if intersection == 0:
        return 0.0
    return intersection / float(np.count_nonzero(a | b))


def _mask_nms(masks: np.ndarray, scores: np.ndarray, threshold: float, limit: int) -> np.ndarray:
    order = np.argsort(-scores)
    kept: list[int] = []
    for index in order:
        if any(_mask_iou(masks[index], masks[other]) > threshold for other in kept):
            continue
        kept.append(int(index))
        if len(kept) >= limit:
            break
    return np.asarray(kept, dtype=np.int64)


def _label_map(masks: np.ndarray, scores: np.ndarray) -> np.ndarray:
    labels = np.zeros(masks.shape[1:], dtype=np.uint16)
    score_map = np.full(masks.shape[1:], -np.inf, dtype=np.float32)
    for local_id in np.argsort(scores):
        update = masks[local_id] & (scores[local_id] >= score_map)
        labels[update] = int(local_id) + 1
        score_map[update] = scores[local_id]
    return labels


def _save_proposals(path: Path, proposals: ProposalSet) -> None:
    flat = proposals.masks.reshape(proposals.masks.shape[0], -1)
    packed = np.packbits(flat, axis=1)
    np.savez_compressed(
        path,
        packed_masks=packed,
        mask_shape=np.asarray(proposals.masks.shape, dtype=np.int32),
        scores=proposals.scores,
        predicted_iou=proposals.predicted_iou,
        stability=proposals.stability,
        boxes=proposals.boxes,
    )


def _load_proposals(path: Path) -> ProposalSet:
    data = np.load(path)
    shape = tuple(int(x) for x in data["mask_shape"])
    flat = np.unpackbits(data["packed_masks"], axis=1, count=shape[1] * shape[2])
    return ProposalSet(
        masks=flat.reshape(shape).astype(bool),
        scores=data["scores"],
        predicted_iou=data["predicted_iou"],
        stability=data["stability"],
        boxes=data["boxes"],
    )


def _save_raw_visualization(
    image_path: Path,
    view_index: int,
    proposals: ProposalSet,
    raw_dir: Path,
    overlay_dir: Path,
    output_resolution: int,
    image: np.ndarray | None = None,
) -> None:
    raw_labels = _label_map(proposals.masks, proposals.scores)
    raw_labels = cv2.resize(
        raw_labels, (output_resolution, output_resolution), interpolation=cv2.INTER_NEAREST
    )
    Image.fromarray(raw_labels).save(raw_dir / f"view_{view_index:03d}.png")
    if image is None:
        image = np.asarray(Image.open(image_path).convert("RGB")).copy()
    if image.shape[:2] != raw_labels.shape:
        image = cv2.resize(image, raw_labels.shape[::-1], interpolation=cv2.INTER_LANCZOS4)
    Image.fromarray(_overlay(image, raw_labels)).save(
        overlay_dir / f"view_{view_index:03d}.jpg", quality=92
    )


def generate_sam_proposals(
    image_paths: list[Path],
    segmentation_dir: Path,
    args: argparse.Namespace,
) -> list[ProposalSet]:
    raw_dir = segmentation_dir / "masks_raw"
    raw_overlay_dir = segmentation_dir / "overlays_raw"
    cache_dir = segmentation_dir / "proposals"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_overlay_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_paths = [cache_dir / f"view_{i:03d}.npz" for i in range(len(image_paths))]
    cache_signature = {
        "checkpoint": str(args.checkpoint.resolve()),
        "sam_config": args.sam_config,
        "work_resolution": args.work_resolution,
        "points_per_side": args.points_per_side,
        "pred_iou_threshold": args.pred_iou_threshold,
        "stability_threshold": args.stability_threshold,
        "mask_nms_threshold": args.mask_nms_threshold,
        "max_masks_per_view": args.max_masks_per_view,
        "min_mask_area": args.min_mask_area,
    }
    manifest_path = cache_dir / "manifest.json"
    cached_signature = None
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            cached_signature = json.load(handle)
    cache_valid = cached_signature == cache_signature
    if not args.force_sam and cache_valid and all(path.exists() for path in cache_paths):
        print(f"[SAM2] loading {len(cache_paths)} cached proposal sets")
        cached = [_load_proposals(path) for path in cache_paths]
        for view_index, (image_path, proposals) in enumerate(zip(image_paths, cached)):
            _save_raw_visualization(
                image_path,
                view_index,
                proposals,
                raw_dir,
                raw_overlay_dir,
                args.output_resolution,
            )
        return cached

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {args.checkpoint}")
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    model = build_sam2(args.sam_config, str(args.checkpoint), device=args.device)
    generator = SAM2AutomaticMaskGenerator(
        model,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_threshold,
        stability_score_thresh=args.stability_threshold,
        min_mask_region_area=args.min_mask_area,
    )
    result: list[ProposalSet] = []
    for view_index, (image_path, cache_path) in enumerate(zip(image_paths, cache_paths)):
        if cache_path.exists() and not args.force_sam and cache_valid:
            proposals = _load_proposals(cache_path)
            _save_raw_visualization(
                image_path,
                view_index,
                proposals,
                raw_dir,
                raw_overlay_dir,
                args.output_resolution,
            )
            result.append(proposals)
            continue
        image = np.asarray(Image.open(image_path).convert("RGB")).copy()
        started = time.time()
        annotations = generator.generate(image)
        if not annotations:
            raise RuntimeError(f"SAM2 returned no proposals for {image_path}")
        masks = np.stack([_resize_bool(item["segmentation"], args.work_resolution) for item in annotations])
        predicted_iou = np.asarray([item["predicted_iou"] for item in annotations], dtype=np.float32)
        stability = np.asarray([item["stability_score"] for item in annotations], dtype=np.float32)
        boxes = np.asarray([item["bbox"] for item in annotations], dtype=np.float32)
        scores = predicted_iou * stability
        keep = _mask_nms(masks, scores, args.mask_nms_threshold, args.max_masks_per_view)
        proposals = ProposalSet(masks[keep], scores[keep], predicted_iou[keep], stability[keep], boxes[keep])
        _save_proposals(cache_path, proposals)
        _save_raw_visualization(
            image_path,
            view_index,
            proposals,
            raw_dir,
            raw_overlay_dir,
            args.output_resolution,
            image,
        )
        _json_dump(
            raw_dir / f"view_{view_index:03d}.json",
            {
                "image": image_path.name,
                "num_sam_masks": len(annotations),
                "num_masks_after_nms": len(proposals.masks),
                "predicted_iou": proposals.predicted_iou.tolist(),
                "stability_score": proposals.stability.tolist(),
                "bbox_xywh": proposals.boxes.tolist(),
            },
        )
        print(
            f"[SAM2] view {view_index:03d}: {len(annotations)} -> {len(proposals.masks)} masks "
            f"({time.time() - started:.1f}s)"
        )
        result.append(proposals)
    _json_dump(manifest_path, cache_signature)
    del generator, model
    torch.cuda.empty_cache()
    return result


def _unique_edges(vertex_superpoints: np.ndarray, faces: np.ndarray) -> np.ndarray:
    face_sp = vertex_superpoints[faces]
    pairs = np.concatenate((face_sp[:, [0, 1]], face_sp[:, [1, 2]], face_sp[:, [2, 0]]), axis=0)
    pairs.sort(axis=1)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    return np.unique(pairs, axis=0).astype(np.int32)


def build_superpoints(
    mesh: trimesh.Trimesh, voxel_size: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    voxel_keys = np.floor(vertices / voxel_size).astype(np.int32)
    _, vertex_superpoints, counts = np.unique(
        voxel_keys, axis=0, return_inverse=True, return_counts=True
    )
    vertex_superpoints = vertex_superpoints.astype(np.int32)
    n_superpoints = int(counts.size)
    centers = np.stack(
        [
            np.bincount(vertex_superpoints, weights=vertices[:, axis], minlength=n_superpoints) / counts
            for axis in range(3)
        ],
        axis=1,
    ).astype(np.float32)
    face_vertices = vertices[faces]
    face_centers = face_vertices.mean(axis=1)
    nearest_corner = np.argmin(np.linalg.norm(face_vertices - face_centers[:, None], axis=2), axis=1)
    face_superpoints = vertex_superpoints[faces[np.arange(len(faces)), nearest_corner]]
    edges = _unique_edges(vertex_superpoints, faces)
    print(
        f"[geometry] {len(vertices):,} vertices, {len(faces):,} faces, "
        f"{n_superpoints:,} superpoints, {len(edges):,} graph edges"
    )
    return vertices, faces, vertex_superpoints, face_superpoints, counts.astype(np.int32), centers, edges


def _clip_vertices(vertices: torch.Tensor, world_to_camera: np.ndarray, intrinsics: np.ndarray, resolution: int):
    transform = torch.as_tensor(world_to_camera, device=vertices.device, dtype=torch.float32)
    ones = torch.ones((len(vertices), 1), device=vertices.device, dtype=torch.float32)
    camera = torch.cat((vertices, ones), dim=1) @ transform.T
    x, y, z = camera[:, 0], camera[:, 1], camera[:, 2]
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    near, far = 0.01, 100.0
    clip_x = (2.0 * fx / resolution) * x + (2.0 * cx / resolution - 1.0) * z
    # nvdiffrast row zero corresponds to NDC y=-1, matching OpenCV row zero with this sign.
    clip_y = (2.0 * fy / resolution) * y + (2.0 * cy / resolution - 1.0) * z
    clip_z = ((far + near) / (far - near)) * z - (2.0 * far * near / (far - near))
    return torch.stack((clip_x, clip_y, clip_z, z), dim=1), camera[:, :3]


def render_and_project(
    context,
    vertices_gpu: torch.Tensor,
    faces_gpu: torch.Tensor,
    vertex_superpoints: np.ndarray,
    world_to_camera: np.ndarray,
    intrinsics: np.ndarray,
    resolution: int,
    depth_tolerance: float,
) -> Projection:
    import nvdiffrast.torch as dr

    clip, camera = _clip_vertices(vertices_gpu, world_to_camera, intrinsics, resolution)
    rast, _ = dr.rasterize(context, clip[None], faces_gpu, resolution=[resolution, resolution])
    face_ids = rast[0, :, :, 3].to(torch.int64) - 1
    depth, _ = dr.interpolate(camera[None, :, 2:3].contiguous(), rast, faces_gpu.contiguous())
    depth = depth[0, :, :, 0]
    depth = torch.where(face_ids >= 0, depth, torch.zeros_like(depth))

    z = camera[:, 2]
    u = intrinsics[0, 0] * camera[:, 0] / z + intrinsics[0, 2]
    v = intrinsics[1, 1] * camera[:, 1] / z + intrinsics[1, 2]
    px = torch.round(u).to(torch.int64)
    py = torch.round(v).to(torch.int64)
    in_frame = (z > 0) & (px >= 0) & (px < resolution) & (py >= 0) & (py < resolution)
    candidate = torch.nonzero(in_frame, as_tuple=False).squeeze(1)
    sampled_depth = depth[py[candidate], px[candidate]]
    rel_error = torch.abs(z[candidate] - sampled_depth) / torch.clamp(sampled_depth, min=1e-6)
    visible = (sampled_depth > 0) & (rel_error < depth_tolerance)
    candidate = candidate[visible]
    weights = 1.0 - rel_error[visible] / depth_tolerance

    candidate_cpu = candidate.cpu().numpy()
    return Projection(
        superpoints=vertex_superpoints[candidate_cpu],
        pixels_x=px[candidate].cpu().numpy().astype(np.int32),
        pixels_y=py[candidate].cpu().numpy().astype(np.int32),
        depth_weights=weights.cpu().numpy().astype(np.float32),
        face_ids=face_ids.cpu().numpy().astype(np.int32),
        depth=depth.cpu().numpy().astype(np.float32),
    )


def prepare_projections(
    segmentation_dir: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_superpoints: np.ndarray,
    world_to_camera: list[np.ndarray],
    intrinsics: list[np.ndarray],
    args: argparse.Namespace,
) -> list[Projection]:
    import nvdiffrast.torch as dr

    visibility_dir = segmentation_dir / "visibility"
    visibility_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    context = dr.RasterizeCudaContext(device=device)
    vertices_gpu = torch.as_tensor(vertices, device=device, dtype=torch.float32)
    faces_gpu = torch.as_tensor(faces, device=device, dtype=torch.int32)
    result: list[Projection] = []
    for view_index, (extrinsics, intr) in enumerate(zip(world_to_camera, intrinsics)):
        projection = render_and_project(
            context,
            vertices_gpu,
            faces_gpu,
            vertex_superpoints,
            extrinsics,
            intr,
            args.work_resolution,
            args.depth_tolerance,
        )
        np.savez_compressed(
            visibility_dir / f"view_{view_index:03d}.npz",
            face_ids=projection.face_ids,
            depth=projection.depth.astype(np.float16),
            visible_superpoints=projection.superpoints,
            pixels_x=projection.pixels_x,
            pixels_y=projection.pixels_y,
            depth_weights=projection.depth_weights.astype(np.float16),
        )
        print(
            f"[projection] view {view_index:03d}: "
            f"{np.count_nonzero(projection.face_ids >= 0):,} raster pixels, "
            f"{len(projection.superpoints):,} visible vertices"
        )
        result.append(projection)
    return result


def _foreground_label_map(proposals: ProposalSet) -> np.ndarray:
    return _label_map(proposals.masks, proposals.scores)


def compute_affinity(
    label_maps: list[np.ndarray],
    projections: list[Projection],
    edges: np.ndarray,
    superpoint_counts: np.ndarray,
    n_superpoints: int,
) -> np.ndarray:
    numerator = np.zeros(len(edges), dtype=np.float64)
    denominator = np.zeros(len(edges), dtype=np.float64)
    edge_a, edge_b = edges[:, 0], edges[:, 1]
    for view_index, (labels, projection) in enumerate(zip(label_maps, projections)):
        sp = projection.superpoints
        point_labels = labels[projection.pixels_y, projection.pixels_x].astype(np.int32)
        foreground = point_labels > 0
        max_label = int(point_labels.max(initial=0))
        if max_label == 0:
            continue
        hist = sparse.coo_matrix(
            (projection.depth_weights[foreground], (sp[foreground], point_labels[foreground] - 1)),
            shape=(n_superpoints, max_label),
            dtype=np.float32,
        ).tocsr()
        norm = np.sqrt(np.asarray(hist.multiply(hist).sum(axis=1)).ravel())
        inv_norm = np.zeros_like(norm)
        inv_norm[norm > 0] = 1.0 / norm[norm > 0]
        hist = sparse.diags(inv_norm) @ hist
        similarity = np.asarray(hist[edge_a].multiply(hist[edge_b]).sum(axis=1)).ravel()

        visible_count = np.bincount(sp, minlength=n_superpoints).astype(np.float32)
        depth_sum = np.bincount(sp, weights=projection.depth_weights, minlength=n_superpoints)
        depth_mean = np.divide(depth_sum, visible_count, out=np.zeros_like(depth_sum), where=visible_count > 0)
        visibility = np.minimum(visible_count / np.maximum(superpoint_counts, 1), 1.0)
        reliability = depth_mean * visibility
        edge_weight = reliability[edge_a] * reliability[edge_b]
        numerator += edge_weight * similarity
        denominator += edge_weight
        print(f"[affinity] accumulated view {view_index:03d}")
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0).astype(np.float32)


def _directed_graph(
    edges: np.ndarray, affinity: np.ndarray, centers: np.ndarray, counts: np.ndarray
) -> list[list[tuple[int, float, float]]]:
    graph: list[list[tuple[int, float, float]]] = [[] for _ in range(len(counts))]
    distances = np.linalg.norm(centers[edges[:, 0]] - centers[edges[:, 1]], axis=1)
    for (left, right), score, distance in zip(edges, affinity, distances):
        weight_lr = float(counts[right]) / max(float(distance), 1e-4)
        weight_rl = float(counts[left]) / max(float(distance), 1e-4)
        graph[int(left)].append((int(right), float(score), weight_lr))
        graph[int(right)].append((int(left), float(score), weight_rl))
    return graph


def region_growing(
    edges: np.ndarray,
    affinity: np.ndarray,
    centers: np.ndarray,
    counts: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, list[list[tuple[int, float, float]]]]:
    graph = _directed_graph(edges, affinity, centers, counts)
    regions = np.full(len(counts), -1, dtype=np.int32)
    seed_order = np.argsort(-counts)
    region_id = 0
    for seed in seed_order:
        if regions[seed] >= 0:
            continue
        regions[seed] = region_id
        queue: deque[int] = deque([int(seed)])
        while queue:
            current = queue.popleft()
            for candidate, _, _ in graph[current]:
                if regions[candidate] >= 0:
                    continue
                weighted_sum = 0.0
                weight_sum = 0.0
                for neighbor, edge_score, region_weight in graph[candidate]:
                    if regions[neighbor] == region_id:
                        weighted_sum += region_weight * edge_score
                        weight_sum += region_weight
                if weight_sum and weighted_sum / weight_sum >= threshold:
                    regions[candidate] = region_id
                    queue.append(candidate)
        region_id += 1
    print(f"[region-growing] {region_id:,} regions at threshold {threshold:.2f}")
    return regions, graph


def guided_mask_selection(
    proposals: list[ProposalSet],
    projections: list[Projection],
    coarse_regions: np.ndarray,
    vertex_superpoints: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[np.ndarray], dict, dict]:
    n_regions = int(coarse_regions.max()) + 1
    vertex_regions = coarse_regions[vertex_superpoints]
    region_sizes = np.bincount(vertex_regions, minlength=n_regions).astype(np.float32)
    all_vectors: list[sparse.csr_matrix] = []
    proposal_views: list[int] = []
    proposal_locals: list[int] = []
    candidate_by_region: list[list[int]] = [[] for _ in range(n_regions)]
    for view_index, (proposal_set, projection) in enumerate(zip(proposals, projections)):
        point_regions = coarse_regions[projection.superpoints]
        visible_counts = np.bincount(point_regions, minlength=n_regions).astype(np.float32)
        frame_visibility = np.minimum(visible_counts / np.maximum(region_sizes, 1), 1.0)
        start = len(all_vectors)
        for local_id, mask in enumerate(proposal_set.masks):
            inside = mask[projection.pixels_y, projection.pixels_x]
            inside_counts = np.bincount(point_regions[inside], minlength=n_regions).astype(np.float32)
            weighted_inside = np.bincount(
                point_regions[inside], weights=projection.depth_weights[inside], minlength=n_regions
            ).astype(np.float32)
            coverage = np.divide(
                weighted_inside,
                visible_counts,
                out=np.zeros_like(weighted_inside),
                where=visible_counts > 0,
            )
            nonzero = np.flatnonzero(coverage > 0)
            vector = sparse.csr_matrix(
                (coverage[nonzero], (np.zeros(len(nonzero), dtype=np.int32), nonzero)),
                shape=(1, n_regions),
            )
            all_vectors.append(vector)
            proposal_views.append(view_index)
            proposal_locals.append(local_id)
            mask_visibility = np.divide(
                inside_counts,
                visible_counts,
                out=np.zeros_like(inside_counts),
                where=visible_counts > 0,
            )
            candidate_regions = np.flatnonzero(
                (frame_visibility > args.frame_visibility) & (mask_visibility > args.mask_visibility)
            )
            global_id = start + local_id
            for region in candidate_regions:
                candidate_by_region[int(region)].append(global_id)

    matrix = sparse.vstack(all_vectors, format="csr")
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    matrix = sparse.diags(np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)) @ matrix
    score_sum = np.zeros(matrix.shape[0], dtype=np.float64)
    score_count = np.zeros(matrix.shape[0], dtype=np.int32)
    groups_used = 0
    for candidate_ids in candidate_by_region:
        if len(candidate_ids) < 2:
            continue
        ids = np.asarray(sorted(set(candidate_ids)), dtype=np.int32)
        similarities = (matrix[ids] @ matrix[ids].T).toarray()
        scores = (similarities.sum(axis=1) - 1.0) / max(len(ids) - 1, 1)
        score_sum[ids] += scores
        score_count[ids] += 1
        groups_used += 1
    consistency = np.divide(
        score_sum, score_count, out=np.zeros_like(score_sum), where=score_count > 0
    ).astype(np.float32)

    refined_maps: list[np.ndarray] = []
    selected_metadata: list[dict] = []
    selected_global_ids: list[int] = []
    for view_index, proposal_set in enumerate(proposals):
        ids = np.flatnonzero(np.asarray(proposal_views) == view_index)
        local_scores = consistency[ids]
        eligible = np.flatnonzero(local_scores > 0)
        if len(eligible):
            keep_relative = _mask_nms(
                proposal_set.masks[eligible],
                local_scores[eligible],
                args.mask_nms_threshold,
                args.max_masks_per_view,
            )
            keep = eligible[keep_relative]
            refined = _label_map(proposal_set.masks[keep], local_scores[keep])
        else:
            keep = np.asarray([], dtype=np.int64)
            refined = np.zeros((args.work_resolution, args.work_resolution), dtype=np.uint16)
        refined_maps.append(refined)
        selected_metadata.append(
            {
                "view": view_index,
                "selected_local_mask_ids": keep.tolist(),
                "consistency_scores": local_scores[keep].tolist(),
            }
        )
        selected_global_ids.extend(ids[keep].tolist())
        print(f"[3DG-MM] view {view_index:03d}: kept {len(keep)}/{len(proposal_set.masks)} masks")
    metadata = {"candidate_groups_used": groups_used, "views": selected_metadata}
    runtime = {
        "coverage_matrix": matrix,
        "candidate_by_region": candidate_by_region,
        "consistency": consistency,
        "proposal_views": np.asarray(proposal_views, dtype=np.int32),
        "proposal_locals": np.asarray(proposal_locals, dtype=np.int32),
        "selected_global_ids": np.asarray(selected_global_ids, dtype=np.int32),
    }
    return refined_maps, metadata, runtime


def refine_regions(
    regions: np.ndarray,
    graph: list[list[tuple[int, float, float]]],
    max_changes: int,
) -> np.ndarray:
    result = regions.copy()
    change_counts = np.zeros(len(result), dtype=np.int8)
    for iteration in range(max_changes + 1):
        changed = 0
        for node in range(len(result)):
            if change_counts[node] >= max_changes:
                continue
            candidates = {int(result[node])}
            candidates.update(int(result[neighbor]) for neighbor, _, _ in graph[node])
            best_region = int(result[node])
            best_score = -math.inf
            for candidate_region in candidates:
                weighted_sum = 0.0
                weight_sum = 0.0
                for neighbor, score, weight in graph[node]:
                    if result[neighbor] == candidate_region:
                        weighted_sum += score * weight
                        weight_sum += weight
                if weight_sum and weighted_sum / weight_sum > best_score:
                    best_score = weighted_sum / weight_sum
                    best_region = candidate_region
            if best_region != result[node]:
                result[node] = best_region
                change_counts[node] += 1
                changed += 1
        print(f"[region-refinement] iteration {iteration + 1}: {changed:,} changes")
        if changed == 0:
            break
    _, result = np.unique(result, return_inverse=True)
    return result.astype(np.int32)


def cluster_mask_instances(
    proposals: list[ProposalSet],
    projections: list[Projection],
    regions: np.ndarray,
    superpoint_counts: np.ndarray,
    association: dict,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[dict]]:
    matrix: sparse.csr_matrix = association["coverage_matrix"]
    candidate_by_region: list[list[int]] = association["candidate_by_region"]
    consistency: np.ndarray = association["consistency"]
    proposal_views: np.ndarray = association["proposal_views"]
    proposal_locals: np.ndarray = association["proposal_locals"]
    selected = set(int(value) for value in association["selected_global_ids"])
    if not selected:
        raise RuntimeError("3D-guided mask matching selected no proposals")

    pair_scores: dict[tuple[int, int], float] = {}
    for group in candidate_by_region:
        ids = sorted(selected.intersection(group))
        if len(ids) < 2:
            continue
        similarities = (matrix[ids] @ matrix[ids].T).toarray()
        for row in range(len(ids)):
            for col in range(row + 1, len(ids)):
                if proposal_views[ids[row]] == proposal_views[ids[col]]:
                    continue
                score = float(similarities[row, col])
                if score >= args.cluster_similarity:
                    pair = (ids[row], ids[col])
                    pair_scores[pair] = max(pair_scores.get(pair, 0.0), score)

    parent = {item: item for item in selected}
    members = {item: {item} for item in selected}
    cluster_views = {item: {int(proposal_views[item])} for item in selected}
    vector_sums = {item: matrix[item].copy() for item in selected}

    def find(item: int) -> int:
        root = item
        while parent[root] != root:
            root = parent[root]
        while parent[item] != item:
            next_item = parent[item]
            parent[item] = root
            item = next_item
        return root

    for (left, right), _ in sorted(pair_scores.items(), key=lambda item: -item[1]):
        root_left, root_right = find(left), find(right)
        if root_left == root_right or cluster_views[root_left] & cluster_views[root_right]:
            continue
        sum_left, sum_right = vector_sums[root_left], vector_sums[root_right]
        numerator = float(sum_left.multiply(sum_right).sum())
        denominator = math.sqrt(float(sum_left.multiply(sum_left).sum()) * float(sum_right.multiply(sum_right).sum()))
        centroid_similarity = numerator / denominator if denominator else 0.0
        if centroid_similarity < args.cluster_similarity:
            continue
        if len(members[root_left]) < len(members[root_right]):
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left
        members[root_left].update(members.pop(root_right))
        cluster_views[root_left].update(cluster_views.pop(root_right))
        vector_sums[root_left] = vector_sums[root_left] + vector_sums.pop(root_right)

    roots = [root for root in members if len(cluster_views[root]) >= args.min_cluster_views]
    roots.sort(key=lambda root: (-len(cluster_views[root]), -len(members[root])))
    if not roots:
        raise RuntimeError(
            f"No mask cluster appeared in at least {args.min_cluster_views} views at similarity "
            f"{args.cluster_similarity}"
        )
    global_to_cluster: dict[int, int] = {}
    for cluster_id, root in enumerate(roots):
        for global_id in members[root]:
            global_to_cluster[global_id] = cluster_id

    n_superpoints = len(superpoint_counts)
    votes = np.zeros((n_superpoints, len(roots)), dtype=np.float32)
    support = np.zeros(len(roots), dtype=np.int64)
    for global_id, cluster_id in global_to_cluster.items():
        view = int(proposal_views[global_id])
        local = int(proposal_locals[global_id])
        projection = projections[view]
        mask = proposals[view].masks[local]
        inside = mask[projection.pixels_y, projection.pixels_x]
        sp = projection.superpoints[inside]
        weights = projection.depth_weights[inside]
        visible_count = np.bincount(projection.superpoints, minlength=n_superpoints).astype(np.float32)
        weighted_inside = np.bincount(sp, weights=weights, minlength=n_superpoints).astype(np.float32)
        coverage = np.divide(
            weighted_inside, visible_count, out=np.zeros_like(weighted_inside), where=visible_count > 0
        )
        quality = max(float(consistency[global_id]), 0.05)
        votes[:, cluster_id] += quality * coverage
        support[cluster_id] += int(len(sp))

    n_regions = int(regions.max()) + 1
    region_votes = np.zeros((n_regions, len(roots)), dtype=np.float64)
    for cluster_id in range(len(roots)):
        np.add.at(region_votes[:, cluster_id], regions, votes[:, cluster_id] * superpoint_counts)
    region_vertices = np.bincount(regions, weights=superpoint_counts, minlength=n_regions)
    region_scores = np.divide(
        region_votes,
        region_vertices[:, None],
        out=np.zeros_like(region_votes),
        where=region_vertices[:, None] > 0,
    )
    region_cluster = np.argmax(region_scores, axis=1)
    region_confidence = region_scores[np.arange(n_regions), region_cluster]
    region_cluster[region_confidence <= 0] = -1
    superpoint_cluster = region_cluster[regions]

    cluster_vertices = np.asarray(
        [np.sum(superpoint_counts[superpoint_cluster == cluster_id]) for cluster_id in range(len(roots))],
        dtype=np.int64,
    )
    keep_clusters = np.flatnonzero(
        (cluster_vertices >= args.min_instance_vertices) & (support >= args.min_instance_support)
    )
    keep_clusters = keep_clusters[np.argsort(-cluster_vertices[keep_clusters])]
    cluster_to_instance = np.zeros(len(roots), dtype=np.uint16)
    metadata: list[dict] = []
    for instance_id, cluster_id in enumerate(keep_clusters, start=1):
        cluster_to_instance[cluster_id] = instance_id
        root = roots[int(cluster_id)]
        mask_records = [
            {
                "view": int(proposal_views[global_id]),
                "local_mask_id": int(proposal_locals[global_id]),
                "consistency_score": float(consistency[global_id]),
            }
            for global_id in sorted(members[root])
        ]
        metadata.append(
            {
                "instance_id": instance_id,
                "num_vertices": int(cluster_vertices[cluster_id]),
                "mask_support_samples": int(support[cluster_id]),
                "visible_views": len(cluster_views[root]),
                "source_masks": mask_records,
            }
        )
    instance_per_superpoint = np.zeros(n_superpoints, dtype=np.uint16)
    assigned = superpoint_cluster >= 0
    instance_per_superpoint[assigned] = cluster_to_instance[superpoint_cluster[assigned]]
    print(
        f"[instances] {len(selected)} selected masks -> {len(roots)} 3D mask clusters -> "
        f"{len(metadata)} supported instances"
    )
    return instance_per_superpoint, metadata


def _palette(instance_id: int) -> tuple[int, int, int]:
    hue = (instance_id * 0.618033988749895) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.72, 1.0)
    return int(red * 255), int(green * 255), int(blue * 255)


def _overlay(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
    colors = np.zeros_like(image)
    for instance_id in np.unique(labels):
        if instance_id:
            colors[labels == instance_id] = _palette(int(instance_id))
    alpha = (labels > 0)[..., None].astype(np.float32) * 0.55
    return np.clip(image * (1.0 - alpha) + colors * alpha, 0, 255).astype(np.uint8)


def export_consistent_masks(
    segmentation_dir: Path,
    image_paths: list[Path],
    vertices: np.ndarray,
    faces: np.ndarray,
    face_superpoints: np.ndarray,
    instance_per_superpoint: np.ndarray,
    world_to_camera: list[np.ndarray],
    intrinsics_work: list[np.ndarray],
    args: argparse.Namespace,
) -> tuple[list[dict], list[np.ndarray]]:
    import nvdiffrast.torch as dr

    consistent_dir = segmentation_dir / "masks_consistent"
    overlay_dir = segmentation_dir / "overlays"
    consistent_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    context = dr.RasterizeCudaContext(device=device)
    vertices_gpu = torch.as_tensor(vertices, device=device, dtype=torch.float32)
    faces_gpu = torch.as_tensor(faces, device=device, dtype=torch.int32)
    view_records: list[dict] = []
    output_labels: list[np.ndarray] = []
    scale = args.output_resolution / args.work_resolution
    for view_index, (image_path, extrinsics, intr_work) in enumerate(
        zip(image_paths, world_to_camera, intrinsics_work)
    ):
        intr = intr_work.copy()
        intr[0] *= scale
        intr[1] *= scale
        clip, _ = _clip_vertices(vertices_gpu, extrinsics, intr, args.output_resolution)
        rast, _ = dr.rasterize(
            context, clip[None], faces_gpu, resolution=[args.output_resolution, args.output_resolution]
        )
        face_ids = (rast[0, :, :, 3].to(torch.int64) - 1).cpu().numpy()
        labels = np.zeros(face_ids.shape, dtype=np.uint16)
        valid = face_ids >= 0
        labels[valid] = instance_per_superpoint[face_superpoints[face_ids[valid]]]
        Image.fromarray(labels).save(consistent_dir / f"view_{view_index:03d}.png")
        image = np.asarray(Image.open(image_path).convert("RGB").resize(labels.shape[::-1], Image.Resampling.LANCZOS))
        Image.fromarray(_overlay(image, labels)).save(overlay_dir / f"view_{view_index:03d}.jpg", quality=92)
        ids, counts = np.unique(labels[labels > 0], return_counts=True)
        view_records.append(
            {
                "view": view_index,
                "image": image_path.name,
                "instance_pixel_counts": {str(int(i)): int(c) for i, c in zip(ids, counts)},
            }
        )
        output_labels.append(labels)
        print(f"[export] view {view_index:03d}: {len(ids)} visible instances")
    return view_records, output_labels


def export_instance_ply(
    path: Path,
    mesh: trimesh.Trimesh,
    vertex_superpoints: np.ndarray,
    instance_per_superpoint: np.ndarray,
) -> None:
    labels = instance_per_superpoint[vertex_superpoints]
    colors = np.zeros((len(labels), 4), dtype=np.uint8)
    colors[:, 3] = 255
    for instance_id in np.unique(labels):
        if instance_id:
            colors[labels == instance_id, :3] = _palette(int(instance_id))
    colored = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces), vertex_colors=colors, process=False
    )
    colored.export(path)


def evaluate_against_scannetpp(
    scan_dir: Path,
    vertex_superpoints: np.ndarray,
    instance_per_superpoint: np.ndarray,
) -> dict:
    segments_path = scan_dir / "segments.json"
    annotations_path = scan_dir / "segments_anno.json"
    if not segments_path.exists() or not annotations_path.exists():
        return {"available": False}
    with segments_path.open("r", encoding="utf-8") as handle:
        segment_ids = np.asarray(json.load(handle)["segIndices"], dtype=np.int64)
    with annotations_path.open("r", encoding="utf-8") as handle:
        groups = json.load(handle)["segGroups"]
    segment_to_gt: dict[int, int] = {}
    labels: dict[int, str] = {}
    for gt_id, group in enumerate(groups, start=1):
        labels[gt_id] = group.get("label", "")
        for segment in group["segments"]:
            segment_to_gt[int(segment)] = gt_id
    gt = np.fromiter((segment_to_gt.get(int(segment), 0) for segment in segment_ids), dtype=np.int32)
    pred = instance_per_superpoint[vertex_superpoints].astype(np.int32)
    gt_ids = np.unique(gt[gt > 0])
    pred_ids = np.unique(pred[pred > 0])
    gt_sizes = np.bincount(gt)
    pred_sizes = np.bincount(pred)
    intersections: dict[tuple[int, int], int] = {}
    both = (gt > 0) & (pred > 0)
    for pair, count in zip(*np.unique(np.stack((gt[both], pred[both]), axis=1), axis=0, return_counts=True)):
        intersections[(int(pair[0]), int(pair[1]))] = int(count)
    best_records: list[dict] = []
    for gt_id in gt_ids:
        best_iou = 0.0
        best_pred = 0
        for pred_id in pred_ids:
            intersection = intersections.get((int(gt_id), int(pred_id)), 0)
            if intersection == 0:
                continue
            union = int(gt_sizes[gt_id] + pred_sizes[pred_id] - intersection)
            iou = intersection / union
            if iou > best_iou:
                best_iou, best_pred = iou, int(pred_id)
        best_records.append(
            {
                "gt_instance_id": int(gt_id),
                "label": labels[int(gt_id)],
                "best_pred_instance_id": best_pred,
                "best_iou": best_iou,
            }
        )
    best_ious = np.asarray([record["best_iou"] for record in best_records], dtype=np.float32)
    return {
        "available": True,
        "note": "GT is evaluation-only; it is never used for proposal association or region merging.",
        "num_gt_instances": len(best_records),
        "num_pred_instances": int(len(pred_ids)),
        "mean_best_iou": float(best_ious.mean()) if len(best_ious) else 0.0,
        "recall_at_iou_0.25": float(np.mean(best_ious >= 0.25)) if len(best_ious) else 0.0,
        "recall_at_iou_0.50": float(np.mean(best_ious >= 0.50)) if len(best_ious) else 0.0,
        "matches": best_records,
    }


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    segmentation_dir = output_dir / "segmentation"
    segmentation_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted((output_dir / "scene").glob("view_*.png"))
    if len(image_paths) != 16:
        raise ValueError(f"Expected exactly 16 scene views in {output_dir / 'scene'}, found {len(image_paths)}")
    source_dir = _source_scene_dir(output_dir)
    scan_dir = source_dir / "scans"
    mesh_path = scan_dir / "mesh_aligned_0.05.ply"
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)

    started = time.time()
    _, world_to_camera, intrinsics = _load_cameras(output_dir, args.work_resolution)
    proposals = generate_sam_proposals(image_paths, segmentation_dir, args)
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    (
        vertices,
        faces,
        vertex_superpoints,
        face_superpoints,
        superpoint_counts,
        superpoint_centers,
        edges,
    ) = build_superpoints(mesh, args.voxel_size)
    projections = prepare_projections(
        segmentation_dir,
        vertices,
        faces,
        vertex_superpoints,
        world_to_camera,
        intrinsics,
        args,
    )

    coarse_maps = [_foreground_label_map(item) for item in proposals]
    coarse_affinity = compute_affinity(
        coarse_maps, projections, edges, superpoint_counts, len(superpoint_counts)
    )
    coarse_regions, _ = region_growing(
        edges, coarse_affinity, superpoint_centers, superpoint_counts, args.merge_threshold
    )
    refined_maps, matching_metadata, association = guided_mask_selection(
        proposals, projections, coarse_regions, vertex_superpoints, args
    )
    refined_affinity = compute_affinity(
        refined_maps, projections, edges, superpoint_counts, len(superpoint_counts)
    )
    refined_regions, refined_graph = region_growing(
        edges, refined_affinity, superpoint_centers, superpoint_counts, args.merge_threshold
    )
    refined_regions = refine_regions(refined_regions, refined_graph, args.max_region_changes)
    instance_per_superpoint, instance_metadata = cluster_mask_instances(
        proposals, projections, refined_regions, superpoint_counts, association, args
    )
    view_records, output_labels = export_consistent_masks(
        segmentation_dir,
        image_paths,
        vertices,
        faces,
        face_superpoints,
        instance_per_superpoint,
        world_to_camera,
        intrinsics,
        args,
    )
    if not args.skip_ply:
        export_instance_ply(
            segmentation_dir / "mesh_instances.ply", mesh, vertex_superpoints, instance_per_superpoint
        )

    evaluation = evaluate_against_scannetpp(scan_dir, vertex_superpoints, instance_per_superpoint)
    valid_ids = set(int(value) for value in np.unique(instance_per_superpoint) if value)
    mask_ids = set(int(value) for labels in output_labels for value in np.unique(labels) if value)
    validation = {
        "all_mask_ids_exist_in_3d": mask_ids.issubset(valid_ids),
        "num_3d_instances": len(valid_ids),
        "num_instances_visible_in_masks": len(mask_ids),
        "num_views": len(output_labels),
        "mask_shape": list(output_labels[0].shape),
        "mask_dtype": str(output_labels[0].dtype),
        "id_source": "Every 2D pixel ID is rasterized from one shared 3D superpoint instance label.",
        "scannetpp_gt_geometric_best_match": evaluation,
    }
    if not validation["all_mask_ids_exist_in_3d"]:
        raise AssertionError("A 2D mask contains an instance ID that is absent from the 3D result")

    _json_dump(
        segmentation_dir / "instances.json",
        {
            "method": "MV3DIS-inspired geometry-first multi-view mask matching",
            "source_scene": str(source_dir),
            "input_views": [str(path) for path in image_paths],
            "parameters": {
                "sam_model": "SAM2.1 Hiera Small",
                "work_resolution": args.work_resolution,
                "output_resolution": args.output_resolution,
                "voxel_size_m": args.voxel_size,
                "depth_relative_tolerance": args.depth_tolerance,
                "frame_visibility_threshold": args.frame_visibility,
                "mask_visibility_threshold": args.mask_visibility,
                "merge_threshold": args.merge_threshold,
                "cluster_similarity_threshold": args.cluster_similarity,
                "minimum_cluster_views": args.min_cluster_views,
                "max_region_changes": args.max_region_changes,
            },
            "matching": matching_metadata,
            "instances": instance_metadata,
            "views": view_records,
        },
    )
    _json_dump(segmentation_dir / "validation.json", validation)
    print(
        f"[done] {output_dir.name}: {len(instance_metadata)} instances, 16 consistent masks, "
        f"{time.time() - started:.1f}s"
    )


if __name__ == "__main__":
    main()
