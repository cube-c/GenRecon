# Object-Aware Generation for Multi-View 3D Scene Reconstruction

## Objective

Reconstruct and generate an object-aware PBR scene from posed multi-view RGB
images. The target representation should preserve scene-level consistency while
making individual objects separable for editing and simulation.

## Motivation

Scene reconstruction and generation require two properties that are often in
tension:

- **Scene consistency:** objects must have coherent relative placement, scale,
  and geometry across the full scene.
- **Object separability:** individual objects should have explicit geometry and
  materials so they can be selected, edited, replaced, or used in downstream
  simulation.

We focus on the scene-to-object level of the hierarchy. Fine-grained,
within-object part decomposition and articulation are out of scope for the
initial system. Where needed, object-level supervision can be derived from
synthetic datasets with ground-truth 3D instance segmentation, such as
3D-FRONT.

## Findings

- GenRecon shows that conditioning a TRELLIS.2 generative prior with posed
  multi-view images and geometry-aware 3D features can produce high-quality
  scene reconstructions.
- Its joint scene representation is valuable because it preserves spatial and
  geometric consistency among multiple objects. However, a single fused mesh
  limits editability and simulatability.
- Recovering object-aware geometry therefore requires dense 3D assignments at
  the level of points, mesh faces, or an equivalent spatial representation.
  These assignments can guide per-object refinement, including completion of
  surfaces that are occluded or fused with adjacent structures in the global
  reconstruction.
- Existing approaches such as OmniPart use axis-aligned bounding-box
  annotations for objects or parts. This is a poor fit for scene-level
  segmentation and refinement when objects have arbitrary orientations or
  non-axis-aligned extents.

## Proposed Direction

Use GenRecon for scene-level generation, then derive a dense 3D object
segmentation from its intermediate or final representation. Each object segment
is refined into a separable PBR asset while retaining the global scene layout.

```text
posed multi-view RGB images
  -> GenRecon scene reconstruction / generation
  -> dense 3D object segmentation
  -> per-object geometry completion and refinement
  -> separable PBR scene assets
```

The key research question is how to represent and predict the dense object
assignments so that object boundaries are reliable without sacrificing the
global consistency provided by scene-level generation.

## Input

- Posed multi-view RGB images.
- Optionally, intermediate or final GenRecon representations, such as sparse
  structure (SS), shape/texture SLat features, meshes, or point samples.

## Output

- A PBR scene with separable, object-aware geometry and materials.
- Dense object-instance assignments for the generated 3D representation.
