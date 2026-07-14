# Multi-view RGB segmentation consistency 조사

조사 기준일: **2026-07-14**

범위: 2024–2026년 논문·공식 프로젝트를 중심으로, 여러 RGB 뷰에서 SAM/SAM2/SAM3 등이 만든 mask의 경계·instance ID·granularity를 일관되게 만드는 방법과 GenRecon 적용 가능성을 검토한다.

## 요약

결론부터 말하면, **GenRecon에는 SAM2 video tracking만 단독으로 붙이는 것보다, 재구성된 3D mesh를 공통 기준으로 삼아 mask를 다시 연결하는 geometry-first 방식이 더 적합하다.** 구체적으로는 2026년 CVPR의 **MV3DIS**처럼 다음 과정을 쓰는 방안이 가장 현실적이다.

1. 각 입력 뷰에서 SAM2.1 또는 SAM3로 다중 mask proposal을 만든다.
2. GenRecon의 원본 고해상도 mesh를 각 카메라로 depth/triangle-ID rasterization한다.
3. mesh를 작은 superface/superpoint로 과분할하고, 각 mask가 실제로 덮는 3D primitive 집합을 구한다.
4. depth consistency, visible area, mask confidence를 가중치로 사용해 뷰 간 mask를 하나의 3D instance ID로 묶는다.
5. 합의된 3D instance를 모든 뷰로 다시 투영하고, 이 projection을 SAM2의 mask/box prompt로 넣어 경계를 한 번 정제한다.

이 선택의 이유는 다음과 같다.

- SAM2의 memory는 **시간적으로 인접한 video frame**에는 강하지만, 큰 viewpoint jump가 있는 sparse multi-view에는 3D correspondence를 보장하지 않는다. SAM2Object도 continuous/dense views가 필요하다는 한계를 명시한다.
- 현재 GenRecon은 카메라 순서에서 `np.linspace`로 뷰를 균등 샘플링하므로, 16/32-view inference 입력은 원본 영상보다 frame gap이 크다.
- 반면 GenRecon에는 posed camera와 reconstruction mesh가 이미 있다. 즉, 최신 방법들이 별도로 구해야 하는 3D anchor를 후처리 단계에서 바로 활용할 수 있다.
- MV3DIS는 ScanNet++에서 전체 뷰의 5%만 사용한 설정에서도 geometry-guided mask matching의 효과를 보고했고, 코드도 공개되어 있다.

SAM2 tracking은 버릴 필요가 없다. 원본 iPhone capture처럼 dense하고 시간 순서가 확실한 경우에는 **SAM2/SAM2Long의 bidirectional track을 proposal 생성기**로 사용하고, 마지막 ID 결정은 여전히 3D 합의가 맡는 hybrid가 좋다.

## 먼저 구분해야 할 네 가지 consistency

“segmentation consistency”는 하나의 문제가 아니다.

| 종류 | 실패 예 | 필요한 해결책 |
| --- | --- | --- |
| 경계 consistency | 같은 의자가 한 뷰에서는 통째로, 다른 뷰에서는 등받이만 선택됨 | multi-view vote, 3D boundary/visibility, SAM 재-prompt |
| instance-ID consistency | 같은 의자가 뷰마다 다른 ID를 받거나 두 의자의 ID가 바뀜 | tracking 또는 3D mask association |
| semantic consistency | 동일 물체가 `chair`/`stool`처럼 다른 class를 받음 | SAM3/CLIP 계열 semantic aggregation과 3D instance별 voting |
| granularity consistency | `table`과 `table top/leg`가 뷰마다 다른 단계로 분할됨 | hierarchical representation, scale-conditioned/ultrametric feature field |

SAM2는 주로 두 번째 문제를 **시간축**에서 완화한다. 반면 GenRecon 같은 static scene의 sparse views에서는 네 문제를 모두 3D 공간에서 정리할 필요가 있다. 또한 SAM2의 기본 출력은 class-agnostic mask이므로, semantic label이 필요하면 SAM3, GroundingDINO/CLIP 등의 별도 recognition 단계가 필요하다.

## 2024–2026 연구 흐름

### 1. Video memory와 tracking으로 mask ID를 전파

| 연구 | 핵심 | 장점 | multi-view에서의 한계 |
| --- | --- | --- | --- |
| [SAM 2 (2024)](https://arxiv.org/abs/2408.00714) | streaming memory로 prompt mask를 video 전체에 전파 | 바로 사용 가능하고 interactive correction 지원 | camera geometry를 모르며 sparse/unordered view를 video로 간주할 근거가 없음 |
| [Cutie, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Cheng_Putting_the_Object_Back_into_Video_Object_Segmentation_CVPR_2024_paper.html) | pixel memory에 object-level query memory를 추가 | distractor가 있는 VOS에서 identity 유지가 강함 | 역시 시간적 연속성을 가정하며 3D consistency 자체를 강제하지 않음 |
| [SAM2Long, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Ding_SAM2Long_Enhancing_SAM_2_for_Long_Video_Segmentation_with_a_ICCV_2025_paper.pdf) | 여러 mask hypothesis를 유지하는 constrained memory tree, IoU/occlusion 기반 memory 선택 | training-free이고 장기 occlusion·오류 누적에 SAM2보다 강함 | sparse viewpoint jump나 서로 만나지 않는 camera trajectory를 해결하지는 않음 |
| [SAM 3, ICLR 2026](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/) | text/exemplar concept로 해당하는 모든 instance를 검출·분할·추적 | semantic target 자동화와 multi-object 초기화에 유리 | video tracker는 SAM2 계열 memory이므로 3D geometric identity 보장은 별도 문제 |

이 계열은 원본 capture가 촘촘한 영상일 때 강력하다. 특히 SAM2Object 구현은 forward/backward tracking 결과를 합친다. 그러나 GenRecon inference에서 선택된 8/16/32개 뷰만 tracker에 넣으면 viewpoint gap 때문에 mask drift와 ID switch가 늘 수 있다. 가능하면 **선택 전 원본 dense sequence에서 tracking하고**, 선택된 GenRecon view의 mask만 취하는 편이 낫다.

### 2. 2D mask를 3D primitive에 올린 뒤 합의

| 연구 | consistency mechanism | 입력/출력 | GenRecon 적합성 |
| --- | --- | --- | --- |
| [SAI3D, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Yin_SAI3D_Segment_Any_Instance_in_3D_Scenes_CVPR_2024_paper.html) | scene을 geometric primitive로 과분할한 뒤 multi-view SAM mask affinity로 hierarchical region growing | posed RGB-D + 3D scene → class-agnostic 3D instances | 구현이 비교적 단순한 baseline으로 좋지만 framewise SAM inconsistency를 직접 고치지는 않음 |
| [SAM2Object, CVPR 2025](https://jihuaizhaohd.github.io/SAM2Object/) | keyframe SAM2 mask를 양방향 tracking하고, distance-weighted mask consolidation 후 superpoint graph clustering | dense posed RGB-D sequence + point cloud → 3D instances | ScanNet/ScanNet++용 [공식 코드](https://github.com/jihuaizhaohd/SAM2Object)가 있어 출발점으로 좋음. 단, 논문이 continuous/dense views 필요성을 한계로 명시 |
| [Any3DIS, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Nguyen_Any3DIS_Class-Agnostic_3D_Instance_Segmentation_by_2D_Mask_Tracking_CVPR_2025_paper.html) | 3D superpoint별 최적 pivot view 선택 → SAM2 양방향 track → dynamic programming으로 유용한 view만 선택 | RGB-D sequence + point cloud → proposal bank | 모든 frame을 같은 가중치로 합치지 않는 점이 유용. 공식 프로젝트가 연결한 코드는 현재 unofficial임 |
| [MV3DIS, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Zhao_MV3DIS_Multi-View_Mask_Matching_via_3D_Guides_for_Zero-Shot_3D_CVPR_2026_paper.html) | coarse 3D segment projection을 공통 anchor로 삼아 mask를 매칭하고, depth/visibility weight로 occlusion을 억제한 뒤 region refinement | sparse posed RGB-D + point cloud → 3D instances | **가장 추천.** tracker보다 sparse views에 적합하며 [공식 코드](https://github.com/zybjn/MV3DIS) 공개 |

이 흐름의 핵심 변화는 “2D mask끼리 IoU를 비교”하는 데서 “두 2D mask가 **같은 3D surface를 덮는지** 비교”하는 쪽으로 이동한 것이다. MV3DIS는 coarse 3D segment를 각 뷰에 투영해 공통 reference로 사용하고, z-buffer depth와 맞지 않는 projection을 낮은 가중치로 처리한다. 이는 의자 뒤의 벽이 의자 mask에 섞이는 occlusion 오류를 줄인다.

동일한 MV3DIS 논문의 ScanNet++ 표에서는 SAM2Object가 `mAP/AP50/AP25 = 20.2/34.1/48.7`, MV3DIS가 `22.0/36.7/51.7`을 기록한다. 특히 ScanNet++에서는 SAM2 automatic mask와 **5% view sampling**을 사용했다. 서로 다른 논문의 숫자를 임의로 섞은 비교가 아니라 동일 논문·동일 표의 비교라는 점에서, GenRecon의 sparse-view 조건에 가장 직접적인 근거다. 다만 sensor depth 대신 생성 mesh depth를 쓰면 geometry error가 추가되므로 별도 검증이 필요하다.

### 3. inconsistent 2D mask를 3D feature field가 흡수

| 연구 | 표현과 학습 방식 | 잘 맞는 목적 | 비용/한계 |
| --- | --- | --- | --- |
| [GARField, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Kim_GARField_Group_Anything_with_Radiance_Fields_CVPR_2024_paper.html) | physical scale-conditioned affinity field로 서로 충돌하는 SAM mask를 coarse-to-fine hierarchy로 융합 | object/part/group을 여러 granularity로 선택 | scene별 radiance-field optimization 필요 |
| [OmniSeg3D, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Ying_OmniSeg3D_Omniversal_3D_Segmentation_via_Hierarchical_Contrastive_Learning_CVPR_2024_paper.html) | mask hierarchy로 3D feature를 pull/push하는 hierarchical contrastive learning | 전체 scene의 hierarchical segmentation과 interactive selection | scene별 feature-field optimization과 differentiable rendering 필요. [공식 코드](https://github.com/THU-luvision/OmniSeg3D)는 MIT이나 환경이 구형 CUDA/NeRF stack 중심 |
| [Ultrametric Feature Fields, ECCV 2024](https://arxiv.org/abs/2405.19678) | triangle inequality보다 강한 ultrametric 구조를 학습해 threshold만으로 계층적 grouping | transitive하고 안정적인 object-part hierarchy | instance ID만 필요한 경우에는 과한 per-scene optimization |
| [Gaussian Grouping, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04195.pdf) | 각 3D Gaussian에 compact identity encoding을 두고 SAM mask supervision + 3D spatial regularization | 3DGS scene의 segmentation과 editing | GenRecon은 mesh/O-Voxel 출력이므로 Gaussian scene을 새로 최적화해야 함 |

이 계열은 서로 다른 뷰에서 SAM이 object와 part를 다르게 자르는 **granularity ambiguity**를 가장 잘 다룬다. 반면 사용자가 원하는 것이 “각 물체에 고정 ID를 부여한 PLY/GLB” 정도라면, feature field를 새로 학습하는 것보다 geometry graph voting이 훨씬 가볍다. GenRecon 내부 학습에 segmentation feature를 장기적으로 통합할 때는 OmniSeg3D의 contrastive formulation이 더 참고할 만하다.

### 4. 처음부터 multi-view segmentation을 예측

| 연구 | 핵심 | 상태 | GenRecon 관점 |
| --- | --- | --- | --- |
| [PanSt3R, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Zust_PanSt3R_Multi-view_Consistent_Panoptic_Segmentation_ICCV_2025_paper.html) | MUSt3R 3D feature + DINOv2 2D feature + 모든 view가 공유하는 instance query로 geometry와 panoptic mask를 한 번에 예측 | [공식 코드](https://github.com/naver/panst3r) 공개, checkpoint는 non-commercial license | unposed RGB에서도 빠르지만 GenRecon reconstruction을 대체/병렬 수행하며 license 확인 필요 |
| [MV-SAM (arXiv 2026)](https://arxiv.org/abs/2601.17866) | unposed image의 pointmap으로 SAM image/prompt embedding을 3D에 lift하고 3D positional embedding으로 mask decoding | 2026-07-14 현재 [project page](https://jaesung-choe.github.io/mv_sam/)의 code는 TBU | 방향은 가장 이상적이지만 지금 당장 재현 가능한 dependency로 보기 어려움 |
| [V²-SAM, CVPR 2026](https://arxiv.org/abs/2511.20886) | DINOv3 기반 anchor prompt와 visual prompt 전문가를 만들고 cycle consistency로 더 믿을 출력을 선택 | [project page](https://jianchengpan.space/projects/V2-SAM/)에 code/model 공개 | geometry가 없는 큰 viewpoint 변화에 유용하지만 ego–exo correspondence 중심이며 전체 indoor instance 자동 분할기는 아님 |

이 계열은 per-scene optimization 없이 consistency를 모델 자체에 넣는 최신 방향이다. 다만 현재 GenRecon에 바로 결합하기에는 별도 대형 모델·checkpoint·license 부담이 크고, GenRecon이 이미 알고 있는 정확한 camera pose를 충분히 활용하지 못할 수 있다.

## GenRecon에서 확인되는 조건

현재 코드 기준으로 중요한 사실은 다음과 같다.

- [`inference/get_images.py`](inference/get_images.py#L170-L208)는 camera list가 capture order라고 가정하고 `np.linspace`로 균등한 sparse subset을 고른다. 즉, 선택된 view에는 temporal continuity가 약하다.
- scene image와 per-chunk 2D condition은 같은 제한된 pool에서 선택된다([`get_images.py`](inference/get_images.py#L251-L304)). segmentation을 reconstruction conditioning과 정확히 맞추려면 이 동일한 view list를 사용해야 한다.
- 각 view의 normalized intrinsics와 OpenCV world-to-camera extrinsics는 `cameras.json`에 저장된다([`get_images.py`](inference/get_images.py#L306-L332)).
- `--save_imgs`를 주면 실제 모델에 들어간 1024×1024 crop/resize 이미지를 `scene/view_*.png`로 저장한다([`reconstruct_scene.py`](reconstruct_scene.py#L183-L187), [`reconstruct_scene.py`](reconstruct_scene.py#L384-L394)).
- reconstruction 후에는 world-space `mesh.ply`와 chunk PLY가 생성된다([`reconstruct_scene.py`](reconstruct_scene.py#L450-L473)).
- GenRecon은 이미 동일한 intrinsics/extrinsics로 image patch를 3D point/voxel에 투영하는 공통 함수를 갖고 있다([`genrecon/modules/cond_3D/projection.py`](genrecon/modules/cond_3D/projection.py#L7-L47)). segmentation feature를 모델 내부로 넣는 장기안에서 이 convention을 재사용할 수 있다.

### 좌표와 crop에서 주의할 점

후처리 구현에서 가장 쉽게 생길 수 있는 오류다.

1. `cameras.json`의 `extrinsics_c0`는 **chunk-0 local frame → camera**이고, 저장된 `mesh.ply`는 world frame이다. `chunk_inputs.pt`의 `M_chunk_to_original = M_c2o`를 사용하면 column-vector convention에서 다음처럼 바꿔야 한다.

   ```text
   E_world_to_cam = E_c0_to_cam @ inverse(M_c0_to_world)
   ```

2. `cameras.json`의 intrinsics는 crop/resize 후 image와 대응하지만 `img_path`는 원본 파일을 가리킬 수 있다. 따라서 가장 안전한 방법은 reconstruction에 `--save_imgs`를 사용하고 `scene/view_*.png`를 segment하는 것이다. 그렇지 않으면 crop box도 metadata에 저장하도록 코드를 확장해야 한다.
3. geometry consensus에는 decimated `scene_viewer.glb`가 아니라 **원본 `mesh.ply` 또는 `to_glb_inputs.pt` geometry**를 써야 한다. viewer decimation은 작은 물체와 instance boundary를 없앨 수 있다. 최종 label만 decimated mesh로 nearest-triangle/barycentric transfer한다.

## 권장 구현: Geometry-first mask consensus

### 입력

- `scene/view_XXX.png` 또는 crop 정보가 보존된 원본 RGB
- `cameras.json`
- `mesh.ply` 또는 `to_glb_inputs.pt`
- `chunk_inputs.pt`의 좌표 변환
- 선택 사항: 원본 dense video, sensor depth, ScanNet++ source mesh/point cloud

### 단계 1 — 2D proposal 생성

- 모든 물체를 자동으로 찾는 목적이면 SAM2.1 automatic mask generator를 기본 baseline으로 둔다.
- 찾을 class가 정해져 있으면 SAM3 text concept으로 instance 후보를 만들고 SAM2식 mask refinement를 사용한다.
- dense iPhone video가 있으면 full sequence에서 SAM2Long 또는 SAM2Object식 forward/backward tracking을 먼저 수행한다.
- 각 mask에 `view_id`, local `mask_id`, predicted IoU/stability, bbox, area, optional DINO/CLIP embedding을 저장한다.

### 단계 2 — mesh visibility buffer 생성

각 카메라에 대해 원본 mesh를 rasterize해 다음 map을 만든다.

- `depth[v, y, x]`
- `face_id[v, y, x]`
- `normal[v, y, x]` (선택)

한 pixel의 3D primitive가 mask에 포함되더라도, projected depth와 z-buffer depth 차이가 tolerance보다 크면 occluded sample로 버린다. tolerance는 world-space 고정값 하나보다 depth와 pixel footprint에 비례시키는 편이 안전하다.

### 단계 3 — 3D primitive와 mask association

mesh face graph를 normal/curvature/color discontinuity로 과분할해 superface를 만든다. 이후 각 2D mask를 “보이는 superface의 weighted coverage vector”로 바꾼다.

```text
coverage(mask m, superface s)
  = Σ visible pixels of s [pixel ∈ m] · depth_weight · incidence_weight
    / Σ visible pixels of s depth_weight · incidence_weight
```

두 mask의 affinity는 다음 항을 조합한다.

- 동일한 3D superface를 덮는 정도
- depth-consistent visible area
- mask embedding cosine similarity
- 2D boundary와 projected 3D boundary의 일치도
- mask confidence/stability

같은 view의 서로 겹치는 두 instance는 동일 cluster에 들어가지 못하게 하고, 전체 mask graph를 connected components 또는 constrained agglomerative clustering으로 묶는다. 단순 pairwise union은 transitive error에 약하므로, 각 cluster가 설명하는 3D support와 새 mask의 support를 함께 비교한다.

### 단계 4 — 3D label 결정과 재투영

- superface별 instance probability를 weighted vote로 구한다.
- 작은 isolated component를 제거하고, geometry adjacency를 사용해 hole을 채운다.
- 합의된 3D label을 각 view로 투영해 `consistent_mask`를 만든다.
- 이 mask 또는 bbox/positive-negative points를 SAM2에 다시 prompt해 RGB boundary를 정제한다.
- 정제 mask로 3D vote를 한 번 더 수행한다. 반복은 1회부터 시작하고, confidence가 실제로 좋아질 때만 추가한다.

이 방식은 SAM이 제공하는 정교한 image boundary와 mesh가 제공하는 global instance identity를 분리해 사용한다. 생성 mesh가 관측 image와 다른 부분은 무조건 합의시키지 않고 `unknown/low_confidence`로 남기는 것이 중요하다.

### 권장 출력 형식

```text
<output>/segmentation/
├── masks_raw/
│   └── view_XXX.npz
├── masks_consistent/
│   └── view_XXX.png
├── visibility/
│   └── view_XXX.npz
├── instances.json
├── mesh_instances.ply
└── scene_instances.glb
```

`instances.json`에는 최소한 다음을 둔다.

```json
{
  "coordinate_frame": "world",
  "instances": [
    {
      "instance_id": 1,
      "semantic_label": null,
      "confidence": 0.91,
      "source_views": [0, 3, 7],
      "face_count": 12345
    }
  ]
}
```

PLY에는 기존 PBR attribute를 유지하면서 vertex 또는 face 단위 `instance_id`, `semantic_id`, `confidence`를 추가한다. GLB는 instance별 primitive/node로 분리하거나, 원본 textured GLB와 별도의 integer ID texture/metadata를 함께 제공할 수 있다. viewer GLB를 instance별로 물리 분할하면 draw call이 늘 수 있으므로, 많은 instance에는 ID texture 방식이 더 효율적이다.

## 단계별 도입안

### P0 — training-free 후처리 baseline

새 스크립트 하나로 reconstruction 이후에 실행한다.

```text
reconstruct_scene.py
  → mesh.ply + cameras.json + saved scene images
  → SAM2/SAM3 raw proposals
  → mesh rasterization
  → geometry-weighted graph consensus
  → mesh_instances.ply + consistent 2D masks
```

처음에는 MV3DIS 전체를 이식하기보다 다음 최소 기능으로 성능을 확인한다.

- SAM2.1 automatic masks
- face-ID/depth rasterization
- superface graph
- depth/visibility-weighted co-membership affinity
- region growing + reprojected masks

### P1 — dense-video hybrid

ScanNet/ScanNet++ iPhone처럼 원본 순서가 있는 경우에만 SAM2Long/SAM2Object식 bidirectional tracking을 추가한다. tracker ID는 최종 정답이 아니라 graph의 affinity prior로 취급한다. tracking과 3D support가 충돌하면 3D support 또는 `uncertain` 판정을 우선한다.

### P2 — GenRecon conditioning에 통합

후처리 효과가 확인된 뒤에만 학습 파이프라인을 건드리는 것이 좋다.

- `SelectedImages`에 mask/instance feature tensor를 추가한다.
- image encoder patch와 같은 해상도로 mask embedding을 만들고 기존 `project_points_to_patches` convention으로 voxel에 투영한다.
- 같은 surface를 보는 view feature에는 pull loss, 서로 다른 3D instance에는 push loss를 적용한다.
- chunk overlap에서는 동일 world-space instance embedding을 공유한다.
- geometry/texture generation이 segmentation boundary를 침범하지 않도록 boundary-aware loss를 추가한다.

이 단계는 OmniSeg3D의 hierarchical contrastive learning을 GenRecon의 projection-based conditioning에 맞게 단순화한 형태다. 잘못된 pseudo-mask가 생성 geometry 자체를 망칠 수 있으므로, P0 결과와 confidence filtering 없이 바로 학습에 넣는 것은 권하지 않는다.

## 평가 계획

### 데이터

- ScanNet++ validation scene을 우선 사용한다. 공식 benchmark는 5% decimated scan mesh vertex의 instance mask를 `AP`, `AP50`, `AP25`로 평가한다([공식 문서](https://scannetpp.mlsg.cit.tum.de/scannetpp/benchmark/insseg)).
- 생성 mesh와 GT scan mesh의 topology가 다르므로, GenRecon mesh label을 nearest-surface 방식으로 GT mesh에 옮기거나 양쪽을 공통 point sample로 평가해야 한다.
- 현재 생성된 두 ScanNet++ scene은 qualitative smoke test로 쓰고, annotation이 있는 validation scene을 여러 개 추가해 수치 평가한다.

### baseline/ablation

1. framewise SAM2 automatic mask + 단순 3D vote
2. SAM2 bidirectional tracking + 단순 3D vote
3. geometry-first consensus
4. geometry-first consensus + SAM re-prompt
5. 가능하면 source scan geometry와 GenRecon geometry 각각 사용

각 방법을 8/16/32/full views에서 비교한다. source view selection은 현재 GenRecon과 동일하게 유지한다.

### metric

- 공식 3D instance `mAP/AP50/AP25`
- view pair별, 3D correspondence 영역에서 Hungarian-matched mask IoU
- 한 GT instance가 몇 prediction으로 쪼개지는지 보는 fragmentation count
- 한 prediction에 몇 GT instance가 합쳐지는지 보는 merge error
- ID switch 수와 track coverage
- reprojected boundary F-score
- unlabeled/low-confidence surface 비율
- scene당 runtime, peak GPU memory, mask/metadata 저장 용량

## 최종 선택 가이드

| 조건 | 권장안 |
| --- | --- |
| 연속된 iPhone video, 빠른 prototype | SAM2.1/SAM2Long bidirectional tracking + 3D validation |
| 현재처럼 8–32개 sparse posed views | **MV3DIS-inspired geometry-first consensus** |
| object/part hierarchy가 핵심 | OmniSeg3D 또는 ultrametric/scale-conditioned feature field |
| semantic text query가 핵심 | SAM3 proposal + geometry-first instance association |
| camera pose도 없고 forward-only 결과가 필요 | PanSt3R; license와 별도 checkpoint 고려 |
| interactive prompt를 여러 sparse view에 전파 | MV-SAM이 유망하지만 현재 code 공개 대기, 그 전에는 V²-SAM/3D re-prompt 방식 |

따라서 GenRecon의 첫 구현은 **SAM2.1 proposal + 원본 mesh rasterization + MV3DIS식 depth/visibility-weighted graph + SAM2 re-prompt**로 잡는 것이 좋다. 이 경로는 모델 재학습 없이 현재 산출물에 붙일 수 있고, ScanNet++ 공식 3D instance annotation으로 검증할 수 있으며, 이후 PLY/GLB의 물체별 선택·표시에도 직접 연결된다.

## 주요 자료

### 2024

- Ravi et al., [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714).
- Cheng et al., [Putting the Object Back into Video Object Segmentation (Cutie)](https://openaccess.thecvf.com/content/CVPR2024/html/Cheng_Putting_the_Object_Back_into_Video_Object_Segmentation_CVPR_2024_paper.html), CVPR 2024.
- Yin et al., [SAI3D: Segment Any Instance in 3D Scenes](https://openaccess.thecvf.com/content/CVPR2024/html/Yin_SAI3D_Segment_Any_Instance_in_3D_Scenes_CVPR_2024_paper.html), CVPR 2024.
- Kim et al., [GARField: Group Anything with Radiance Fields](https://openaccess.thecvf.com/content/CVPR2024/html/Kim_GARField_Group_Anything_with_Radiance_Fields_CVPR_2024_paper.html), CVPR 2024.
- Ying et al., [OmniSeg3D](https://openaccess.thecvf.com/content/CVPR2024/html/Ying_OmniSeg3D_Omniversal_3D_Segmentation_via_Hierarchical_Contrastive_Learning_CVPR_2024_paper.html), CVPR 2024.
- He et al., [View-Consistent Hierarchical 3D Segmentation Using Ultrametric Feature Fields](https://arxiv.org/abs/2405.19678), ECCV 2024.
- Ye et al., [Gaussian Grouping: Segment and Edit Anything in 3D Scenes](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04195.pdf), ECCV 2024.

### 2025

- Zhao et al., [SAM2Object](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhao_SAM2Object_Consolidating_View_Consistency_via_SAM2_for_Zero-Shot_3D_Instance_CVPR_2025_paper.pdf), CVPR 2025.
- Nguyen et al., [Any3DIS](https://openaccess.thecvf.com/content/CVPR2025/html/Nguyen_Any3DIS_Class-Agnostic_3D_Instance_Segmentation_by_2D_Mask_Tracking_CVPR_2025_paper.html), CVPR 2025.
- Ding et al., [SAM2Long](https://openaccess.thecvf.com/content/ICCV2025/papers/Ding_SAM2Long_Enhancing_SAM_2_for_Long_Video_Segmentation_with_a_ICCV_2025_paper.pdf), ICCV 2025.
- Žust et al., [PanSt3R](https://openaccess.thecvf.com/content/ICCV2025/html/Zust_PanSt3R_Multi-view_Consistent_Panoptic_Segmentation_ICCV_2025_paper.html), ICCV 2025.
- Carion et al., [SAM 3: Segment Anything with Concepts](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/), released 2025, ICLR 2026.

### 2026

- Zhao et al., [MV3DIS: Multi-View Mask Matching via 3D Guides](https://openaccess.thecvf.com/content/CVPR2026/html/Zhao_MV3DIS_Multi-View_Mask_Matching_via_3D_Guides_for_Zero-Shot_3D_CVPR_2026_paper.html), CVPR 2026.
- Pan et al., [V²-SAM: SAM2 with Multi-Prompt Experts for Cross-View Object Correspondence](https://arxiv.org/abs/2511.20886), CVPR 2026.
- Jeong et al., [MV-SAM: Multi-view Promptable Segmentation using Pointmap Guidance](https://arxiv.org/abs/2601.17866), arXiv 2026; 2026-07-14 기준 code TBU.
