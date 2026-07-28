# PERSONA-1 gating research — findings

The RFC (§27.3, R1) flags two questions as architecture-shaping and requires them answered *before*
implementation. Both are now answered. This note records the findings; `ROADMAP_5.0.0.md` reflects the
resulting decisions.

## Survey 1 — Face-landmark / face-mesh ControlNet availability, per family

Category: **(a)** dedicated face-mesh/landmark CN · **(b)** OpenPose CN that *includes* face keypoints · **(c)** none.

| Family | Cat. | Repo(s) | Conditioning | License | Dedicated face-landmark? |
|---|---|---|---|---|---|
| SD 1.5 | a+b | `CrucibleAI/ControlNetMediaPipeFace`; `lllyasviel/control_v11p_sd15_openpose` | MediaPipe 468-pt mesh + pupils; OpenPose incl. face | OpenRAIL-M | **AVAILABLE** (only mesh CN) |
| SD 2.1 | a+b | `CrucibleAI/ControlNetMediaPipeFace` (SD2.1-base is its primary target) | mesh + pupils | OpenRAIL-M | **AVAILABLE** |
| SDXL | b | `xinsir/controlnet-openpose-sdxl-1.0` (**Apache-2.0**); `thibaud/...` | OpenPose incl. face keypoints | Apache / OpenRAIL | ABSENT (coarse keypoints only) |
| SD 3.5 | b | `InstantX/SD3-Controlnet-Pose` | OpenPose incl. face | SD3 non-commercial | ABSENT (official CNs = blur/canny/depth only) |
| PixArt-Σ | c | `PixArt-alpha/PixArt-ControlNet` (HED) | edge only | OpenRAIL-ish | **ABSENT entirely** |
| Flux | b | `Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro` | union "pose" incl. face keypoints | FLUX.1-dev non-commercial | ABSENT (keypoints via union only) |
| Stable Cascade | c | (canny/inpaint/super-res only) | — | Stability community | **ABSENT** (a face CN was staged then deleted, never shipped) |
| Sana | c | `Efficient-Large-Model/Sana_600M_1024px_ControlNet_HED` | edge only | NSCL non-commercial | **ABSENT entirely** |

**Conclusion.** Dedicated face-mesh conditioning exists for **exactly one lineage — SD 1.5 / 2.1**
(one OpenRAIL-M checkpoint: usable + redistributable-with-license, but not pure-permissive, so an
optional separately-licensed add-on, never bundled into the Unlicense tree). **Every DiT/MMDiT family
— SDXL, SD3.5, Flux, PixArt-Σ, Sana, Cascade — lacks a dedicated face-landmark ControlNet**; the best
they offer is coarse face keypoints inside an OpenPose union CN (SDXL/SD3.5/Flux) or nothing
(PixArt/Sana/Cascade). **Face-mesh geometry is NOT a portable cross-family primitive.**

**Architectural decision (RFC §27.3 "negative result reshapes the architecture"):** the **face-swap
bridge (SCRFD → ArcFace → inswapper + restore, already in plakat, Tier B §11.5) is the PRIMARY
geometric/identity path across families.** Layer 2's face-landmark-CN map is an SD1.5/2.1-only
enhancement (Tier A); the geometry engine's cross-family value is its **depth / pose-skeleton / region-
mask / detail-overlay** outputs, not the mesh map. This lowers R1 from a blocker to a per-family bonus.

## Survey 2 — Dense facial-landmark aligner (net-new port)

| Option | Points | ONNX | License | Port effort | Verdict |
|---|---|---|---|---|---|
| InsightFace `2d106det` (buffalo_l) | 106 | yes | **models NON-COMMERCIAL / research-only** | trivial (like SCRFD) | **REJECTED — license-poisoned** |
| **PIPNet** (`yakhyo/pipnet-onnx`) | **98 (WFLW)** / 68 | **prebuilt ONNX** | **MIT** | easy (ResNet-18 + pixel-in-pixel decode) | **RECOMMENDED** |
| MediaPipe FaceMesh | 468 (+iris) | community | Apache-2.0 | moderate (tflite graph, iris submodel) | clean, only if dense 3D mesh needed |
| FAN / face-alignment | 68 (2D+3D) | torch→onnx | BSD-3 | heavier (hourglass) | clean fallback |

**Conclusion.** The canonical 106-pt InsightFace aligner is **license-poisoned** (non-commercial
weights) and must be rejected, exactly as a public-domain project should. **Port PIPNet-98 (WFLW, MIT,
prebuilt ONNX) via the existing `convert-onnx` path** — denser than 68-pt, license-clean, least
effort. This is a **topology change vs the RFC's assumed 106-pt InsightFace set** (§10.1): the landmark
topology becomes **WFLW-98**, and the named anchor regions (§8.2/§10.1) + every probe must be defined
against WFLW-98. Feed this back into the RFC/lexicon as the frozen topology v1. MediaPipe FaceMesh is
the license-clean escape hatch if a dense 3D mesh is later required; FAN is a 68-pt BSD fallback.

## §2.3 baselines — the control numbers

Harness committed: `tools/reference/persona_baseline.py` (renders N seeds via the plakat CLI or reads
a dir → InsightFace SCRFD+ArcFace pairwise-cosine identity variance + detection-failure rate +
best-effort OWL-ViT localized-detail hit/side rate). It is a **compute job** (multi-family render) to
schedule; the measurement half runs on any image dir today. Results are committed as a corpus entry
and every later phase reports the same statistics.

> Detail-hit measurement caveat: OWL-ViT on a 4-pixel mole is exactly the §2.1.4 problem, so the
> baseline's detail-hit number is a noisy proxy until the Phase-1 `local_anomaly` probe (which needs
> the WFLW-98 aligner) replaces it.
