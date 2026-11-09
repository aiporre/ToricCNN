# ToricCNN (Python port)

Pure-Python / PyTorch port of the MATLAB code for Maron et al., *Convolutional
Neural Networks on Surfaces via Seamless Toric Covers* (SIGGRAPH 2017). No
MATLAB and no `.mat` files: every intermediate file is a `.npz` that loads with
`np.load(..., allow_pickle=False)`.

## Install

```bash
conda activate pmconvTG
pip install -e .            # add [test] for pytest
```

## Pipeline

```bash
# 1. meshes -> toric images (per mesh: <stem>.ply + <stem>.txt, one 1-based label per face)
toriccnn-make-data data/train --out data/train_processed --prefix train --ext ply \
    --num-triplets 300 --size 512 --seed 0 --workers 8
toriccnn-make-data data/test  --out data/test_processed  --prefix test

# 2. train FCN-32s (resumes automatically from the last net-epoch-N.pt in --exp-dir)
toriccnn-train --data data/train_processed --exp-dir runs/exp --epochs 50 \
    --lr 1e-4 --batch-size 10 --sub-batches 5 --num-classes 8 [--amp] [--no-pretrained]

# 3. per-triplet predictions lifted to the mesh
toriccnn-predict --exp-dir runs/exp --data data/test_processed --out runs/exp/pred [--plots]

# 4. aggregate the triplets of each mesh (scale-weighted) and report accuracy
toriccnn-aggregate --pred-dir runs/exp/pred --out runs/exp/agg --meshes data/test_processed [--plots]
```

Every command also runs as `python -m toriccnn.<module>` and has `--help`.

## File formats

| file | keys |
|---|---|
| `triplets/{prefix}_{stem}_tri_{j}.npz` | `data` (H,W,26) uint8, `seg` (H,W) int16 0-based, `V`, `F` (area-normalised mesh), flattening: `V_flat`, `T_cut`, `V_cut`, `cut_to_uncut`, `cones`, `path_pairs`, `path_lens` (`Flattening.from_arrays`) |
| `meshes/{prefix}_{stem}.npz` | `functions` (n,26) in [0,255], `labels` (m,) 0-based face labels, `V`, `F`, `triplets` (k,3) |
| `selected_points/*.png` | AGD-coloured mesh with the sampled cones |
| `exp/split.json`, `stats.npz`, `config.json`, `log.csv`, `net-epoch-N.pt` | train/val file lists; channel `mean`, `covariance`, `class_counts`; arguments; per-epoch loss / pixel acc / mean acc / mIoU; model + optimizer state |
| `pred/pred_{name}.npz` | `scores_on_mesh` (n,K) raw max-projected logits, `prediction` (n,), `gt_on_mesh` (n,), `scale` (n,), `V`, `F`, `cones` |
| `agg/{mesh}_combined.npz` | `combined_scores`, `prediction`, `gt_on_mesh`, `vertex_accuracy`, and with `--meshes`: `face_labels`, `face_prediction`, `face_accuracy` |

Accuracy in `toriccnn-aggregate`: vertex GT is the majority label of incident
faces; the area-weighted face accuracy uses, per face, the argmax of the
combined scores summed over its three vertices.

## Differences from the MATLAB code

* **WKS**: the Voronoi-area bug of the original WKS code is fixed.
* **Loss**: the class-distribution-scaled loss uses the prior 4000 consistently
  (MATLAB used 4000 in forward and 4 in backward); gradients come from autograd.
* **Max-projection**: exact autograd gradient of the max over the 4 torus
  copies instead of the hand-written backward.
* **Initialisation**: ImageNet VGG16 (torchvision) instead of the PASCAL-VOC
  FCN-32s MatConvNet weights (`--no-pretrained` for random init). The model
  multiplies its mean-subtracted [0,255] input by `1/(255*0.226)` (a buffer
  saved in the checkpoint) to match the input scale torchvision's VGG16 expects.
* **Network mode**: `--net-mode matlab` (default) is the faithful FCN-32s with a
  100-px first pad; it is only translation-equivariant away from the image
  border. `--net-mode toric` circularly pads every layer on the true torus and
  is exactly equivariant to toric shifts (H, W multiples of 32).
* **Rasterisation**: a software rasteriser (`toriccnn.raster`) replaces
  capturing MATLAB figures; images are written exactly at `--size`.
* **Lifting**: images are lifted to the mesh by sampling the full periodic
  image with torus wrap-around (`toriccnn.lift`) instead of the `Tiler` +
  quarter-image crop.
* Per-triplet failures in data generation are logged and skipped instead of
  aborting the run; gradient accumulation (`--sub-batches`) mirrors
  MatConvNet's `numSubBatches` with the derivative divided by batch size.
