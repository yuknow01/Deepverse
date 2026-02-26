# Dataset information
# Basic Deepverse
## Deepverse Dataset (DT31) – Data Shape & Configuration

This repo assumes the Deepverse dataset is organized as:

- Root folder: `scenarios/`
- Scenario name: `DT31`
- Scenario path: `scenarios/DT31`

---

## 1) Dataset Root & Scenario

- `dataset.params['dataset_folder'] = "scenarios"`
- `dataset.params['scenario'] = "DT31"`
- `dataset.scenario_path = "scenarios/DT31"`
- `dataset.scenario = ScenarioManage(...)`

---

## 2) Coverage (Scenes / Basestations)

- Scenes are configured via:
  - `dataset.params['scenes'] = [1..100]`
  - You can change the range in: `scenarios/DT31/param/config.m`
- Basestations:
  - `dataset.params['basestations'] = [1]`

---

## 3) Enabled Modalities (Sub-dataset Handles)

When enabled, each modality is exposed as a dataset handle:

- `dataset.camera_dataset   = CameraDataset(...)`  — Camera data
- `dataset.lidar_dataset    = LiDARDataset(...)`   — LiDAR data
- `dataset.mobility_dataset = MobilityDataset(...)`— Mobility / pose / position data
- [dataset.comm_dataset = CommunicationDataset(...)](#dataset-wireless-communication) — Wireless communication (OFDM channel) data
- `dataset.radar_dataset    = RadarDataset(...)`   — Radar (FMCW) data

---

## 4) Sensor / Data Toggles

Example toggles used in this setup:

- `dataset.params['camera']    = True`
- `dataset.params['camera_id'] = ['unit1_cam1']`
- `dataset.params['lidar']     = True`
- `dataset.params['lidar_id']  = ['unit1_lidar1']`
- `dataset.params['position']  = True`

---

## 5) Communication Settings (`dataset.params['comm']`)

- `enable = True`

### Antenna Setup
- **BS antenna**
  - `shape = [16, 1]`
  - `rotation = [0, 0, -45.04]`
  - `spacing = 0.5`
  - `FoV = [360, 180]`
- **UE antenna**
  - `shape = [1, 1]`
  - `rotation = [0, 0, 0]`
  - `spacing = 0.5`
  - `FoV = [360, 180]`

### OFDM
- `bandwidth = 0.05`
- `subcarriers = 512`
- `selected_subcarriers = [0..7]`

### Channel Options
- `activate_RX_filter = 0`
- `generate_OFDM_channels = 1`
- `num_paths = 25`
- `enable_Doppler = 1`

---

## 6) Radar Settings (`dataset.params['radar']`)

### Antenna Setup
- **TX antenna**
  - `shape = [1, 1]`
  - `rotation = [0, 0, -45.04]`
  - `spacing = 0.5`
  - `FoV = [180, 180]`
- **RX antenna**
  - `shape = [16, 1]`
  - `rotation = [0, 0, -45.04]`
  - `spacing = 0.5`
  - `FoV = [180, 180]`

### FMCW
- `chirp_slope = 8.014e12`
- `Fs = 6.2e6`
- `n_samples_per_chirp = 256`
- `n_chirps = 256`

### Ray Paths
- `num_paths = 50000`

---

# Wireless Data Analyze
## Dataset wireless communication
`dataset.comm_dataset = CommunicationDataset(...)` — Wireless communication (OFDM channel) data

## Location / Position (Reference)
### loc data array
<img width="648" height="418" alt="loc data array" src="https://github.com/user-attachments/assets/b97e5fb3-5b95-459c-beb6-f62ed6075fd7" />

### scene loc
<img width="289" height="252" alt="scene loc" src="https://github.com/user-attachments/assets/227fb9cc-7e6c-43e5-aff3-0fddbfe77a0c" />

## Channel information
**Tensor shape:** `(Rx, Tx, Subcarrier)`
- `Rx`: BS antenna index (e.g., 1)
- `Tx`: UE antenna index (e.g., 16)
- `Subcarrier`: OFDM subcarrier index (e.g., 512)
- Optional subset: `selected_subcarriers = [0..7]`

<img width="745" height="350" alt="channel shape (Rx, Tx, subcarrier)" src="https://github.com/user-attachments/assets/86387a14-eefb-48db-b575-e6a7632ffbde" />

---

### 수식 설명
https://chatgpt.com/s/t_694d3f0b31fc819195d061e9b84e8846

-------------------------------


# Multi-Modal Next-Step Channel Prediction (DeepVerse + `multi_modal_lwm` Fine-tuning)
````md
This repo implements a **one-step forecasting** pipeline that predicts the **next wireless channel vector** from:
- **Past channel coefficients** (complex OFDM CSI → real/imag features)
- **Past camera frames** (RGB images)

The codebase includes:
1) DeepVerse dataset loading (DT31)  
2) Channel/image preprocessing  
3) Time-series multi-modal `Dataset` (past → next)  
4) Fine-tuning wrapper on top of `multi_modal_lwm` backbone  
5) Training loop (AMP + grad clipping + cosine scheduler + best checkpoint)

---

## Experiment Setup (TL;DR)

- **Scenario:** `DT31`
- **Scenes:** `range(100)` (configured in code via `param_manager.params["scenes"]`)
- **Subcarriers:** `selected_subcarriers = range(64)` (in code)
- **Task:** **Next-step prediction**
  - input window covers time `t-15 ... t` (**16 frames**)
  - target is channel at `t+1`
- **Train/Val split:** **75% / 25%** (by dataset index order)
- **Channel feature dimension:**
  - `F_in = F_out = 2 * N_selected_subcarriers`
  - with 64 selected subcarriers → `F_in = F_out = 128` (real/imag concat)
- **Image input (from dataset):** time-stacked frames
  - `img_past`: `(B, 16, 3, 224, 224)`

---

## 0) End-to-End Pipeline (At a Glance)

```text
DeepVerse Scenario (DT31)
  ├─ comm_dataset   : per frame -> ue -> coeffs (complex)
  └─ camera_dataset : per time -> image file path

Preprocess
  ├─ coeffs complex -> [real, imag] -> min-max scaling (train stats) -> concat -> flatten
  └─ image path -> resize(224) -> ImageNet normalize -> tensor

Dataset (past_len = 16)
  input:
    ch_past  : (B, T=16, F_in)
    img_past : (B, T=16, 3, 224, 224)
  target:
    y_next   : (B, F_out)  where F_out == F_in

Model
  FinetuneChannelPredictor:
    ch_past -> in_proj(F_in→...→16) -> backbone(multi_modal_lwm) -> tokens(B,T,64)
            -> pool(last/mean) -> head(64→F_out) -> yhat(B,F_out)

Train
  loss  : MSE(yhat, y)
  metric: NMSE(dB)
  AMP + grad clip + AdamW + CosineAnnealingLR + best checkpoint saving
````

---

## 1) Environment & Imports

**Dependencies**

* Python, PyTorch
* Pillow (`PIL`) for image loading
* DeepVerse
* Backbone: `lwm_multi_model.py` providing `multi_modal_lwm`

---

## 2) Data Loading (DeepVerse)

```python
scenarios_name = "DT31"
config_path = f"scenarios/{scenarios_name}/param/config.m"
param_manager = ParameterManager(config_path)

# example configuration
param_manager.params["scenes"] = list(range(100))
param_manager.params["comm"]["OFDM"]["selected_subcarriers"] = list(range(64))

dataset = Dataset(param_manager)

comm = dataset.comm_dataset
cam_sensor = dataset.camera_dataset.sensors["unit1_cam1"]
```

---

## 3) Channel Extraction & Preprocessing

### 3.1 Robust coeffs accessor

DeepVerse frame structures can vary (list/tuple vs single object). This helper extracts `coeffs` safely.

```python
def get_coeffs_from_frame(frame, ue_idx=0):
    ue_obj = frame["ue"]

    if isinstance(ue_obj, (list, tuple)):
        ch_obj = ue_obj[ue_idx]
    else:
        ch_obj = ue_obj

    if hasattr(ch_obj, "coeffs"):
        return ch_obj.coeffs

    if isinstance(ch_obj, dict) and "coeffs" in ch_obj:
        return ch_obj["coeffs"]

    raise TypeError(f"Cannot get coeffs. ue type={type(ue_obj)}, ch type={type(ch_obj)}")
```

### 3.2 Train-only min/max stats (avoid leakage)

Compute scaling statistics **only on training indices**.

```python
def get_train_min_max_realimag(frames, train_idx, us_idx=0):
    rmin, rmax = float("inf"), float("-inf")
    imin, imax = float("inf"), float("-inf")

    for t in train_idx:
        frame = frames[t]
        coeffs = get_coeffs_from_frame(frame, us_idx)  # (N_subcarriers,)

        rmin = min(rmin, float(coeffs.real.min()))
        rmax = max(rmax, float(coeffs.real.max()))
        imin = min(imin, float(coeffs.imag.min()))
        imax = max(imax, float(coeffs.imag.max()))

    return (rmin, rmax), (imin, imax)
```

### 3.3 Min-Max scaling + concat real/imag

Transforms complex channel coeffs into a real-valued vector.

* input: complex `(N_subcarriers,)`
* output: real `(2*N_subcarriers,)`

```python
def preprocess_channel_coeffs_minmax(coeffs_np, r_min, r_max, i_min, i_max, device="cuda", eps=1e-12):
    coeffs = torch.from_numpy(coeffs_np).to(torch.complex64)

    r = coeffs.real
    i = coeffs.imag

    r_scaled = (r - r_min) / max(r_max - r_min, eps)
    i_scaled = (i - i_min) / max(i_max - i_min, eps)

    H = torch.cat([r_scaled, i_scaled], dim=-1).to(device)
    return H
```

---

## 4) Image Preprocessing (Resize + ImageNet Normalization)

```python
IMG_SIZE = 224

def preprocess_img(path, img_size=IMG_SIZE, device="cuda"):
    img = Image.open(path).convert("RGB")
    arr = np.array(img)

    x = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float() / 255.0
    x = x.unsqueeze(0)  # (1,3,H,W)

    x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype).view(1, 3, 1, 1)
    x = (x - mean) / std

    return x.to(device, non_blocking=True)  # (1,3,224,224)
```
출처
https://docs.pytorch.org/vision/0.8/models.html

---

## 5) Dataset: Multi-Modal Time-Series (Past → Next)

### Task Definition & Train/Val Split

Each dataset sample corresponds to an index `idx` mapped to:

* `t = valid_start + idx`

**Inputs**

* Channel history: `H(t-15), ..., H(t)`  → `(16, F_in)`
* Image history:   `I(t-15), ..., I(t)`  → `(16, 3, 224, 224)`

**Target**

* Next channel: `H(t+1)` → `(F_out,)`
  In this pipeline: `F_out == F_in`.

**Valid index range**

* `valid_start = past_len - 1`
* `valid_end   = N - 2`  (since target uses `t+1`)

**Split**

* `train = first 75%`, `val = last 25%` (index order)

> Note: The dataset returns **time-stacked images** `(B, 16, 3, 224, 224)`.
> If your backbone expects `(B, 3, 224, 224)`, apply temporal pooling (e.g. `img.mean(dim=1)`) or encode per-frame then fuse.

### 5.1 Flatten comm frames

DeepVerse comm frames can be nested; flatten into a 1D time list.

```python
def flatten_comm_frames(comm):
    frames = []
    for row in comm.data:
        for d in row:
            frames.append(d)
    return frames
```

### 5.2 Dataset implementation

```python
class MultiModalNextStepDatasetGPU(TorchDataset):
    def __init__(self, comm_frames, cam_files, ue_idx=0, past_len=15, device="cuda",
                 r_min=0.0, r_max=1.0, i_min=0.0, i_max=1.0):
        self.comm_frames = comm_frames
        self.cam_files = list(cam_files)
        self.ue_idx = ue_idx
        self.past_len = past_len
        self.device = device

        self.r_min, self.r_max = r_min, r_max
        self.i_min, self.i_max = i_min, i_max

        self.N = min(len(self.comm_frames), len(self.cam_files))
        self.valid_start = past_len - 1
        self.valid_end = self.N - 2  # because target uses t+1

    def __len__(self):
        return self.valid_end - self.valid_start + 1

    def __getitem__(self, idx):
        t = self.valid_start + idx

        # 1) image past: (T,3,224,224)
        img_list = []
        for k in range(t - self.past_len + 1, t + 1):
            img_k = preprocess_img(self.cam_files[k], device=self.device).squeeze(0)
            img_list.append(img_k)
        img = torch.stack(img_list, dim=0)

        # 2) channel past: (T,F_in)
        ch_list = []
        for k in range(t - self.past_len + 1, t + 1):
            coeffs_np = get_coeffs_from_frame(self.comm_frames[k], ue_idx=self.ue_idx)
            h = preprocess_channel_coeffs_minmax(
                coeffs_np, self.r_min, self.r_max, self.i_min, self.i_max, device=self.device
            ).reshape(-1)
            ch_list.append(h)
        channel_past = torch.stack(ch_list, dim=0)

        # 3) target: (F_out)
        coeffs_np_next = get_coeffs_from_frame(self.comm_frames[t + 1], ue_idx=self.ue_idx)
        target = preprocess_channel_coeffs_minmax(
            coeffs_np_next, self.r_min, self.r_max, self.i_min, self.i_max, device=self.device
        ).reshape(-1)

        return channel_past, img, target
```

---

## 6) Model: `FinetuneChannelPredictor` (Backbone Adapter)

### 6.1 Motivation

* Dataset channel vector size: `F_in = 2 * N_selected_subcarriers` (often large)
* Backbone expects a small per-step embedding dimension (e.g., `element_length=16`)

So the wrapper provides:

* `in_proj` : `(F_in → 16)` per time step
* `head`    : `(d_model=64 → F_out)` to match the next-step target vector

### 6.2 Architecture

```text
ch:  (B,T,F_in)
  -> in_proj: (F_in -> 512 -> 128 -> 16)
  -> backbone: multi_modal_lwm(ch_emb, img) -> tokens (B,T,64)  (assumed)
  -> pool: last or mean over T
  -> head: (64 -> F_out)
  -> yhat (B,F_out)
```

### 6.3 Implementation

```python
class FinetuneChannelPredictor(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        F_in: int,
        F_out: int,
        pool: str = "last",
        freeze_image: bool = False,
        freeze_backbone: bool = False,
        element_length: int = 16,
        d_model: int = 64
    ):
        super().__init__()
        self.backbone = backbone
        self.pool = pool

        self.in_proj = nn.Sequential(
            nn.Linear(F_in, 512),
            nn.GELU(),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, element_length)
        )

        self.head = nn.Linear(d_model, F_out)

        if freeze_image:
            for p in self.backbone.image_embedding.parameters():
                p.requires_grad = False

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            for p in self.in_proj.parameters():
                p.requires_grad = True
            for p in self.head.parameters():
                p.requires_grad = True

    def forward(self, ch, img):
        ch = self.in_proj(ch)               # (B,T,16)
        tokens = self.backbone(ch, img)     # (B,T,64)  (assumed)
        if self.pool == "last":
            z = tokens[:, -1, :]            # (B,64)
        elif self.pool == "mean":
            z = tokens.mean(dim=1)          # (B,64)
        else:
            raise ValueError(f"Unknown pool={self.pool}")
        return self.head(z)                 # (B,F_out)
```

---

## 7) Training: Loss, Metric, AMP, Scheduler, Checkpoint

### 7.1 Metric: NMSE (dB)

```python
@torch.no_grad()
def nmse_db(yhat, y, eps=1e-12):
    num = torch.sum((yhat - y) ** 2, dim=1)
    den = torch.sum(y ** 2, dim=1).clamp_min(eps)
    nmse = num / den
    return 10.0 * torch.log10(nmse.clamp_min(eps)).mean()
```

### 7.2 Training loop (AMP + grad clipping)

* **Loss:** MSE
* **Optimizer:** AdamW (`lr=1e-4`, `weight_decay=1e-4`)
* **Scheduler:** CosineAnnealingLR (`T_max=epochs`)
* **Checkpoint:** save `best_finetune.pt` based on **best val loss**

```python
def train_one_epoch(model, loader, optimizer, device, use_amp=True, grad_clip=1.0):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    total_loss, total_nmse, n = 0.0, 0.0, 0

    for ch, img, y in loader:
        ch = ch.to(device, non_blocking=True)
        img = img.to(device, non_blocking=True)
        y  = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            yhat = model(ch, img)
            loss = F.mse_loss(yhat, y)

        scaler.scale(loss).backward()

        if grad_clip and grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_nmse += nmse_db(yhat.detach(), y).item()
        n += 1

    return total_loss / max(n, 1), total_nmse / max(n, 1)
```

---

## 8) Shapes & Sanity Checks

| Tensor | Shape                  |
| ------ | ---------------------- |
| `ch`   | `(B, 16, F_in)`        |
| `img`  | `(B, 16, 3, 224, 224)` |
| `y`    | `(B, F_out)`           |

**Sanity recommendation**

* After applying min-max scaling, `y` should be approximately within `[0, 1]` (numerical tolerances apply).
* Confirm `model(ch, img)` produces `(B, F_out)`.

---

## 9) Output Artifacts

* `best_finetune.pt`

  * `epoch`
  * `model_state`
  * `optimizer_state`
  * `F_in`, `F_out`

---

## 10) Common Pitfalls

1. **Image time dimension mismatch**

* Dataset returns `(B, 16, 3, 224, 224)`
* Backbone may expect `(B, 3, 224, 224)`

2. **Scaling leakage**

* Always compute min/max on training indices only, then apply to train/val.

3. **GPU-in-dataset**

* If dataset returns CUDA tensors, `pin_memory=True` is not useful.
* `num_workers>0` may break if CUDA tensors are created inside workers.

---

## 11) TODO (Nice-to-have)

* [ ] Confirm `multi_modal_lwm.forward(ch, img)` expected image shape
* [ ] Decide image fusion strategy: time-pooling vs per-frame encoding + temporal fusion
* [ ] Add deterministic seeding + logging + TensorBoard/W&B
* [ ] Add evaluation plots: NMSE(dB) vs epoch

```
```
