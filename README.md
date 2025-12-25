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
- `Rx`: BS antenna index (e.g., 16)
- `Tx`: UE antenna index (e.g., 1)
- `Subcarrier`: OFDM subcarrier index (e.g., 512)
- Optional subset: `selected_subcarriers = [0..7]`

<img width="745" height="350" alt="channel shape (Rx, Tx, subcarrier)" src="https://github.com/user-attachments/assets/86387a14-eefb-48db-b575-e6a7632ffbde" />

---


