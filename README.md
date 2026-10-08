# RadIOCD Point Cloud Experiments

## Dataset Summary
- 5,776 recordings from 10 subjects.
- 5 classes: backpack, chair, desk, human, wall.
- ~1,150 recordings per class, so the raw recordings are well balanced.
- Radar operates at 10 FPS.
- Each recording is ~8 seconds (~81 frames on average).
- Very sparse point clouds: ~7 points per frame on average.

## Preprocessing
- Each recording is divided into 1-second samples (10 frames).
- 90% overlap: each new sample moves forward by only 1 frame.
- Example: frames 1–10 → 2–11 → 3–12 → ...
- Presence/clustering/filtering removes unusable samples.
- Final processed dataset: 76,821 1-second samples.
- The processed samples become class-imbalanced because different classes survive preprocessing differently.
### RadIOCD Filtering
- `window = 10`: 10 frames = 1 second at 10 FPS.
- `threshold_frames = 8`: target must be present in at least 8/10 frames (80%).
- `DBSCAN eps = 0.5`, `min_samples = 8`: groups nearby radar points and removes isolated/noisy points.
- Only `Presence = 1` and non-zero Doppler points are considered.
- Target cluster is expected near the walking direction (`|X center| < 0.3 m`).
- Closest valid cluster is selected, approximately within **1–2.5 m**.
- Final cluster is summarized using **13 features**: mean/min/max/std of X,Y,Z + number of points.
- Published result: **76,821 final 1-second samples**.

## Starting Experiment
**RadIOCD:** Point cloud → 13 handcrafted XYZ statistical features → DDNN → object class

“We reproduced the published RadIOCD preprocessing exactly and obtained DDNN performance close to the reported baseline, with minor differences likely due to framework/version and stochastic training differences.”


**Ours:** Point cloud → PointNet / modern point-cloud model → object class

### Question
Can a point-cloud DL model learn directly from the radar points and improve classification without manually reducing the point cloud to statistical features?


RadIOCD DDNN baseline reproduction
----------------------------------
Features: /Users/nagashayanaramamurthy/GitHub/radiocd-pointcloud/outputs/features.csv
Samples : 76821
Input   : 13 handcrafted features
Network : 256 -> 256 -> 128 -> 128 -> 5
Dropout : 0.1 after every hidden layer
Batch   : 128
Epochs  : up to 1000
Stop    : val_accuracy patience=50

Class counts:
  backpack  : 16027
  chair     : 19825
  desk      : 6258
  human     : 21393
  wall      : 13318

10-fold cross-validation
------------------------
Split unit: original CSV recording (overlapping 1-second windows stay together).
Fold  1: accuracy=0.7577, macro_f1=0.7272, epochs=65
Fold  2: accuracy=0.7321, macro_f1=0.7088, epochs=63
Fold  3: accuracy=0.7418, macro_f1=0.7142, epochs=70
Fold  4: accuracy=0.7622, macro_f1=0.7290, epochs=60
Fold  5: accuracy=0.7113, macro_f1=0.6867, epochs=92
Fold  6: accuracy=0.7278, macro_f1=0.7041, epochs=86
Fold  7: accuracy=0.7324, macro_f1=0.7075, epochs=120
Fold  8: accuracy=0.7567, macro_f1=0.7304, epochs=65
Fold  9: accuracy=0.7332, macro_f1=0.7150, epochs=60
Fold 10: accuracy=0.7654, macro_f1=0.7365, epochs=74

10-fold mean
Accuracy : 74.21%
Macro F1 : 71.59%
Paper    : 76.53% accuracy, 72.55% F1


PY
count    76821.000000
mean        14.664389
std          4.721539
min          2.000000
1%           8.000000
5%           9.000000
10%         10.000000
25%         11.000000
50%         14.000000
75%         17.000000
90%         21.000000
95%         24.000000
99%         29.000000
max         42.000000
Name: n_points, dtype: float64

By class:
          count       mean  median  min  max
object                                      
backpack  16027  13.237474    13.0    8   25
chair     19825  16.873190    16.0    5   42
desk       6258  11.327421    11.0    7   28
human     21393  14.777077    14.0    2   34
wall      13318  14.480553    13.0    5   39


pointcloud with xyz only



Fold  1: accuracy=0.7473, macro_f1=0.7088, epochs=67
Fold  2: accuracy=0.7237, macro_f1=0.6882, epochs=45
Fold  3: accuracy=0.7351, macro_f1=0.6994, epochs=69
Fold  4: accuracy=0.7556, macro_f1=0.7256, epochs=120
Fold  5: accuracy=0.7089, macro_f1=0.6793, epochs=61
Fold  6: accuracy=0.7169, macro_f1=0.6893, epochs=147
Fold  7: accuracy=0.7379, macro_f1=0.7095, epochs=69
Fold  8: accuracy=0.7318, macro_f1=0.6964, epochs=83
Fold  9: accuracy=0.7248, macro_f1=0.7020, epochs=75
Fold 10: accuracy=0.7620, macro_f1=0.7333, epochs=61

PointNet 10-fold mean
Accuracy : 73.44%
Macro F1 : 70.32%

Reference DDNN reproduction:
Accuracy : 74.21%
Macro F1 : 71.59%


2026-09-29 09:02:26.992660: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:117] Plugin optimizer for device_type GPU is enabled.


Fold  1: accuracy=0.7620, macro_f1=0.7235, epochs=55

Fold-1 reference from XYZ baseline:
Accuracy : 74.73%
Macro F1 : 70.88%
Saved    : /Users/nagashayanaramamurthy/GitHub/radiocd-pointcloud/outputs/pointnet_ablation/results_4ch.csv
((.venv-metal) ) nagashayanaramamurthy@Nagas-MBP radiocd-pointcloud % 
((.venv-metal) ) nagashayanaramamurthy@Nagas-MBP radiocd-pointcloud % 
((.venv-metal) ) nagashayanaramamurthy@Nagas-MBP radiocd-pointcloud % python scripts/train_pointnet_ablation.py --channels 5 --fold 1
RadIOCD PointNet channel ablation
--------------------------------
Data     : /Users/nagashayanaramamurthy/GitHub/radiocd-pointcloud/outputs/pointnet_xyzdi_32.npz
Channels : ['X', 'Y', 'Z', 'Doppler', 'Intensity']
Shape    : (76821, 32, 5)
Folds    : [1]

2026-09-29 09:26:46.815685: I metal_plugin/src/device/metal_device.cc:1154] Metal device set to: Apple M1 Pro
2026-09-29 09:26:46.815726: I metal_plugin/src/device/metal_device.cc:296] systemMemory: 16.00 GB
2026-09-29 09:26:46.815734: I metal_plugin/src/device/metal_device.cc:313] maxCacheSize: 5.92 GB
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1790688406.815747 2343601 pluggable_device_factory.cc:305] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
I0000 00:00:1790688406.815768 2343601 pluggable_device_factory.cc:271] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)
2026-09-29 09:26:47.763103: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:117] Plugin optimizer for device_type GPU is enabled.
Fold  1: accuracy=0.8839, macro_f1=0.8661, epochs=55

Fold-1 reference from XYZ baseline:
Accuracy : 74.73%
Macro F1 : 70.88%
Saved    : /Users/nagashayanaramamurthy/GitHub/radiocd-pointcloud/outputs/pointnet_ablation/results_5ch.csv
((.venv-metal) ) nagashayanaramamurthy@Nagas-MBP radiocd-pointcloud % 

intensity is boosting it by a lot
I0000 00:00:1790690012.824976 2359513 pluggable_device_factory.cc:271] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)
2026-09-29 09:53:33.826209: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:117] Plugin optimizer for device_type GPU is enabled.
Fold  1: accuracy=0.8839, macro_f1=0.8661, epochs=55


Fold  2: accuracy=0.8762, macro_f1=0.8736, epochs=79
Fold  3: accuracy=0.8868, macro_f1=0.8707, epochs=76
Fold  4: accuracy=0.8847, macro_f1=0.8657, epochs=99
Fold  5: accuracy=0.8538, macro_f1=0.8492, epochs=86
Fold  6: accuracy=0.8468, macro_f1=0.8336, epochs=55
Fold  7: accuracy=0.8780, macro_f1=0.8640, epochs=73
Fold  8: accuracy=0.8533, macro_f1=0.8374, epochs=95
Fold  9: accuracy=0.8849, macro_f1=0.8735, epochs=67
Fold 10: accuracy=0.8811, macro_f1=0.8694, epochs=79

10-fold mean
Accuracy : 87.30%
Macro F1 : 86.03%

Fold-1 reference from XYZ baseline:
Accuracy : 74.73%
Macro F1 : 70.88%
Saved    : /Users/nagashayanaramamurthy/GitHub/radiocd-pointcloud/outputs/pointnet_ablation/results_5ch.csv

 -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)
Features: i Shape: (76821, 32, 1) Parameters: 152005
2026-09-29 16:55:20.364550: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:117] Plugin optimizer for device_type GPU is enabled.


Only intensity

Accuracy=0.6008 MacroF1=0.5744 Epochs=77 BestEpoch=47
Training=29.3 min, 22.87 s/epoch; inference=0.0772 ms/sample

without doppler

python scripts/train_pointnet_ablation_features.py --features xyzi --fold 1
X dtype: float32 y dtype: int64
2026-09-29 18:48:23.437053: I metal_plugin/src/device/metal_device.cc:1154] Metal device set to: Apple M1 Pro
2026-09-29 18:48:23.437305: I metal_plugin/src/device/metal_device.cc:296] systemMemory: 16.00 GB
2026-09-29 18:48:23.437317: I metal_plugin/src/device/metal_device.cc:313] maxCacheSize: 5.92 GB
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1790722103.437494 2741450 pluggable_device_factory.cc:305] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
I0000 00:00:1790722103.437717 2741450 pluggable_device_factory.cc:271] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)
Features: xyzi Shape: (76821, 32, 4) Parameters: 152197
2026-09-29 18:48:24.409794: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:117] Plugin optimizer for device_type GPU is enabled.


Accuracy=0.8724 MacroF1=0.8533 Epochs=89 BestEpoch=59
Training=33.4 min, 22.55 s/epoch; inference=0.0775 ms/sample

