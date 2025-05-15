# nnUNet Results

Here are a few instructions to reproduce the nnUNet results from our NeurIPS2025 submission.

## Semantic Segmentation

In the `nnUNet` folder, you will find the `dataset.json` and `splits_final.json` files for all of our training setups:
- Training only on the BHSD dataset
- Training only on the CQ500 dataset
- Training only on the HemSeg200 dataset
- Training only on the INSTANCE2022 dataset
- Training on all four datasets

You will have to change a few lines of code in the `nnunetv2` library though:
1. In `nnunetv2\preprocessing\normalization/default_normalization_schemes.py`, replace the definition of `CTNormalization`:
```python
class CTNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False
    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        return (np.clip(image, a_min=-50, a_max=200) + 50) / 250
```

2. In `nnunetv2\preprocessing\resampling/default_resampling.py`, replace the definition of `resample_data_or_seg_to_spacing`:
```python
def resample_data_or_seg_to_spacing(data: np.ndarray,
                                    current_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                                    new_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                                    is_seg: bool = False,
                                    order: int = 3, order_z: int = 0,
                                    force_separate_z: Union[bool, None] = False,
                                    separate_z_anisotropy_threshold: float = ANISO_THRESHOLD):
    if force_separate_z is not None:
        do_separate_z = force_separate_z
        if force_separate_z:
            axis = get_lowres_axis(current_spacing)
        else:
            axis = None
    else:
        if get_do_separate_z(current_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(current_spacing)
        elif get_do_separate_z(new_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(new_spacing)
        else:
            do_separate_z = False
            axis = None

    if axis is not None:
        if len(axis) == 3:
            # every axis has the same spacing, this should never happen, why is this code here?
            do_separate_z = False
        elif len(axis) == 2:
            # this happens for spacings like (0.24, 1.25, 1.25) for example. In that case we do not want to resample
            # separately in the out of plane axis
            do_separate_z = False
        else:
            pass

    if data is not None:
        assert data.ndim == 4, "data must be c x y z"

    shape = np.array(data.shape)
    new_shape = compute_new_shape(shape[1:], current_spacing, new_spacing)

    # These two lines are the important stuff!
    new_shape = list(new_shape)
    new_shape[0] = data.shape[1]

    data_reshaped = resample_data_or_seg(data, new_shape, is_seg, axis, order, do_separate_z, order_z=order_z)
    return data_reshaped
```

Note: only the last few lines are relevant.

Now you're ready to train your models. We don't include commands to run prediction on the test data.

## Object Detection

Here, we assume that you already trained your models and have predicted segmentations for all test images.
Below is the code that we used to convert the semantic segmentation to a `BoxList` instance to use with our evaluation code.
We assume the following folder structure:
```
path to one dir with predictions/
├─ BHSD/
│  ├─ all predictions for BHSD
├─ CQ500/
│  ├─ all predictions for CQ500
├─ HemSeg200/
│  ├─ all predictions for HemSeg200
├─ INSTANCE2022/
│  ├─ all predictions for INSTANCE2022
├─ PhysioNet/
│  ├─ all predictions for PhysioNet
```


```python
from pathlib import Path
import tqdm

from scene_graph_api.utils.nifti_io import NiftiImageWrapper
from scene_graph_api.tensor_structures import BoxList, BoxListConverter
from scene_graph_api.scene import SceneGraph
from scene_graph_api.knowledge import KnowledgeGraph
import cc3d
import logging
import torch
import numpy as np

logger = logging.getLogger(__name__)
# TODO: change the path to your knowledge graph
knowledge_graph = KnowledgeGraph.load("path to your knowledge graph", logger)
assert knowledge_graph

# TODO: change the path to your output root folder
out_root = Path("output path")

# TODO: fill this list with all folder containing the predicted segmentations
for in_root in [
    Path(r"path to a prediction folder"),
]:
    print(in_root)
    for ds in ["BHSD", "CQ500", "HemSeg200", "INSTANCE2022", "PhysioNet"]:
        print(f"\t{ds}")
        (out_root / ds).mkdir(parents=True, exist_ok=True)
        for pred_p in tqdm.tqdm(sorted((in_root / ds).rglob("*.nii.gz"))):
            out_p = out_root / ds / pred_p.name.replace(".nii.gz", ".pth")
            
            # Skip any existing BoxList
            if out_p.exists():
                continue

            # Load the prediction
            pred = NiftiImageWrapper.load_depth_first(pred_p)
            pred_arr = pred.get_mask_data()
            scores_arr = np.load(pred_p.with_name(pred_p.name.replace(".nii.gz", ".npz")))["probabilities"]

            # Get the mask for the ventricle system
            ventricle_mask = pred_arr == 1
            cc3d.dust(ventricle_mask, threshold=25, in_place=True)
            # Get the mask for the midline
            midline_mask = pred_arr == 2
            cc3d.dust(midline_mask, threshold=25, in_place=True)

            # Get the mask for ICH and remove any connected component with 25 voxels or less
            all_bleeds_masks = pred_arr == 3
            cc3d.dust(all_bleeds_masks, threshold=25, in_place=True)
            # Split into connected components
            bleed_labelmap, n_bleed = cc3d.connected_components(all_bleeds_masks, return_N=True)
            # Remap ids to have the ICH instance after the ventricle system and midline
            bleed_labelmap[bleed_labelmap > 0] += 2
            bleed_labelmap[ventricle_mask > 0] = 1
            bleed_labelmap[midline_mask > 0] = 2

            # Create a dict with object id mapping to class id
            ids_to_class = {1: 1, 2: 2} | {i + 3: 3 for i in range(n_bleed)}
            # Convert to SceneGraph
            graph = SceneGraph.create_fom_labelmap(
                knowledge_graph,
                NiftiImageWrapper.from_array(bleed_labelmap, pred.affine, pred.header),
                ids_to_class,
                logger
            )
            # Convert to BoxList (the process assumes that is a GT annotation)
            boxlist = BoxListConverter(BoxList).from_scene_graph(graph)
            # Set the predicted object label field
            boxlist.PRED_LABELS = boxlist.LABELS
            # Compute the predicted score field
            scores = [np.max(scores_arr[i if i <= 2 else 3][bleed_labelmap == i]) for i in range(1, n_bleed + 3)]
            boxlist.PRED_SCORES = torch.tensor(scores)
            boxlist.save(out_p)
```

Then to use `scene-graph-prediction`'s `sgpred_detector_offline_eval` script to evaluate these BoxLists, we need to change the folder structure to:

```
experiment output folder/
├─ BHSD/
├─ ├─ test/
│  ├─ ├─ all predictions for BHSD
├─ CQ500/
├─ ├─ test/
│  ├─ ├─ all predictions for CQ500
├─ HemSeg200/
├─ ├─ test/
│  ├─ ├─ all predictions for HemSeg200
├─ INSTANCE2022/
├─ ├─ test/
│  ├─ ├─ all predictions for INSTANCE2022
├─ PhysioNet/
├─ ├─ test/
│  ├─ ├─ all predictions for PhysioNet
│  seg_cfg.yaml
```

You can find `seg_cfg.yaml` in the `nnUNet` folder. Please make sure to replace "{output}" with the actual experiment output folder.
Since this is mostly file handling, we let this as an exercise for the reader.
