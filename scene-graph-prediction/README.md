![method.png](images/method.png)

# Scene Graph Prediction

This is the library for Deep Learning experiments for Voxel Scene Graph and is based off the well-known
[Scene Graph Benchmark](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) framework.
Though, the code has been thoroughly restructured, documented and type hinted.

## Installation

First, create a Python >3.10 environment (using Conda is recommended).
Then install [PyTorch](https://pytorch.org/get-started/locally/) with CUDA.
As we also have to compile CUDA code, we'll need Nvidia build tools too.
So either install the CUDA version corresponding to your drivers/PyTorch install or if you used conda:
```bash
conda install -c nvidia cuda-python
```

To check that the build tools are correctly configured, you can print your `CUDA_HOME` path:
```python
from torch.utils.cpp_extension import CUDA_HOME
print(CUDA_HOME)  # Shouldn't be None
```

Next, install out `pycocotools3d` and `scene_graph_api` libraries by cloning our other repositories and following
our instructions.

Finally, to install as an integrative framework, clone the repo and run `make interactive` or run:
```bash
pip install -e .
rm -rf build
```

## Setting up your datasets

To set up your datasets, you can edit the `scene_graph_prediction.data.paths_catalog.py` file.
`DATASETS_DIR` is the root folder, where your data will be stored.
Then you can add entries in the `DATSETS` dict to add individual datasets.
Currently, we only use the `RelationDetectionDataset` class, which requires:
- `img_dir`: the path to image folder.
- `annotation_dir`: the path to BoxList folder.
- `knowledge_graph_file`: the path to knowledge graph.
- `spliter`: a `DatasetSpliter` to compute train/val/test splits given the annotation files.

Anyway, here is an example from one of our papers:
```python
DATASETS = {
  "obj_detect_train": {
      "img_dir": DATASETS_DIR + "sgg_full_rot/images",
      "annotation_dir": DATASETS_DIR + "sgg_full_rot/boxLists",
      "knowledge_graph_file": DATASETS_DIR + "sgg_full_rot/knowledge_graph.json",
      "spliter": FixedSpliter(DATASETS_DIR + "sgg_full_rot/split.json"),
      _DATASET_TYPE_KEY: RelationDetectionDataset
  },
}
```
where `split.json` is a JSON file containing out splits. You can find this file in `configs/MICCAI2024`.

## Model Configuration

All models/experiments are configured using YAML configuration files.
You can find the files we used in our papers in the `configs` folder.
For more details, you can check the comments in `scene_graph_prediction.config`.

Note: we support recursive configuration imports using the `BASE_CFG` key.

## Scripts

Note: you can override any config file parameter by adding the (parameter path, value) pair as argument in the command.
For example:
```bash
python sgpred_relation_train_net.py --config-file "configs/my_config.yaml" SOLVER.MAX_ITER 50000
```
will override the max number of iteration.

### Training

Currently, only two-stage SGG methods are implemented. This means, that you first have to train an object detector and
only then can you train the relation predictor.

- `sgpred_detector_pretrain_net`: script to train the object detector (one-stage or two-stage object detector).
- `sgpred_relation_train_net`: script to train the relation predictor.
- `sgpred_detector_detector_one_stage_box_feature_extractor_pretrain_net`: SGG methods have been designed to work with 
two-stage object detectors and to reuse the pretrained encoder of the 2nd stage of the object detector. If you're using
a one-stage object detector and wish to pretrain this encoder, this is the script that you want: after having trained
your one-stage object detector, it simulates a two-stage detector by adding a detector head on top of it. You rarely 
need to do this tough.

Here is an example to train our V-IMP model on the Predicate Classification task:
```bash
python sgpred_detector_pretrain_net --config-file configs/MICCAI2024/object_detector.yaml
python sgpred_relation_train_net --config-file configs/MICCAI2024/sgg_imp_use_gt_with_mask.yaml
```
You would need the data though 😉 (TBA)

### Evaluation

- `sgpred_test_net`: script to evaluate a trained model on the test split. 
It is already done automatically at the end of the training though.
- `sgpred_detector_offline_eval`: if you saved your predictions for validation/test during/after the training process,
you can use this script to compute the metrics for the predictions without needing to use GPU.

### Utils

- `sgpred_boxlist_prediction_to_slicer_rois`: given a knowledge graph and a path to a BoxList 
(ground truth or prediction), produces Python code that can be pasted in [3D Slicer](https://www.slicer.org/) to
visualize the bounding boxes.

## Citation

If you use this library, please cite our **[paper](https://arxiv.org/abs/2407.21580)**:
```
Sanner, A. P., Grauhan, N. F., Brockmann, M. A., Othman, A. E., & Mukhopadhyay, A. (2024). 
Voxel Scene Graph for Intracranial Hemorrhage. arXiv [Cs.CV]. Retrieved from https://arxiv.org/abs/2407.21580
```

