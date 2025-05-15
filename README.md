# Voxel Scene Graph

This is our central hub for all our work on Voxel Scene Graph. It is still a WIP and fancy illustrations may come in
the future. Anyway here is an overview of the repository that you can find here:

- `pycocotools3d`: object detection evaluation for 3d bounding boxes and MS COCO format abstractions
- `scene-graph-api`: data structures and abstractions to bridge data annotation and Deep Learning applications
- `scene-graph-prediction`: framework for Voxel Scene Graph applications
- `federated-scene-graph-prediction`: framework for Voxel Scene Graph applications trained 
using Federated Learning
- `theoden`: our fork of `TheODen` [framework](https://github.com/MECLabTUDA/TheODen) for Federated Learning
Made with love by our colleagues ❤️
- `3d-slicer-modules`: any custom module we design for 3D Slicer, currently only to make the segmentation of cohorts easier
- `scene-graph-annotation`: tool for Voxel Scene Graph annotation and more
- `documentation`: guides, tutorials or instructions to reproduce our results (WIP)

Our **Scene Graph annotation is openly available** on [Kaggle](https://www.kaggle.com/datasets/sannera/bleedscene3d).
It includes Scene Graphs for 574 volumes from five open source datasets.

**Note:** our dataset submission is under review at NeurIPS2025. We'll make the Kaggle dataset public upon acceptance
(decision on 18th of Sep. 2025). 
Until then, we'll do our best to have everything to document all our code/tools.

![figure2.png](images/figure2.png)

## Our Accepted Papers

### Federated Voxel Scene Graph for Intracranial Hemorrhage

![datasets_WACV2025.png](images/datasets_WACV2025.png)

This paper has been **accepted at [WACV2025](https://wacv2025.thecvf.com/)**.
To the best of our knowledge, this is the first application of Federated Learning to Scene Graph Generation of any kind.
For this paper, we had to gather publicly available datasets with head CTs of ICH patients from around the world.
We then **curated** and **annotated** these, which allowed to have **~450 annotated images** 
(compared to the ~150 from the MICCAI paper).
In this paper, we not only show that models trained centrally on a single dataset fail to generalize to other datasets,
but that Federated Learning allows bridging this gap without breach of data privacy.
You can find the pre-print **[here](https://arxiv.org/abs/2411.00578)**:
```
Sanner, A. P., Stieber, J., Grauhan, N. F., Kim, S., Brockmann, M. A., Othman, A. E., & Mukhopadhyay, A. (2024). 
Federated Voxel Scene Graph for Intracranial Hemorrhage. arXiv [Cs.CV]. Retrieved from https://arxiv.org/abs/2411.00578
```

![method_MICCAI2024.png](images/method_MICCAI2024.png)

This paper has been **accepted at [MICCAI2024](https://conferences.miccai.org/2024/en/)**.
To the best of our knowledge, this is the first application of Scene Graph to voxel data.
In particular, we show how only detecting Intracranial Hemorrhage (ICH) is clinically insufficient, 
as bleedings interact with neighboring brain structures, potentially causing deadly complications.
Scene Graphs can capture the entire clinical cerebral scene and model these complex relations.
You can find the pre-print **[here](https://arxiv.org/abs/2407.21580)**:
```
Sanner, A. P., Grauhan, N. F., Brockmann, M. A., Othman, A. E., & Mukhopadhyay, A. (2024). 
Voxel Scene Graph for Intracranial Hemorrhage. arXiv [Cs.CV]. Retrieved from https://arxiv.org/abs/2407.21580
```

### Detection of Intracranial Hemorrhage for Trauma Patients

![method_ICPR2024.png](images/method_ICPR2024.png)

This paper has been **accepted at [ICPR2024](https://icpr2024.org/)**. 
It focuses on the challenges of Intracranial Hemorrhage detection
in voxel data, and introduced the novel VC-IoU loss for bounding box regression. 
You can find the pre-print [here](https://www.arxiv.org/abs/2408.10768):
```
Sanner, A. P., Grauhan, N. F., Brockmann, M. A., Othman, A. E., & Mukhopadhyay, A. (2024). 
Detection of Intracranial Hemorrhage for Trauma Patients. arXiv [Cs.CV]. 
Retrieved from https://arxiv.org/abs/2408.10768
```

