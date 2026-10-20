# Scene Graph API

This library defines useful **data structures** and **format conversion** methods to bridge the gap between data 
annotation (scene-graph-annotation (TBA)) and Deep Learning experiments (scene-graph-prediction).
Both natural images, and voxel data are supported.
Additionally, we consider five possible data formats, which are described further below.

## Installation

To install as an integrative framework, clone the repo and run:
```bash
pip install -e .
```

To use the pyTorch-based data representation, please install `torch` manually.
If you wish to convert data to/from the COCO format, please also install `pycocotools3d`.

Note: these libraries are not part of the requirements as they are application-specific, 
e.g. not required for data annotation.

## Knowledge Graphs

Before we can start talking about Scene Graphs, we need to define some application knowledge about **what** we want to
know. This is known as a **Knowledge Graph**. If you're acquainted with the 
[nnUNet](https://github.com/MIC-DKFZ/nnUNet) framework, you have to define a `dataset.json` file with labels, and
modalities before you can train your segmentation model. Knowledge Graphs are the equivalent for Scene Graph
application, though they are fully customizable. We can currently support the following list of annotation:
- image-level attributes (i.e. image classification)
- object bounding boxes (can be computed from masks)
- object attributes
- relations

Knowledge Graphs are stored in the JSON format, and can be generated using a small Python script. 
Examples from our published papers are available in the `examples` folder.
For more details, you can look at the classes defined in `scene_graph_api.knowledge`.

## Annotation Formats

All data formats use masks for object localization. These masks can be either precise or just rough label maps that
provide the bounds of each object. For more details, or to see which data annotation method suits your needs best,
you can check out our **[paper](https://www.arxiv.org/abs/2408.10768)**
```
Sanner, A. P., Grauhan, N. F., Brockmann, M. A., Othman, A. E., & Mukhopadhyay, A. (2024). 
Detection of Intracranial Hemorrhage for Trauma Patients. arXiv [Cs.CV]. 
Retrieved from https://arxiv.org/abs/2408.10768
```

### Label Map

The first data format is **label maps**, i.e. a segmentation with its corresponding JSON file with object classes 
(similar to [nnDetection](https://github.com/MIC-DKFZ/nnDetection)).
You will only encounter this format when:
- Annotating a new Scene Graphs dataset from segmentations
- Extracting a label map from a scene graph

You can mark class ids as ignored in the knowledge graph using the `is_ignored` option, when defining the object class.

### Semantic Segmentation

**Semantic segmentations** can be automatically converted to label maps through a connected component analysis.
If you want to prevent this analysis on a per-class-id basis, you can use the `is_unique` option in the knowledge graph.

### Scene Graph (JSON)

The **SceneGraph** is the first format, which can store a scene graph. The annotation for each image is stored as an
individual lightweight JSON file and also contains the labelmap.
This format is best-suited for data annotation (TBA).
Similarly to knowledge graphs, such JSON files should only be created using the Python API.
Relevant classes can be found in `scene_graph_api.scene`.

Note: a scene graph only makes sense given its corresponding knowledge graph.

### BoxList (PyTorch)

**BoxLists** are PyTorch tensor-based representation of a scene graph and the implementation is based off the well-known
[Scene Graph Benchmark](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) framework.
The main concept is that a BoxList contains the bounding boxes for a given image, as well as _fields_ are per
bounding-box annotation. E.g. the `LABELS` field contains a tensor with the label of each bbox.
The usefulness is that: **when a BoxList is indexed not only are the bboxes indexed, 
but also all fields in the BoxList**.
This makes sampling bboxes very convenient, at the price of checking the relevant fields are present in the BoxList.

The relevant implementation can be found in `scene_graph_api.tensor_structures`:
- `box_list.py`: data structure definition and I/O.
- `box_list_converter.py`: conversion methods to/from BoxList.
- `box_list_field_extractor.py`: the information can be stored using multiple fields combination. This class provides
methods to extract this information, e.g. extract a semantic segmentation from the `LABELMAP` and `LABELS` fields.
- `box_list_fields.py`: enum with fields defined for ground truth and predictions. 
These have special behavior already implemented, but feel free to use your own fields.
- `box_list_ops.py`: common operations on BoxLists, e.g. cropping, flipping, IoU computation, concatenation... 
for both 2D and 3D (voxel) data.

Note: don't forget to install PyTorch yourself!

### MS COCO

The [MS COCO](https://cocodataset.org/#format-data) format is currently only partially supported.
You can export your dataset in the `SceneGraph` format to MS COCO using the `script/graph_dataset_to_coco.py` script
or the `sgapi_graph_dataset_to_coco` command.
Otherwise, it's only used internally for the evaluation of object detection using the `pycocotools3d` library.

## Data Conversion

You can use the `scripts/format_conversion.py` script or the `sgapi_format_conversion` command to convert from any
data format to any other (except MS COCO). Of course, the conversion may not be loss-less, e.g. `BoxList`→`labelmap`.

Note: BoxList predictions from the `scene_graph_prediction` framework to `SceneGraph` are only partially supported.

## Citation

If you use this library, please cite our **[paper](https://arxiv.org/abs/2407.21580)**:
```
Sanner, A. P., Grauhan, N. F., Brockmann, M. A., Othman, A. E., & Mukhopadhyay, A. (2024). 
Voxel Scene Graph for Intracranial Hemorrhage. arXiv [Cs.CV]. Retrieved from https://arxiv.org/abs/2407.21580
```
