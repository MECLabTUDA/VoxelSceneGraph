# pycocotools3d

This is a fork of the original PyPI [cocoapi](https://github.com/ppwwyyxx/cocoapi), with **3D support**.
While support for 2D boxes is retained, all coordinates are expected to be **width last** (whether for 2D or 3D), 
e.g. (z1, y1, x1, z2, y2, x2) or (z, y, x, d, h, w)
Our reasoning is that most libraries for computer vision on natural images are hardcoded for 2D computation anyway.
So we might as well use a sensible coordinate representation and stop weird combinations like (y1, x1, z1, y2, x2, z2)
to enable some sort of compatibility.

Anyway, just make sure that your coordinates are ordered correctly.

Note: 3D polygons are not supported and 3D keypoints are still a WIP!

## Installation

For a local installation, simply run `make interactive` or run:
To install as an integrative framework, clone the repo and run:
```bash
pip install -e .
rm -rf build
```

For a full installation, run `make install` or run:
```bash
python -m pip install .
rm -rf build
```

## Non-Exhaustive Changelog

- Add support for 3D bounding boxes
- All bounding boxes are width-last
- Add some abstractions for the expected dict structures
- Add type hints
- Add some tests

## Citation

If you use this library, please cite our **[paper](https://arxiv.org/abs/2407.21580)**:
```
Sanner, A. P., Grauhan, N. F., Brockmann, M. A., Othman, A. E., & Mukhopadhyay, A. (2024). 
Voxel Scene Graph for Intracranial Hemorrhage. arXiv [Cs.CV]. Retrieved from https://arxiv.org/abs/2407.21580
```
