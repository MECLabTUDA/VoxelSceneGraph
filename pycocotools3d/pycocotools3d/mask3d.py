"""
Copyright 2023 Antoine Sanner, Technical University of Darmstadt, Darmstadt, Germany

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

__author__ = 'asanner'

import numpy as np

from ._masks import mask3d, AnyRLE, CompressedRLE, UncompressedRLE

AnyRLE = AnyRLE
CompressedRLE = CompressedRLE
UncompressedRLE = UncompressedRLE

# Interface for manipulating masks stored in RLE format.
#
# RLE is a simple yet efficient format for storing binary masks. RLE
# first divides a vector (or vectorized image) into a series of piecewise
# constant regions and then for each piece simply stores the length of
# that piece. For example, given M=[0 0 1 1 1 0 1] the RLE counts would
# be [2 3 1 1], or for M=[1 1 1 1 1 1 0] the counts would be [0 6 1]
# (note that the odd counts are always the numbers of zeros). Instead of
# storing the counts directly, additional compression is achieved with a
# variable bitrate representation based on a common scheme called LEB128.
#
# Compression is the greatest given large piecewise constant regions.
# Specifically, the size of the RLE is proportional to the number of
# *boundaries* in M (or for an image the number of boundaries in the y
# direction). Assuming fairly simple shapes, the RLE representation is
# O(sqrt(n)) where n is number of pixels in the object. Hence, space usage
# is substantially lower, especially for large simple objects (large n).
#
# Many common operations on masks can be computed directly using the RLE
# (without need for decoding). This includes computations such as area,
# union, intersection, etc. All of these operations are linear in the
# size of the RLE, in other words they are O(sqrt(n)) where n is the area
# of the object. Computing these operations on the original mask is O(n).
# Thus, using the RLE can result in substantial computational savings.
#
# The following API functions are defined:
#  encode         - Encode binary masks using RLE.
#  decode         - Decode binary masks encoded via RLE.
#  merge          - Compute union or intersection of encoded masks.
#  iou            - Compute intersection over union between masks.
#  area           - Compute area of encoded masks.
#  toBbox         - Get bounding boxes surrounding encoded masks.
#  frPyObjects    - Convert polygon, bbox, and uncompressed RLE to encoded RLE mask.
#
# Usage:
#  Rs     = encode( masks )
#  masks  = decode( Rs )
#  R      = merge( Rs, intersect=false )
#  o      = iou( dt, gt, iscrowd )
#  a      = area( Rs )
#  bbs    = toBbox( Rs )
#  Rs     = frPyObjects( [pyObjects], d, h, w )
#
# In the API the following formats are used:
#  Rs      - [dict] Run-length encoding of binary masks
#  R       - dict Run-length encoding of a binary mask
#  masks   - [d x h x w x n] Binary mask(s) (must have type np.ndarray(dtype=uint8) in column-major order)
#  iscrowd - [nx1] list of np.ndarray. 1 indicates corresponding gt image has crowd region to ignore
#  bbs     - [nx6] Bounding box(es) stored as [x y z w h d]
#  dt,gt   - May be either bounding boxes or encoded masks
# Both poly and bbs are 0-indexed (bbox=[0 0 1 1] encloses first pixel).
#
# Finally, a note about the intersection over union (iou) computation.
# The standard iou of a ground truth (gt) and detected (dt) object is
#  iou(gt,dt) = area(intersect(gt,dt)) / area(union(gt,dt))
# For "crowd" regions, we use a modified criteria. If a gt object is
# marked as "iscrowd", we allow a dt to match any subregion of the gt.
# Choosing gt' in the crowd gt that best matches the dt can be done using
# gt'=intersect(dt,gt). Since by definition union(gt',dt)=dt, computing
#  iou(gt,dt,iscrowd) = iou(gt',dt) = area(intersect(gt,dt)) / area(dt)
# For crowd gt regions we use this modified criteria above for the iou.
#
# To compile run "python setup.py build_ext --inplace"
# Please do not contact us for help with compiling.
#
# Microsoft COCO Toolbox.      version 2.0
# Data, paper, and tutorials available at:  http://mscoco.org/
# Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
# Licensed under the Simplified BSD License [see coco/license.txt]

iou = mask3d.iou
merge = mask3d.merge
frPyObjects = mask3d.frPyObjects


def encode(bimask: np.ndarray) -> CompressedRLE:
    """:param bimask: binary mask (d, h, w) as np.uint8. Supports batches on last dim."""
    if len(bimask.shape) == 4:
        # noinspection PyTypeChecker
        return mask3d.encode(np.asfortranarray(bimask, np.uint8))
    elif len(bimask.shape) == 3:
        d, h, w = bimask.shape
        return mask3d.encode(np.asfortranarray(bimask, np.uint8).reshape((d, h, w, 1), order='F'))[0]
    raise ValueError(f"Invalid number of dimensions ({len(bimask.shape)})")


# noinspection PyPep8Naming
def decode(rleObjs: AnyRLE | list[AnyRLE]) -> np.ndarray:
    if isinstance(rleObjs, list):
        return mask3d.decode(rleObjs)
    return mask3d.decode([rleObjs])[..., 0]


# noinspection PyPep8Naming,DuplicatedCode
def area(rleObjs: AnyRLE | list[AnyRLE]):
    if isinstance(rleObjs, list):
        return mask3d.area(rleObjs)
    return mask3d.area([rleObjs])[0]


# noinspection PyPep8Naming,DuplicatedCode
def toBbox(rleObjs: AnyRLE | list[AnyRLE]):
    if isinstance(rleObjs, list):
        return mask3d.toBbox(rleObjs)
    return mask3d.toBbox([rleObjs])[0]