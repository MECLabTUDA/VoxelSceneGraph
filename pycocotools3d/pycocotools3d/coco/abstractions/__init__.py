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

from . import common, image_captioning, keypoint_detection, object_detection, panoptic_segmentation, relation_detection

# Note: avoid including image_captioning.Category since it's an alias for Any
AnyCategory = keypoint_detection.Category | \
              object_detection.Category | \
              panoptic_segmentation.Category | \
              relation_detection.SSGCategory | \
              relation_detection.PSGCategory

AnyAnnotation = keypoint_detection.Annotation | \
                object_detection.Annotation | \
                panoptic_segmentation.Annotation | \
                relation_detection.SSGAnnotation | \
                relation_detection.PSGAnnotation

AnyDataset = keypoint_detection.Dataset | \
             object_detection.Dataset | \
             panoptic_segmentation.Dataset | \
             relation_detection.SSGDataset | \
             relation_detection.PSGDataset

__all__ = [
    "common", "image_captioning", "keypoint_detection",
    "object_detection", "panoptic_segmentation", "relation_detection",
    "AnyCategory", "AnyAnnotation", "AnyDataset"
]
