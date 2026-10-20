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

from abc import ABC, abstractmethod

import nibabel as nib
import numpy as np
from PIL import Image

# Need to be careful with imports to scene_graph_annotation from .utils
from scene_graph_annotation.knowledge.KnowledgeGraph import RadiologyImageKG

_size = tuple[int, int]  # width, height ordering


class ArrayView(ABC):
    """
    This interface is here to solve two main problems:
    - images may or may not have a channel dimension, can be RGB float vs int, or even be 2D vs 3D,
      so we need a harmonized way of retrieving a 2D RGBA slice to display
    - it will implement the logic to pan/zoom/scale images to a desired view
      it should also be able to perform the same processing on a segmentation mask (of known data type)


    Interface used to get a 2D view from an N-D volume for display.
    For convenience, all data read will be converted to Nifti (even from DICOM or PNG).
    Object ids can take any value in [1, 255].
    """

    def __init__(self, n_dim: int, array: np.ndarray):
        self._n_dim = n_dim
        self.array = array

        # Pixel size that the array view should have (width, height)
        self.target_size: _size = 1, 1

        # For UI zoom
        self._zoom_factor = 1.

        # For panning: Float offset for the original array, accommodates for px fraction offsets
        self._anchor = np.zeros(self._n_dim, dtype=float)

    @property
    def n_dim(self) -> int:
        """Returns the number of dimensions (excluding channel dim). Typically, 2 or 3."""
        return self._n_dim

    @property
    def zoom_factor(self) -> float:
        """Getter for the zoom factor."""
        return self._zoom_factor

    @zoom_factor.setter
    def zoom_factor(self, zoom_factor: float):
        """Setter for the zoom_factor. The new value will only be set if it is strictly positive."""
        # No need to move the anchor after the zoom in/out, because the anchor is zoom invariant
        if zoom_factor > 0:
            self._zoom_factor = zoom_factor

    @abstractmethod
    def move_anchor(self, dx: int, dy: int):
        """Convert a (mouse) pixel offset to the true array offset (independent of scaling)."""
        raise NotImplementedError

    def reset_anchor(self):
        """Resets the panning."""
        self._anchor[:] = 0

    @abstractmethod
    def view_coordinates_to_orig_array(self, x: int, y: int) -> tuple[int, ...]:
        """Maps cursor coordinates (from the view) back to coordinates in the original ND array."""
        raise NotImplementedError


class RadiologyImageArrayView(ArrayView):
    """Array view for grayscale 2D or 3D Nifti images."""

    def __init__(self, knowledge_graph: RadiologyImageKG, image: nib.Nifti1Image):
        self.knowledge_graph = knowledge_graph

        # For value windowing
        self.window_center = 0
        self.window_width = 0
        self.reset_value_windowing()

        self._image = image

        array: np.ndarray = np.asarray(image.get_fdata())
        super().__init__(n_dim=len(array.shape), array=array)

        # For intrinsic image scaling if voxels are not cubes
        self._zooms = np.array(image.header.get_zooms())

        # For slice selection, ignored if data is 2D
        # The orientation is fixed when loading the nifti image, axial should always be 0
        self._selected_axis = 0
        self._selected_idx = 0
        self.center_selected_idx()

    @property
    def selected_axis(self) -> int:
        """Getter for the axis selected. Ony used for 3D images."""
        return self._selected_axis

    @selected_axis.setter
    def selected_axis(self, axis: int):
        """
        Setter for the axis selected. Ony used for 3D images.
        Will also center the selected slice index.
        """
        old_axis = self._selected_axis
        self._selected_axis = axis
        # Update selected slice if axis changed
        if old_axis != self._selected_axis:
            self.center_selected_idx()

    def center_selected_idx(self):
        """Sets the selected slice index to be the middle slice in this axis."""
        self._selected_idx = self.array.shape[self._selected_axis] // 2

    @property
    def selected_idx(self) -> int:
        """Getter for the selected slice index."""
        return self._selected_idx

    @selected_idx.setter
    def selected_idx(self, idx: int):
        """Setter for the selected slice index. Checks that the index is in bounds."""
        if 0 <= idx < self.array.shape[self._selected_axis]:
            self._selected_idx = idx

    def reset_value_windowing(self):
        """Resets the window center and width to the values set in the template."""
        self.window_center = self.knowledge_graph.window_center
        self.window_width = self.knowledge_graph.window_width

    def move_anchor(self, dx: int, dy: int):
        """Convert a (mouse) pixel offset to the true array offset (independent of scaling)."""
        zoom_ax0, zoom_ax1 = self._get_zooms_for_view()
        d = [dx / zoom_ax1, dy / zoom_ax0]
        if self.selected_axis != 0:
            d[1] *= -1
        if self._n_dim == 3:
            d.insert(2 - self.selected_axis, 0)
        # Compute true offset
        self._anchor += d

    def view_coordinates_to_orig_array(self, x: int, y: int) -> tuple[int, ...]:
        """Maps cursor coordinates (from the view) back to coordinates in the 2D or 3D array."""
        # Compute the size of the slice before rescaling
        orig_slice_shape = list(self._image.shape)
        if self._n_dim == 3:
            orig_slice_shape.pop(self.selected_axis)
        # Compute the size of the slice after rescaling
        zoom_ax0, zoom_ax1 = self._get_zooms_for_view()
        zoomed_shape = round(orig_slice_shape[0] * zoom_ax0), round(orig_slice_shape[1] * zoom_ax1)
        # Get anchor coordinates
        anc = list(self._anchor)
        if self._n_dim == 3:
            anc.pop(2 - self.selected_axis)
        dx, dy = anc
        # Compute padding
        pad_x, pad_y = (self.target_size[0] - zoomed_shape[1]) / 2, (self.target_size[1] - zoomed_shape[0]) / 2
        # Map x, y coordinates back
        x = dx + (x - pad_x) / zoom_ax1
        if self.selected_axis == 0:
            y = dy + (y - pad_y) / zoom_ax0
        else:
            y = self.array.shape[0] + dy + (pad_y - y) / zoom_ax0

        coordinates = [y, x]
        if self._n_dim == 3:
            coordinates.insert(self.selected_axis, self._selected_idx)
        return tuple(map(lambda c: int(round(c)), coordinates))

    def take_slice(self, array: np.ndarray, from_rgba: bool = False) -> np.ndarray:
        """
        Given a 2D or 3D array of the same shape as self._image. Returns the slice currently selected.
        In this slice, everything called a slice is a raw 2D cut of a 2D or 3D array.
        Nothing was moved or scaled.
        :param array: image
        :param from_rgba: whether the input array is greyscale or has an extra channel dim
        """
        if self._n_dim == 2:
            # No slicing required, but we need to return a copy as changes may occur
            return np.copy(array)
        # Slice the volume along the right axis
        slicer: list[slice | int] = [slice(None)] * (3 + (1 if from_rgba else 0))
        slicer[self._selected_axis] = self._selected_idx
        return array[tuple(slicer)]

    def get_rgba_slice(self) -> np.ndarray:
        """Returns the slice of the grayscale image as RGBA and the alpha channel is set to 255."""
        # Compute value range
        min_v = self.window_center - self.window_width / 2
        max_v = self.window_center + self.window_width / 2
        # Take a slice, still float grayscale
        sliced = self.take_slice(self.array)
        windowed_sliced = np.clip(sliced, min_v, max_v)
        # Convert to int [0-255]
        intarray = (255 * (windowed_sliced - min_v) / (max_v - min_v)).astype(np.uint8)
        # Convert to rgba
        rgb_view = np.stack((intarray,) * 3, axis=-1)
        rgba_view = np.dstack((rgb_view, np.full(rgb_view.shape[:2], 255, dtype=np.uint8)))
        return rgba_view

    def get_view_from_slice(self, array: np.ndarray) -> np.ndarray:
        """
        Compute a view of a 2D slice (RGBA or grayscale). If your array is 3D check the take_slice method.
        In this file a view denotes a slice that was:
        - shifted to account for the panning
        - zoomed to the correct scaling factor for each axis
        - padded or cropped so that the final array has the size self.target_size
        """
        zoom_ax0, zoom_ax1 = self._get_zooms_for_view()
        zoomed_shape = round(array.shape[0] * zoom_ax0), round(array.shape[1] * zoom_ax1)

        # Compute the correct fill color of the pixmap depending on whether it is grayscale, RGB or RGBA
        if len(array.shape) == 3:
            im_type = "RGBA"
            fillcolor = (0, 0, 0, 255) if array.shape[-1] == 3 else (0, 0, 0)
        else:
            im_type = "L"
            fillcolor = (0,)

        # noinspection PyTypeChecker
        im = Image.fromarray(array, im_type)
        # Translate image and set array shape to target shape with padding
        anc = list(self._anchor)
        if self._n_dim == 3:
            anc.pop(2 - self.selected_axis)
        dx, dy = anc
        # Compute padding: ensures that the original pixmap content is centered no matter the target size
        # Also makes the zoom feature always zoom towards the center of the pixmap content
        pad_x, pad_y = -(self.target_size[0] - zoomed_shape[1]) / 2, -(self.target_size[1] - zoomed_shape[0]) / 2

        # Does both panning and rescaling at once
        # (ax+by+c, dx+ey+f) with (a, b, c, d, e, f) transform
        if self.selected_axis == 0:
            transform = 1 / zoom_ax1, 0, dx + pad_x / zoom_ax1, \
                0, 1 / zoom_ax0, dy + pad_y / zoom_ax0
        else:
            # If we're not in the axial view, we need to invert the y-axis to make the image upside down
            transform = 1 / zoom_ax1, 0, dx + pad_x / zoom_ax1, \
                0, -1 / zoom_ax0, array.shape[0] + dy - pad_y / zoom_ax0

        im = im.transform(self.target_size, Image.AFFINE, transform, resample=Image.BILINEAR, fillcolor=fillcolor)

        return np.asarray(im)

    def center_anchor_on(self, array: np.ndarray, value: int, coordinates: tuple[float, ...]):
        """
        Given an array, finds the slice the most occurrences of the given value.
        Then also center the anchor on the bounding box for the selected slice.
        """
        if self._n_dim == 2:
            # The anchor is in orig array space
            self._anchor[0] = coordinates[1] - self.array.shape[1] / 2
            self._anchor[1] = coordinates[0] - self.array.shape[0] / 2
            return 0

        # Assumes that the image can only be 2D or 3D
        axes = [0, 1, 2]
        axes.remove(self._selected_axis)
        matches = np.sum(array == value, axis=tuple(axes))

        self._selected_idx = np.argmax(matches)

        # Ignore the coordinate for the axis where we selected the slice
        coordinates = list(coordinates)
        coordinates.pop(self.selected_axis)
        slice_shape = list(self.array.shape)
        slice_shape.pop(self.selected_axis)

        # Now just update the anchor
        anc = [coordinates[1] - slice_shape[0] / 2, coordinates[0] - slice_shape[1] / 2]
        anc.insert(2 - self.selected_axis, self._anchor[2 - self.selected_axis])
        self._anchor[:] = anc

    def _get_zooms_for_view(self) -> tuple[float, float]:
        """Returns the scaling factor (zoom factor included) for each axis in the view."""
        # Compute zoom level by axis
        zooms = list(self._zooms)

        if self._n_dim == 3:
            zooms.pop(self._selected_axis)
        zoom_ax0, zoom_ax1 = zooms

        # If in 3D and the first axis is selected, then the zoom ordering needs to be swapped.
        if self._n_dim == 3 and self._selected_axis == 0:
            zoom_ax1, zoom_ax0 = zoom_ax0, zoom_ax1

        # Multiply by the zoom factor after the min computation
        return zoom_ax0 / min(zooms) * self._zoom_factor, zoom_ax1 / min(zooms) * self._zoom_factor


class NaturalImageArrayView(ArrayView):
    """Array view for RGBA (H, W, 4) images."""

    def __init__(self, image: np.ndarray):
        """:param image: RGBA (H, W, 4) array"""
        if len(image.shape) != 3 and image.shape[2] != 4:
            raise ValueError(f"Expected a RGBA (W, H, 4) array.")
        super().__init__(n_dim=2, array=image)

    def move_anchor(self, dx: int, dy: int):
        """Convert a (mouse) pixel offset to the true array offset (independent of scaling)."""
        # Compute true offset
        self._anchor += [dx / self._zoom_factor, dy / self._zoom_factor]

    def center_anchor_on(self, y: float, x: float):
        """
        Set anchor such that the zyx coordinates given are now at the center of the view.
        May change the selected slice.
        """
        # The anchor is in orig array space
        self._anchor[0] = x - self.array.shape[1] / 2
        self._anchor[1] = y - self.array.shape[0] / 2

    def view_coordinates_to_orig_array(self, x: int, y: int) -> tuple[int, ...]:
        """Maps cursor coordinates (from the view) back to coordinates in the 2D or 3D array."""
        # Compute the size of the image before rescaling
        orig_shape = self.array.shape[:-1]
        # Compute the size of the image after rescaling
        zoomed_shape = round(orig_shape[0] * self._zoom_factor), round(orig_shape[1] * self._zoom_factor)
        # Get anchor coordinates
        dx, dy = list(self._anchor)
        # Compute padding
        pad_x, pad_y = (self.target_size[0] - zoomed_shape[1]) / 2, (self.target_size[1] - zoomed_shape[0]) / 2
        # Map x, y coordinates back
        x = dx + (x - pad_x) / self._zoom_factor
        y = dy + (y - pad_y) / self._zoom_factor
        return tuple(map(lambda c: int(round(c)), [y, x]))

    def get_view(self, array: np.ndarray) -> np.ndarray:
        """
        Compute the view. In this file a view denotes a slice that was:
        - shifted to account for the panning
        - zoomed to the correct scaling factor for each axis
        - padded or cropped so that the final array has the size self.target_size
        :param array: RGBA image
        """
        orig_shape = array.shape[:-1]
        zoomed_shape = round(orig_shape[0] * self._zoom_factor), round(orig_shape[1] * self._zoom_factor)

        # noinspection PyTypeChecker
        im = Image.fromarray(array, "RGBA")
        # Translate image and set array shape to target shape with padding
        dx, dy = list(self._anchor)

        # Compute padding: ensures that the original pixmap content is centered no matter the target size
        # Also makes the zoom feature always zoom towards the center of the pixmap content
        pad_x, pad_y = -(self.target_size[0] - zoomed_shape[1]) / 2, -(self.target_size[1] - zoomed_shape[0]) / 2

        # Does both panning and rescaling at once
        # (ax+by+c, dx+ey+f) with (a, b, c, d, e, f) transform
        transform = 1 / self._zoom_factor, 0, dx + pad_x / self._zoom_factor, \
            0, 1 / self._zoom_factor, dy + pad_y / self._zoom_factor

        im = im.transform(self.target_size, Image.AFFINE, transform, resample=Image.BILINEAR, fillcolor=(0, 0, 0, 255))

        return np.asarray(im)
