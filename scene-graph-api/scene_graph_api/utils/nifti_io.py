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

import gzip
import io
from os import PathLike

import nibabel as nib
import numpy as np
from nibabel.spatialimages import SpatialFirstSlicer
from typing_extensions import Self

PATH_T = str | PathLike


# TODO fix .as_reoriented call on the wrapper

class NiftiImageWrapper(nib.Nifti1Image):
    """Wrapper class with many utility functions."""

    # https://stackoverflow.com/questions/44659851/unicodedecodeerror-utf-8-codec-cant-decode-byte-0x8b-in-position-1-invalid
    _encoding = "ISO-8859–1"

    # We definitely do not want to do that super().__init__() call because of self.__getattr__
    # But we still have nib.Nifti1Image as a base class for auto-completion and since we technically support all methods
    # noinspection PyMissingConstructor
    def __init__(self, img: nib.Nifti1Image, is_depth_first: bool):
        self._img = img
        self._is_depth_first = is_depth_first

    @classmethod
    def from_array(
            cls,
            array: np.ndarray,
            affine: np.ndarray,
            header: None | nib.Nifti1Header,
            is_depth_first: bool = True
    ) -> Self:
        """Shorthand for avoiding to manually create a NiftiImage, when creating a NiftiImageWrapper from an array."""
        return cls(nib.Nifti1Image(array, affine=affine, header=header), is_depth_first)

    @property
    def is_depth_first(self) -> bool:
        return self._is_depth_first

    def __getattr__(self, item):
        return self._img.__getattribute__(item)

    @property
    def slicer(self) -> "SpatialFirstSlicerWrapper":
        return SpatialFirstSlicerWrapper(self._img.slicer, self._is_depth_first)

    @classmethod
    def load(cls, path: PATH_T) -> Self:
        """Load a Nifti image width-first (as per default)."""
        return cls(nib.load(path), False)

    @classmethod
    def load_depth_first(cls, path: PATH_T) -> Self:
        """Load a Nifti image width-first and convert it to depth-first."""
        return cls.load(path).swap_ordering()

    def swap_ordering(self) -> Self:
        """Convert in-place a width-first image into a depth-first one and vice-versa."""
        self._img = self._img.as_reoriented(np.array([[2, 1], [1, 1], [0, 1]]))
        self._is_depth_first = not self._is_depth_first
        return self

    def as_width_first(self) -> Self:
        """Convert the image to width-first. No-op if already the case. A new Wrapper is created."""
        if self._is_depth_first:
            return type(self)(self._img, self._is_depth_first).swap_ordering()
        return self

    def as_depth_first(self) -> Self:
        """Convert the image to depth-first. No-op if already the case. A new Wrapper is created."""
        if self._is_depth_first:
            return self
        return type(self)(self._img, self._is_depth_first).swap_ordering()

    def save(self, path: PATH_T):
        """Save the Nifti image as width-first (canonical form for saving)."""
        nib.save(self.as_width_first()._img, path)

    def save_as_is(self, path: PATH_T):
        """
        Save the Nifti image in it's current ordering.
        Warning: be careful how you use it...
        """
        nib.save(self._img, path)

    def to_str(self) -> str:
        """
        Convert the image to depth-first, convert it to a string and gzip it.
        Warning: this fis only intended for int data.
        """
        bio = io.BytesIO()
        # Seems to lead to corruption if we do it this way
        # zz = gzip.GzipFile(fileobj=bio, mode='w')
        # file_map = img.make_file_map({'header': zz, 'image': zz})
        # Instead compress everything afterward
        img = self.as_depth_first()._img
        file_map = img.make_file_map({"header": bio, "image": bio})
        img.to_file_map(file_map, dtype=np.int32)
        return gzip.compress(bio.getvalue()).decode(self._encoding)

    @classmethod
    def from_str(cls, string_bytes: str) -> Self:
        """
        Convert the string back to a depth-first image containing int32 data.
        """
        gzipped_bytes: bytes = string_bytes.encode(cls._encoding)
        fh = nib.FileHolder(fileobj=gzip.GzipFile(fileobj=io.BytesIO(gzipped_bytes)))
        return cls(nib.Nifti1Image.from_file_map({"header": fh, "image": fh}), True)

    @classmethod
    def empty(cls) -> Self:
        """Create a dummy empty image. Can be used for error handling elsewhere."""
        return cls(nib.Nifti1Image(np.empty((0, 0, 0)), np.eye(4)), False)

    def get_mask_data(self) -> np.ndarray:
        """Shorthand for getting the fdata from the array, rounding and casting to np.uint8."""
        # noinspection PyUnresolvedReferences
        return self._img.get_fdata().round().astype(np.uint8)


class SpatialFirstSlicerWrapper(SpatialFirstSlicer):
    # noinspection PyMissingConstructor
    def __init__(self, slicer: SpatialFirstSlicer, is_depth_first: bool):
        self._slicer = slicer
        self._is_depth_first = is_depth_first

    @property
    def is_depth_first(self) -> bool:
        return self._is_depth_first

    def __getattr__(self, item):
        return self._img.__getattribute__(item)

    def __getitem__(self, item) -> NiftiImageWrapper:
        return NiftiImageWrapper(self._slicer[item], self._is_depth_first)


__all__ = [NiftiImageWrapper]
