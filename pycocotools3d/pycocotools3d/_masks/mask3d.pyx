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

# distutils: language = c

#**************************************************************************
# Microsoft COCO Toolbox.      version 3.0
# Data, paper, and tutorials available at:  http://mscoco.org/
# Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
# Licensed under the Simplified BSD License [see coco/license.txt]
#**************************************************************************

__author__ = 'asanner'

# import both Python-level and C-level symbols of Numpy
# the API uses Numpy to interface C and Python
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

# Initialized Numpy. must do.
np.import_array()

# import numpy C function
# we use PyArray_ENABLEFLAGS to make Numpy ndarray responsible to memory management
cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)

# Declare the prototype of the C functions in MaskApi.h
cdef extern from "maskApi3d.h":
    ctypedef unsigned int uint
    ctypedef unsigned long siz
    ctypedef unsigned char byte
    ctypedef double * BB3D
    ctypedef struct RLE3D:
        siz d,
        siz h,
        siz w,
        siz m,
        uint * cnts,
    void rlesInit3D(RLE3D ** R, siz n)
    void rleEncode3D(RLE3D *R, const byte *M, siz d, siz h, siz w, siz n)
    byte rleDecode3D(const RLE3D *R, byte *mask, siz n)
    void rleMerge3D(const RLE3D *R, RLE3D *M, siz n, int intersect)
    void rleArea3D(const RLE3D *R, siz n, uint *a)
    void rleIou3D(RLE3D *dt, RLE3D *gt, siz m, siz n, byte *iscrowd, double *o)
    void bbIou3D(BB3D dt, BB3D gt, siz m, siz n, byte *iscrowd, double *o)
    void rleToBbox3D(const RLE3D *R, BB3D bb, siz n)
    void rleFrBbox3D(RLE3D *R, const BB3D bb, siz d, siz h, siz w, siz n)
    char * rleToString3D(const RLE3D *R)
    void rleFrString3D(RLE3D *R, char *s, siz d, siz h, siz w)

# python class to wrap RLE3D array in C
# the class handles the memory allocation and deallocation
cdef class RLE3Ds:
    cdef RLE3D *R
    cdef siz n

    def __cinit__(self, siz n =0):
        rlesInit3D(&self.R, n)
        self.n = n

    # free the RLE3D array here
    def __dealloc__(self):
        if self.R is not NULL:
            for i in range(self.n):
                free(self.R[i].cnts)
            free(self.R)

    def __getattr__(self, key):
        if key == 'n':
            return self.n
        raise AttributeError(key)

# python class to wrap Mask array in C
# the class handles the memory allocation and deallocation
cdef class Masks3D:
    cdef byte *mask
    cdef siz d
    cdef siz h
    cdef siz w
    cdef siz n

    def __cinit__(self, d, h, w, n):
        self.mask = <byte *> malloc(d * h * w * n * sizeof(byte))
        self.d = d
        self.h = h
        self.w = w
        self.n = n
    # def __dealloc__(self):
    # the memory management of _mask has been passed to np.ndarray
    # it doesn't need to be freed here

    # called when passing into np.array() and return a np.ndarray in column-major order
    def __array__(self):
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.d * self.h * self.w * self.n
        # Create a 1D array, and reshape it to fortran/Matlab column-major array
        ndarray = np.PyArray_SimpleNewFromData(1, shape, np.NPY_UINT8, self.mask) \
            .reshape((self.d, self.h, self.w, self.n), order='F')
        # The mask allocated by Masks is now handled by ndarray
        PyArray_ENABLEFLAGS(ndarray, np.NPY_OWNDATA)
        return ndarray

# internal conversion from Python RLE3Ds object to compressed RLE3D format
def _to_json(RLE3Ds Rs):
    cdef siz n = Rs.n
    cdef bytes py_string
    cdef char * c_string
    objs = []
    for i in range(n):
        c_string = rleToString3D(<RLE3D *> &Rs.R[i])
        py_string = c_string
        objs.append({
            'size': [Rs.R[i].d, Rs.R[i].h, Rs.R[i].w],
            'counts': py_string
        })
        free(c_string)
    return objs

# internal conversion from compressed RLE3D format to Python RLE3Ds object
def _from_json(rleObjs):
    cdef siz n = len(rleObjs)
    Rs = RLE3Ds(n)
    cdef bytes py_string
    cdef char * c_string
    for i, obj in enumerate(rleObjs):
        py_string = str.encode(obj['counts']) if type(obj['counts']) == str else obj['counts']
        c_string = py_string
        rleFrString3D(<RLE3D *> &Rs.R[i], <char *> c_string, obj['size'][0], obj['size'][1], obj['size'][2])
    return Rs

# encode mask to RLE3Ds objects
# list of RLE3D string can be generated by RLE3Ds member function
def encode(np.ndarray[np.uint8_t, ndim=4, mode='fortran'] mask):
    d, h, w, n = mask.shape[0], mask.shape[1], mask.shape[2], mask.shape[3]
    cdef RLE3Ds Rs = RLE3Ds(n)
    rleEncode3D(Rs.R, <byte *> mask.data, d, h, w, n)
    objs = _to_json(Rs)
    return objs

# decode mask from compressed list of RLE3D string or RLE3Ds object
def decode(rleObjs):
    cdef RLE3Ds Rs = _from_json(rleObjs)
    d, h, w, n = Rs.R[0].d, Rs.R[0].h, Rs.R[0].w, Rs.n
    masks = Masks3D(d, h, w, n)
    if rleDecode3D(<RLE3D *> Rs.R, masks.mask, n) != 1:
        raise ValueError("Invalid RLE3D mask representation")
    return np.array(masks)

def merge(rleObjs, intersect=0):
    cdef RLE3Ds Rs = _from_json(rleObjs)
    cdef RLE3Ds R = RLE3Ds(1)
    rleMerge3D(<RLE3D *> Rs.R, <RLE3D *> R.R, <siz> Rs.n, intersect)
    obj = _to_json(R)[0]
    return obj

def area(rleObjs):
    cdef RLE3Ds Rs = _from_json(rleObjs)
    cdef uint * _a = <uint *> malloc(Rs.n * sizeof(uint))
    rleArea3D(Rs.R, Rs.n, _a)
    cdef np.npy_intp shape[1]
    shape[0] = <np.npy_intp> Rs.n
    a = np.array((Rs.n,), dtype=np.uint8)
    a = np.PyArray_SimpleNewFromData(1, shape, np.NPY_UINT32, _a)
    PyArray_ENABLEFLAGS(a, np.NPY_OWNDATA)
    return a

# iou computation. support function overload (RLE3Ds-RLE3Ds and bbox-bbox).
def iou(dt, gt, pyiscrowd):
    def _preproc(objs):
        if len(objs) == 0:
            return objs
        if type(objs) == np.ndarray:
            if len(objs.shape) == 1:
                objs = objs.reshape((objs[0], 1))
            # check if it's Nx6 bbox
            if len(objs.shape) != 2 or objs.shape[1] != 6:
                raise Exception('numpy ndarray input is only for *bounding boxes* and should have Nx6 dimension')
            objs = objs.astype(np.double)
        elif type(objs) == list:
            # check if list is in box format and convert it to np.ndarray
            is_box = np.all(np.array([(len(obj) == 6) and
                                      ((type(obj) == list) or (type(obj) == np.ndarray))
                                      for obj in objs]))
            is_rle = np.all(np.array([type(obj) == dict for obj in objs]))
            if is_box:
                objs = np.array(objs, dtype=np.double)
                if len(objs.shape) == 1:
                    objs = objs.reshape((1, objs.shape[0]))
            elif is_rle:
                objs = _from_json(objs)
            else:
                raise Exception('list input can be bounding box (Nx6) or RLE3Ds ([RLE3D])')
        else:
            raise Exception(
                'unrecognized type.  The following type: RLE3Ds (rle), np.ndarray (box), and list (box) are supported.')
        return objs
    def _rleIou(RLE3Ds dt, RLE3Ds gt, np.ndarray[np.uint8_t, ndim=1] iscrowd, siz m, siz n,
                np.ndarray[np.double_t, ndim=1] _iou):
        rleIou3D(<RLE3D *> dt.R, <RLE3D *> gt.R, m, n, <byte *> iscrowd.data, <double *> _iou.data)
    def _bbIou(np.ndarray[np.double_t, ndim=2] dt, np.ndarray[np.double_t, ndim=2] gt,
               np.ndarray[np.uint8_t, ndim=1] iscrowd, siz m, siz n, np.ndarray[np.double_t, ndim=1] _iou):
        bbIou3D(<BB3D> dt.data, <BB3D> gt.data, m, n, <byte *> iscrowd.data, <double *> _iou.data)
    def _len(obj):
        cdef siz N = 0
        if type(obj) == RLE3Ds:
            N = obj.n
        elif len(obj) == 0:
            pass
        elif type(obj) == np.ndarray:
            N = obj.shape[0]
        return N
    # convert iscrowd to numpy array
    cdef np.ndarray[np.uint8_t, ndim=1] iscrowd = np.array(pyiscrowd, dtype=np.uint8)
    # simple type checking
    cdef siz m, n, crowd_length
    dt = _preproc(dt)
    gt = _preproc(gt)
    m = _len(dt)
    n = _len(gt)
    crowd_length = len(pyiscrowd)
    assert crowd_length == n, "iou(iscrowd=) must have the same length as gt"
    if m == 0 or n == 0:
        return []
    if type(dt) != type(gt):
        raise Exception('The dt and gt should have the same data type, either RLE3Ds, list or np.ndarray')

    # define local variables
    cdef double * _iou = <double *> 0
    cdef np.npy_intp shape[1]
    # check type and assign iou function
    if type(dt) == RLE3Ds:
        _iouFun = _rleIou
    elif type(dt) == np.ndarray:
        _iouFun = _bbIou
    else:
        raise Exception('input data type not allowed.')
    _iou = <double *> malloc(m * n * sizeof(double))
    iou = np.zeros((m * n,), dtype=np.double)
    shape[0] = <np.npy_intp> m * n
    iou = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, _iou)
    PyArray_ENABLEFLAGS(iou, np.NPY_OWNDATA)
    _iouFun(dt, gt, iscrowd, m, n, iou)
    return iou.reshape((m, n), order='F')

def toBbox(rleObjs):
    cdef RLE3Ds Rs = _from_json(rleObjs)
    cdef siz n = Rs.n
    cdef BB3D _bb = <BB3D> malloc(6 * n * sizeof(double))
    rleToBbox3D(<const RLE3D *> Rs.R, _bb, n)
    cdef np.npy_intp shape[1]
    shape[0] = <np.npy_intp> 6 * n
    bb = np.array((1, 6 * n), dtype=np.double)
    bb = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, _bb).reshape((n, 6))
    PyArray_ENABLEFLAGS(bb, np.NPY_OWNDATA)
    return bb

def frBbox(np.ndarray[np.double_t, ndim=2] bb, siz d, siz h, siz w):
    cdef siz n = bb.shape[0]
    Rs = RLE3Ds(n)
    rleFrBbox3D(<RLE3D *> Rs.R, <const BB3D> bb.data, d, h, w, n)
    objs = _to_json(Rs)
    return objs

def frPoly(poly, siz d, siz h, siz w):
    raise NotImplementedError

def frUncompressedRLE3D(ucRles, siz d, siz h, siz w):
    cdef np.ndarray[np.uint32_t, ndim=1] cnts
    cdef RLE3D R
    cdef uint *data
    n = len(ucRles)
    objs = []
    for i in range(n):
        Rs = RLE3Ds(1)
        cnts = np.array(ucRles[i]['counts'], dtype=np.uint32)
        # time for malloc can be saved here, but it's fine
        data = <uint *> malloc(len(cnts) * sizeof(uint))
        for j in range(len(cnts)):
            data[j] = <uint> cnts[j]
        R = RLE3D(ucRles[i]['size'][0], ucRles[i]['size'][1], ucRles[i]['size'][2], len(cnts), <uint *> data)
        Rs.R[0] = R
        objs.append(_to_json(Rs)[0])
    return objs

def frPyObjects(pyobj, d, h, w):
    # encode rle from a list of python objects
    if type(pyobj) == np.ndarray:
        objs = frBbox(pyobj, d, h, w)
    # Note: as of Python 3.10, isinstance(pyobj[0], list | np.ndarray) is not supported
    # (not syntactically but semantically)
    elif isinstance(pyobj, list) and \
            (isinstance(pyobj[0], list) or isinstance(pyobj[0], np.ndarray)) and \
            len(pyobj[0]) == 6:
        objs = frBbox(np.array(pyobj, np.float64), d, h, w)
    elif isinstance(pyobj, list) and \
            (isinstance(pyobj[0], list) or isinstance(pyobj[0], np.ndarray)) and \
            len(pyobj[0]) > 6:
        objs = frPoly(pyobj, d, h, w)
    elif type(pyobj) == list and type(pyobj[0]) == dict and 'counts' in pyobj[0] and 'size' in pyobj[0]:
        objs = frUncompressedRLE3D(pyobj, d, h, w)
    # encode rle from single python object
    elif type(pyobj) == list and len(pyobj) == 6:
        objs = frBbox(np.array([pyobj], np.float64), d, h, w)[0]
    elif type(pyobj) == list and len(pyobj) > 6:
        objs = frPoly([pyobj], d, h, w)[0]
    elif type(pyobj) == dict and 'counts' in pyobj and 'size' in pyobj:
        objs = frUncompressedRLE3D([pyobj], d, h, w)[0]
    else:
        raise Exception('input type is not supported.')
    return objs
