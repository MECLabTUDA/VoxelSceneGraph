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

__author__ = "asanner"

import copy
import datetime
import time
from collections import defaultdict
from logging import Logger, getLogger
from typing import TypedDict

import numpy as np

from . import mask as mask_utils
from . import mask3d as mask_utils3d
from .IouType import IouType
from .coco import COCOBase, COCO3d
from .coco.abstractions import AnyAnnotation
from .params import EvaluationParams as _EvaluationParams, DefaultParams as _DefaultParams


class EvalResults(TypedDict):
    params: _EvaluationParams
    counts: list
    date: str
    precision: np.ndarray
    recall: np.ndarray
    scores: np.ndarray


class EvalImgResults(TypedDict):
    image_id: int
    category_id: int
    aRng: str
    maxDet: int
    dtIds: list[int]
    gtIds: list[int]
    dtMatches: np.ndarray
    gtMatches: np.ndarray
    dtScores: list[float]
    gtIgnore: np.ndarray
    dtIgnore: np.ndarray


_ANNOTATION_HOLDER_T = dict[tuple[int, int], list[AnyAnnotation]]  # Dict: (img id, cat id) -> [AnyAnnotation]


class StatsSummary(dict[str, float]):
    """
    Dict used to contain the results of the COCOeval.summarize().
    Note: Replaces the np.ndarray that was previously used.
    Note: Provides a better interface by proposing a method to build keys corresponding to specific metrics.
    """

    def __init__(self, params: _EvaluationParams):
        super().__init__()
        self._params = params

    # noinspection PyPep8Naming
    def build_key(self,
                  ap: bool = True,
                  iou_thr: float | None = None,
                  area_rng: str = "all",
                  max_dets: int = 100) -> str:
        """
        Method used to build unique keys for each specific metric.
        Note: uses the same arguments as COCOeval.summarize._summarize().
        """
        metric_str = "AP" if ap else "AR"
        thr_str = f"{self._params.iouThrs[0]:0.2f}:{self._params.iouThrs[-1]:0.2f}" \
            if iou_thr is None else f"{iou_thr:0.2f}"
        area_str = f"[\"{area_rng}\"]"
        maxDets_str = f"@{max_dets}"
        return metric_str + thr_str + area_str + maxDets_str


# noinspection PyPep8Naming
class COCOeval:
    # Interface for evaluating detection on the Microsoft COCO dataset.
    #
    # The usage for CocoEval is as follows:
    #  cocoGt=..., cocoDt=...       # load dataset and results
    #  E = CocoEval(cocoGt,cocoDt); # initialize CocoEval object
    #  E.params.recThrs = ...;      # set parameters as desired
    #  E.evaluate();                # run per image evaluation
    #  E.accumulate();              # accumulate per image results
    #  E.summarize();               # display summary metrics of results
    # For example usage see evalDemo.m and http://mscoco.org/.
    #
    # The evaluation parameters are as follows (defaults in brackets):
    #  imgIds     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
    #  iouType    - ["segm"] set iouType to "segm", "bbox" or "keypoints"
    #  useCats    - [1] if true use category labels for evaluation
    # Note: if useCats=0 category labels are ignored as in proposal scoring.
    # Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
    #
    # evaluate(): evaluates detections on every image and every category and
    # concatenates the results into the "evalImgs" with fields:
    #  dtIds      - [1xD] id for each of the D detections (dt)
    #  gtIds      - [1xG] id for each of the G ground truths (gt)
    #  dtMatches  - [TxD] matching gt id at each IoU or 0
    #  gtMatches  - [TxG] matching dt id at each IoU or 0
    #  dtScores   - [1xD] confidence of each dt
    #  gtIgnore   - [1xG] ignore flag for each gt
    #  dtIgnore   - [TxD] ignore flag for each dt at each IoU
    #
    # accumulate(): accumulates the per-image, per-category evaluation
    # results in "evalImgs" into the dictionary "eval" with fields:
    #  params     - parameters used for evaluation
    #  date       - date evaluation was performed
    #  counts     - [T,R,K,A,M] parameter dimensions (see above)
    #  precision  - [TxRxKxAxM] precision for every evaluation setting
    #  recall     - [TxKxAxM] max recall for every evaluation setting
    # Note: precision and recall==-1 for settings with no gt objects.
    #
    # See also coco, mask, pycocoDemo, pycocoEvalDemo
    #
    # Microsoft COCO Toolbox.      version 2.0
    # Data, paper, and tutorials available at:  http://mscoco.org/
    # Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
    # Licensed under the Simplified BSD License [see coco/license.txt]

    _NO_SCORE = -1.  # Needs to be negative, since comparing for being greater is easier than equality with float

    def __init__(
            self,
            cocoGt: COCOBase,
            cocoDt: COCOBase,
            iouType: IouType = IouType.Segmentation,
            params: _EvaluationParams | None = None,
            logger: Logger | None = None
    ):
        """
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :param logger:
        """
        if logger is None:
            logger = getLogger(__file__)
        self.logger = logger

        self.params = params if params is not None else _DefaultParams(iouType=iouType)  # parameters
        self._paramsEval = copy.deepcopy(self.params)  # parameters for evaluation
        self._gts: _ANNOTATION_HOLDER_T = defaultdict(list)  # (image id, cat id) to ann
        self._dts: _ANNOTATION_HOLDER_T = defaultdict(list)  # (image id, cat id) to ann

        self.cocoGt = cocoGt  # ground truth COCO API
        self.cocoDt = cocoDt  # detections COCO API
        self.evalImgs: list[EvalImgResults | None] = []  # per-image per-category evaluation results [KxAxI] elements
        self.eval: EvalResults = {
            "params": self._paramsEval,
            "counts": [],
            "date": "",
            "precision": np.array([]),
            "recall": np.array([]),
            "scores": np.array([])
        }  # accumulated evaluation results
        self.stats = StatsSummary(self.params)  # result summarization
        self.ious: dict[tuple[int, int], np.ndarray] = {}  # ious between all gts and dts
        if cocoGt is not None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())

    def _prepare(self) -> None:
        """Prepare ._gts and ._dts for evaluation based on params."""

        def _to_mask(anns: list[AnyAnnotation], coco: COCOBase):
            # Modify ann["segmentation"] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann["segmentation"] = rle

        p = self.params
        if p.useCats:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        else:
            gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # Convert ground truth to mask if iouType == "segm"
        if p.iouType == IouType.Segmentation:
            _to_mask(gts, self.cocoGt)
            _to_mask(dts, self.cocoDt)
        # Set ignore flag
        for gt in gts:
            gt["ignore"] = gt["ignore"] if "ignore" in gt else 0
            gt["ignore"] = "iscrowd" in gt and gt["iscrowd"]
            if p.iouType == IouType.Keypoints:
                gt["ignore"] = (gt["num_keypoints"] == 0) or gt["ignore"]
        self._gts: _ANNOTATION_HOLDER_T = defaultdict(list)  # gt for evaluation
        self._dts: _ANNOTATION_HOLDER_T = defaultdict(list)  # dt for evaluation
        for gt in gts:
            self._gts[gt["image_id"], gt["category_id"]].append(gt)
        for dt in dts:
            self._dts[dt["image_id"], dt["category_id"]].append(dt)
        self.evalImgs: list[EvalImgResults | None] = []  # per-image per-category evaluation results
        self.eval = {
            "params": self._paramsEval,
            "counts": [],
            "date": "",
            "precision": np.array([]),
            "recall": np.array([]),
            "scores": np.array([])
        }  # accumulated evaluation results

    def evaluate(self):
        """Run per image evaluation on given images and store results (a list of dict) in self.evalImgs."""
        tic = time.time()
        self.logger.debug("Running per image evaluation...")
        params = self.params

        self.logger.info(f"Evaluate annotation type *{params.iouType}*")
        params.imgIds = list(np.unique(params.imgIds))
        if params.useCats:
            params.catIds = list(np.unique(params.catIds))
        params.maxDets = sorted(params.maxDets)

        self._prepare()

        # Use dummy value if not p.useCats
        cat_ids = params.catIds if params.useCats else [-1]

        match params.iouType:
            case IouType.Segmentation | IouType.BoundingBox:
                self.ious = {
                    (img_id, cat_id): self.computeIoU(img_id, cat_id)
                    for img_id in params.imgIds for cat_id in cat_ids
                }
            case IouType.Keypoints:
                self.ious = {
                    (img_id, cat_id): self.computeOks(img_id, cat_id)
                    for img_id in params.imgIds for cat_id in cat_ids
                }
            case _:
                raise ValueError(f"Invalid iouType ({params.iouType})")

        max_det = params.maxDets[-1]  # Maximum max det
        # Loop through images, area range, max detection number
        self.evalImgs = [
            self.evaluateImg(img_id, cat_id, area_rng, max_det)
            for cat_id in cat_ids
            for area_rng in params.areaRng
            for img_id in params.imgIds
        ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        self.logger.debug(f"DONE (t={toc - tic:0.2f}s).")

    def computeIoU(self, imgId: int, catId: int) -> np.ndarray:
        """
        Return an array with IoUs for a given (image id, category id) pair.
        :param imgId: image id
        :param catId: category id, is ignored if self.params.useCats is True.
        """
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt: list[dict] = [_ for cat_id in p.catIds for _ in self._gts[imgId, cat_id]]
            dt: list[dict] = [_ for cat_id in p.catIds for _ in self._dts[imgId, cat_id]]
        if len(gt) == 0 and len(dt) == 0:
            return np.array([])

        # Sort detections by decreasing score
        indexes = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in indexes]

        # Only keep up to maximum max det considered for evaluation
        if len(dt) > p.maxDets[-1]:
            dt = dt[0: p.maxDets[-1]]

        # Get appropriate detection type
        if p.iouType == IouType.Segmentation:
            g = [g["segmentation"] for g in gt]
            d = [d["segmentation"] for d in dt]
        elif p.iouType == IouType.BoundingBox:
            g = [g["bbox"] for g in gt]
            d = [d["bbox"] for d in dt]
        else:
            raise ValueError(f"Invalid iouType ({p.iouType})")

        # Compute iou between each dt and gt region
        iscrowd = [int(o["iscrowd"]) for o in gt]
        if isinstance(self.cocoGt, COCO3d):
            ious = mask_utils3d.iou(d, g, iscrowd)
        else:
            ious = mask_utils.iou(d, g, iscrowd)

        return ious

    def computeOks(self, imgId: int, catId: int) -> np.ndarray:
        if self.params.kpt_oks_sigmas is None:
            raise RuntimeError("Keypoint sigma cannot be None. Please update the parameters.")
        if isinstance(self.cocoGt, COCO3d):
            return self._computeOks3d(imgId, catId)
        return self._computeOks2d(imgId, catId)

    # noinspection DuplicatedCode
    def _computeOks2d(self, imgId: int, catId: int) -> np.ndarray:
        p = self.params
        assert p.kpt_oks_sigmas is not None
        # dimension here should be Nxm
        gts = self._gts[imgId, catId]
        dts = self._dts[imgId, catId]
        indices = np.argsort([-d["score"] for d in dts], kind="mergesort")
        dts: list[dict] = [dts[i] for i in indices]
        if len(dts) > p.maxDets[-1]:
            dts = dts[0:p.maxDets[-1]]
        # if len(gts) == 0 and len(dts) == 0:
        if len(gts) == 0 or len(dts) == 0:
            return np.array([])
        ious = np.zeros((len(dts), len(gts)))
        sigmas = p.kpt_oks_sigmas
        variances = (sigmas * 2) ** 2
        k = len(sigmas)
        # compute oks between each detection and ground truth object
        for j, gt in enumerate(gts):
            # create bounds for ignore regions(double the gt bbox)
            g = np.array(gt["keypoints"])
            yg = g[0::3]
            xg = g[1::3]
            vg = g[2::3]
            k1 = np.count_nonzero(vg > 0)
            bb = gt["bbox"]
            y0 = bb[0] - bb[2]
            y1 = bb[0] + bb[2] * 2
            x0 = bb[1] - bb[3]
            x1 = bb[1] + bb[3] * 2
            for i, dt in enumerate(dts):
                d = np.array(dt["keypoints"])
                yd = d[0::3]
                xd = d[1::3]
                if k1 > 0:
                    # measure the per-keypoint distance if keypoints visible
                    dx = xd - xg
                    dy = yd - yg
                else:
                    # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                    z = np.zeros(k)
                    dx = np.max((z, x0 - xd), axis=0) + np.max((z, xd - x1), axis=0)
                    dy = np.max((z, y0 - yd), axis=0) + np.max((z, yd - y1), axis=0)
                e = (dx ** 2 + dy ** 2) / variances / (gt["area"] + np.spacing(1)) / 2
                if k1 > 0:
                    e = e[vg > 0]
                ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
        return ious

    # noinspection DuplicatedCode
    def _computeOks3d(self, imgId: int, catId: int) -> np.ndarray:
        p = self.params
        assert p.kpt_oks_sigmas is not None
        # dimension here should be Nxm
        gts = self._gts[imgId, catId]
        dts = self._dts[imgId, catId]
        indices = np.argsort([-d["score"] for d in dts], kind="mergesort")
        dts: list[dict] = [dts[i] for i in indices]
        if len(dts) > p.maxDets[-1]:
            dts = dts[0:p.maxDets[-1]]
        # if len(gts) == 0 and len(dts) == 0:
        if len(gts) == 0 or len(dts) == 0:
            return np.array([])
        ious = np.zeros((len(dts), len(gts)))
        sigmas = p.kpt_oks_sigmas
        variances = (sigmas * 2) ** 2
        k = len(sigmas)
        # compute oks between each detection and ground truth object
        for j, gt in enumerate(gts):
            # create bounds for ignore regions(double the gt bbox)
            g = np.array(gt["keypoints"])
            zg = g[0::4]
            yg = g[1::4]
            xg = g[2::4]
            vg = g[3::4]
            k1 = np.count_nonzero(vg > 0)
            bb = gt["bbox"]
            z0 = bb[0] - bb[3]
            z1 = bb[0] + bb[3] * 2
            y0 = bb[1] - bb[4]
            y1 = bb[1] + bb[4] * 2
            x0 = bb[2] - bb[5]
            x1 = bb[2] + bb[5] * 2
            for i, dt in enumerate(dts):
                d = np.array(dt["keypoints"])
                zd = d[0::4]
                yd = d[1::4]
                xd = d[2::4]
                if k1 > 0:
                    # measure the per-keypoint distance if keypoints visible
                    dx = xd - xg
                    dy = yd - yg
                    dz = zd - zg
                else:
                    # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                    zz = np.zeros(k)
                    dx = np.max((zz, x0 - xd), axis=0) + np.max((zz, xd - x1), axis=0)
                    dy = np.max((zz, y0 - yd), axis=0) + np.max((zz, yd - y1), axis=0)
                    dz = np.max((zz, z0 - zd), axis=0) + np.max((zz, zd - z1), axis=0)
                e = (dx ** 2 + dy ** 2 + dz ** 2) / variances / (gt["area"] + np.spacing(1)) / 2  # Or / 3?
                if k1 > 0:
                    e = e[vg > 0]
                ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
        return ious

    def evaluateImg(self, imgId: int, catId: int, aRng: tuple[float, float], maxDet: int) -> EvalImgResults | None:
        """
        Perform evaluation for single category and image.
        :param imgId: image id
        :param catId: category id
        :param aRng: area range as float values
        :param maxDet: max number of detections considered
        :return: dict (single image results)
        """
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]  # likely list[FromNumpyAnnotationDict]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]

        if len(gt) == 0 and len(dt) == 0:
            return None

        # Gt ignore mask, either ignored already or not in area range
        gt_ignore_mask = [int(g["ignore"] or not (aRng[0] <= g["area"] <= aRng[1])) for g in gt]

        # Sort gt ignore last
        gt_ind = np.argsort(gt_ignore_mask, kind="mergesort")
        gt: list[dict] = [gt[i] for i in gt_ind]
        # Sort detections by highest score first
        dt_ind = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt: list[dict] = [dt[i] for i in dt_ind[0:maxDet]]
        iscrowd = [int(o["iscrowd"]) for o in gt]
        # load computed ious
        ious = self.ious[imgId, catId][:, gt_ind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm = np.zeros((T, G))
        dtm = np.zeros((T, D))
        gt_ig = np.array(gt_ignore_mask)
        dt_ig = np.zeros((T, D))
        if not len(ious) == 0:
            for t_ind, t in enumerate(p.iouThrs):
                for d_ind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t, 1 - 1e-10])
                    m = -1
                    for g_ind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[t_ind, g_ind] > 0 and not iscrowd[g_ind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m > -1 and gt_ig[m] == 0 and gt_ig[g_ind] == 1:
                            break
                        # continue to next gt unless better match made
                        if ious[d_ind, g_ind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou = ious[d_ind, g_ind]
                        m = g_ind
                    # if match made store id of match for both dt and gt
                    if m == -1:
                        continue
                    dt_ig[t_ind, d_ind] = gt_ig[m]
                    dtm[t_ind, d_ind] = gt[m]["id"]
                    gtm[t_ind, m] = d["id"]
        # set unmatched detections outside of area range to ignore
        a = np.array([d["area"] < aRng[0] or d["area"] > aRng[1] for d in dt]).reshape((1, len(dt)))
        dt_ig = np.logical_or(dt_ig, np.logical_and(dtm == 0, np.repeat(a, T, 0)))
        # store results for given image and category
        return {
            "image_id": imgId,
            "category_id": catId,
            "aRng": aRng,
            "maxDet": maxDet,
            "dtIds": [d["id"] for d in dt],
            "gtIds": [g["id"] for g in gt],
            "dtMatches": dtm,
            "gtMatches": gtm,
            "dtScores": [d["score"] for d in dt],
            "gtIgnore": gt_ig,
            "dtIgnore": dt_ig,
        }

    def accumulate(self, p: _EvaluationParams | None = None):
        """
        Accumulate per image evaluation results and store the result in self.eval
        :param p: optional parameters for accumulating the results. If None, self.params is used.
        """
        self.logger.debug("Accumulating evaluation results...")
        tic = time.time()
        if not self.evalImgs:
            self.logger.error("Please run evaluate() first. Running it now...")
            self.evaluate()

        # Allows input customized parameters
        if p is None:
            p = self.params

        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T = len(p.iouThrs)  # T = #IoU thresholds
        R = len(p.recThrs)  # R = #recall thresholds
        K = len(p.catIds) if p.useCats else 1  # K = #categories
        A = len(p.areaRng)  # A = #area ranges
        M = len(p.maxDets)  # M = #max dets
        precision = np.full((T, R, K, A, M), self._NO_SCORE)  # self._NO_SCORE for the precision of absent categories
        recall = np.full((T, K, A, M), self._NO_SCORE)
        scores = np.full((T, R, K, A, M), self._NO_SCORE)

        # Create dictionary for future indexing
        params_eval = self._paramsEval
        cat_ids = params_eval.catIds if params_eval.useCats else [-1]
        # Get indices to evaluate
        k_list = [n for n, k in enumerate(p.catIds) if k in set(cat_ids)]
        m_list = [m for m in p.maxDets if m in set(params_eval.maxDets)]
        a_list = [n for n, a in enumerate(p.areaRng) if a in set(params_eval.areaRng)]
        i_list = [n for n, i in enumerate(p.imgIds) if i in set(params_eval.imgIds)]
        I0 = len(params_eval.imgIds)
        A0 = len(params_eval.areaRng)

        # Retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0 * A0 * I0
            for a, a0 in enumerate(a_list):
                Na = a0 * I0
                for m, max_det in enumerate(m_list):
                    E: list[dict] = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E: list[EvalImgResults] = [e for e in E if e is not None]
                    if len(E) == 0:
                        continue

                    # Different sorting method generates slightly different results
                    # Merge sort is used to be consistent with Matlab implementation
                    dt_scores = np.concatenate([e["dtScores"][0:max_det] for e in E])
                    indices = np.argsort(-dt_scores, kind="mergesort")
                    dtScoresSorted = dt_scores[indices]

                    dtm = np.concatenate([e["dtMatches"][:, 0:max_det] for e in E], axis=1)[:, indices]
                    dt_ig = np.concatenate([e["dtIgnore"][:, 0:max_det] for e in E], axis=1)[:, indices]
                    gt_ig = np.concatenate([e["gtIgnore"] for e in E])
                    np_ig = np.sum(gt_ig == 0)
                    if np_ig == 0:
                        continue
                    tps = np.logical_and(dtm, np.logical_not(dt_ig))
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dt_ig))

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / np_ig
                        pr = tp / (fp + tp + np.spacing(1))

                        recall[t, k, a, m] = rc[-1] if nd else 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist()
                        q_list = [0.] * R
                        ss = [0.] * R

                        for i in range(nd - 1, 0, -1):
                            if pr[i] > pr[i - 1]:
                                pr[i - 1] = pr[i]

                        indices = np.searchsorted(rc, p.recThrs, side="left")
                        try:
                            for ri, pi in enumerate(indices):
                                q_list[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except IndexError:
                            pass
                        precision[t, :, k, a, m] = np.array(q_list)
                        scores[t, :, k, a, m] = np.array(ss)
        self.eval: EvalResults = {
            "params": p,
            "counts": [T, R, K, A, M],
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "precision": precision,
            "recall": recall,
            "scores": scores,
        }
        toc = time.time()
        self.logger.debug(f"DONE (t={toc - tic:0.2f}s).")

    def summarize(self, catId: int | None = None, omit_missing: bool = False):
        """
        Compute and display summary metrics for evaluation results.
        Note: after call, self.stats is overwritten with a new StatsSummary object.
        :param catId: if None, metrics are averaged over all images and categories.
                      if not None, only the category catId will be summarized.
        :param omit_missing: whether to not log any metric with no score (-1.0).
                             can be useful if not all area ranges are populated.
        """

        def _summarize(
                ap: bool = True,
                iou_thr: float | None = None,
                area_rng: str = "all",
                max_dets: int = 100
        ) -> float:
            iStr = " {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}"
            title_str = "Average Precision" if ap == 1 else "Average Recall"
            type_str = "(AP)" if ap else "(AR)"
            iou_str = f"{params.iouThrs[0]:0.2f}:{params.iouThrs[-1]:0.2f}" if iou_thr is None else f"{iou_thr:0.2f}"

            a_ind = [i for i, aRng in enumerate(params.areaRngLbl) if aRng == area_rng]
            m_ind = [i for i, mDet in enumerate(params.maxDets) if mDet == max_dets]
            if ap:
                # Dimension of precision: [TxRxKxAxM]
                s = self.eval["precision"]
            else:
                # Dimension of recall: [TxKxAxM]
                s = self.eval["recall"]

            # Select correct IoU threshold index
            if iou_thr is not None:
                s = s[np.where(np.abs(iou_thr - params.iouThrs) < 1e-5)[0]]

            s = s[..., cat_slice, a_ind, m_ind]
            has_score = s > self._NO_SCORE
            if len(s[has_score]) == 0:
                mean_s = self._NO_SCORE
            else:
                mean_s = np.mean(s[has_score])

            if not omit_missing or mean_s != self._NO_SCORE:
                self.logger.info(iStr.format(title_str, type_str, iou_str, area_rng, max_dets, mean_s))
            return mean_s

        # noinspection DuplicatedCode
        def _summarizeDets() -> StatsSummary:
            stats = StatsSummary(self.params)
            # Precision: All
            stats[stats.build_key(True)] = _summarize(True)
            # Precision: At all IoU thresholds
            for thr in params.iouThrsSummary:
                stats[stats.build_key(True, iou_thr=thr, max_dets=self.params.maxDets[-1])] = \
                    _summarize(True, iou_thr=thr, max_dets=self.params.maxDets[-1])
            # At all area ranges (except "all")
            for rng in params.areaRngLbl[1:]:
                for thr in params.iouThrsSummary:
                    stats[stats.build_key(True, area_rng=rng, iou_thr=thr, max_dets=self.params.maxDets[-1])] = \
                        _summarize(True, area_rng=rng, iou_thr=thr, max_dets=self.params.maxDets[-1])
            # Recall: at all max dets
            for max_det in params.maxDets:
                stats[stats.build_key(False, max_dets=max_det)] = _summarize(False, max_dets=max_det)
            # Recall: At all IoU thresholds
            for thr in params.iouThrsSummary:
                stats[stats.build_key(False, iou_thr=thr, max_dets=self.params.maxDets[-1])] = \
                    _summarize(False, iou_thr=thr, max_dets=self.params.maxDets[-1])
            # Recall: at all area ranges (except "all")
            for rng in params.areaRngLbl[1:]:
                for thr in params.iouThrsSummary:
                    stats[stats.build_key(False, area_rng=rng, iou_thr=thr, max_dets=self.params.maxDets[-1])] = \
                        _summarize(False, area_rng=rng, iou_thr=thr, max_dets=self.params.maxDets[-1])
            return stats

        # noinspection DuplicatedCode
        def _summarizeKps() -> StatsSummary:
            stats = StatsSummary(self._paramsEval)
            # Precision: All
            stats[stats.build_key(True, max_dets=self.params.maxDets[-1])] = \
                _summarize(True, max_dets=self.params.maxDets[-1])
            # Precision: At all IoU thresholds
            for thr in params.iouThrsSummary:
                stats[stats.build_key(True, iou_thr=thr, max_dets=self.params.maxDets[-1])] = \
                    _summarize(True, iou_thr=thr, max_dets=self.params.maxDets[-1])
            # Precision: At all area ranges (except "all")
            for rng in params.areaRngLbl[1:]:
                stats[stats.build_key(True, area_rng=rng, max_dets=self.params.maxDets[-1])] = \
                    _summarize(True, area_rng=rng, max_dets=self.params.maxDets[-1])
            # Recall: At maximum max dets
            stats[stats.build_key(False, max_dets=self.params.maxDets[-1])] = \
                _summarize(False, max_dets=self.params.maxDets[-1])
            # Recall: At all IoU thresholds
            for thr in params.iouThrsSummary:
                stats[stats.build_key(False, iou_thr=thr, max_dets=self.params.maxDets[-1])] = \
                    _summarize(False, iou_thr=thr, max_dets=self.params.maxDets[-1])
            # Recall: at all area ranges
            for rng in params.areaRngLbl:
                stats[stats.build_key(False, area_rng=rng, max_dets=self.params.maxDets[-1])] = \
                    _summarize(False, area_rng=rng, max_dets=self.params.maxDets[-1])

            return stats

        if not self.eval:
            # Update self.eval if the user hasn't done it first...
            self.accumulate()

        params = self.eval["params"]
        K = len(params.catIds) if params.useCats else 1
        if catId is not None and (catId > K or K < 1):
            raise ValueError(f"catId ({catId}) is invalid. (Needs to be within [1, {K}])")
        cat_slice = catId - 1 if catId is not None else slice(None)  # Need to offset because of the missing bg class

        match self.params.iouType:
            case IouType.Segmentation | IouType.BoundingBox:
                self.stats: StatsSummary = _summarizeDets()
            case IouType.Keypoints:
                self.stats: StatsSummary = _summarizeKps()
            case _:
                raise ValueError(f"Invalid iouType ({self.params.iouType})")
