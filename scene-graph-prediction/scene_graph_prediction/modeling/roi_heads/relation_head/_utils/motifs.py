import array
import os
import sys
import zipfile
from typing import Sequence, Callable
from urllib.request import urlretrieve

import numpy as np
import torch
from tqdm import tqdm

from scene_graph_prediction.data import DatasetCatalog
from scene_graph_prediction.structures import BoxList
from scene_graph_prediction.utils.logger import setup_logger
from ....utils import cat

_logger = setup_logger("modeling.roi_heads.relation_head._utils.motifs", save_dir="", distributed_rank=0)


def normalize_sigmoid_logits(orig_logits: torch.Tensor) -> torch.Tensor:
    orig_logits = torch.sigmoid(orig_logits)
    orig_logits = orig_logits / (orig_logits.sum(1).unsqueeze(-1) + 1e-12)
    return orig_logits


# noinspection DuplicatedCode
def generate_attributes_target(attributes: torch.Tensor, num_attributes_cat: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    From list of (int) attribute indexes to one-hot encoding.
    :returns:
        one-hot attributes
        indexes for positive cases
    """
    max_num_attributes = attributes.shape[1]
    num_obj = attributes.shape[0]

    with_attributes_idx = (attributes.sum(-1) > 0).long()
    attribute_targets = torch.zeros((num_obj, num_attributes_cat), device=attributes.device).float()

    for idx in torch.nonzero(with_attributes_idx).squeeze(1).tolist():
        for k in range(max_num_attributes):
            att_id = int(attributes[idx, k])
            if att_id == 0:
                break
            else:
                attribute_targets[idx, att_id] = 1

    return attribute_targets, with_attributes_idx


def _transpose_packed_sequence_indexes(lengths: list[int]) -> tuple[np.ndarray, list[int]]:
    """
    Get a TxB indices from sorted lengths.
    Fetch new_indexes, split by new_lengths, padding to max(new_lengths), and stack.
    :returns:
        new_indexes: array of shape [sum(lengths), ]
        new_lengths: number of elements of each time step, in descending order
    """
    new_indexes = []
    new_lengths = []
    cum_add = np.cumsum([0] + lengths)
    max_len = lengths[0]
    length_pointer = len(lengths) - 1
    for i in range(max_len):
        while length_pointer > 0 and lengths[length_pointer] <= i:
            length_pointer -= 1
        new_indexes.append(cum_add[:(length_pointer + 1)].copy())
        cum_add[:(length_pointer + 1)] += 1
        new_lengths.append(length_pointer + 1)
    new_indexes = np.concatenate(new_indexes, 0)
    return new_indexes, new_lengths


def sort_by_score(
        proposals: list[BoxList],
        scores: torch.FloatTensor
) -> tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
    """
    We'll sort everything score-wise from Hi->low, BUT we need to keep images together and sort LSTM.
    :param proposals:
    :param scores: Goodness ranging between [0, 1]. Higher numbers come FIRST.
    :returns:
        Permutation to put everything in the right order for the LSTM
        Inverse permutation
        Lengths for the TxB packed sequence.
    """
    num_rois = [len(b) for b in proposals]
    num_im = len(num_rois)

    ordered_scores = []
    for i, (score, num_roi) in enumerate(zip(scores.split(num_rois, dim=0), num_rois)):
        ordered_scores.append(score + 2. * (num_roi * 2 * num_im + i))
    ordered_scores = cat(ordered_scores, dim=0)
    _, perm = torch.sort(ordered_scores, 0, descending=True)

    num_rois = sorted(num_rois, reverse=True)
    indexes, ls_transposed = _transpose_packed_sequence_indexes(num_rois)  # move it to TxB form
    indexes = torch.tensor(indexes, dtype=torch.int64, device=scores.device)
    ls_transposed = torch.tensor(ls_transposed, dtype=torch.int64, device=scores.device)

    perm = perm[indexes]  # (batch_num_box, )
    _, inv_perm = torch.sort(perm)

    return perm, inv_perm, ls_transposed


def to_onehot(vec: torch.LongTensor, num_classes: int, fill: int = 1000) -> torch.FloatTensor:
    """
    Create a [size, num_classes] torch FloatTensor where one_hot[i, vec[i]] = fill
    :param vec: 1d torch tensor
    :param num_classes: int
    :param fill: value that we want + and - things to be.
    :returns: vec as one-hot encoded
    """
    onehot_result = vec.new(vec.size(0), num_classes).float().fill_(-fill)
    range_indexes = vec.new(vec.size(0)).long()
    torch.arange(0, vec.size(0), out=range_indexes)

    onehot_result.view(-1)[vec.long() + num_classes * range_indexes] = fill
    return onehot_result


def get_dropout_mask(dropout_probability: float, tensor_shape: tuple, device: torch.device) -> torch.Tensor:
    """Once gotten, the dropout mask remains fixed."""
    binary_mask = torch.rand(tensor_shape) > dropout_probability
    # Scale mask by 1/keep_prob to preserve output statistics.
    # noinspection PyUnresolvedReferences
    dropout_mask = binary_mask.float().to(device).div(1.0 - dropout_probability)
    return dropout_mask


def encode_box_info(proposals: list[BoxList]) -> torch.Tensor:
    """
    Encode proposed box information ("zyxzyx" mode) to [
      *centers ratios to image size,
      *lengths ratios to image size,
      *starts ratios to image size,
      *ends ratios to images size,
      area ratio to image are
    ]

    Example in 2D: [
      cy/img_height,
      cx/img_width,
      h/img_height,
      w/img_width,
      y1/img_height,
      x1/img_width,
      y2/img_height,
      x2/img_width,
      w*h/img_width*img_height
    ].
    :param proposals: list of BoxList in "zyxzyx" mode.
    """
    # TODO this could be customizable
    assert proposals and proposals[0].mode == BoxList.Mode.zyxzyx
    boxes_info = []
    for proposal in proposals:
        boxes = proposal.boxes
        img_size = proposal.size
        n_dim = proposal.n_dim
        lengths = boxes[:, n_dim:] - boxes[:, :n_dim] + 1.
        centers = boxes[:, :n_dim] + .5 * lengths
        lengths = lengths.split([1] * n_dim, dim=-1)
        centers = centers.split([1] * n_dim, dim=-1)
        zyxzyx = boxes.split([1, 1] * n_dim, dim=-1)
        assert np.prod(img_size) != 0
        info = torch.cat(
            [lengths[dim] / img_size[dim] for dim in range(n_dim)] +
            [centers[dim] / img_size[dim] for dim in range(n_dim)] +
            [zyxzyx[dim] / img_size[dim] for dim in range(n_dim)] +
            [zyxzyx[n_dim + dim] / img_size[dim] for dim in range(n_dim)] +
            [torch.prod(torch.hstack(lengths), 1).view(-1, 1) / np.prod(img_size)], dim=-1)
        boxes_info.append(info)

    return torch.cat(boxes_info, dim=0)


def obj_edge_vectors(names: Sequence[str], wv_dir: str, wv_type: str = 'glove.6B', wv_dim: int = 300) -> torch.Tensor:
    wv_dict, wv_arr, wv_size = _load_word_vectors(wv_dir, wv_type, wv_dim)

    vectors = torch.Tensor(len(names), wv_dim)
    vectors.normal_(0, 1)

    for i, token in enumerate(names):
        wv_index = wv_dict.get(token.lower())
        if wv_index is not None:
            vectors[i] = wv_arr[wv_index]
        else:
            # Try the longest word
            lw_token = sorted(token.split(" "), key=lambda x: len(x), reverse=True)[0]
            _logger.debug(f"obj_edge_vectors matching: {token} -> {lw_token}")
            wv_index = wv_dict.get(lw_token, None)
            if wv_index is not None:
                vectors[i] = wv_arr[wv_index]
            else:
                _logger.debug(f"Fail on {token}")

    return vectors


def _load_word_vectors(root: str, wv_type: str, dim: int) -> tuple[dict[str, int], torch.Tensor, int]:
    """Load word vectors from a path, trying .pt, .txt, and .zip extensions."""
    urls = {
        'glove.42B': 'https://nlp.stanford.edu/data/glove.42B.300d.zip',
        'glove.840B': 'https://nlp.stanford.edu/data/glove.840B.300d.zip',
        'glove.twitter.27B': 'https://nlp.stanford.edu/data/glove.twitter.27B.zip',
        'glove.6B': 'https://nlp.stanford.edu/data/glove.6B.zip',
    }
    if root == "":
        root = DatasetCatalog.CACHE_DIR

    if isinstance(dim, int):
        dim = str(dim) + 'd'
    f_name = os.path.join(root, wv_type + '.' + dim)

    # Loading from torch .pt format
    if os.path.isfile(f_name + '.pt'):
        f_name_pt = f_name + '.pt'
        _logger.debug('Loading word vectors from', f_name_pt)
        try:
            return torch.load(f_name_pt, map_location=torch.device("cpu"))
        except Exception as e:
            _logger.error(f"Error loading the word vectors from {f_name_pt}:\n{e}")
            sys.exit(-1)
    else:
        _logger.info(f"File not found: {f_name}.pt")

    if not os.path.isfile(f_name + '.txt'):
        _logger.info(f"File not found: {f_name}.txt")
    if os.path.isfile(f_name + '.txt'):
        f_name_txt = f_name + '.txt'
        cm = open(f_name_txt, 'rb')
        cm = [line for line in cm]
    elif os.path.basename(wv_type) in urls:
        url = urls[wv_type]
        _logger.info(f'Downloading word vectors from {url}')
        filename = os.path.basename(f_name)
        if not os.path.exists(root):
            os.makedirs(root)
        with tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
            f_name, _ = urlretrieve(url, f_name, reporthook=_reporthook(t))
            with zipfile.ZipFile(f_name, "r") as zf:
                _logger.info(f'Extracting word vectors into {root}')
                zf.extractall(root)
        if not os.path.isfile(f_name + '.txt'):
            raise RuntimeError('No word vectors of requested dimension found')
        return _load_word_vectors(root, wv_type, dim)
    else:
        raise RuntimeError('unable to load word vectors')

    wv_tokens, wv_arr, wv_size = [], array.array('d'), None
    if cm is not None:
        for line in tqdm(range(len(cm)), desc=f"loading word vectors from {f_name_txt}"):
            entries = cm[line].strip().split(b' ')
            word, entries = entries[0], entries[1:]
            if wv_size is None:
                wv_size = len(entries)
            try:
                if isinstance(word, bytes):
                    word = word.decode('utf-8')
            except Exception as e:
                _logger.error(f'Non-UTF8 token {repr(word)} ignored:\n{e}')
                continue
            wv_arr.extend(float(x) for x in entries)
            wv_tokens.append(word)

    wv_dict = {word: i for i, word in enumerate(wv_tokens)}
    wv_arr = torch.Tensor(wv_arr).view(-1, wv_size)
    ret = wv_dict, wv_arr, wv_size
    torch.save(ret, f_name + '.pt')
    return ret


def _reporthook(t: tqdm) -> Callable:
    """https://github.com/tqdm/tqdm"""
    last_b = [0]

    def inner(b=1, bsize=1, t_size=None):
        if t_size is not None:
            t.total = t_size
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner
