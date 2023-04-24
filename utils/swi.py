from typing import Callable, Union

import torch
import torch.nn.functional as F
from monai.data.utils import compute_importance_map, dense_patch_slices, get_valid_patch_size
from monai.utils import BlendMode, PytorchPadMode, fall_back_tuple

def sliding_window_inference(
    inputs: torch.Tensor,
    roi_size,
    sw_batch_size: int,
    predictor: Callable,
    overlap: float = 0.25,
    mode: Union[BlendMode, str] = BlendMode.CONSTANT,
    padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
    cval=0,
):
    """
    Sliding window inference on `inputs` with `predictor`.

    When roi_size is larger than the inputs' spatial size, the input image are padded during inference.
    To maintain the same spatial sizes, the output image will be cropped to the original input size.

    Args:
        inputs: input image to be processed (assuming NCHW[D])
        roi_size (list, tuple): the spatial window size for inferences.
            When its components have None or non-positives, the corresponding inputs dimension will be used.
            if the components of the `roi_size` are non-positive values, the transform will use the
            corresponding components of img size. For example, `roi_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        sw_batch_size: the batch size to run window slices.
        predictor: given input tensor `patch_data` in shape NCHW[D], `predictor(patch_data)`
            should return a prediction with the same spatial shape and batch_size, i.e. NMHW[D];
            where HW[D] represents the patch spatial size, M is the number of output channels, N is `sw_batch_size`.
        overlap: Amount of overlap between scans.
        mode: {``"constant"``, ``"gaussian"``}
            How to blend output of overlapping windows. Defaults to ``"constant"``.

            - ``"constant``": gives equal weight to all predictions.
            - ``"gaussian``": gives less weight to predictions on edges of windows.

        padding_mode: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}
            Padding mode when ``roi_size`` is larger than inputs. Defaults to ``"constant"``
            See also: https://pytorch.org/docs/stable/nn.functional.html#pad
        cval: fill value for 'constant' padding mode. Default: 0

    Raises:
        NotImplementedError: inputs must have batch_size=1.

    Note:
        - input must be channel-first and have a batch dim, support both spatial 2D and 3D.
        - currently only supports `inputs` with batch_size=1.
    """
    num_spatial_dims = len(inputs.shape) - 2
    assert 0 <= overlap < 1, "overlap must be >= 0 and < 1."

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    image_size_ = list(inputs.shape[2:])
    batch_size = inputs.shape[0]

    # TODO: Enable batch sizes > 1 in future
    if batch_size > 1:
        raise NotImplementedError("inputs must have batch_size=1.")

    roi_size = fall_back_tuple(roi_size, image_size_)
    # in case that image size is smaller than roi size
    image_size = tuple(max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims))
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    inputs = F.pad(inputs, pad=pad_size, mode=PytorchPadMode(padding_mode).value, value=cval)

    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)

    # Store all slices in list
    slices = dense_patch_slices(image_size, roi_size, scan_interval)

    slice_batches = []
    for slice_index in range(0, len(slices), sw_batch_size):
        slice_index_range = range(slice_index, min(slice_index + sw_batch_size, len(slices)))
        input_slices = []
        for curr_index in slice_index_range:
            curr_slice = slices[curr_index]
            if len(curr_slice) == 3:
                input_slices.append(inputs[0, :, curr_slice[0], curr_slice[1], curr_slice[2]])
            else:
                input_slices.append(inputs[0, :, curr_slice[0], curr_slice[1]])
        slice_batches.append(torch.stack(input_slices))

    # Perform predictions
    output_rois = list()
    mgmt_pred = 0
    feature = 0
    for data in slice_batches:
        if predictor.out_feature_vector:
            seg_prob, mgmt_prob, feature_vec = predictor(data)
            output_rois.append(seg_prob)
            mgmt_pred += mgmt_prob / len(slice_batches)
            feature += feature_vec / len(slice_batches)
        else:
            seg_prob, mgmt_prob  = predictor(data)  # batched patch segmentation
            output_rois.append(seg_prob)
            mgmt_pred += mgmt_prob

    # stitching output image
    output_classes = output_rois[0].shape[1]
    output_shape = [batch_size, output_classes] + list(image_size)

    # Create importance map
    importance_map = compute_importance_map(get_valid_patch_size(image_size, roi_size), mode=mode, device=inputs.device)

    # allocate memory to store the full output and the count for overlapping parts
    output_image = torch.zeros(output_shape, dtype=torch.float32, device=inputs.device)
    count_map = torch.zeros(output_shape, dtype=torch.float32, device=inputs.device)

    for window_id, slice_index in enumerate(range(0, len(slices), sw_batch_size)):
        slice_index_range = range(slice_index, min(slice_index + sw_batch_size, len(slices)))

        # store the result in the proper location of the full output. Apply weights from importance map.
        for curr_index in slice_index_range:
            curr_slice = slices[curr_index]
            if len(curr_slice) == 3:
                output_image[0, :, curr_slice[0], curr_slice[1], curr_slice[2]] += (
                    importance_map * output_rois[window_id][curr_index - slice_index, :]
                )
                count_map[0, :, curr_slice[0], curr_slice[1], curr_slice[2]] += importance_map
            else:
                output_image[0, :, curr_slice[0], curr_slice[1]] += (
                    importance_map * output_rois[window_id][curr_index - slice_index, :]
                )
                count_map[0, :, curr_slice[0], curr_slice[1]] += importance_map

    # account for any overlapping sections
    output_image /= count_map

    mgmt_pred = mgmt_pred.mean(dim=0, keepdim=True)
    mgmt_pred = torch.sigmoid(mgmt_pred)[0]

    if predictor.out_feature_vector:
        return output_image[
                    ...,
                    pad_size[4] : image_size_[0] + pad_size[4],
                    pad_size[2] : image_size_[1] + pad_size[2],
                    pad_size[0] : image_size_[2] + pad_size[0],
                ], mgmt_pred, feature
    else:
        return output_image[
                    ...,
                    pad_size[4] : image_size_[0] + pad_size[4],
                    pad_size[2] : image_size_[1] + pad_size[2],
                    pad_size[0] : image_size_[2] + pad_size[0],
            ], mgmt_pred



def _get_scan_interval(image_size, roi_size, num_spatial_dims: int, overlap: float):
    assert len(image_size) == num_spatial_dims, "image coord different from spatial dims."
    assert len(roi_size) == num_spatial_dims, "roi coord different from spatial dims."

    scan_interval = []
    for i in range(num_spatial_dims):
        if roi_size[i] == image_size[i]:
            scan_interval.append(int(roi_size[i]))
        else:
            # scan interval is (1-overlap)*roi_size
            scan_interval.append(int(roi_size[i] * (1 - overlap)))
    return tuple(scan_interval)