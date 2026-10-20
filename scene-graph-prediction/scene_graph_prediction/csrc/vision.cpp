// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include "nms.h"
#include "ROIAlign.h"
#include "deform_conv.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms", &nms, "non-maximum suppression");
  m.def("nms_3d", &nms_3d, "non-maximum suppression 3D");
  m.def("roi_align_forward", &ROIAlign_forward, "ROIAlign_forward");
  m.def("roi_align_backward", &ROIAlign_backward, "ROIAlign_backward");
  m.def("roi_align_forward_3d", &ROIAlign_forward_3d, "ROIAlign_forward_3d");
  m.def("roi_align_backward_3d", &ROIAlign_backward_3d, "ROIAlign_backward_3d");
  // dcn-v2
  m.def("deform_conv_forward", &deform_conv_forward, "deform_conv_forward");
  m.def("deform_conv_backward_input", &deform_conv_backward_input, "deform_conv_backward_input");
  m.def("deform_conv_backward_parameters", &deform_conv_backward_parameters, "deform_conv_backward_parameters");
  m.def("modulated_deform_conv_forward", &modulated_deform_conv_forward, "modulated_deform_conv_forward");
  m.def("modulated_deform_conv_backward", &modulated_deform_conv_backward, "modulated_deform_conv_backward");
}
