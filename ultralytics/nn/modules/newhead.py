import math

import torch
from torch import nn
import torch.nn.functional as F

from ultralytics.nn.modules import DFL
from ultralytics.nn.modules.conv import autopad, Conv
# from ultralytics.nn.modules.newblock import DyDCNv2
from ultralytics.utils.tal import make_anchors, dist2bbox


class Conv_GN(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.gn = nn.GroupNorm(16, c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.gn(self.conv(x)))


class TaskDecomposition(nn.Module):
    def __init__(self, feat_channels, stacked_convs, la_down_rate=8):
        super(TaskDecomposition, self).__init__()
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.in_channels = self.feat_channels * self.stacked_convs
        self.la_conv1 = nn.Conv2d(self.in_channels, self.in_channels // la_down_rate, 1)
        self.relu = nn.ReLU(inplace=True)
        self.la_conv2 = nn.Conv2d(self.in_channels // la_down_rate, self.stacked_convs, 1, padding=0)
        self.sigmoid = nn.Sigmoid()

        self.reduction_conv = Conv_GN(self.in_channels, self.feat_channels, 1)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.normal_(self.la_conv1.weight.data, mean=0, std=0.001)
        torch.nn.init.normal_(self.la_conv2.weight.data, mean=0, std=0.001)
        torch.nn.init.zeros_(self.la_conv2.bias.data)
        torch.nn.init.normal_(self.reduction_conv.conv.weight.data, mean=0, std=0.01)

    def forward(self, feat, avg_feat=None):
        b, c, h, w = feat.shape
        if avg_feat is None:
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
        weight = self.relu(self.la_conv1(avg_feat))
        weight = self.sigmoid(self.la_conv2(weight))

        conv_weight = weight.reshape(b, 1, self.stacked_convs, 1) * \
                      self.reduction_conv.conv.weight.reshape(1, self.feat_channels, self.stacked_convs,
                                                              self.feat_channels)
        conv_weight = conv_weight.reshape(b, self.feat_channels, self.in_channels)
        feat = feat.reshape(b, self.in_channels, h * w)
        feat = torch.bmm(conv_weight, feat).reshape(b, self.feat_channels, h, w)
        feat = self.reduction_conv.gn(feat)
        feat = self.reduction_conv.act(feat)

        return feat


class Scale(nn.Module):
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class Detect_LSCD(nn.Module):
    # Lightweight Shared Convolutional Detection Head
    """YOLOv8 Detect head for detection models."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, hidc=256, ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        self.conv = nn.ModuleList(nn.Sequential(Conv_GN(x, hidc, 1)) for x in ch)
        self.share_conv = nn.Sequential(Conv_GN(hidc, hidc, 3), Conv_GN(hidc, hidc, 3))
        self.cv2 = nn.Conv2d(hidc, 4 * self.reg_max, 1)
        self.cv3 = nn.Conv2d(hidc, self.nc, 1)
        self.scale = nn.ModuleList(Scale(1.0) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = self.conv[i](x[i])
            x[i] = self.share_conv(x[i])
            x[i] = torch.cat((self.scale[i](self.cv2(x[i])), self.cv3(x[i])), 1)
        if self.training:  # Training path
            return x

        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(box)

        if self.export and self.format in ("tflite", "edgetpu"):
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            img_h = shape[2]
            img_w = shape[3]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * img_size)
            dbox = dist2bbox(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2], xywh=True, dim=1)

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        # for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
        m.cv2.bias.data[:] = 1.0  # box
        m.cv3.bias.data[: m.nc] = math.log(5 / m.nc / (640 / 16) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes):
        """Decode bounding boxes."""
        return dist2bbox(self.dfl(bboxes), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

#
# class Detect_TADDH(nn.Module):
#     # Task Dynamic Align Detection Head
#     """YOLOv8 Detect head for detection models."""
#
#     dynamic = False  # force grid reconstruction
#     export = False  # export mode
#     shape = None
#     anchors = torch.empty(0)  # init
#     strides = torch.empty(0)  # init
#
#     def __init__(self, nc=80, hidc=256, ch=()):
#         """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
#         super().__init__()
#         self.nc = nc  # number of classes
#         self.nl = len(ch)  # number of detection layers
#         self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
#         self.no = nc + self.reg_max * 4  # number of outputs per anchor
#         self.stride = torch.zeros(self.nl)  # strides computed during build
#         self.share_conv = nn.Sequential(Conv_GN(hidc, hidc // 2, 3), Conv_GN(hidc // 2, hidc // 2, 3))
#         self.cls_decomp = TaskDecomposition(hidc // 2, 2, 16)
#         self.reg_decomp = TaskDecomposition(hidc // 2, 2, 16)
#         self.DyDCNV2 = DyDCNv2(hidc // 2, hidc // 2)
#         self.spatial_conv_offset = nn.Conv2d(hidc, 3 * 3 * 3, 3, padding=1)
#         self.offset_dim = 2 * 3 * 3
#         self.cls_prob_conv1 = nn.Conv2d(hidc, hidc // 4, 1)
#         self.cls_prob_conv2 = nn.Conv2d(hidc // 4, 1, 3, padding=1)
#         self.cv2 = nn.Conv2d(hidc // 2, 4 * self.reg_max, 1)
#         self.cv3 = nn.Conv2d(hidc // 2, self.nc, 1)
#         self.scale = nn.ModuleList(Scale(1.0) for x in ch)
#         self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
#
#     def forward(self, x):
#         """Concatenates and returns predicted bounding boxes and class probabilities."""
#         for i in range(self.nl):
#             stack_res_list = [self.share_conv[0](x[i])]
#             stack_res_list.extend(m(stack_res_list[-1]) for m in self.share_conv[1:])
#             feat = torch.cat(stack_res_list, dim=1)
#
#             # task decomposition
#             avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
#             cls_feat = self.cls_decomp(feat, avg_feat)
#             reg_feat = self.reg_decomp(feat, avg_feat)
#
#             # reg alignment
#             offset_and_mask = self.spatial_conv_offset(feat)
#             offset = offset_and_mask[:, :self.offset_dim, :, :]
#             mask = offset_and_mask[:, self.offset_dim:, :, :].sigmoid()
#             reg_feat = self.DyDCNV2(reg_feat, offset, mask)
#
#             # cls alignment
#             cls_prob = self.cls_prob_conv2(F.relu(self.cls_prob_conv1(feat))).sigmoid()
#
#             x[i] = torch.cat((self.scale[i](self.cv2(reg_feat)), self.cv3(cls_feat * cls_prob)), 1)
#         if self.training:  # Training path
#             return x
#
#         # Inference path
#         shape = x[0].shape  # BCHW
#         x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
#         if self.dynamic or self.shape != shape:
#             self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
#             self.shape = shape
#
#         if self.export and self.format in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):  # avoid TF FlexSplitV ops
#             box = x_cat[:, : self.reg_max * 4]
#             cls = x_cat[:, self.reg_max * 4:]
#         else:
#             box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
#         dbox = self.decode_bboxes(box)
#
#         if self.export and self.format in ("tflite", "edgetpu"):
#             img_h = shape[2]
#             img_w = shape[3]
#             img_size = torch.tensor([img_w, img_h, img_w, img_h], device=box.device).reshape(1, 4, 1)
#             norm = self.strides / (self.stride[0] * img_size)
#             dbox = dist2bbox(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2], xywh=True, dim=1)
#
#         y = torch.cat((dbox, cls.sigmoid()), 1)
#         return y if self.export else (y, x)
#
#     def bias_init(self):
#         """Initialize Detect() biases, WARNING: requires stride availability."""
#         m = self  # self.model[-1]  # Detect() module
#         m.cv2.bias.data[:] = 1.0  # box
#         m.cv3.bias.data[: m.nc] = math.log(5 / m.nc / (640 / 16) ** 2)  # cls (.01 objects, 80 classes, 640 img)
#
#     def decode_bboxes(self, bboxes):
#         """Decode bounding boxes."""
#         return dist2bbox(self.dfl(bboxes), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides


class Pose_LSCD(Detect_LSCD):
    """YOLOv8 Pose head for keypoints models."""

    def __init__(self, nc=80, kpt_shape=(17, 3), hidc=256, ch=()):
        """Initialize YOLO network with default parameters and Convolutional Layers."""
        super().__init__(nc, hidc, ch)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total
        self.detect = Detect_LSCD.forward
        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 1), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x):
        """Perform forward pass through YOLO model and return predictions."""
        bs = x[0].shape[0]  # batch size
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
        x = self.detect(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(bs, kpt)
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, bs, kpts):
        """Decodes keypoints."""
        ndim = self.kpt_shape[1]
        if self.export:  # required for TFLite export to avoid 'PLACEHOLDER_FOR_GREATER_OP_CODES' bug
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                y[:, 2::3] = y[:, 2::3].sigmoid()  # sigmoid (WARNING: inplace .sigmoid_() Apple MPS bug)
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y

#
# class Pose_TADDH(Detect_TADDH):
#     """YOLOv8 Pose head for keypoints models."""
#
#     def __init__(self, nc=80, kpt_shape=(17, 3), hidc=256, ch=()):
#         """Initialize YOLO network with default parameters and Convolutional Layers."""
#         super().__init__(nc, hidc, ch)
#         self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
#         self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total
#         self.detect = Detect_TADDH.forward
#
#         c4 = max(ch[0] // 4, self.nk)
#         self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 1), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)
#
#     def forward(self, x):
#         """Perform forward pass through YOLO model and return predictions."""
#         bs = x[0].shape[0]  # batch size
#         kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
#         x = self.detect(self, x)
#         if self.training:
#             return x, kpt
#         pred_kpt = self.kpts_decode(bs, kpt)
#         return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))
#
#     def kpts_decode(self, bs, kpts):
#         """Decodes keypoints."""
#         ndim = self.kpt_shape[1]
#         if self.export:  # required for TFLite export to avoid 'PLACEHOLDER_FOR_GREATER_OP_CODES' bug
#             y = kpts.view(bs, *self.kpt_shape, -1)
#             a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
#             if ndim == 3:
#                 a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
#             return a.view(bs, self.nk, -1)
#         else:
#             y = kpts.clone()
#             if ndim == 3:
#                 y[:, 2::3] = y[:, 2::3].sigmoid()  # sigmoid (WARNING: inplace .sigmoid_() Apple MPS bug)
#             y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
#             y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
#             return y
