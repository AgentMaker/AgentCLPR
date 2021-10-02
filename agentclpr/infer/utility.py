import os
import cv2
import base64
import numpy as np
import onnxruntime as ort


file_path = os.path.abspath(__file__)
file_dir = os.path.dirname(file_path)
package_dir = os.path.dirname(file_dir)


clp_det_model = os.path.join(
    package_dir,
    'resources',
    'pretrained_models',
    'clp_det.onnx'
)

det_model = os.path.join(
    package_dir,
    'resources',
    'pretrained_models',
    'ch_mul_v2_c_det.onnx'
)
cls_model = os.path.join(
    package_dir,
    'resources',
    'pretrained_models',
    'ch_mul_m_cls.onnx'
)
rec_model = os.path.join(
    package_dir,
    'resources',
    'pretrained_models',
    'clp_v2_c_rec.onnx'
)
rec_char_dict_path = os.path.join(
    package_dir,
    'resources',
    'char_dicts',
    'clp_dict.txt'
)


def preprocess(img):
    img = img.astype('float32')
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = img[None, ...]
    return img


def str2providers(str):
    available_providers = ort.get_available_providers()

    if str.lower() == 'auto':
        return available_providers

    providers_dict = {
        provider.lower(): provider
        for provider in available_providers
    }

    provider_strs = [(provider_str + 'ExecutionProvider').lower()
                     for provider_str in str.split(',')]

    select_providers = [
        providers_dict[provider_str] for provider_str in provider_strs
        if provider_str in providers_dict.keys()
    ]

    if len(select_providers) == 0:
        select_providers = available_providers

    return select_providers


def load_onnx(model_path, providers='auto'):
    providers = str2providers(providers)
    sess_options = ort.SessionOptions()
    if 'DmlExecutionProvider' in providers:
        sess_options.enable_mem_pattern = False
    session = ort.InferenceSession(model_path, sess_options=sess_options)
    inputs_name = [input.name for input in session.get_inputs()]
    outputs_name = [output.name for output in session.get_outputs()]
    return session, inputs_name, outputs_name


def gen_scale(h, w, num=2):
    return np.array([w, h] * num)


def nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = np.concatenate((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_landm(pre, priors, variances):
    """Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (tensor): landm predictions for loc layers,
            Shape: [num_priors,10]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    """
    landms = np.concatenate((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                             priors[:, :2] + pre[:, 2:4] *
                             variances[0] * priors[:, 2:],
                             #priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                             priors[:, :2] + pre[:, 4:6] * \
                             variances[0] * priors[:, 2:],
                             priors[:, :2] + pre[:, 6:8] * \
                             variances[0] * priors[:, 2:],
                             ), axis=1)
    return landms


def base64_to_cv2(b64str):
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.fromstring(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data