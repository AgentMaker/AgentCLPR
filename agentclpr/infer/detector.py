import numpy as np

from math import ceil
from itertools import product

from .utility import load_onnx, preprocess, gen_scale, decode, decode_landm, nms


class CLPDetector:
    def __init__(
            self,
            model_path,
            confidence_threshold=0.02,
            top_k=1000,
            nms_threshold=0.4,
            vis_threshold=0.7,
            keep_top_k=500,
            min_sizes=[[24, 48], [96, 192], [384, 768]],
            steps=[8, 16, 32],
            clip=False,
            variance=[0.1, 0.2],
            providers='auto'):
        self.session, self.inputs_name, self.outputs_name = load_onnx(
            model_path, providers)
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        self.nms_threshold = nms_threshold
        self.keep_top_k = keep_top_k
        self.vis_threshold = vis_threshold

        self.min_sizes = min_sizes
        self.steps = steps
        self.clip = clip
        self.variance = variance

    def __call__(self, image):
        h, w, _ = image.shape
        image = preprocess(image)
        loc, conf, landms = self.session.run(
            self.outputs_name,
            input_feed={
                self.inputs_name[0]: image
            }
        )
        priors = self.gen_priors(h, w)

        boxes = decode(loc.squeeze(0), priors, self.variance)
        boxes = boxes * gen_scale(h, w)

        landms = decode_landm(landms.squeeze(0), priors, self.variance)
        landms = landms * gen_scale(h, w, 4)

        scores = conf.squeeze(0)[:, 1]

        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(
            np.float32, copy=False)
        keep = nms(dets, self.nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:self.keep_top_k, :]
        landms = landms[:self.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)

        inds = np.where(dets[:, 4] > self.vis_threshold)[0]
        dets = dets[inds]
        return dets

    def gen_priors(self, h, w):
        anchors = []
        self.feature_maps = [
            [ceil(h/step), ceil(w/step)] for step in self.steps]
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / w
                    s_ky = min_size / h
                    dense_cx = [x * self.steps[k] / w
                                for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / h
                                for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = np.array(anchors).reshape(-1, 4)
        if self.clip:
            np.clip(output, 0, 1)
        return output
