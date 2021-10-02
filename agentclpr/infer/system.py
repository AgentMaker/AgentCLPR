import cv2
import json
import numpy as np

from agentocr import OCRSystem

from .detector import CLPDetector
from .utility import clp_det_model, det_model, cls_model, rec_model, rec_char_dict_path, base64_to_cv2


class CLPSystem:
    def __init__(
            self,
            clp_det_model=clp_det_model,
            det_model=det_model,
            cls_model=cls_model,
            rec_model=rec_model,
            rec_char_dict_path=rec_char_dict_path,
            det_db_score_mode='slow',
            det_db_unclip_ratio=1.3,
            **kwarg):
        self.det = CLPDetector(clp_det_model, **kwarg)
        self.ocr = OCRSystem(det_model=det_model,
                             cls_model=cls_model,
                             rec_model=rec_model,
                             rec_char_dict_path=rec_char_dict_path,
                             det_db_score_mode=det_db_score_mode,
                             det_db_unclip_ratio=det_db_unclip_ratio,
                             **kwarg)

    def __call__(self, image):
        if isinstance(image, np.ndarray):
            results = self.det_ocr(image)
        elif isinstance(image, str):
            image = cv2.imdecode(np.fromfile(image, dtype=np.uint8), 1)
            results = self.det_ocr(image)
        else:
            raise ValueError('Please Check the image format.')
        return results

    def det_ocr(self, image):
        results = []
        det_results = self.det(image)
        for det_result in det_results:
            bbox = list(map(int, det_result[:4]))
            points = np.array(
                list(map(int, det_result[5:13]))
            ).reshape(4, 2).tolist()
            x1, y1, x2, y2 = tuple(bbox)
            w, h = x2-x1+20, y2-y1+20
            img_ocr_infer = np.pad(
                cv2.resize(image[y1-10:y2 + 11, x1-10:x2 + 11, :], (256, int(256/w*h))),
                ((100, 100), (100, 100), (0, 0)),
                mode='constant',
                constant_values=(255, 255)
            )
            ocr_results = self.ocr(img_ocr_infer)
            text = ''.join([ocr_result[1][0] for ocr_result in ocr_results])
            rec_score = np.prod(
                [ocr_result[1][1] for ocr_result in ocr_results]
            )
            results.append([points, [text, rec_score]])
        return results
