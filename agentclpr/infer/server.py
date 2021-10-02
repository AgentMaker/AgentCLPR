import json

from flask import Flask, request
from .system import CLPSystem
from .utility import base64_to_cv2

clp = CLPSystem()
clp_server = Flask(__name__)


@clp_server.route("/ocr", methods=['POST'])
def predict_server():
    if not request.data:
        return json.dumps([], ensure_ascii=False)

    request_data = request.data.decode('utf-8')
    request_json = json.loads(request_data)

    if 'image' not in request_json:
        return json.dumps([], ensure_ascii=False)

    image_base64 = request_json.pop('image')
    image = base64_to_cv2(image_base64)
    results = clp(image, **request_json)

    return json.dumps(results, ensure_ascii=False)
