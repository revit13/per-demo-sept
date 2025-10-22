# Copyright 2021 The KServe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

from torchvision import models, transforms
from typing import Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import base64
import io
import numpy as np
import os
from kserve.storage import Storage
import torch
import numpy as np
import random
import os
import time
import json

from kserve import Model, ModelServer, model_server, InferRequest, InferOutput, InferResponse
from kserve.errors import InvalidInput
from kserve.utils.utils import generate_uuid

from .bdq_agent import BDQNetwork
MODEL_EXTENSIONS = ".pt"
ENVIRONMENT = {
    "map_name": "10_nodes_osm",  # AutoMap
}

NNETWORKS = {
    "hidden_size": 1024,
    "layers": 10
}

# This custom predictor example implements the custom model following KServe REST v1/v2 protocol,
# the input can be raw image base64 encoded bytes or image tensor which is pre-processed by transformer
# and then passed to the custom predictor, the output is the prediction response.


class PerModel(Model):
    def __init__(self, name: str,
                 model_dir: str,):
        super().__init__(name)
        self.model_dir = model_dir
        self.actor = None
        self.ready = False


        # Set up device (GPU if available, else CPU)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n\t--> Running model in {self.device}\n")

        # Build networks
        initial_state = np.load("/home/kserve/initial_state.npy")
        state_size = initial_state.flatten().size

        with open("/home/kserve/branching_nodes_candidate_splits.json", "r") as f:
            data = json.load(f)

        self.branching_nodes = {
            int(k): v for k, v in data["branching_nodes"].items()}
        self.candidate_splits = {
            int(k): v for k, v in data["candidate_splits"].items()}

        self.hidden_size = NNETWORKS["hidden_size"]
        self.layers = NNETWORKS["layers"]

        self.model = BDQNetwork(state_size, self.branching_nodes,
                                hidden_size=self.hidden_size,
                                layers=self.layers,
                                device=self.device)

        self.load()

    def load(self):
        model_path = Storage.download(self.model_dir)
        model_files = []
        for file in os.listdir(model_path):
            file_path = os.path.join(model_path, file)
            if os.path.isfile(file_path) and file.endswith(MODEL_EXTENSIONS):
                model_files.append(file_path)
        if len(model_files) == 0:
            raise ModelMissingError(model_path)
        elif len(model_files) > 1:
            raise RuntimeError(
                "More than one model file is detected, "
                f"Only one is allowed within model_dir: {model_files}"
            )
        saved_model_path = model_files[0]

        checkpoint = torch.load(saved_model_path, map_location="cpu")

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)

        self.ready = True
        print("Model loaded")

    def preprocess(self, payload: Union[Dict, InferRequest], headers: Dict[str, str] = None) -> torch.Tensor:
        raw_img_data = None
        if isinstance(payload, Dict) and "instances" in payload:
            headers["request-type"] = "v1"
            if "data" in payload["instances"][0]:
                # assume the data is already preprocessed in transformer
                np_array = np.asarray(payload["instances"][0]["data"])
                input_tensor = torch.Tensor(np_array)
                return input_tensor.unsqueeze(0)
            elif "image" in payload["instances"][0]:
                # Input follows the Tensorflow V1 HTTP API for binary values
                # https://www.tensorflow.org/tfx/serving/api_rest#encoding_binary_values
                img_data = payload["instances"][0]["image"]["b64"]
                raw_img_data = base64.b64decode(img_data)
        elif isinstance(payload, InferRequest):
            infer_input = payload.inputs[0]
            if infer_input.datatype == "BYTES":
                if payload.from_grpc:
                    raw_img_data = infer_input.data[0]
                else:
                    raw_img_data = base64.b64decode(infer_input.data[0])
            elif infer_input.datatype == "FP32":
                # assume the data is already preprocessed in transformer
                input_np = infer_input.as_numpy()
                return torch.Tensor(input_np)
        else:
            raise InvalidInput("invalid payload")

        input_image = Image.open(io.BytesIO(raw_img_data))
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        return input_tensor.unsqueeze(0)

    def predict(self, state: np.array, headers: Dict[str, str] = None) -> Union[Dict, InferResponse]:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_branches = self.model(state_tensor)

        action_dict = {node: torch.argmax(
            q_branches[node]).item() for node in self.branching_nodes}

        print(f"ACTION_DICT {action_dict.keys()}")

        response_id = generate_uuid()
        #infer_output = InferOutput(name="output-0", shape=list(values.shape), datatype="FP32", data=result)
        infer_response= InferResponse(model_name=self.name, response_id=response_id, infer_outputs=[  InferOutput(        name="bdq_output",       shape=[1],        datatype="BYTES",       data=[json.dumps(action_dict)]     )   ])
#        infer_output = InferOutput(name="output-0", datatype="FP32", data=action_dict["data"], shape=list(action_dict["shape"])
 #       infer_response = InferResponse(model_name=self.name, infer_outputs=[infer_output], response_id=response_id)
        if "request-type" in headers and headers["request-type"] == "v1":
            return {"predictions": action_dict}
        else:
            return infer_response



parser = argparse.ArgumentParser(parents=[model_server.parser])
parser.add_argument(
    "--model_dir", required=True, help="A local path to the model directory"
)
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    model = PerModel(args.model_name, args.model_dir)
    model.load()
    ModelServer().start([model])
