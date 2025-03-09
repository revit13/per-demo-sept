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

from kserve import Model, ModelServer, model_server, InferRequest, InferOutput, InferResponse
from kserve.errors import InvalidInput
from kserve.utils.utils import generate_uuid

from brevitas import nn as qnn
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat

N_BITS = 3

MODEL_EXTENSIONS = ".pt"

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 qlinear_args={
                     "weight_bit_width": N_BITS,
                     "weight_quant": Int8WeightPerTensorFloat,
                     "bias": True,
                     "bias_quant": None,
                     "narrow_range": True,
                     "return_quant_tensor": False,
                 },
                 qidentity_args={"bit_width": N_BITS,
                                 "act_quant": Int8ActPerTensorFloat},
                 ):
        super().__init__()
        self.quant_inp = qnn.QuantIdentity(**qidentity_args)
        self.fc1 = qnn.QuantLinear(input_size, hidden_size, **qlinear_args)
        self.relu1 = qnn.QuantReLU(**qidentity_args)
        self.fc2 = qnn.QuantLinear(hidden_size, hidden_size, **qlinear_args)
        self.relu2 = qnn.QuantReLU(**qidentity_args)
        self.fc3 = qnn.QuantLinear(hidden_size, output_size, **qlinear_args)

    def forward(self, state):
        x = self.quant_inp(state)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

# This custom predictor example implements the custom model following KServe REST v1/v2 protocol,
# the input can be raw image base64 encoded bytes or image tensor which is pre-processed by transformer
# and then passed to the custom predictor, the output is the prediction response.
class PerModel(Model):
    def __init__(self, name: str, model_dir: str):
        super().__init__(name)
        self.model_dir = model_dir
        self.actor = None
        self.ready = False
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
        actor_file_path=model_files[0]
        self.model = torch.load(actor_file_path, weights_only=True, map_location=torch.device('cpu'))
        # The ready flag is used by model ready endpoint for readiness probes,
        # set to True when model is loaded successfully without exceptions.
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

    def predict(self, input_tensor: torch.Tensor, headers: Dict[str, str] = None) -> Union[Dict, InferResponse]:
        state_size = input_tensor.numel()
        action_size = input_tensor.shape[0]
        hidden_size = 64
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_size, hidden_size, action_size).to(device)
        self.actor.load_state_dict(self.model)

        input_tensor_flattened=input_tensor.reshape(input_tensor.numel())
        output=self.actor(input_tensor_flattened)
        #output = self.model(input_tensor)
        #torch.nn.functional.softmax(output, dim=1)
        #values, top_5 = torch.topk(output, 5)
        #result = values.flatten().tolist()
        print(output)
        values, top_5 = torch.topk(output, 5)
        result = values.flatten().tolist()

        response_id = generate_uuid()
        #infer_output = InferOutput(name="output-0", shape=list(values.shape), datatype="FP32", data=result)
        infer_output = InferOutput(name="output-0", datatype="FP32", data=result, shape=list(values.shape))
        infer_response = InferResponse(model_name=self.name, infer_outputs=[infer_output], response_id=response_id)
        if "request-type" in headers and headers["request-type"] == "v1":
            return {"predictions": result}
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
