import torch
import json
import requests
import json
import os
import numpy as np



array = torch.load('./reduced_tronchetto_array.pt')

# Create request message to be sent to the predictor
message_data = {}
inputs = {}
message_data["inputs"] = []
inputs["name"]="input1"
inputs["shape"] = array.shape
inputs["datatype"]="FP32" # as the given per model expects float32
inputs["data"]=array.tolist()
message_data["inputs"].append(inputs)
print(message_data)
# Call predictor

service_hostname=os.environ["SERVICE_HOSTNAME"]
model_name=os.environ["MODEL_NAME"]
ingress_ip="localhost"
ingress_port=os.environ["INGRESS_PORT"]
predictor_url = f"http://{ingress_ip}:{ingress_port}/v2/models/{model_name}/infer"
print(predictor_url)
request_headers = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "Host": service_hostname,
}
print(json.dumps(message_data))
response = requests.post(predictor_url, headers=request_headers, data=json.dumps(message_data))
print(response)
response_message = json.loads(response.text)
output1 = np.array(response_message["outputs"][0]['data'], dtype=np.float32)

# postprocess
print(output1)
