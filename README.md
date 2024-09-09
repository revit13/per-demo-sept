# PER model Inference with kserve

This document contains instruction for PER model inference managed with [kserve](https://kserve.github.io/website/0.12/).

## Before you begin

Deploy kserve 1.2.0 and its prerequisites on the cluster using kserve [quick start](https://kserve.github.io/website/0.12/get_started/#install-the-kserve-quickstart-environment).

## Create a namespace

```bash
kubectl create namespace per
```

## Create an InferenceService

Before exeuting the following command make sure the s3 credentials of the bucket holding the model are stored in a secret similar to the description
in [this](https://kserve.github.io/website/latest/modelserving/kafka/kafka/#create-s3-secret-for-minio-and-attach-to-service-account) section.
```bash
kubectl apply -f isvc-per.yaml -n per 
```

## Check InferenceService status.
```bash
kubectl get inferenceservices per-custom-model -n per
```

## Determine the ingress IP and ports

Execute [section](https://kserve.github.io/website/0.12/get_started/first_isvc/#4-determine-the-ingress-ip-and-ports) from kserve quick start.
```bash
INGRESS_GATEWAY_SERVICE=$(kubectl get svc --namespace istio-system --selector="app=istio-ingressgateway" --output jsonpath='{.items[0].metadata.name}')
kubectl port-forward --namespace istio-system svc/${INGRESS_GATEWAY_SERVICE} 8080:80
```

## Verify the service is healthy

```bash
curl -H "Content-Type: application/json" -H "Host: per-custom-model.per.example.com"  localhost:8080/v1/models/per-custom-model
```

## Run the notebook

To run Jupyter notebook execute the commands from the notebook directory:

```bash
cd notebook
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the sample notebook in jupyter

Before executing the notebook make sure the input is located in `notebook` directory.

```bash
jupyter notebook
```

# Update PER custom perdictor

PER custom perdictor is written based on the instructions in the kserve [tutorial](https://kserve.github.io/website/0.12/modelserving/v1beta1/custom/custom_model/). 

To make changes in the predictor please do the following:

```bash
git clone https://github.com/kserve/kserve.git
cd python
cp custom_model_code/model.py custom_model/
```
create the docker image of the perdictor and push it to the registry:
 
```bash
docker build . -t ${DOCKER_USER}/per-custom-model:v1 -f custom_model.Dockerfile
docker push ${DOCKER_USER}/per-custom-model:v1
```

Next, update the [isvc-per.yaml](./isvc-per.yaml) inferenceservice with the new docker image tag and redeploy it.
