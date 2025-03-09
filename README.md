# PER model Inference with kserve

This document contains instructions for PER model inference managed with [kserve](https://kserve.github.io/website/0.12/).

## Before you begin

Deploy kserve 1.2.0 and its prerequisites on the cluster using kserve [quick start](https://kserve.github.io/website/0.12/get_started/#install-the-kserve-quickstart-environment).

## Create a namespace

```bash
kubectl create namespace per
kubectl config set-context --current --namespace=per
```

## Deploy Minio

To deploy Minio, follow this [link](https://kserve.github.io/website/latest/modelserving/kafka/kafka/#deploy-minio).
Make sure the s3 credentials of the bucket holding the model are stored in a secret in per namespace similar to the description
in [this](https://kserve.github.io/website/latest/modelserving/kafka/kafka/#create-s3-secret-for-minio-and-attach-to-service-account) section.
Next, Create a port-forward to communicate with minio server:

```bash
# Run port forwarding command in a different terminal
kubectl port-forward $(kubectl get pod --selector="app=minio" --output jsonpath='{.items[0].metadata.name}') 9000:9000

```
### Upload the models to Minio using the GUI


1. Open a Browser Tab: Navigate to http://localhost:9000/ in your web browser to access the Minio server.
2. Use the default login credentials to login to Minio:
   
        Access key: minio
        Secert key: minio123
3. After successfully login, click on the red circle located at the bottom right
corner of the page to create a new bucket as shown in the image below. 
Then, press the `Create bucket` button to create a new bucket named `per-bucket`.



![minio_create_bucket](images/minio_create_bucket.jpg)

4. Upload `actor_quantized_DDPGAgent_38nodes_500eps.pt` and `actor_DDPGAgent_38nodes_500eps.pt` models to the 
newly created bucket by clicking on the red circle located at 
the bottom right corner of the page as shown in the image below and pressing the `Upload file` button.

![minio_create_file](images/minio_create_file.jpg)


## Create a ClusterServingRuntime and InferenceService

```bash
kubectl apply -f csr-per.yaml -n per
kubectl apply -f isvc-per.yaml -n per 
```

To apply encrypted model deploy:

```bash
kubectl apply -f csr-per-encrypted.yaml -n per
kubectl apply -f isvc-per-encrypted.yaml -n per 
```

## Check InferenceService status.
```bash
kubectl get inferenceservices per-custom-model -n per
```

The following when using the encrypted model:
```bash
kubectl get inferenceservices per-custom-encrypted-model -n per
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


```bash
{"name":"per-custom-model","ready":true}
```
## Run the notebook

To run Jupyter notebook execute the commands from the notebook directory:

```bash
cd notebook
```

### Install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run the sample notebook in jupyter

Before executing the notebook make sure the input is located in `notebook` directory.

```bash
pip install jupyter
jupyter notebook
```

**Note:** for encrypted model inference the first two lines in the nodebook should be:

```bash
%env MODEL_NAME=per-custom-encrypted-model
HOSTNAME=!(kubectl get inferenceservice "per-custom-encrypted-model" -o jsonpath='{.status.url}' | cut -d "/" -f 3)
```

# Update PER custom predictor

PER custom predictor is written based on the instructions in the kserve [tutorial](https://kserve.github.io/website/0.12/modelserving/v1beta1/custom/custom_model/). 

To make changes to the predictor's code execute the following commands
which copies files from [custom_model_code](./custom_model_code) directory:

```bash
cd custom_model_code
ROOT_DIR=$PWD
git clone https://github.com/kserve/kserve.git
cd python
cp $ROOT_DIR/custom_model.Dockerfile .
cp $ROOT_DIR/model.py custom_model/
```

Create the docker image of the predictor and push it to the registry:
 
```bash
docker build . -t ${DOCKER_USER}/per-custom-model:v1 -f custom_model.Dockerfile
docker push ${DOCKER_USER}/per-custom-model:v1
```

Next, update the [csr-per.yaml](./csr-per.yaml) ClusterServingRuntime with the new docker image tag and redeploy it.

### Encrypted predictor

To update the encrypted predictor code, execute the following commands
which copies files from [custom_encrypted_model_code](./custom_encrypted_model_code) directory into the relevant directory in kserve repository.

```bash
cd custom_encrypted_model_code
ROOT_DIR=$PWD
git clone https://github.com/kserve/kserve.git
cd python
cp $ROOT_DIR/custom_model.Dockerfile .
cp $ROOT_DIR/model.py custom_model/
```
Create the docker image of the predictor and push it to the registry:
 
```bash
docker build . -t ${DOCKER_USER}/per-encrypted-model:v1 -f custom_model.Dockerfile
docker push ${DOCKER_USER}/per-encrypted-model:v1
```

Next, update the [csr-per-encrypted.yaml](./csr-per-encrypted.yaml) ClusterServingRuntime with the new docker image tag and redeploy it.
