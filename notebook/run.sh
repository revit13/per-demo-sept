export ACCESS_KEY="minio"
export SECRET_KEY="minio123"

export ENDPOINT="http://127.0.0.1:9000"
export BUCKET_NAME=per-input
export FILENAME=initial_state.npy
export MODEL_NAME=per-custom-model 
export HOSTNAME=per-custom-model.per.example.com 
export INGRESS_HOST=localhost 
export INGRESS_PORT=8080 
export SERVICE_HOSTNAME=per-custom-model.per.example.com 
python3.11 notebook_edge.py
