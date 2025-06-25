export ACCESS_KEY="minio"
export SECRET_KEY="minio123"

export ENDPOINT="http://127.0.0.1:9000"
export BUCKET_NAME=per-input
export FILENAME=reduced_tronchetto_array.pt
MODEL_NAME=per-custom-model HOSTNAME=per-custom-model.per.example.com INGRESS_HOST=localhost INGRESS_PORT=8080 SERVICE_HOSTNAME=per-custom-model.per.example.com python notebook_edge.py
