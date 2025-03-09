ARG PYTHON_VERSION=3.9
ARG BASE_IMAGE=python:${PYTHON_VERSION}-slim-bullseye
ARG VENV_PATH=/home/kserve

FROM ${BASE_IMAGE} as builder

# Install Poetry
ARG POETRY_HOME=/opt/poetry
ARG POETRY_VERSION=1.7.1

RUN python3 -m venv ${POETRY_HOME} && ${POETRY_HOME}/bin/pip install poetry==${POETRY_VERSION}
ENV PATH="$PATH:${POETRY_HOME}/bin"

# Activate virtual env
ARG VENV_PATH
ENV VIRTUAL_ENV=${VENV_PATH}
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY kserve/pyproject.toml kserve/poetry.lock kserve/
RUN cd kserve && poetry install --no-root --no-interaction --no-cache  --extras "storage"
COPY kserve kserve
RUN cd kserve && poetry install --extras "storage" --no-interaction --no-cache

COPY custom_model/pyproject.toml custom_model/poetry.lock custom_model/
RUN cd custom_model && poetry install --no-root --no-interaction --no-cache
COPY custom_model custom_model
RUN cd custom_model && poetry install --no-interaction --no-cache

#RUN pip install torch==2.4.1
#RUN pip install numpy
RUN pip install brevitas
RUN pip install concrete_ml

FROM ${BASE_IMAGE} as prod
RUN apt update && apt install -y binutils
RUN ld --version
RUN which ld

COPY third_party third_party

# Activate virtual env
ARG VENV_PATH
ENV VIRTUAL_ENV=${VENV_PATH}
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN useradd kserve -m -u 1000 -d /home/kserve
COPY --from=builder --chown=kserve:kserve $VIRTUAL_ENV $VIRTUAL_ENV
COPY --from=builder kserve kserve
COPY --from=builder custom_model custom_model
COPY custom_model/model_encrypted.py custom_model/model.py
COPY --chmod=775 reduced_tronchetto_array.pt /reduced_tronchetto_array.pt
#RUN ld --version  # Verify ld is installed
USER 1000
ENTRYPOINT ["python", "-m", "custom_model.model"]
