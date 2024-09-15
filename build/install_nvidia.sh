#!/usr/bin/env bash
ARG RELEASE
ARG LAUNCHPAD_BUILD_ARCH
LABEL org.opencontainers.image.ref.name=ubuntu
LABEL org.opencontainers.image.version=22.04
#ADD file:63d5ab3ef0aab308c0e71cb67292c5467f60deafa9b0418cbb220affcd078444 in /
CMD ["/bin/bash"]
ENV NVARCH=x86_64
ENV NVIDIA_REQUIRE_CUDA=cuda>=12.1 brand=tesla,driver>=470,driver<471 brand=unknown,driver>=470,driver<471 brand=nvidia,driver>=470,driver<471 brand=nvidiartx,driver>=470,driver<471 brand=geforce,driver>=470,driver<471 brand=geforcertx,driver>=470,driver<471 brand=quadro,driver>=470,driver<471 brand=quadrortx,driver>=470,driver<471 brand=titan,driver>=470,driver<471 brand=titanrtx,driver>=470,driver<471 brand=tesla,driver>=525,driver<526 brand=unknown,driver>=525,driver<526 brand=nvidia,driver>=525,driver<526 brand=nvidiartx,driver>=525,driver<526 brand=geforce,driver>=525,driver<526 brand=geforcertx,driver>=525,driver<526 brand=quadro,driver>=525,driver<526 brand=quadrortx,driver>=525,driver<526 brand=titan,driver>=525,driver<526 brand=titanrtx,driver>=525,driver<526
ENV NV_CUDA_CUDART_VERSION=12.1.105-1
ENV NV_CUDA_COMPAT_PACKAGE=cuda-compat-12-1
ARG TARGETARCH
RUN |1 TARGETARCH=amd64 /bin/sh -c apt-get update && apt-get install -y --no-install-recommends     cuda-cudart-12-1=${NV_CUDA_CUDART_VERSION}     ${NV_CUDA_COMPAT_PACKAGE}     && rm -rf /var/lib/apt/lists/* # buildkit
ENV CUDA_VERSION=12.1.1
RUN |1 TARGETARCH=amd64 /bin/sh -c apt-get update && apt-get install -y --no-install-recommends     cuda-cudart-12-1=${NV_CUDA_CUDART_VERSION}     ${NV_CUDA_COMPAT_PACKAGE}     && rm -rf /var/lib/apt/lists/* # buildkit
RUN |1 TARGETARCH=amd64 /bin/sh -c echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf     && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf # buildkit
RUN |1 TARGETARCH=amd64 /bin/sh -c echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf     && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf # buildkit
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
COPY NGC-DL-CONTAINER-LICENSE / # buildkit
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NV_CUDA_LIB_VERSION=12.1.1-1
ENV NV_NVTX_VERSION=12.1.105-1
ENV NV_LIBNPP_VERSION=12.1.0.40-1
ENV NV_LIBNPP_PACKAGE=libnpp-12-1=12.1.0.40-1
ENV NV_LIBCUSPARSE_VERSION=12.1.0.106-1
ENV NV_LIBCUBLAS_PACKAGE_NAME=libcublas-12-1
ENV NV_LIBCUBLAS_VERSION=12.1.3.1-1
ENV NV_LIBCUBLAS_PACKAGE=libcublas-12-1=12.1.3.1-1
ENV NV_LIBNCCL_PACKAGE_NAME=libnccl2
ENV NV_LIBNCCL_PACKAGE_VERSION=2.17.1-1
ENV NCCL_VERSION=2.17.1-1
ENV NV_LIBNCCL_PACKAGE=libnccl2=2.17.1-1+cuda12.1
ARG TARGETARCH
LABEL maintainer=NVIDIA CORPORATION <cudatools@nvidia.com>
RUN |1 TARGETARCH=amd64 /bin/sh -c apt-get update && apt-get install -y --no-install-recommends     cuda-libraries-12-1=${NV_CUDA_LIB_VERSION}     ${NV_LIBNPP_PACKAGE}     cuda-nvtx-12-1=${NV_NVTX_VERSION}     libcusparse-12-1=${NV_LIBCUSPARSE_VERSION}     ${NV_LIBCUBLAS_PACKAGE}     ${NV_LIBNCCL_PACKAGE}     && rm -rf /var/lib/apt/lists/* # buildkit
RUN |1 TARGETARCH=amd64 /bin/sh -c apt-mark hold ${NV_LIBCUBLAS_PACKAGE_NAME} ${NV_LIBNCCL_PACKAGE_NAME} # buildkit
COPY entrypoint.d/ /opt/nvidia/entrypoint.d/ # buildkit
COPY nvidia_entrypoint.sh /opt/nvidia/ # buildkit
ENV NVIDIA_PRODUCT_NAME=CUDA
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
ENV NV_CUDA_LIB_VERSION=12.1.1-1
ENV NV_CUDA_CUDART_DEV_VERSION=12.1.105-1
ENV NV_NVML_DEV_VERSION=12.1.105-1
ENV NV_LIBCUSPARSE_DEV_VERSION=12.1.0.106-1
ENV NV_LIBNPP_DEV_VERSION=12.1.0.40-1
ENV NV_LIBNPP_DEV_PACKAGE=libnpp-dev-12-1=12.1.0.40-1
ENV NV_LIBCUBLAS_DEV_VERSION=12.1.3.1-1
ENV NV_LIBCUBLAS_DEV_PACKAGE_NAME=libcublas-dev-12-1
ENV NV_LIBCUBLAS_DEV_PACKAGE=libcublas-dev-12-1=12.1.3.1-1
ENV NV_CUDA_NSIGHT_COMPUTE_VERSION=12.1.1-1
ENV NV_CUDA_NSIGHT_COMPUTE_DEV_PACKAGE=cuda-nsight-compute-12-1=12.1.1-1
ENV NV_NVPROF_VERSION=12.1.105-1
ENV NV_NVPROF_DEV_PACKAGE=cuda-nvprof-12-1=12.1.105-1
ENV NV_LIBNCCL_DEV_PACKAGE_NAME=libnccl-dev
ENV NV_LIBNCCL_DEV_PACKAGE_VERSION=2.17.1-1
ENV NCCL_VERSION=2.17.1-1
ENV NV_LIBNCCL_DEV_PACKAGE=libnccl-dev=2.17.1-1+cuda12.1
ARG TARGETARCH
LABEL maintainer=NVIDIA CORPORATION <cudatools@nvidia.com>
RUN |1 TARGETARCH=amd64 /bin/sh -c apt-get update && apt-get install -y --no-install-recommends     cuda-cudart-dev-12-1=${NV_CUDA_CUDART_DEV_VERSION}     cuda-command-line-tools-12-1=${NV_CUDA_LIB_VERSION}     cuda-minimal-build-12-1=${NV_CUDA_LIB_VERSION}     cuda-libraries-dev-12-1=${NV_CUDA_LIB_VERSION}     cuda-nvml-dev-12-1=${NV_NVML_DEV_VERSION}     ${NV_NVPROF_DEV_PACKAGE}     ${NV_LIBNPP_DEV_PACKAGE}     libcusparse-dev-12-1=${NV_LIBCUSPARSE_DEV_VERSION}     ${NV_LIBCUBLAS_DEV_PACKAGE}     ${NV_LIBNCCL_DEV_PACKAGE}     ${NV_CUDA_NSIGHT_COMPUTE_DEV_PACKAGE}     && rm -rf /var/lib/apt/lists/* # buildkit
RUN |1 TARGETARCH=amd64 /bin/sh -c apt-mark hold ${NV_LIBCUBLAS_DEV_PACKAGE_NAME} ${NV_LIBNCCL_DEV_PACKAGE_NAME} # buildkit
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs
ENV NV_CUDNN_VERSION=8.9.0.131
ENV NV_CUDNN_PACKAGE_NAME=libcudnn8
ENV NV_CUDNN_PACKAGE=libcudnn8=8.9.0.131-1+cuda12.1
ENV NV_CUDNN_PACKAGE_DEV=libcudnn8-dev=8.9.0.131-1+cuda12.1
ARG TARGETARCH
LABEL maintainer=NVIDIA CORPORATION <cudatools@nvidia.com>
LABEL com.nvidia.cudnn.version=8.9.0.131
RUN |1 TARGETARCH=amd64 /bin/sh -c apt-get update && apt-get install -y --no-install-recommends     ${NV_CUDNN_PACKAGE}     ${NV_CUDNN_PACKAGE_DEV}     && apt-mark hold ${NV_CUDNN_PACKAGE_NAME}     && rm -rf /var/lib/apt/lists/* # buildkit
SHELL [/bin/bash -o pipefail -c]
ENV DEBIAN_FRONTEND=noninteractive TZ=Europe/London PYTHONUNBUFFERED=1 SHELL=/bin/bash
ARG PYTHON_VERSION=3.10
COPY --chmod=755 ../../build/packages.sh /packages.sh # buildkit
RUN |1 PYTHON_VERSION=3.10 /bin/bash -o pipefail -c /packages.sh && rm /packages.sh # buildkit
ARG INDEX_URL=https://download.pytorch.org/whl/cu121
ARG TORCH_VERSION=2.1.2+cu121
ARG XFORMERS_VERSION=0.0.23.post1
RUN |4 PYTHON_VERSION=3.10 INDEX_URL=https://download.pytorch.org/whl/cu121 TORCH_VERSION=2.1.2+cu121 XFORMERS_VERSION=0.0.23.post1 /bin/bash -o pipefail -c pip3 install --no-cache-dir torch==${TORCH_VERSION} torchvision torchaudio --index-url ${INDEX_URL} &&     pip3 install --no-cache-dir xformers==${XFORMERS_VERSION} --index-url ${INDEX_URL} # buildkit
ARG RUNPODCTL_VERSION=v1.14.3
ENV RUNPODCTL_VERSION=v1.14.3
COPY ../../code-server/vsix/*.vsix /tmp/ # buildkit
COPY ../../code-server/settings.json /root/.local/share/code-server/User/settings.json # buildkit
COPY --chmod=755 ../../build/apps.sh /apps.sh # buildkit
RUN |5 PYTHON_VERSION=3.10 INDEX_URL=https://download.pytorch.org/whl/cu121 TORCH_VERSION=2.1.2+cu121 XFORMERS_VERSION=0.0.23.post1 RUNPODCTL_VERSION=v1.14.3 /bin/bash -o pipefail -c /apps.sh && rm /apps.sh # buildkit
RUN |5 PYTHON_VERSION=3.10 INDEX_URL=https://download.pytorch.org/whl/cu121 TORCH_VERSION=2.1.2+cu121 XFORMERS_VERSION=0.0.23.post1 RUNPODCTL_VERSION=v1.14.3 /bin/bash -o pipefail -c rm -f /etc/ssh/ssh_host_* # buildkit
COPY ../../nginx/502.html /usr/share/nginx/html/502.html # buildkit
WORKDIR /
COPY --chmod=755 ../../scripts/* ./ # buildkit
RUN |5 PYTHON_VERSION=3.10 INDEX_URL=https://download.pytorch.org/whl/cu121 TORCH_VERSION=2.1.2+cu121 XFORMERS_VERSION=0.0.23.post1 RUNPODCTL_VERSION=v1.14.3 /bin/bash -o pipefail -c mv /manage_venv.sh /usr/local/bin/manage_venv # buildkit
ARG REQUIRED_CUDA_VERSION=12.1
ENV REQUIRED_CUDA_VERSION=12.1
SHELL [/bin/bash --login -c]
CMD ["/start.sh"]
SHELL [/bin/bash -o pipefail -c]
ENV DEBIAN_FRONTEND=noninteractive TZ=Europe/London PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=on SHELL=/bin/bash
RUN /bin/bash -o pipefail -c mkdir -p /sd-models # buildkit
COPY sd_xl_base_1.0.safetensors /sd-models/sd_xl_base_1.0.safetensors # buildkit
WORKDIR /
ARG KOHYA_VERSION=v24.1.6
RUN |1 KOHYA_VERSION=v24.1.6 /bin/bash -o pipefail -c git clone https://github.com/bmaltais/kohya_ss.git /kohya_ss &&     cd /kohya_ss &&     git checkout ${KOHYA_VERSION} &&     git submodule update --init --recursive # buildkit
ARG INDEX_URL=https://download.pytorch.org/whl/cu121
ARG TORCH_VERSION=2.1.2+cu121
ARG XFORMERS_VERSION=0.0.23.post1
WORKDIR /kohya_ss
COPY kohya_ss/requirements* ./ # buildkit
RUN |4 KOHYA_VERSION=v24.1.6 INDEX_URL=https://download.pytorch.org/whl/cu121 TORCH_VERSION=2.1.2+cu121 XFORMERS_VERSION=0.0.23.post1 /bin/bash -o pipefail -c python3 -m venv --system-site-packages /venv &&     source /venv/bin/activate &&     pip3 install torch==${TORCH_VERSION} torchvision torchaudio --index-url ${INDEX_URL} &&     pip3 install xformers==${XFORMERS_VERSION} --index-url ${INDEX_URL} &&     pip3 install bitsandbytes==0.43.0         tensorboard==2.15.2 tensorflow==2.15.0.post1         wheel packaging tensorrt &&     pip3 install tensorflow[and-cuda] &&     pip3 install -r requirements.txt &&     deactivate # buildkit
WORKDIR /
RUN |4 KOHYA_VERSION=v24.1.6 INDEX_URL=https://download.pytorch.org/whl/cu121 TORCH_VERSION=2.1.2+cu121 XFORMERS_VERSION=0.0.23.post1 /bin/bash -o pipefail -c pip3 uninstall -y tensorboard tb-nightly &&     pip3 install tensorboard==2.15.2 tensorflow==2.15.0.post1 # buildkit
ARG APP_MANAGER_VERSION=1.2.1
WORKDIR /
RUN |5 KOHYA_VERSION=v24.1.6 INDEX_URL=https://download.pytorch.org/whl/cu121 TORCH_VERSION=2.1.2+cu121 XFORMERS_VERSION=0.0.23.post1 APP_MANAGER_VERSION=1.2.1 /bin/bash -o pipefail -c git clone https://github.com/ashleykleynhans/app-manager.git /app-manager &&     cd /app-manager &&     git checkout tags/${APP_MANAGER_VERSION} &&     npm install # buildkit
COPY app-manager/config.json /app-manager/public/config.json # buildkit
COPY --chmod=755 app-manager/*.sh /app-manager/scripts/ # buildkit
RUN |5 KOHYA_VERSION=v24.1.6 INDEX_URL=https://download.pytorch.org/whl/cu121 TORCH_VERSION=2.1.2+cu121 XFORMERS_VERSION=0.0.23.post1 APP_MANAGER_VERSION=1.2.1 /bin/bash -o pipefail -c rm -f /etc/ssh/ssh_host_* # buildkit
COPY nginx/nginx.conf /etc/nginx/nginx.conf # buildkit
ARG RELEASE=24.1.6
ENV TEMPLATE_VERSION=24.1.6
COPY --chmod=755 scripts/* ./ # buildkit
COPY kohya_ss/accelerate.yaml ./ # buildkit
SHELL [/bin/bash --login -c]
CMD ["/start.sh"]