variable "REGISTRY" {
    default = "ghcr.io"
}

variable "REGISTRY_USER" {
    default = "themountain427"
}

variable "APP" {
    default = "stable-diffusion-webui"
}

variable "RELEASE" {
    default = "6.9.0"
}

variable "CU_VERSION" {
    default = "121"
}

variable "BASE_IMAGE_REPOSITORY" {
    default = "docker.io/ubuntu"
}

variable "BASE_IMAGE_VERSION" {
    default = "22.04"
}

variable "CUDA_VERSION" {
    default = "12.1.1"
}

variable "TORCH_VERSION" {
    default = "2.3.1"
}

variable "PYTHON_VERSION" {
    default = "3.10"
}

target "default" {
    dockerfile = "Dockerfile"
    tags = ["${REGISTRY}/${REGISTRY_USER}/${APP}:${RELEASE}"]
    args = {
        RELEASE = "${RELEASE}"
        BASE_IMAGE = "${BASE_IMAGE_REPOSITORY}:${BASE_IMAGE_VERSION}-python${PYTHON_VERSION}-cuda${CUDA_VERSION}-torch${TORCH_VERSION}"
        INDEX_URL = "https://download.pytorch.org/whl/cu${CU_VERSION}"
        TORCH_VERSION = "${TORCH_VERSION}+cu${CU_VERSION}"
        XFORMERS_VERSION = "0.0.27"
        WEBUI_VERSION = "v1.9.4"
        CONTROLNET_COMMIT = "b63899a654ee2f70d475c259691f35ac67c320d4"
        DREAMBOOTH_COMMIT = "45a12fe5950bf93205b6ef2b7511eb94052a241f"
        CIVITAI_BROWSER_PLUS_VERSION = "v3.5.4"
        KOHYA_VERSION = "v24.1.5"
        KOHYA_TORCH_VERSION = "2.1.2+cu${CU_VERSION}"
        KOHYA_XFORMERS_VERSION = "0.0.23.post1"
        COMFYUI_COMMIT = "33346fd9b8856942187ad8c818498a080a2027b5"
        INVOKEAI_VERSION = "4.2.6post1"
        APP_MANAGER_VERSION = "1.2.1"
        CIVITAI_DOWNLOADER_VERSION = "2.1.0"
        VENV_PATH = "/workspace/venvs/${APP}"
    }
}
