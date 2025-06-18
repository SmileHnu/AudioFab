FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# 设置环境变量，防止apt-get等工具在构建过程中进行交互式提问
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# 1. 安装系统级依赖
# 包括 python, pip, git, ffmpeg, portaudio等
# build-essential 包含了编译C/C++代码所需的工具链
RUN apt-get update && apt-get install -y \
    wget \
    git \
    ffmpeg \
    portaudio19-dev \
    libsm6 \
    libxext6 \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. 安装 Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh
ENV PATH /opt/conda/bin:$PATH

# 设置工作目录
WORKDIR /app

# --- 3. 创建主控 Conda 环境 ---
# 复制主环境的 yml 文件
COPY environment.yml .

# 使用 yml 文件创建名为 AudioFab 的主环境
RUN echo "--- Creating main environment: AudioFab ---" && \
    conda env create -f environment.yml


# --- 4. 依次创建各个工具的 Conda 环境 ---
# 复制所有的 requirements.txt 文件
COPY mcp_servers/yml_requirements/*_requirements.yml ./

RUN for f in *_requirements.yml; do \
    echo "Creating conda env from $f"; \
    conda env create -f "$f"; \
done


# 5. 复制项目代码
COPY . .

# 6. 设置默认的Shell和入口点
SHELL ["conda", "run", "-n", "AudioFab", "/bin/bash", "-c"]

CMD ["python", "scripts/start_all.py"]
