# Step 1: 选择 Ubuntu 22.04 作为基础镜像
FROM ubuntu:22.04

# Step 2: 安装系统级依赖
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    unzip \
    git \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Step 3: 安装最新版的 Anaconda (完整版)
RUN wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh
ENV PATH="/opt/conda/bin:$PATH"

# Step 4: 拷贝并解压 Isaac Sim 
COPY isaac_sim-2022.2.0.zip /tmp/isaac_sim-2022.2.0.zip
RUN mkdir -p /root/.local/share/ov/pkg/
RUN unzip /tmp/isaac_sim-2022.2.0.zip -d /root/.local/share/ov/pkg/ && rm /tmp/isaac_sim-2022.2.0.zip

# 设置 Isaac Sim 相关的环境变量
ENV ISAACSIM_PATH="/root/.local/share/ov/pkg/isaac_sim-2022.2.0"

# Step 5: 设置工作目录并拷贝你的项目代码
WORKDIR /app
COPY . /app/


# Step 6: 创建并配置 Conda 环境
SHELL ["/bin/bash", "-c"]
RUN conda create -n sim python=3.7 -y
SHELL ["conda", "run", "-n", "sim", "/bin/bash", "-c"]
RUN cp -r conda_setup/etc $CONDA_PREFIX
# Step 7: 在 conda 环境中安装项目和第三方包
RUN pip install -e .
RUN git submodule update --init --recursive
RUN cd third_party/tensordict && git checkout 5e6205c && pip install -e . --no-build-isolation
RUN cd third_party/torchrl && git checkout e39e701 && pip install -e . --no-build-isolation

# Step 8: 设置容器启动时的默认命令
SHELL ["/bin/bash", "-c"]
CMD ["bash"]

