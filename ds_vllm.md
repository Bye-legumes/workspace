start docker
```
sudo usermod -aG docker $USER
newgrp docker
docker create \
  --gpus all \
  --net=host \
  --shm-size=200g \
  --cap-add=SYS_ADMIN \
  -v /shared_workspace_mfs:/shared_workspace_mfs \
  -v /home/original_models:/home/original_models \
  --name verl \
  --entrypoint bash \
  verlai/verl:base-verl0.6-cu128-cudnn9.8-torch2.8.0-fa2.7.4 \
  -c "while true; do sleep 3600; done"

docker start verl

docker exec -it verl bash

```
change `vim ~/.config/pip/pip.conf`

```

[global]
index-url = https://pypi.org/simple
extra-index-url = https://wheels.vllm.ai/nightly
no-cache-dir = true
```


set up uv on every machine
```
wget -qO- https://astral.sh/uv/install.sh | sh
bash /shared_workspace_mfs/zhilong/set_uv.sh
source ~/uv_env/bin/activate

export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/shared_workspace_mfs/zhilong/DeepGEMM:$PYTHONPATH


uv pip install vllm --extra-index-url https://wheels.vllm.ai/nightly
```


try ```export VLLM_USE_DEEP_GEMM=1
``` first, if it failed, try `export VLLM_USE_DEEP_GEMM=0`

On one machine, start ray head
```
export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/shared_workspace_mfs/zhilong/DeepGEMM:$PYTHONPATH
export MASTER_ADDR=10.10.100.90
export NUM_NODES=2
export GPUS_PER_NODE=8
export NODE_RANK=0
export GLOO_SOCKET_IFNAME=bond0
export NCCL_SOCKET_IFNAME=ib4

ray start --head --port 6479

```

On another machine, start ray
```
export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/shared_workspace_mfs/zhilong/DeepGEMM:$PYTHONPATH
export MASTER_ADDR=10.10.100.90
export NUM_NODES=2
export GPUS_PER_NODE=8
export NODE_RANK=1
export GLOO_SOCKET_IFNAME=bond0
export NCCL_SOCKET_IFNAME=ib4

ray start --address='10.10.100.90:6479'
```

then start load model

```
bash /shared_workspace_mfs/zhilong/load_ds.sh
```
