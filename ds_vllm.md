set up uv on every machine
```
pkill -9 -f python
pkill -9 -f VLLM
bash /shared_workspace_mfs/zhilong/set_uv.sh
source ~/uv_env/bin/activate

export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64/:$LD_LIBRARY_PATH
uv pip uninstall vllm
uv pip install vllm --extra-index-url https://wheels.vllm.ai/nightly


python3 -c "import vllm; print(vllm.__version__)"


uv pip install --force-reinstall --no-cache-dir --no-build-isolation git+https://github.com/deepseek-ai/DeepGEMM.git@v2.1.1.post3
```

On one machine, start ray head
```
export VLLM_USE_DEEP_GEMM=0
export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64/:$LD_LIBRARY_PATH
export MASTER_ADDR=10.10.100.178
export NUM_NODES=2
export GPUS_PER_NODE=8
export NODE_RANK=0
export GLOO_SOCKET_IFNAME=bond0
export NCCL_SOCKET_IFNAME=ib4

ray start --head --port 6479

```

On another machine, start ray
```
export VLLM_USE_DEEP_GEMM=0
export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64/:$LD_LIBRARY_PATH
export MASTER_ADDR=10.10.100.178
export NUM_NODES=2
export GPUS_PER_NODE=8
export NODE_RANK=1
export GLOO_SOCKET_IFNAME=bond0
export NCCL_SOCKET_IFNAME=ib4

ray start --address='10.10.100.178:6479'
```

then start load model

```
bash /shared_workspace_mfs/zhilong/load_ds.sh
```
