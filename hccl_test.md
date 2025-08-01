# Create container with this image 

# swr.cn-southwest-2.myhuaweicloud.com/modelfoundry/dev_images/h00874875/ubuntu:30a87d4

Set mount path to huafengchun and use large memory
<img width="1320" height="1606" alt="image" src="https://github.com/user-attachments/assets/e856c26c-cbf3-4a3f-887e-8bdc75cf4246" />
<img width="1356" height="1542" alt="image" src="https://github.com/user-attachments/assets/74d702ed-37a1-4574-8506-eccd8d5f8ac6" />


#install 

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh

chmod +x Miniconda3-latest-*.sh
./Miniconda3-latest-*.sh


chmod +x Miniconda3-latest-*.sh
./Miniconda3-latest-*.sh


source ~/miniconda3/bin/activate



source ~/miniconda3/bin/activate
conda create -n myenv python=3.11

conda activate myenv
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple attrs cython numpy==1.24.0 decorator sympy cffi pyyaml pathlib2 psutil protobuf==3.20 scipy requests absl-py



bash /shared/cann/Ascend-cann-toolkit_8.2.RC1.alpha003_linux-aarch64.run --full -q
source /home/huafengchun/Ascend/ascend-toolkit/set_env.sh
bash /shared/cann/Ascend-cann-kernels-910b_8.2.RC1.alpha003_linux-aarch64.run --install -q

# install torch
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple torch==2.4.0

# install torch-npu
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple torch-npu==2.4.0.post2
```

test

```
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu


def init_process(rank, world_size):
    """初始化分布式环境（Ascend HCCL 后端）"""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['HCCL_WHITELIST_DISABLE'] = '1'   # 关闭白名单校验，单机调试时常用

    torch_npu.npu.set_device(rank)               # 每个子进程绑定一张 NPU
    dist.init_process_group(
        backend='hccl',
        rank=rank,
        world_size=world_size
    )


def send_recv(rank, world_size):
    init_process(rank, world_size)
    dist.barrier()                               # 等待所有进程就绪

    tensor_size = (2, 2)
    src, dst = 0, 1                              # 0 号卡发送，1 号卡接收

    if rank == src:
        tensor = torch.randn(tensor_size).to(f"npu:{rank}")
        print(f"[Rank {rank}] sending tensor:\n{tensor}")
        dist.send(tensor, dst=dst)

    elif rank == dst:
        tensor = torch.empty(tensor_size).to(f"npu:{rank}")
        dist.recv(tensor, src=src)
        print(f"[Rank {rank}] received tensor:\n{tensor}")

    dist.barrier()                               # 再同步一次，确保双方都结束


def main():
    world_size = 2                               # 只用 2 张卡
    mp.spawn(
        send_recv,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    main()

```

<img width="1912" height="586" alt="image" src="https://github.com/user-attachments/assets/d66cdf2c-e19a-4878-8107-f32533f35bf4" />



# Ray build

install redis
```
sudo apt-get update
sudo apt-get install -y redis-server redis-tools   # redis-cli is in redis-tools
```

Replace the ray/bazel/redis.BUILD
with the following rules

```
########################################
# redis/BUILD.bazel – use system-installed redis
########################################

# Only the Windows binaries are real source files in the archive.
exports_files(
    [
        "redis-server.exe",
        "redis-cli.exe",
    ],
    visibility = ["//visibility:public"],
)

# Linux / macOS: copy the redis binaries already on the host into Bazel’s
# output tree.  Tagged "local" so it never runs on a remote worker that might
# not have Redis installed.
genrule(
    name = "bin",
    srcs = [],
    outs = [
        "redis-server",
        "redis-cli",
    ],
    cmd = select({
        "@platforms//os:osx": """
            set -euo pipefail
            SRV=$$(command -v redis-server)
            CLI=$$(command -v redis-cli)
            [ -x "$$SRV" ] || { echo "redis-server not found on PATH"; exit 1; }
            [ -x "$$CLI" ] || { echo "redis-cli not found on PATH";   exit 1; }
            cp "$$SRV" "$(location redis-server)"
            cp "$$CLI" "$(location redis-cli)"
            chmod +x "$(location redis-server)" "$(location redis-cli)"
        """,
        # Default covers Linux and any other non-macOS Unix-like host.
        "//conditions:default": """
            set -euo pipefail
            SRV=$$(command -v redis-server)
            CLI=$$(command -v redis-cli)
            [ -x "$$SRV" ] || { echo "redis-server not found on PATH"; exit 1; }
            [ -x "$$CLI" ] || { echo "redis-cli not found on PATH";   exit 1; }
            cp "$$SRV" "$(location redis-server)"
            cp "$$CLI" "$(location redis-cli)"
            chmod +x "$(location redis-server)" "$(location redis-cli)"
        """,
    }),
    visibility = ["//visibility:public"],
    tags = ["local"],
)


```


Solution for `OSError: /home/huafengchun/miniconda3/envs/myenv/bin/../lib/libstdc++.so.6: version GLIBCXX_3.4.30' not found (required by /home/huafengchun/ray/python/ray/_raylet.so)`

```
conda install -c conda-forge libstdcxx-ng=12
```
