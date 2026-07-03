"""在训练进程中查看 CUDA 内存分配器的实时状态。

通过 nvidia-smi 获取进程 GPU 显存占用，
再通过 PyTorch 的 memory_stats() 获取分配器内部账本。
"""

import subprocess
import re

# nvidia-smi 查看进程 GPU 占用
result = subprocess.run(
    ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
    capture_output=True, text=True
)
print("nvidia-smi 进程 GPU 占用:")
print(result.stdout)

# nvidia-smi 总体状态
result2 = subprocess.run(
    ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total", "--format=csv,noheader,nounits"],
    capture_output=True, text=True
)
print("nvidia-smi GPU 总体:")
print(result2.stdout)
