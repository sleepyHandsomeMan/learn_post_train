"""attach 到训练进程的 CUDA context，读取分配器内部账本。

注意: PyTorch 的 CUDA memory_stats() 只能在同一进程内调用。
不能从外部进程读取另一个进程的分配器状态。

所以这里通过 nvidia-smi 获取外部可见信息，
再结合进程的内存使用来推断。
"""

import subprocess
import re
import os

def run_cmd(cmd: str) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    return result.stdout + result.stderr

# 1. nvidia-smi GPU 总体
print("=" * 60)
print("  nvidia-smi GPU 总体状态")
print("=" * 60)
out = run_cmd("nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader")
print(out)

# 2. 训练进程 PID 15816 的系统内存
print("=" * 60)
print("  训练进程 (PID 15816) 系统内存")
print("=" * 60)
out = run_cmd('wmic process where "ProcessId=15816" get WorkingSetSize,PrivateMemorySize /format:list')
print(out)

# 3. nvidia-smi 的 Dedicated + Shared GPU Memory
# WDDM 模式下无法从 nvidia-smi 直接获取 Shared GPU Memory
# 需要用 PowerShell 的 Get-Process
print("=" * 60)
print("  PowerShell Get-Process GPU Memory (Dedicated + Shared)")
print("=" * 60)
out = run_cmd(
    'powershell -Command "Get-Process -Id 15816 | Select-Object Name,Id,'
    'WorkingSet64,PrivateMemorySize64 | Format-List"'
)
print(out)

# 4. GPU Dedicated + Shared (从任务管理器的数据源)
print("=" * 60)
print("  GPU 进程 Dedicated/Shared (Performance Counter)")
print("=" * 60)
# WDDM 模式下通过 GPU performance counter 获取
out = run_cmd(
    'powershell -Command "'
    'Get-Counter \'\\GPU Process Memory(pid 15816*)\\Dedicated Usage\' '
    '-ErrorAction SilentlyContinue | Select-Object -ExpandProperty CounterSamples | Format-List"'
)
print("Dedicated:", out)

out = run_cmd(
    'powershell -Command "'
    'Get-Counter \'\\GPU Process Memory(pid 15816*)\\Shared Usage\' '
    '-ErrorAction SilentlyContinue | Select-Object -ExpandProperty CounterSamples | Format-List"'
)
print("Shared:", out)

# 5. GPU 总体 Dedicated + Shared
print("=" * 60)
print("  GPU 总体 Dedicated/Shared")
print("=" * 60)
out = run_cmd(
    'powershell -Command "'
    'Get-Counter \'\\GPU Adapter Memory(*)\\Dedicated Usage\' '
    '-ErrorAction SilentlyContinue | Select-Object -ExpandProperty CounterSamples | Format-List"'
)
print("Dedicated:", out)

out = run_cmd(
    'powershell -Command "'
    'Get-Counter \'\\GPU Adapter Memory(*)\\Shared Usage\' '
    '-ErrorAction SilentlyContinue | Select-Object -ExpandProperty CounterSamples | Format-List"'
)
print("Shared:", out)

print("\n注意: PyTorch 的 torch.cuda.memory_stats() 只能在训练进程内部调用。")
print("要从训练进程内获取分配器账本，需要在训练代码中插入打印。")
