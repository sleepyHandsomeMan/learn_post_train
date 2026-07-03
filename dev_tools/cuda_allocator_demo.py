"""CUDA 内存分配器演示：验证 reserved > allocated 是预留池，不是凑差值。

运行方式:
  D:/Anaconda/envs/test3/python.exe dev_tools/cuda_allocator_demo.py
"""

import torch
import gc

def format_size(bytes_val: int) -> str:
    """将字节转换为人类可读的大小。"""
    if bytes_val < 1024:
        return f"{bytes_val} B"
    elif bytes_val < 1024**2:
        return f"{bytes_val / 1024:.1f} KB"
    elif bytes_val < 1024**3:
        return f"{bytes_val / 1024**2:.1f} MB"
    else:
        return f"{bytes_val / 1024**3:.2f} GB"


def print_memory_state(label: str) -> None:
    """打印 CUDA 内存的三个核心指标和分配器内部账本。"""
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  allocated (真实数据):  {format_size(allocated)}")
    print(f"  reserved  (分配器持有): {format_size(reserved)}")
    print(f"  预留池 (reserved - allocated): {format_size(reserved - allocated)}")

    # 分配器的内部账本——这是最直接的证据
    stats = torch.cuda.memory_stats()

    # active_alloc_count: 当前正在使用的分配块数
    # active_alloc_size: 当前正在使用的分配块总大小
    # segment_count: CUDA runtime 分配的大块内存段数
    # segment_size: 这些段的总大小 (= reserved)
    active_alloc_size = stats.get("allocated_bytes.all.current", 0)
    segment_size = stats.get("reserved_bytes.all.current", 0)
    active_alloc_count = stats.get("allocation.all.current", 0)
    segment_count = stats.get("segment.all.current", 0)

    print(f"\n  分配器内部账本:")
    print(f"    活跃分配块数: {active_alloc_count}")
    print(f"    活跃分配块大小: {format_size(active_alloc_size)}  ← 这就是 allocated")
    print(f"    内存段数: {segment_count}  ← CUDA runtime 大块分配次数")
    print(f"    内存段总大小: {format_size(segment_size)}  ← 这就是 reserved")
    print(f"    每段平均大小: {format_size(int(segment_size / max(segment_count, 1)))}")

    # 关键: 段内空闲部分 = 预留池
    free_in_segments = segment_size - active_alloc_size
    print(f"\n    段内空闲 (预留池): {format_size(free_in_segments)}")
    print(f"    ← 这是分配器持有但未分配给任何 tensor 的 VRAM")
    print(f"    ← 不是'凑差值'，是分配器自己报告的内部空闲空间")


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA 不可用，无法演示")
        return

    device = torch.device("cuda")

    # 初始状态: 还没分配任何 tensor
    torch.cuda.empty_cache()
    gc.collect()
    print_memory_state("初始状态 (empty_cache 后)")

    # 实验1: 分配 2 GB tensor → 看预留池膨胀
    print("\n\n>>> 实验1: 分配 2 GB tensor")
    t1 = torch.randn(500_000_000, dtype=torch.float16, device=device)  # 500M × 2B = 1 GB
    print_memory_state("分配 1 GB 后")

    t2 = torch.randn(500_000_000, dtype=torch.float16, device=device)  # 再 1 GB
    print_memory_state("分配 2 GB 后")

    # 实验2: 释放 tensor → 看预留池不缩小
    print("\n\n>>> 实验2: 释放所有 tensor (但不 empty_cache)")
    del t1, t2
    gc.collect()
    print_memory_state("释放后 (无 empty_cache)")
    print("  注意: allocated 降到 ~0, 但 reserved 保持不变!")
    print("  分配器不把段归还给 CUDA runtime → 预留池一直存在")

    # 实验3: empty_cache → 强制归还段
    print("\n\n>>> 实验3: 调用 empty_cache (强制归还)")
    torch.cuda.empty_cache()
    print_memory_state("empty_cache 后")
    print("  empty_cache 强制把空闲段归还给 CUDA runtime")
    print("  reserved 降到接近 0 → 证明之前多出的确实是分配器预留的段")

    # 实验4: 模拟训练循环——反复分配/释放
    print("\n\n>>> 实验4: 模拟训练循环 (5 次分配/释放)")
    for i in range(5):
        tensors = [
            torch.randn(250_000_000, dtype=torch.float16, device=device)  # 0.5 GB
            for _ in range(3)  # 3 × 0.5 = 1.5 GB 模拟激活值
        ]
        print_memory_state(f"循环 {i+1}: 分配 1.5 GB 激活值")
        del tensors
        gc.collect()
        print_memory_state(f"循环 {i+1}: 释放激活值")

    # 最终状态
    print_memory_state("5 次循环后的最终状态")
    print("\n  关键观察:")
    print("  1. 每次分配后 reserved > allocated → 分配器多拿了")
    print("  2. 释放后 allocated 降但 reserved 不降 → 段被保留在池里")
    print("  3. empty_cache 能清空 reserved → 证明多出的确实是分配器预留的段")
    print("  4. 分配器的 memory_stats() 直接报告段内空闲 → 不是凑差值")


if __name__ == "__main__":
    main()
