import os
import time
import subprocess
import pynvml

# 配置区域
YAML_FILES = [
    "ultralytics/cfg/models/11/yolo11-closs1.yaml",
    # 在此处添加更多 YAML 文件
]

CHECK_INTERVAL = 30  # 每 30 秒检查一次
MEM_THRESHOLD = 500  # 显存阈值 (MB)
NUM_GPUS = 8  # 检查的显卡数量


def get_free_gpus():
    """获取显存占用小于阈值的 GPU 列表"""
    free_gpus = []
    pynvml.nvmlInit()
    try:
        for i in range(NUM_GPUS):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used_mb = info.used / 1024 / 1024
            if used_mb < MEM_THRESHOLD:
                free_gpus.append(i)
    except Exception as e:
        print(f"Error checking GPU memory: {e}")
    finally:
        pynvml.nvmlShutdown()
    return free_gpus


def main():
    queue = YAML_FILES.copy()
    active_processes = {}  # {gpu_index: process}

    print(f"Starting GPU Task Scheduler...")
    print(f"Queue: {queue}")

    while queue or active_processes:
        # 1. 检查已运行的进程是否结束
        finished_gpus = []
        for gpu, proc in active_processes.items():
            if proc.poll() is not None:  # 进程已结束
                print(f"Task on GPU {gpu} finished with exit code {proc.returncode}")
                finished_gpus.append(gpu)

        for gpu in finished_gpus:
            del active_processes[gpu]

        # 2. 如果队列中有任务，检查是否有空闲 GPU
        if queue:
            free_gpus = get_free_gpus()
            # 过滤掉当前正在运行任务的 GPU
            available_gpus = [g for g in free_gpus if g not in active_processes]

            for gpu in available_gpus:
                if not queue:
                    break

                yaml_file = queue.pop(0)
                # 使用 tmux new-window 在当前会话中为每个任务创建一个新窗口
                # 窗口名称包含 GPU ID 和模型文件名，方便查看
                if os.environ.get("TMUX"):
                    window_name = f"GPU{gpu}_{os.path.basename(yaml_file)}"
                    shell_cmd = f"python train.py {yaml_file} --device={gpu} --patience=300; read -p 'Task Finished. Press enter to close...'"
                    cmd = ["tmux", "new-window", "-n", window_name, shell_cmd]
                else:
                    # 如果不在 tmux 中，回退到普通子进程模式
                    cmd = ["python", "train.py", yaml_file, f"--device={gpu}"]

                print(
                    f"Launching task on GPU {gpu} via {'tmux' if os.environ.get('TMUX') else 'subprocess'}: {' '.join(cmd)}"
                )
                # 使用 Popen 非阻塞运行
                proc = subprocess.Popen(cmd)
                active_processes[gpu] = proc

        if queue or active_processes:
            time.sleep(CHECK_INTERVAL)

    print("All tasks completed.")


if __name__ == "__main__":
    main()
