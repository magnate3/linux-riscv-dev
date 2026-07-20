import subprocess

NEOHOST_PATH = "/opt/neohost/sdk/get_device_performance_counters.py"

def call_device_script():
    # 构造命令和参数列表
    cmd = [
        "python2", NEOHOST_PATH,
        "--mode=shell",
        "--dev-uid=0000:04:00.0",
        "--show-description",
        "--run-loop"
    ]

    try:
        # 调用子进程，并捕获输出
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
        if result.returncode != 0:
            # 如果返回非0，说明有错误
            print("Error occurred:", result.stderr)
            return None
        return result.stdout

    except subprocess.TimeoutExpired:
        print("Process timed out!")
        return None

if __name__ == "__main__":
    output = call_device_script()
    if output:
        print("Script output:")
        print(output)
