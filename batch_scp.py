import os
import subprocess
import shlex
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob

# Cloudflare R2 Credentials & Bucket Info
R2_ACCOUNT_ID = "7eed45822bca811e3ac90ebf02a1a862"
R2_ACCESS_KEY_ID = "7e0440d339254e40858fb8739892357a"
R2_SECRET_ACCESS_KEY = "30edd4ff9ca11a5aeee9a2eaf548e96e0fc2f447c4113e60ebcd751c499aaef1"
R2_BUCKET = "sn-shards-private"

# Define nodes with explicit GPU counts alongside connection details
NODES = [
    # {"host": "ssh9.vast.ai", "port": 26905, "user": "root", "gpus": 1},
    {"host": "ssh9.vast.ai", "port": 34741, "user": "root", "gpus": 1}
]

def provision_and_run(node, assigned_videos):
    host = node["host"]
    port = node["port"]
    user = node["user"]
    node_id = f"{host}:{port}"

    if not assigned_videos:
        return node_id, "Skipped (0 videos assigned)"

    # Build --include flags for spawn_jpg.py + only this node's assigned .tar.gz shards
    files_to_pull = [os.path.basename(f) for f in assigned_videos] + ["spawn_jpg.py"]
    include_flags = " ".join(f"--include {shlex.quote(f)}" for f in files_to_pull)

    # Inside tmux:
    # 1. Wait for Vast on-start script (/workspace/finished) if still initializing
    # 2. Export stateless R2 credentials
    # 3. Pull assigned .tar.gz shards + spawn_jpg.py from R2 at line speed
    # 4. Activate .venv and run spawn_jpg.py
    worker_pipeline = (
        f"echo 'Waiting for /workspace/finished...' && "
        f"while [ ! -f /workspace/finished ]; do sleep 5; done && "
        f"which rclone >/dev/null 2>&1 || (apt-get update && apt-get install -y rclone) && "
        f"export RCLONE_CONFIG_R2_TYPE=s3 && "
        f"export RCLONE_CONFIG_R2_PROVIDER=Cloudflare && "
        f"export RCLONE_CONFIG_R2_ACCESS_KEY_ID={shlex.quote(R2_ACCESS_KEY_ID)} && "
        f"export RCLONE_CONFIG_R2_SECRET_ACCESS_KEY={shlex.quote(R2_SECRET_ACCESS_KEY)} && "
        f"export RCLONE_CONFIG_R2_ENDPOINT='https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com' && "
        f"export RCLONE_CONFIG_R2_ACL=private && "
        f"export RCLONE_CONFIG_R2_NO_CHECK_BUCKET=true && "
        f"cd /workspace/sn-gamestate && "
        f"echo 'Pulling {len(assigned_videos)} shards from Cloudflare R2...' && "
        f"rclone copy r2:{R2_BUCKET}/ /workspace/sn-gamestate/ {include_flags} "
        f"--transfers 8 --multi-thread-streams 8 --multi-thread-cutoff 64M -v && "
        f"echo 'Starting inference...' && "
        f"source .venv/bin/activate || source /workspace/.venv/bin/activate && "
        f"python spawn_jpg.py"
    )

    # Kill any previous tmux session of the same name and start fresh, logging everything to run.log
    remote_cmd = (
        f"tmux kill-session -t inference_job 2>/dev/null || true; "
        f"tmux new-session -d -s inference_job "
        f"{shlex.quote(f'bash -c \"{worker_pipeline} > /workspace/sn-gamestate/run.log 2>&1\"')}"
    )

    ssh_cmd = [
        "ssh", "-p", str(port),
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=15",
        f"{user}@{host}",
        remote_cmd
    ]

    subprocess.run(ssh_cmd, capture_output=True, text=True, check=True)
    return node_id, f"Initialized ({len(assigned_videos)} shards queued)"


if __name__ == "__main__":
    # 0. Sync the latest local spawn_jpg.py up to R2 once (takes ~1 second)
    print("Uploading latest spawn_jpg.py to R2...")
    subprocess.run(["rclone", "copy", "spawn_jpg.py", f"r2:{R2_BUCKET}/"], check=True)

    # 1. Gather shards and distribute proportionally across GPUs (including remainder)
    targzs = sorted(glob.glob("*.tar.gz"))
    total_gpus = sum(node["gpus"] for node in NODES)

    base_per_gpu = len(targzs) // total_gpus
    remainder = len(targzs) % total_gpus
    current_idx = 0
    gpu_cursor = 0

    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = []
        for node in NODES:
            node_gpus = node["gpus"]
            # Give +1 extra file to the first `remainder` GPUs so no files are left behind
            extra = sum(1 for g in range(gpu_cursor, gpu_cursor + node_gpus) if g < remainder)
            gpu_cursor += node_gpus

            node_targz_amount = (base_per_gpu * node_gpus) + extra
            assigned_targzs = targzs[current_idx : current_idx + node_targz_amount]
            current_idx += node_targz_amount

            futures.append(executor.submit(provision_and_run, node, assigned_targzs))

        for future in as_completed(futures):
            try:
                node_id, status = future.result()
                print(f"Node {node_id} -> {status}")
            except subprocess.CalledProcessError as e:
                print(f"SSH Error: {e.stderr}")