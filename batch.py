import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

NODES = [
    {"host": "192.168.1.101", "port": 2222, "user": "root"},
    {"host": "192.168.1.102", "port": 2222, "user": "root"},
    # ... add all 15 nodes here
]

# Commands to run inside the persistent session
COMMANDS = [
    "git clone https://github.com/your-repo/ball.git",
    "cd ball && pip install -r requirements.txt",
    "python -m ball.mine.src.reconstruct_ball_flight --snippet SNGS-073_snippet_f15_to_703 > run.log 2>&1"
]

def initialize_node(node):
    target = f"{node['user']}@{node['host']}"
    port = node['port']
    
    # Chain commands and wrap them in a detached tmux session
    joined_cmds = " && ".join(COMMANDS)
    remote_cmd = f"tmux new-session -d -s ball_flight_job '{joined_cmds}'"
    
    ssh_cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-p", str(port), target, remote_cmd]
    
    try:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, check=True)
        return node['host'], True, "Launched in detached tmux session."
    except subprocess.CalledProcessError as e:
        return node['host'], False, e.stderr

# Execute across all nodes concurrently
with ThreadPoolExecutor(max_workers=15) as executor:
    futures = {executor.submit(initialize_node, node): node for node in NODES}
    for future in as_completed(futures):
        host, success, output = future.result()
        status = "SUCCESS" if success else "FAILED"
        print(f"[{status}] Node {host}: {output}")