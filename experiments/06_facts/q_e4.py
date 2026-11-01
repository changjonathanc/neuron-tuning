"""E4 job queue: N worker threads in one process (run it under the GPU flock). A job is done when its "out" file
exists; claim by mkdir; 1 retry. Re-reads the job file between jobs (jobs can be appended).

    flock LOCK python q_e4.py jobs/local.jsonl --workers 2 > runs/q_local.log 2>&1
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"


def log(*a):
    print(time.strftime("%m-%d %H:%M:%S"), *a, flush=True)


def worker(name, path, lock, py):
    while True:
        with lock:
            job = None
            for j in [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]:
                if (HERE / j["out"]).exists():
                    continue
                try:
                    (RUNS / "claims" / j["id"]).mkdir(parents=True)
                except FileExistsError:
                    continue
                job = j
                break
        if job is None:
            log(name, "nothing left")
            return
        logf = RUNS / "logs" / f"{job['id']}.log"
        for attempt in range(2):
            log(name, "start", job["id"], "attempt", attempt)
            rc = subprocess.run(f"cd {HERE} && export PY={py} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
                                f"HF_HUB_OFFLINE=1 && bash -c '{job['cmd']}' >> {logf} 2>&1", shell=True).returncode
            if rc == 0 and (HERE / job["out"]).exists():
                log(name, "done", job["id"])
                break
            log(name, "FAILED", job["id"], "rc", rc)
            time.sleep(20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--py", default=os.environ.get("PYTHON", sys.executable))
    a = ap.parse_args()
    (RUNS / "logs").mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    ts = [threading.Thread(target=worker, args=(f"w{k}", a.jobs, lock, a.py)) for k in range(a.workers)]
    for t in ts:
        t.start()
        time.sleep(30)
    for t in ts:
        t.join()
    log("queue finished")


if __name__ == "__main__":
    main()
