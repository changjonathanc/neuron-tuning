"""E2 job queue, run on each GPU machine: N workers, claim by mkdir, done when the job's "out" file exists, 1 retry.
Re-reads the job file between jobs (new jobs can be appended).

    nohup python experiments/04_activity_selection/q_e2.py experiments/04_activity_selection/jobs/q_gpu2.jsonl --workers 3 \
        > experiments/04_activity_selection/runs/q_gpu2.log 2>&1 &
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "experiments/04_activity_selection/runs"
PY = os.environ.get("PYTHON", sys.executable)   # the Python of the jobs ($PY in the job commands)
ENV = ("export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_XET=1 "
       f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PY={PY}")


def log(*a):
    print(time.strftime("%m-%d %H:%M:%S"), *a, flush=True)


def worker(name, path, lock):
    while True:
        with lock:
            jobs = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
            job = None
            for j in jobs:
                if (REPO / j["out"]).exists():
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
            rc = subprocess.run(f"cd {REPO} && {ENV} && bash -c '{job['cmd']}' >> {logf} 2>&1", shell=True).returncode
            if rc == 0 and (REPO / job["out"]).exists():
                log(name, "done", job["id"])
                break
            log(name, "FAILED", job["id"], "rc", rc)
            time.sleep(20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    (RUNS / "logs").mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    ts = [threading.Thread(target=worker, args=(f"w{k}", a.jobs, lock)) for k in range(a.workers)]
    for t in ts:
        t.start()
        time.sleep(30)
    for t in ts:
        t.join()
    log("queue finished")


if __name__ == "__main__":
    main()
