"""Phase 2b job queue (a copy of the job queue of an earlier study, not in this repository; only the directory and the process names differ), one per GPU machine (each machine ran its own queue). Jobs are JSON lines {"id", "cmd", "needs": [paths]}
(cmd and needs paths are relative to experiments/02_conditional_rules; $PY is set to $PYTHON, default: this Python). A job is claimed by mkdir, retried up to 2 times, marked done by a file.
A job whose "needs" files do not exist yet waits. --workers N runs N jobs in parallel on the one GPU.

    nohup python experiments/02_conditional_rules/jobq2b.py experiments/02_conditional_rules/jobs/x.jsonl --workers 3 > experiments/02_conditional_rules/runs/q_x.log 2>&1 &
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "experiments/02_conditional_rules/runs"
PY = os.environ.get("PYTHON", sys.executable)   # the Python of the jobs ($PY in the job commands)
ENV = f"export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PY={PY}"


def log(*a):
    print(time.strftime("%m-%d %H:%M:%S"), *a, flush=True)


def is_done(j):
    return (RUNS / "done" / j["id"]).exists() or ("out" in j and (REPO / "experiments/02_conditional_rules" / j["out"]).exists())


def n_running():
    """model processes of Phase 2b running on this machine (any queue)."""
    r = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True)
    return sum(1 for line in r.stdout.splitlines() if re.match(re.escape(PY) + r" (train2b|gen2b|eval2b)", line))


def launch(name, job, attempt):
    """start the job (called with the lock held; waits until the model process shows in ps)."""
    log(name, "start", job["id"], "attempt", attempt)
    logf = RUNS / "logs" / f"{job['id']}.log"
    proc = subprocess.Popen(f"cd {REPO}/experiments/02_conditional_rules && {ENV} && bash -c '{job['cmd']}' >> {logf} 2>&1", shell=True)
    time.sleep(5)
    return proc


def worker(name, path, lock, forever, max_procs):
    while True:
        with lock:
            jobs = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
            todo = [j for j in jobs if not is_done(j)]
            if not todo and not forever:
                log(name, "no jobs left")
                return
            if n_running() >= max_procs:
                todo = []
                waiting = True
            else:
                waiting = False
            job = None
            for j in todo:
                if (RUNS / "claims" / j["id"]).exists():
                    continue
                if not all((REPO / "experiments/02_conditional_rules" / p).exists() for p in j.get("needs", [])):
                    waiting = True
                    continue
                (RUNS / "claims" / j["id"]).mkdir(parents=True)
                job = j
                break
            if job is not None:
                proc = launch(name, job, 0)
        if job is None:
            if waiting or forever:
                time.sleep(60)
                continue
            log(name, "nothing claimable; exit")
            return
        for attempt in range(3):
            if attempt:
                with lock:
                    proc = launch(name, job, attempt)
            rc = proc.wait()
            if rc == 0:
                (RUNS / "done" / job["id"]).write_text(name)
                log(name, "done", job["id"])
                break
            log(name, "FAILED", job["id"], "rc", rc)
            time.sleep(30)
        else:
            log(name, "GAVE UP", job["id"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--release", action="store_true", help="free claims of unfinished jobs of this file first")
    ap.add_argument("--forever", action="store_true", help="keep polling the (growing) job file")
    ap.add_argument("--max-procs", type=int, default=3, help="max model processes on this machine")
    a = ap.parse_args()
    jobs = [json.loads(l) for l in Path(a.jobs).read_text().splitlines() if l.strip()]
    for d in ("claims", "done", "logs"):
        (RUNS / d).mkdir(parents=True, exist_ok=True)
    if a.release:
        for j in jobs:
            c = RUNS / "claims" / j["id"]
            if c.exists() and not (RUNS / "done" / j["id"]).exists():
                c.rmdir()
    lock = threading.Lock()
    ts = [threading.Thread(target=worker, args=(f"w{i}", a.jobs, lock, a.forever, a.max_procs)) for i in range(a.workers)]
    for t in ts:
        t.start()
        time.sleep(20)   # stagger model loads
    for t in ts:
        t.join()
    log("queue finished")


if __name__ == "__main__":
    main()
