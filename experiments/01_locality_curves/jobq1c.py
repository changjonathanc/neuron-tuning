"""GPU job queue of this experiment (a copy of the job queue of an earlier study, not in this repository; only the runs directory differs). Jobs are JSON lines {"id", "cmd" or "cmds", "where"}
("any" | a worker name); cmd runs from the repo root with $PY set. "cmds": a list of commands run IN PARALLEL under
one GPU lock (used for generation: two batch-1 generators share a 3090 and give about 2x the throughput). A job is claimed by mkdir, retried up to 3 times,
and marked done by a file. Each job runs under flock $GPU_LOCK (default: runs/job.lock of the study) with
$PY = $PYTHON (default: this Python). The experiments ran jobs on several GPU machines; the remote dispatch code
to other machines was removed from the public version, and all workers now run on this machine.

    nohup python experiments/01_locality_curves/jobq1c.py experiments/01_locality_curves/jobs_main.jsonl > experiments/01_locality_curves/runs/queue_main.log 2>&1 &
"""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "experiments/01_locality_curves/runs"
ENV = "export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
PY = os.environ.get("PYTHON", sys.executable)
LOCK = os.environ.get("GPU_LOCK", str(RUNS / "job.lock"))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sh(cmd):
    return subprocess.run(cmd, shell=True).returncode


def body(job):
    if "cmds" in job:
        return " ".join(f"( {c} ) & p{k}=$!;" for k, c in enumerate(job["cmds"])) + " " + \
            " && ".join(f"wait $p{k}" for k in range(len(job["cmds"])))
    return job["cmd"]


def run_local(job, logf):
    c = f"cd {REPO} && {ENV} && export PY={PY} && flock {LOCK} bash -c '{body(job)}' >> {logf} 2>&1"
    return sh(c)


def worker(name, jobs):
    while True:
        todo = [j for j in jobs if not (RUNS / "done" / j["id"]).exists()]
        if not todo:
            log(name, "no jobs left")
            return
        job, waiting = None, False
        for j in todo:
            if j.get("where", "any") not in ("any", name):
                continue
            if not all((RUNS / "done" / d).exists() for d in j.get("after", [])):
                waiting = waiting or not (RUNS / "claims" / j["id"]).exists()
                continue
            try:
                (RUNS / "claims" / j["id"]).mkdir(parents=True)
                job = j
                break
            except FileExistsError:
                continue
        if job is None:
            if waiting:                      # unclaimed jobs wait for a dependency that another worker runs
                time.sleep(60)
                continue
            log(name, "nothing claimable; exit")
            return
        logf = RUNS / "logs" / f"{job['id']}.log"
        logf.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(3):
            log(name, "start", job["id"], "attempt", attempt)
            rc = run_local(job, logf)
            if rc == 0:
                (RUNS / "done").mkdir(exist_ok=True)
                (RUNS / "done" / job["id"]).write_text(name)
                log(name, "done", job["id"])
                break
            log(name, "FAILED", job["id"], "rc", rc)
            time.sleep(30)
        else:
            log(name, "GAVE UP", job["id"])


def main():
    jobs = [json.loads(l) for l in Path(sys.argv[1]).read_text().splitlines() if l.strip()]
    names = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] != "--release" else ["local"]
    for d in ("claims", "done", "logs"):
        (RUNS / d).mkdir(parents=True, exist_ok=True)
    # --release: free stale claims of THIS file's unfinished jobs (after a crashed queue; never while it still runs)
    if "--release" in sys.argv:
        ids = {j["id"] for j in jobs}
        for c in (RUNS / "claims").iterdir():
            if c.name in ids and not (RUNS / "done" / c.name).exists():
                c.rmdir()
    ts = [threading.Thread(target=worker, args=(n, jobs)) for n in names]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    log("queue finished")


if __name__ == "__main__":
    main()
