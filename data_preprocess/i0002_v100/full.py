"""Full-dataset scheduler for the pilot-approved I0002 kernels.

One process pool computes records; one writer packs and verifies a shard while
the pool computes the next shard. Only manifest commits expose finished files.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED
from collections import Counter
import json
import math
import os
from pathlib import Path
import shutil
import time
import traceback
import uuid

import h5py
import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

from . import pipeline as p
from .core import APPLICABLE, CHANNELS, GROUPS, HARD_CODES, hard_bit

GIB = 2**30


def replace_with_retry(source, destination, attempts=8):
    """Windows readers may briefly deny rename/delete sharing."""
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except OSError as error:
            if not (isinstance(error, PermissionError) or getattr(error, "winerror", None) in (5, 32, 33)) or attempt == attempts-1:
                raise
            time.sleep(min(.05 * 2**attempt, .5))


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        p.write_json(tmp, value)
        replace_with_retry(tmp, path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def write_progress(path, state, log):
    """Telemetry is best effort; durable data/commit writes remain fail-closed."""
    try:
        atomic_json(path, state)
        return True
    except OSError as error:
        log("PROGRESS_WRITE_DEFERRED", error=repr(error))
        return False


def atomic_parquet(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows), tmp)
    replace_with_retry(tmp, path)


def plan_records(table_path, max_records=16, target_gib=6):
    rows = pq.read_table(table_path).to_pylist()
    seen, excluded, planned = set(), [], []
    for row in rows:
        key = row["session_key"]
        p.require(key not in seen, f"duplicate session in inventory: {key}")
        seen.add(key)
        duration = row.get("h5_duration_seconds")
        if key in p.EXCLUSIONS:
            excluded.append({"session_key": key, "status": "FIXED_EXCLUDED", "reason": "SOURCE_DURATION_CONFLICT"})
            continue
        if duration is None or not math.isfinite(duration) or not 30 <= duration <= 72*3600:
            excluded.append({"session_key": key, "status": "RECORD_EXCLUDED", "reason": "INVALID_ROOT_DURATION"})
            continue
        n = math.floor(duration/30)
        planned.append({"session_key": key, "epoch_count": n, "duration_sec": float(duration),
                        "estimated_bytes": n*15*6000*4, "source_h5_bytes": row.get("h5_bytes", 0)})
    return batch_records(planned, max_records, target_gib), excluded


def batch_records(planned, max_records=16, target_gib=6):
    # Stable source-table ordering; a recording is never split across shards.
    shards, batch, size = [], [], 0
    for row in planned:
        p.require(row["estimated_bytes"] <= 8*GIB, "single record exceeds shard hard cap")
        if batch and (len(batch) >= max_records or size + row["estimated_bytes"] > target_gib*GIB):
            shards.append(batch)
            batch, size = [], 0
        batch.append(row)
        size += row["estimated_bytes"]
    if batch:
        shards.append(batch)
    return shards


def recovery_plan(original_shards, commits, max_records=16, target_gib=6):
    """Keep committed record assignments immutable; requeue every unpublished key."""
    p.require(sorted(commits) == list(range(len(commits))), "recovery requires contiguous commits")
    candidates = {r["session_key"]: r for batch in original_shards for r in batch}
    published = [r["session_key"] for c in commits.values() for r in c["records"]]
    p.require(len(set(published)) == len(published), "duplicate published recording")
    p.require(set(published) <= candidates.keys(), "committed recording outside inventory")
    retained = [[candidates[r["session_key"]] for r in commits[n]["records"]] for n in range(len(commits))]
    published = set(published)
    pending = [r for r in candidates.values() if r["session_key"] not in published]
    return retained + batch_records(pending, max_records, target_gib)


def session_job(input_root, session, work_root, threads):
    work = Path(work_root)
    work.mkdir(parents=True, exist_ok=True)
    temp, record = p.process_session(input_root, session, str(work), threads)
    atomic_json(work/(record["recording_id"]+".json"), {"temp":temp, "record":record})
    return temp, record


def load_commits(output, verify=False, progress=None):
    commits = {}
    for path in sorted((output/"manifests/commits").glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        for artifact in value["artifacts"]:
            ap = output/artifact["artifact"]
            p.require(ap.exists() and ap.stat().st_size == artifact["bytes"], f"resume artifact missing/size mismatch: {ap}")
            if verify:
                p.require(p.sha256_file(ap) == artifact["sha256"], f"resume checksum: {ap}")
        commits[value["shard_number"]] = value
        if progress is not None:
            progress("RESUME_COMMIT_VERIFIED",shard=value["shard_number"],
                     artifacts=len(value["artifacts"]),sha256_checked=verify)
    return commits


def shard_report(stage, records, validation, number):
    """Small per-shard tables plus full hard-event intervals in Parquet."""
    lines = [f"# I0002 v1.0.0 shard {number:05d} QC", "",
             f"- Automatic QC: {validation['status']}; records={len(records)}; epochs={validation['total_epochs']}",
             "- Approved pilot algorithms; 200 Hz, float32 physical values, no normalization.",
             "", "## Hard codes", "", "| channel | code | name | evaluated | hard epochs |",
             "|---|---:|---|---:|---:|"]
    summary, events = [], []
    with h5py.File(stage/"signals/qc/qc-00000.h5") as qf, h5py.File(stage/"signals/shards/signal-00000.h5") as sf:
        h, e = qf["hard_flags"][:], qf["hard_evaluated_flags"][:]
        ri, ei = sf["segments/record_index"][:], sf["segments/epoch_in_record"][:]
        for c, name in enumerate(CHANNELS):
            for code in range(1,8):
                hits=(h[:,c]&hard_bit(code))!=0
                checked=(e[:,c]&hard_bit(code))!=0
                item={"shard_id":f"{number:05d}","channel":name,"code":code,"name":HARD_CODES[code],
                      "applicable":bool(APPLICABLE[c]&hard_bit(code)),
                      "evaluated_epochs":int(checked.sum()),"hard_epochs":int(hits.sum())}
                summary.append(item)
                if item["applicable"]:
                    lines.append(f"| {name} | {code} | {HARD_CODES[code]} | {item['evaluated_epochs']} | {item['hard_epochs']} |")
                for k in np.flatnonzero(hits):
                    events.append({"shard_id":f"{number:05d}","session_key":records[ri[k]]["session_key"],
                                   "channel":name,"epoch_in_record":int(ei[k]),
                                   "start_seconds":int(ei[k]*30),"end_seconds":int((ei[k]+1)*30),
                                   "hard_code":code,"name":HARD_CODES[code]})
    lines += ["", "## Channel processing", "",
              "| session | channel | source Hz | effective HP/LP | availability code | usable h |",
              "|---|---|---:|---|---:|---:|"]
    for r in records:
        for c in r["channel_metadata"]:
            lines.append(f"| {r['session_key']} | {c['canonical_channel']} | {c.get('source_fs_hz')} | {c.get('applied_hp_lp') or c.get('parsed_source_hp_lp')} | {c['channel_status_code']} | {c['usable_hours']:.4f} |")
    lines += p.task_report_lines(stage, records)
    (stage/"qc.md").write_text("\n".join(lines),encoding="utf-8")
    if events:
        pq.write_table(pa.Table.from_pylist(events), stage/"epoch_hard.parquet",compression="zstd")
    return summary


def pack_shard(output_text, work_text, number, results, failures):
    output, work = Path(output_text), Path(work_text)
    started = time.perf_counter()
    shard_id=f"{number:05d}"
    stage=work/f"pack-{shard_id}"
    stage.mkdir(parents=True,exist_ok=True)
    for rel in ("signals/shards","signals/qc"):
        (stage/rel).mkdir(parents=True,exist_ok=True)
    temps=[Path(x[0]) for x in results]
    records=[x[1] for x in results]
    if not records:
        commit={"shard_number":number,"shard_id":shard_id,"status":"NO_PUBLISHABLE_RECORDS",
                "records":[],"artifacts":[],"summary":[],"failed_sessions":failures,
                "pack_seconds":time.perf_counter()-started}
        atomic_json(output/f"manifests/commits/shard-{shard_id}.json",commit)
        return commit
    bind=p.binding(output,records)
    bind.update(shard_id=shard_id,release_status="QC_PASSED")
    sp,qp=stage/"signals/shards/signal-00000.h5",stage/"signals/qc/qc-00000.h5"
    p.write_signal_shard(sp,temps,records,bind)
    sh=p.sha256_file(sp)
    p.write_qc_shard(qp,temps,records,bind,sh)
    tasks={}
    for task in p.TASKS:
        path=stage/f"tasks/{task}/shards/task-00000.h5"
        path.parent.mkdir(parents=True,exist_ok=True)
        p.write_task_shard(path,task,temps,records,bind,sh)
        tasks[task]=path
    validation=p.validate_artifacts(sp,qp,tasks,records,temps)
    p.require(sp.stat().st_size <= 8*GIB, "actual shard exceeds 8 GiB")
    summary=shard_report(stage,records,validation,number)
    destinations=[(sp,output/f"signals/shards/signal-{shard_id}.h5"),
                  (qp,output/f"signals/qc/qc-{shard_id}.h5")]
    destinations += [(path,output/f"tasks/{task}/shards/task-{shard_id}.h5") for task,path in tasks.items()]
    artifacts=[]
    for source,dest in destinations:
        dest.parent.mkdir(parents=True,exist_ok=True)
        digest=p.sha256_file(source)
        # Staging may be on another volume: copy under .partial, then atomically expose.
        partial=dest.with_suffix(dest.suffix+".partial")
        shutil.copyfile(source,partial)
        p.require(p.sha256_file(partial)==digest,"cross-volume copy checksum")
        replace_with_retry(partial,dest)
        artifacts.append({"artifact":str(dest.relative_to(output)),"bytes":dest.stat().st_size,
                          "sha256":digest,"epochs":validation["total_epochs"],"shard_id":shard_id})
    report_dest=output/f"reports/shards/shard-{shard_id}.md"
    report_dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(stage/"qc.md",report_dest)
    if (stage/"epoch_hard.parquet").exists():
        shutil.copyfile(stage/"epoch_hard.parquet",output/f"reports/shards/epoch-hard-{shard_id}.parquet")
    commit={"shard_number":number,"shard_id":shard_id,"status":"QC_PASSED","records":records,
            "artifacts":artifacts,"summary":summary,"failed_sessions":failures,"validation":validation,
            "pack_seconds":time.perf_counter()-started,"committed_at":p.utc_now()}
    # The commit is the publication boundary; readers consume manifest entries only.
    atomic_json(output/f"manifests/commits/shard-{shard_id}.json",commit)
    # Remove only task-owned staging files after verified copies and commit.
    for temp,r in zip(temps,records,strict=True):
        p.require(temp.resolve().parent==work.resolve(),"temporary record path escaped work root")
        try:
            temp.unlink(missing_ok=True)
            (work/(r["recording_id"]+".json")).unlink(missing_ok=True)
        except OSError:
            pass  # Commit is durable; a locked staging file must not undo it.
    # Enumerate only this exact pack subtree, validate every file before removal.
    for f in sorted(stage.rglob("*"), key=lambda x:len(x.parts),reverse=True):
        p.require(f.resolve().is_relative_to(stage.resolve()),"pack path safety")
        try:
            if f.is_file():
                f.unlink()
            elif f.is_dir():
                f.rmdir()
        except OSError:
            pass
    try:
        stage.rmdir()
    except OSError:
        pass
    return commit


def refresh_manifests(output,commits,excluded,total_shards,total_records,status,progress,recovery_baseline=0):
    rows,artifacts,failures,summary=[],[],[],{}
    for number,commit in sorted(commits.items()):
        first=0
        for r in commit["records"]:
            rows.append({"session_key":r["session_key"],"recording_id":r["recording_id"],
                "subject_id":r["subject_id"],"session_id":r["session_id"],"status":"QC_PASSED",
                "reason":None,"shard_id":commit["shard_id"],"first_epoch":first,"n_epochs":r["epoch_count"],
                "duration_sec":r["duration_sec"],"night_grade":r["night_grade"],
                "source_h5":r["source_h5"],"source_h5_sha256":r["source_h5_sha256"],
                "source_task_sha256":r["source_task_sha256"]})
            first+=r["epoch_count"]
        artifacts.extend(commit["artifacts"])
        if number >= recovery_baseline:
            failures.extend(commit["failed_sessions"])
        for item in commit["summary"]:
            key=(item["channel"],item["code"])
            if key not in summary:
                summary[key]={k:v for k,v in item.items() if k!="shard_id"}
                summary[key]["evaluated_epochs"]=summary[key]["hard_epochs"]=0
            summary[key]["evaluated_epochs"]+=item["evaluated_epochs"]
            summary[key]["hard_epochs"]+=item["hard_epochs"]
    published = {r["session_key"] for r in rows}
    p.require(len(published) == len(rows), "duplicate manifest recording")
    failures = list({f["session_key"]: f for f in failures if f["session_key"] not in published}.values())
    if rows or excluded or failures:
        atomic_parquet(output/"manifests/records.parquet",rows+excluded+[
            {"session_key":f["session_key"],"status":"FAILED_PENDING_REVIEW","reason":f["error"]} for f in failures])
    if artifacts:
        atomic_parquet(output/"manifests/shards.parquet",artifacts)
    if summary:
        atomic_parquet(output/"reports/qc_summary.parquet",list(summary.values()))
    for task in p.TASKS:
        tr=[a for a in artifacts if str(Path(a["artifact"])).replace("\\","/").startswith(f"tasks/{task}/")]
        if tr:
            atomic_parquet(output/f"tasks/{task}/manifest.parquet",tr)
        atomic_json(output/f"tasks/{task}/dataset.json",{"version":p.VERSION,"status":status,
                    "task":task,"shards":len(tr),"entry":"manifest.parquet"})
    atomic_json(output/"signals/dataset.json",{"version":p.VERSION,"status":status,
                "records":len(rows),"shards":sum(bool(c["records"]) for c in commits.values()),
                "manifest":"../manifests/shards.parquet","read_rule":"only committed manifest entries"})
    atomic_json(output/"manifests/release.json",{"version":p.VERSION,"schema_version":p.SCHEMA_VERSION,
        "status":status,"pilot_review":"USER_APPROVED","total_candidate_records":total_records,
        "total_planned_shards":total_shards,"committed_shards":len(commits),"published_records":len(rows),
        "failed_records":len(failures),"excluded_records":len(excluded),"progress":progress,
        "updated_at":p.utc_now()})
    lines=["# I0002 v1.0.0 full QC progress","",
           f"- Status: {status}; updated: {p.utc_now()}",
           f"- Candidates: {total_records}; published: {len(rows)}; failed/pending review: {len(failures)}; excluded: {len(excluded)}",
           f"- Shards committed: {len(commits)}/{total_shards}",
           "- Kernels and PSD match user-approved pilot; no normalization or training split.",
           "- Per-shard hard codes, channel processing, task codes, concrete source values and time-alignment reasons: reports/shards/shard-xxxxx.md.",
           "- Full epoch hard relative times: reports/shards/epoch-hard-xxxxx.parquet.",
           "- Global progress / ETA / resource use: logs/progress.json. Errors: logs/errors.jsonl.",
           "", "| channel | code | name | evaluated | hard epochs |","|---|---:|---|---:|---:|"]
    for item in summary.values():
        if item["applicable"]:
            lines.append(f"| {item['channel']} | {item['code']} | {item['name']} | {item['evaluated_epochs']} | {item['hard_epochs']} |")
    tmp=output/"reports/qc.md.tmp"
    tmp.write_text("\n".join(lines)+"\n",encoding="utf-8")
    replace_with_retry(tmp,output/"reports/qc.md")


def main(argv=None):
    parser=argparse.ArgumentParser(description="Full I0002 preprocessing with bounded overlapping compute/write.")
    parser.add_argument("--input-root",type=Path,default=Path(r"I:\HSP\I0002"))
    parser.add_argument("--output-root",type=Path,default=Path(r"I:\HSP\I0002-preprocess\processed\v1.0.0"))
    parser.add_argument("--inventory",type=Path,default=Path(r"I:\HSP\I0002-preprocess\source_scan\v1.0.0\tables\sessions.parquet"))
    parser.add_argument("--plan",type=Path,default=Path(r"E:\Code\SleepWorldModel\docs\I0002_PREPROCESS_PLAN.md"))
    parser.add_argument("--pilot",type=Path,default=Path(r"I:\HSP\I0002-preprocess\processed\v1.0.0\pilot-20260905-psd-frozen"))
    parser.add_argument("--work-root",type=Path,default=Path(r"E:\Code\SleepWorldModel\artifacts\i0002-full-work"))
    parser.add_argument("--processes",type=int,default=8)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--recover-task-alignment",action="store_true",
                        help="Authorized one-time migration; retain commits and requeue unpublished records")
    parser.add_argument("--verify-resume",action="store_true")
    parser.add_argument("--max-records",type=int,default=16)
    parser.add_argument("--target-gib",type=float,default=6)
    parser.add_argument("--heartbeat-seconds",type=float,default=5)
    args=parser.parse_args(argv)
    p.require(1<=args.processes<=16 and 1<=args.threads<=8,"concurrency out of bounds")
    output,work=args.output_root.resolve(),args.work_root.resolve()
    for rel in ("config","logs","manifests/commits","reports/shards","signals/shards","signals/qc"):
        (output/rel).mkdir(parents=True,exist_ok=True)
    work.mkdir(parents=True,exist_ok=True)
    lock=output/"logs/full.lock"
    if lock.exists():
        previous=json.loads(lock.read_text())
        if psutil.pid_exists(previous["pid"]):
            raise RuntimeError(f"full process already running: {previous['pid']}")
        p.require(args.resume,"stale run lock; use --resume")
        lock.unlink()
    with lock.open("x") as f:
        json.dump({"pid":os.getpid(),"started_at":p.utc_now()},f)
    started=time.perf_counter()
    process=psutil.Process()
    psutil.cpu_percent()
    event_path=output/"logs/run.log"
    (output/"logs/errors.jsonl").touch(exist_ok=True)
    def log(event,**fields):
        entry={"time":p.utc_now(),"event":event,**fields}
        try:
            with event_path.open("a",encoding="utf-8") as f:
                f.write(json.dumps(entry,ensure_ascii=False)+"\n")
        except OSError as error:
            print(json.dumps({"event":"RUN_LOG_WRITE_DEFERRED","error":repr(error)}),flush=True)
        print(json.dumps(entry,ensure_ascii=False),flush=True)
    def error_entry(session,error,attempt):
        try:
            with (output/"logs/errors.jsonl").open("a",encoding="utf-8") as f:
                f.write(json.dumps({"time":p.utc_now(),"session_key":session,"error":repr(error),"attempt":attempt},ensure_ascii=False)+"\n")
        except OSError as log_error:
            log("ERROR_LOG_WRITE_DEFERRED",session=session,error=repr(error),log_error=repr(log_error))
    state={"status":"RUNNING","phase":"verifying_resume","pid":os.getpid(),"updated_at":p.utc_now()}
    write_progress(output/"logs/progress.json",state,log)
    try:
        code_hashes={name:p.sha256_file(Path(__file__).with_name(name)) for name in ("core.py","pipeline.py","full.py")}
        pilot_runtime=json.loads((args.pilot/"config/runtime.json").read_text(encoding="utf-8"))
        p.require(code_hashes["core.py"]==pilot_runtime["code_sha256"]["core.py"],"signal/QC core changed since approved pilot")
        pilot_qc=json.loads((args.pilot/"config/qc.json").read_text(encoding="utf-8"))
        p.require(pilot_qc["psd"]==p.PSD_CONFIG,"PSD differs from approved pilot")
        shards,excluded=plan_records(args.inventory,args.max_records,args.target_gib)
        total_records=sum(map(len,shards))
        total_epochs=sum(r["epoch_count"] for batch in shards for r in batch)
        build_path=output/"config/full_build.json"
        log("RESUME_VERIFY_START",sha256=args.verify_resume)
        commits=load_commits(output,args.verify_resume,log)
        recovery_baseline=0
        spec={"version":p.VERSION,"code_sha256":code_hashes,"inventory_sha256":p.sha256_file(args.inventory),
              "approved_pilot":str(args.pilot),"approved_pilot_plan_sha256":pilot_runtime.get("plan_sha256"),
              "max_records":args.max_records,"target_gib":args.target_gib,
              "plan_sha256":p.sha256_file(args.plan),"input_root":str(args.input_root.resolve()),
              "work_root":str(work),"processes":args.processes,"threads":args.threads,
              "total_records":total_records,"total_epochs":total_epochs,"total_shards":len(shards)}
        if build_path.exists():
            p.require(args.resume,"full output already initialized; use --resume")
            previous=json.loads(build_path.read_text(encoding="utf-8"))
            for key in ("inventory_sha256","max_records","target_gib","input_root","work_root"):
                p.require(previous[key]==spec[key],f"resume specification mismatch: {key}")
            if args.recover_task_alignment:
                p.require(previous["code_sha256"] == {
                    "core.py":"fba1473a816a9fec829244d862e348a4226103e0bf8bb064d745b0147a13a03d",
                    "pipeline.py":"b328a4a1f1f1f3b46521f09679dfa72c2e69ce4dc50a84481bafcd7344d5e1a8",
                    "full.py":"25852fca7c4c5051a0fcedd0ac753421098f474615ef102b4bea3c6789846bab"},
                    "recovery only supports the reviewed failed runtime")
                archive=output/"config/history/pre-task-alignment-recovery"
                archive.mkdir(parents=True,exist_ok=True)
                # Archive first: old shard hash bindings remain resolvable.
                for source in (output/"config").iterdir():
                    if source.is_file():
                        dest=archive/source.name
                        if not dest.exists():
                            shutil.copy2(source,dest)
                shards=recovery_plan(shards,commits,args.max_records,args.target_gib)
                recovery_baseline=len(commits)
                spec.update(total_shards=len(shards),recovery_baseline=recovery_baseline,
                            recovery_archive=str(archive.relative_to(output)),
                            recovery_at=p.utc_now(),recovery_reason="logging fault tolerance and local task alignment QC")
                p.write_configs(output,spec["plan_sha256"])
                for name in ("core.py","pipeline.py","full.py"):
                    shutil.copy2(Path(__file__).with_name(name),output/"config"/name)
                shutil.copy2(args.plan,output/"config/preprocess.md")
                atomic_json(output/"config/shard_plan.json",{"shards":shards,"excluded":excluded})
                atomic_json(build_path,spec)
                log("RECOVERY_MIGRATED",preserved_shards=len(commits),
                    preserved_records=sum(len(c["records"]) for c in commits.values()),
                    requeued_previous_failures=sum(len(c["failed_sessions"]) for c in commits.values()))
            else:
                for key in ("code_sha256","plan_sha256"):
                    p.require(previous[key]==spec[key],f"resume specification mismatch: {key}")
                saved_plan=json.loads((output/"config/shard_plan.json").read_text(encoding="utf-8"))
                shards,excluded=saved_plan["shards"],saved_plan["excluded"]
                recovery_baseline=previous.get("recovery_baseline",0)
                spec.update(total_shards=len(shards),recovery_baseline=recovery_baseline)
        else:
            p.require(not args.resume,"resume requested but no full build config")
            p.write_configs(output,spec["plan_sha256"])
            for name in ("core.py","pipeline.py","full.py"):
                shutil.copy2(Path(__file__).with_name(name),output/"config"/name)
            shutil.copy2(args.plan,output/"config/preprocess.md")
            atomic_json(build_path,spec)
            atomic_json(output/"config/shard_plan.json",{"shards":shards,"excluded":excluded})
        planned_keys=[r["session_key"] for batch in shards for r in batch]
        p.require(len(planned_keys)==total_records and len(set(planned_keys))==total_records,"recovery plan completeness/uniqueness")
        failures_count=sum(len(c["failed_sessions"]) for n,c in commits.items() if n>=recovery_baseline)
        finished_sessions=sum(len(c["records"]) for c in commits.values())+failures_count
        baseline_epochs=sum(sum(r["epoch_count"] for r in shards[n]) for n in commits)
        processed_epochs=baseline_epochs
        committed_epochs=baseline_epochs
        outcomes={}
        attempts=Counter()
        active={}
        writer_future=None
        writer_number=None
        last_log=0
        baseline_shards=len(commits)
        def publish_progress(phase,status="RUNNING"):
            elapsed=time.perf_counter()-started
            new_epochs=processed_epochs-baseline_epochs
            completed_epochs=committed_epochs-baseline_epochs
            # End-to-end ETA becomes meaningful after the first whole shard commit.
            rate=completed_epochs/elapsed if completed_epochs>0 else None
            eta=(total_epochs-committed_epochs)/rate if rate else None
            memory=psutil.virtual_memory()
            children=process.children(recursive=True)
            rss=process.memory_info().rss
            for child in children:
                try:
                    rss+=child.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            state.update(status=status,phase=phase,updated_at=p.utc_now(),pid=os.getpid(),
                total_records=total_records,finished_records=finished_sessions,failed_records=failures_count,
                published_records=sum(len(c["records"]) for c in commits.values()),
                total_shards=len(shards),committed_shards=len(commits),total_epochs=total_epochs,
                computed_epochs=processed_epochs,committed_epochs=committed_epochs,
                elapsed_seconds=elapsed,compute_epochs_per_second=new_epochs/elapsed if elapsed else None,
                end_to_end_epochs_per_second=rate,eta_seconds=eta,
                eta_scope="remaining compute + packing + validation + verified copy, based on committed throughput",
                eta_quality="WARMUP" if len(commits)-baseline_shards<3 else "ESTIMATE",
                processes=args.processes,threads_per_process=args.threads,active_workers=len(active),
                writer_shard=writer_number,cpu_percent=psutil.cpu_percent(),rss_gib=rss/GIB,
                available_memory_gib=memory.available/GIB,output_free_gib=shutil.disk_usage(output).free/GIB)
            write_progress(output/"logs/progress.json",state,log)
        log("FULL_START",**spec,estimated_uncompressed_gib=total_epochs*360000/GIB)
        publish_progress("initializing")
        refresh_manifests(output,commits,excluded,len(shards),total_records,"RUNNING",state,recovery_baseline)
        with ProcessPoolExecutor(max_workers=args.processes) as pool, ThreadPoolExecutor(max_workers=1,thread_name_prefix="shard-writer") as writer:
            while len(commits)<len(shards):
                remaining=[n for n in range(len(shards)) if n not in commits]
                # Only current and next shard may occupy the SSD work queue.
                window=remaining[:2]
                for n in window:
                    outcomes.setdefault(n,{})
                if writer_future is not None and writer_future.done():
                    commit=writer_future.result()
                    commits[writer_number]=commit
                    committed_epochs+=sum(r["epoch_count"] for r in shards[writer_number])
                    log("SHARD_COMMITTED",shard=writer_number,records=len(commit["records"]),
                        failed=len(commit["failed_sessions"]),pack_seconds=commit["pack_seconds"])
                    outcomes.pop(writer_number,None)
                    writer_future=writer_number=None
                    publish_progress("compute_and_pack")
                    refresh_manifests(output,commits,excluded,len(shards),total_records,"RUNNING",state,recovery_baseline)
                    continue
                if writer_future is None:
                    n=remaining[0]
                    if len(outcomes[n])==len(shards[n]):
                        ordered=[outcomes[n][r["session_key"]] for r in shards[n]]
                        results=[x["result"] for x in ordered if "result" in x]
                        failures=[x["failure"] for x in ordered if "failure" in x]
                        writer_number=n
                        writer_future=writer.submit(pack_shard,str(output),str(work),n,results,failures)
                        log("SHARD_PACK_START",shard=n,records=len(results))
                active_keys={job[1]["session_key"] for job in active.values()}
                for n in window:
                    if n==writer_number:
                        continue
                    for row in shards[n]:
                        key=row["session_key"]
                        if key in outcomes[n] or key in active_keys or len(active)>=args.processes:
                            continue
                        # Keep memory and SSD headroom; do not spawn into swapping.
                        if psutil.virtual_memory().available < 12*GIB or shutil.disk_usage(work).free < 25*GIB:
                            break
                        p.require(shutil.disk_usage(output).free > 20*GIB,"output disk below 20 GiB reserve")
                        cache=work/(key.replace("/","_")+".json")
                        if cache.exists():
                            saved=json.loads(cache.read_text(encoding="utf-8"))
                            temp=Path(saved["temp"])
                            if temp.exists():
                                cached_record=saved["record"]
                                p.require(p.sha256_file(Path(cached_record["source_h5"]))==cached_record["source_h5_sha256"],"cached source H5 changed")
                                p.require(p.sha256_file(Path(cached_record["source_task_csv"]))==cached_record["source_task_sha256"],"cached task CSV changed")
                                with h5py.File(temp) as tf:
                                    p.require(tf["signals"].shape==(row["epoch_count"],15,6000),"cached record shape")
                                outcomes[n][key]={"result":(str(temp),saved["record"])}
                                finished_sessions+=1
                                processed_epochs+=row["epoch_count"]
                                log("SESSION_RESUMED",session=key)
                                continue
                        attempts[key]+=1
                        future=pool.submit(session_job,str(args.input_root),key,str(work),args.threads)
                        active[future]=(n,row)
                        active_keys.add(key)
                if active:
                    done,_=wait(list(active),timeout=args.heartbeat_seconds,return_when=FIRST_COMPLETED)
                    for future in done:
                        n,row=active.pop(future)
                        key=row["session_key"]
                        try:
                            result=future.result()
                            p.require(result[1]["epoch_count"]==row["epoch_count"],"source duration changed since scan")
                            outcomes[n][key]={"result":result}
                            log("SESSION_DONE",session=key,epochs=row["epoch_count"],
                                worker_seconds=result[1]["worker_seconds"],pid=result[1]["worker_pid"])
                            for c in result[1]["channel_metadata"]:
                                for err in c.get("errors",[]):
                                    error_entry(key+"/"+c["canonical_channel"],RuntimeError(err),attempts[key])
                        except Exception as error:
                            error_entry(key,error,attempts[key])
                            log("SESSION_FAILED",session=key,attempt=attempts[key],error=repr(error))
                            if attempts[key]<2:
                                cache=work/(key.replace("/","_")+".json")
                                cache.unlink(missing_ok=True)
                                continue
                            outcomes[n][key]={"failure":{"session_key":key,"error":repr(error)}}
                            failures_count+=1
                        finished_sessions+=1
                        processed_epochs+=row["epoch_count"]
                else:
                    time.sleep(min(args.heartbeat_seconds,1))
                publish_progress("compute_and_pack")
                if time.perf_counter()-last_log>=30:
                    log("PROGRESS",**state)
                    last_log=time.perf_counter()
                # Stop early for a systemic source/schema failure, preserving all evidence.
                p.require(not (finished_sessions>=8 and failures_count>=8 and failures_count/finished_sessions>.5),
                          "systemic failure rate; review errors before continuing")
        final_status="COMPLETE" if failures_count==0 else "COMPLETE_WITH_FAILURES"
        publish_progress("complete",final_status)
        state["eta_seconds"]=0
        write_progress(output/"logs/progress.json",state,log)
        refresh_manifests(output,commits,excluded,len(shards),total_records,final_status,state,recovery_baseline)
        log("FULL_COMPLETE",status=final_status,failed=failures_count,elapsed_seconds=time.perf_counter()-started)
        return 0 if failures_count==0 else 2
    except Exception as error:
        error_entry("_run",error,1)
        log("FULL_FAILED",error=repr(error),traceback=traceback.format_exc())
        state.update(status="FAILED",phase="failed",pid=os.getpid(),error=repr(error),
                     elapsed_seconds=time.perf_counter()-started,updated_at=p.utc_now())
        write_progress(output/"logs/progress.json",state,log)
        release_path=output/"manifests/release.json"
        if release_path.exists():
            release=json.loads(release_path.read_text(encoding="utf-8"))
            release.update(status="FAILED",error=repr(error),progress=state,updated_at=p.utc_now())
            atomic_json(release_path,release)
        return 1
    finally:
        lock.unlink(missing_ok=True)


if __name__=="__main__":
    raise SystemExit(main())
