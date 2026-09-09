import csv
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy import signal

from data_preprocess.i0002_v100.core import (
    APPLICABLE, CHANNELS, EFFECTIVE_FILTER, EPOCH_SAMPLES, PROCESS_RESULT,
    design_supplement, filter_relation, FILTER_RELATION, primary_reason,
    process_channel, profile_accepted, hard_bit, spectral_metrics, audit_epochs,
)
from data_preprocess.i0002_v100.pipeline import (
    TASKS, task_arrays, verify_source_alignment, process_session, write_configs, binding,
    write_signal_shard, write_qc_shard, write_task_shard, validate_artifacts,
    sha256_file, night_grade,
)


def attrs(fs=200, prefilter="HP:0.300Hz LP:35Hz N:60Hz", name="f3-m2"):
    return {"dig_min": -32768, "dig_max": 32767, "phys_min": 0 if name == "spo2" else -3200,
            "phys_max": 100 if name == "spo2" else 3200, "unit": "%" if name == "spo2" else "uV",
            "fs": fs, "prefilter": prefilter, "type": "raw"}


def digital(values, at):
    return np.rint(at["dig_min"] + (values-at["phys_min"]) * (at["dig_max"]-at["dig_min"]) /
                   (at["phys_max"]-at["phys_min"])).astype(np.int16)


def sine(freq=10, amplitude=100, fs=200, seconds=30):
    return amplitude * np.sin(2*np.pi*freq*np.arange(round(seconds*fs))/fs)


def test_profile_gates():
    assert profile_accepted("airflow", attrs())
    for p in (-12003000, -328000000, 0):
        assert not profile_accepted("airflow", {**attrs(), "phys_min":p})
    assert profile_accepted("spo2", attrs(name="spo2"))
    assert not profile_accepted("spo2", {**attrs(name="spo2"), "phys_min":-328,"phys_max":328})


def test_filter_cutoffs_and_explicit_preservation():
    assert design_supplement("snore",200)[2] == (10,95)
    assert design_supplement("snore",500)[2] == (10,100)
    assert filter_relation("f3-m2",(0.53,35)) == FILTER_RELATION["SOURCE_NARROWER"]
    at = attrs()
    raw = digital(sine(),at)
    result = process_channel(raw,at,"f3-m2",1)
    expected = at["phys_min"]+(raw.astype(float)-at["dig_min"])*6400/65535
    assert np.array_equal(result["data"][0],expected.astype("f4"))
    assert result["valid"][0]
    assert result["metadata"]["filter_reapply_result"] == PROCESS_RESULT["NOT_REQUIRED"]


@pytest.mark.parametrize("frequency,line_bad,hf_bad",[(10,False,False),(50,True,False),(60,True,False),(80,False,True)])
def test_psd_tones(frequency,line_bad,hf_bad):
    line,hf = spectral_metrics(sine(frequency)[None])
    assert bool(line[0] > .4) is line_bad
    assert bool(hf[0] > .5) is hf_bad


def test_psd_matches_independent_fft_and_zero_denominator():
    x=np.random.default_rng(19).normal(size=(3,6000))
    windows=np.stack([x[:,i:i+800] for i in range(0,5201,400)],axis=1)
    w=.5-.5*np.cos(2*np.pi*np.arange(800)/800)
    fft=np.fft.rfft((windows-windows.mean(-1,keepdims=True))*w,axis=-1)
    power=(abs(fft)**2)/(200*np.dot(w,w))
    power[:,:,1:-1]*=2
    power=power.mean(1)
    f=np.arange(401)*.25
    denom=power[:,f>=.5].sum(1)
    lm=((f>=49)&(f<=51))|((f>=59)&(f<=61))
    actual=spectral_metrics(x)
    np.testing.assert_allclose(actual[0],power[:,lm].sum(1)/denom,atol=1e-14,rtol=1e-12)
    np.testing.assert_allclose(actual[1],power[:,f>70].sum(1)/denom,atol=1e-14,rtol=1e-12)
    zero=spectral_metrics(np.ones((2,6000))*77)
    assert all(np.array_equal(a,[0,0]) for a in zero)


def test_high_amplitude_requires_all_30_windows_and_preserves_spikes():
    at=attrs()
    data=np.zeros((3,6000),np.float32)
    data[0,10]=1500
    data[1,::200]=1500  # accepted limitation: one short spike in every window
    data[2]=data[1]
    data[2,-200:]=0
    raw=digital(data.ravel(),at)
    h,e,m=audit_epochs(data,raw,at,"f3-m2",np.ones(3,bool),200)
    assert m["high_amplitude_duration_seconds"].tolist()==[1,30,29]
    assert ((h&hard_bit(4))!=0).tolist()==[False,True,False]
    result=process_channel(raw,at,"f3-m2",3)
    assert result["data"][1].max()>1400


def test_emg_does_not_evaluate_high_frequency():
    at=attrs()
    result=process_channel(digital(sine(80),at),at,"lat",1)
    assert (result["hard_evaluated_flags"][0]&hard_bit(6))==0
    assert (result["hard_flags"][0]&hard_bit(6))==0
    assert np.isnan(result["metrics"]["high_frequency_fraction"][0])
    assert result["hard_evaluated_flags"][0]==APPLICABLE[10]


def test_saturation_is_fraction_only():
    at=attrs()
    raw=digital(sine(),at)
    raw[:100]=32767  # 0.5 s, less than 5%: not saturation
    r=process_channel(raw,at,"f3-m2",1)
    assert not (r["hard_flags"][0]&hard_bit(3))
    raw[:300]=32767
    r=process_channel(raw,at,"f3-m2",1)
    assert r["hard_flags"][0]&hard_bit(3)
    assert r["data"].max()>3190


def test_unknown_filter_whole_finite_record_without_artifact_segmentation():
    at=attrs(prefilter="HP:0Hz LP:0Hz")
    values=sine(seconds=90)
    values[6000:12000]=0
    raw=digital(values,at)
    r=process_channel(raw,at,"f3-m2",3)
    x=at["phys_min"]+(raw.astype(float)-at["dig_min"])*6400/65535
    stages,_,_=design_supplement("f3-m2",200)
    for sos in stages:
        x=signal.sosfiltfilt(sos,x,padtype="odd")
    np.testing.assert_array_equal(r["data"],x.reshape(3,6000).astype("f4"))
    assert r["processing_valid"].all()
    assert r["metadata"]["processed_runs"]==1
    assert r["data"][0].any() and r["data"][-1].any()


def test_nan_evidence_before_fill_and_mask_independence():
    at=attrs()
    raw=digital(sine(seconds=90),at).astype(float)
    raw[7000]=np.nan
    r=process_channel(raw,at,"f3-m2",3)
    assert r["metadata"]["channel_available"]
    assert r["coverage_valid"].all()
    assert r["processing_valid"].tolist()==[True,False,True]
    assert r["hard_flags"][1]&hard_bit(1)
    assert r["hard_evaluated_flags"][1]==hard_bit(1)
    assert r["metrics"]["nonfinite_source_count"][1]==1
    assert r["data"][1,1000]==0 and np.isfinite(r["data"]).all()


def test_missing_partial_and_profile_placeholders():
    missing=process_channel(None,{},"f3-m2",2)
    assert not missing["metadata"]["channel_available"]
    assert not missing["artifact_valid"].any()
    assert not missing["hard_evaluated_flags"].any()
    raw=digital(sine(seconds=40),attrs())
    partial=process_channel(raw,attrs(),"f3-m2",2)
    assert partial["metadata"]["channel_available"]
    assert partial["coverage_valid"].tolist()==[True,False]
    assert partial["processing_valid"].tolist()==[True,False]
    excluded=process_channel(None,{**attrs(), "phys_min":-12003000,
        "_source_present":True,"_source_samples":12000},"airflow",2)
    assert excluded["coverage_valid"].all()
    assert not excluded["metadata"]["channel_available"]
    assert not excluded["hard_flags"].any()
    assert excluded["metadata"]["waveform_read"] is False


def test_airflow_flat_and_snore_silence():
    raw=np.zeros(6000,np.int16)
    af=process_channel(raw,attrs(),"airflow",1)
    sn=process_channel(raw,attrs(),"snore",1)
    assert af["hard_flags"][0]&hard_bit(2)
    assert sn["valid"][0] and sn["hard_evaluated_flags"][0]==1


def test_spo2_hold_and_overnight_near_zero():
    at=attrs(fs=25,name="spo2")
    raw=digital(np.repeat([95.,96.,94.],250),at)
    r=process_channel(raw,at,"spo2",1)
    assert r["valid"][0]
    assert np.all(r["data"].reshape(-1,8)==r["data"].reshape(-1,8)[:,0:1])
    low=process_channel(digital(np.ones(750)*2,at),at,"spo2",1)
    assert not low["metadata"]["channel_available"]
    assert low["metadata"]["channel_status_code"]==5
    assert low["data"].mean()>1  # preserve finite rejected waveform


def test_downsample_reference_and_high_code_priority():
    at=attrs(fs=500)
    raw=digital(sine(fs=500),at)
    result=process_channel(raw,at,"f3-m2",1)
    x=at["phys_min"]+(raw.astype(float)-at["dig_min"])*6400/65535
    expected=signal.resample_poly(x,2,5,window=("kaiser",5.0),padtype="line").astype("f4")
    assert np.array_equal(result["data"][0],expected)
    assert primary_reason(np.array([hard_bit(4)|hard_bit(3)|hard_bit(1)],"u1"))[0]==1


def write_csv(path, rows):
    fields=["Epoch",*(f for fields in TASKS.values() for f in fields)]
    with path.open("w",newline="",encoding="utf-8") as fp:
        writer=csv.DictWriter(fp,fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({"Epoch":1,"Stage":"W","Heart Rate Max":90,"Heart Rate Min":60,
                             "Heart Rate Avg":70,"SaO2 Max":98,"SaO2 Min":95,**row})


def test_task_codes_fields_and_sentinels(tmp_path):
    path=tmp_path/"tasks.csv"
    write_csv(path,[{"Epoch":1},{"Epoch":2,"Heart Rate Min":0},
                   {"Epoch":3,"Heart Rate Max":40},{"Epoch":4,"SaO2 Max":101},
                   {"Epoch":5,"Stage":"L"},{"Epoch":8}])
    tasks=task_arrays(path,6)
    assert tasks["heart_rate"]["field_qc_code"][1].tolist()==[0,2,0]
    assert tasks["heart_rate"]["field_qc_code"][2].tolist()==[4,4,4]
    assert tasks["sao2"]["field_qc_code"][3].tolist()==[3,0]
    assert tasks["sleep_stage"]["qc_code"].tolist()==[0,0,0,0,3,1]
    assert tasks["sleep_stage"]["field_valid"].shape==(6,1)
    assert tasks["_coverage"]["extra_source_epochs"]==1
    assert np.isnan(tasks["heart_rate"]["labels"][5]).all()


def test_task_duplicate_fails_and_unverified_time_masks_tasks(tmp_path):
    path=tmp_path/"tasks.csv"
    write_csv(path,[{"Epoch":1},{"Epoch":1}])
    with pytest.raises(ValueError,match="TASK_EPOCH_KEY_MISMATCH"):
        task_arrays(path,1)
    write_csv(path,[{"Epoch":1}])
    tasks=task_arrays(path,1)
    events={"event_map":json.dumps({str(i):v for i,v in enumerate(["Unknown","N3","N2","N1","REM","Wake"])}),
            "starts":[30],"ends":[60],"codes":[5]}
    alignment=verify_source_alignment(events,tasks,1)
    assert alignment["status"]=="PARTIAL"
    assert alignment["time_unverified_epochs"]==[0]
    for task in TASKS:
        assert not tasks[task]["valid"].any()
        assert (tasks[task]["field_qc_code"]==5).all()


def test_realistic_synthetic_session_packing_and_corruption(tmp_path):
    root=tmp_path/"input"
    session="sub-synthetic/ses-1"
    eeg=root/session/"eeg"
    eeg.mkdir(parents=True)
    with h5py.File(eeg/"record.h5","w") as f:
        f.attrs.update(duration_sec=65,subject_id="sub-synthetic",session_id="ses-1")
        for name in CHANNELS:
            at=attrs(fs=25 if name=="spo2" else 200,name=name)
            x=np.ones(65*25)*96 if name=="spo2" else sine(seconds=65)
            ds=f.create_dataset("signals/"+name,data=digital(x,at))
            ds.attrs.update(at)
        g=f.create_group("annotations/expert_1/stage")
        g.attrs["event_map"]=json.dumps({str(i):v for i,v in enumerate(["Unknown","N3","N2","N1","REM","Wake"])})
        for k,v in {"starts":[0,0],"ends":[30,30],"codes":[4,5]}.items():
            g.create_dataset(k,data=v)
    write_csv(eeg/"record_sleep_annotations.csv",[{"Epoch":1},{"Epoch":2}])
    work=tmp_path/"work"
    work.mkdir()
    temp, record=process_session(str(root),session,str(work),2)
    stage=tmp_path/"stage"
    stage.mkdir()
    write_configs(stage,"test_plan")
    bind=binding(stage,[record])
    sp,qp=stage/"signal.h5",stage/"qc.h5"
    write_signal_shard(sp,[Path(temp)],[record],bind)
    write_qc_shard(qp,[Path(temp)],[record],bind,sha256_file(sp))
    tasks={}
    for task in TASKS:
        tasks[task]=stage/f"{task}.h5"
        write_task_shard(tasks[task],task,[Path(temp)],[record],bind,sha256_file(sp))
    assert validate_artifacts(sp,qp,tasks,[record],[Path(temp)])["status"]=="PASS"
    with h5py.File(tasks["sleep_stage"]) as tf:
        assert tf["qc_code"][:].tolist()==[6,5]
    with h5py.File(tasks["heart_rate"]) as tf:
        assert tf["qc_code"][:].tolist()==[0,5]
    assert record["task_alignment"]["status"]=="PARTIAL"
    from data_preprocess.i0002_v100.full import pack_shard, load_commits
    committed=pack_shard(str(stage),str(work),7,[(temp,record)],[])
    assert committed["status"]=="QC_PASSED"
    assert load_commits(stage,verify=True)[7]["records"][0]["recording_id"]==record["recording_id"]
    with h5py.File(stage/"signals/shards/signal-00007.h5") as full_signal:
        assert full_signal.attrs["shard_id"]=="00007"
        assert full_signal.attrs["release_status"]=="QC_PASSED"
    with h5py.File(qp,"r+") as f:
        f["hard_evaluated_flags"][0,10] |= hard_bit(6)
    with pytest.raises(AssertionError,match="inapplicable"):
        validate_artifacts(sp,qp,tasks,[record])


def test_unscored_stage_does_not_invalidate_explicit_task_time_grid(tmp_path):
    path=tmp_path/"tasks.csv"
    write_csv(path,[{"Epoch":1,"Stage":"L"}])
    tasks=task_arrays(path,1)
    events={"event_map":json.dumps({str(i):v for i,v in enumerate(["Unknown","N3","N2","N1","REM","Wake"])}),
            "starts":[0],"ends":[30],"codes":[0]}
    result=verify_source_alignment(events,tasks,1)
    assert result["status"]=="PASS" and result["known_anchors"]==0
    assert not tasks["sleep_stage"]["valid"][0]
    assert tasks["heart_rate"]["valid"][0]


def test_full_plan_exclusions_and_shard_caps(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from data_preprocess.i0002_v100.full import plan_records
    path=tmp_path/"sessions.parquet"
    rows=[{"session_key":f"sub-{i}/ses-1","h5_duration_seconds":28800.0,"h5_bytes":100} for i in range(35)]
    rows.append({"session_key":"sub-I0002150026681/ses-2","h5_duration_seconds":79.0,"h5_bytes":100})
    pq.write_table(pa.Table.from_pylist(rows),path)
    shards,excluded=plan_records(path)
    assert [len(s) for s in shards]==[16,16,3]
    assert len(excluded)==1 and excluded[0]["status"]=="FIXED_EXCLUDED"
    assert all(sum(r["estimated_bytes"] for r in batch)<=6*2**30 for batch in shards)


def test_local_stage_conflict_and_mismatch_preserve_numeric_tasks(tmp_path):
    path=tmp_path/"tasks.csv"
    write_csv(path,[{"Epoch":1},{"Epoch":2},{"Epoch":3}])
    tasks=task_arrays(path,3)
    events={"event_map":json.dumps({str(i):v for i,v in enumerate(["Unknown","N3","N2","N1","REM","Wake"])}),
            "starts":[0,30,30,60],"ends":[30,60,60,90],"codes":[5,4,5,2]}
    alignment=verify_source_alignment(events,tasks,3)
    assert alignment["h5_stage_conflict_epochs"]==[1]
    assert alignment["source_stage_mismatch_epochs"]==[2]
    assert alignment["known_anchors"]==1
    assert tasks["sleep_stage"]["qc_code"].tolist()==[0,6,7]
    assert tasks["sleep_stage"]["labels"].tolist()==[5,5,5]
    assert tasks["heart_rate"]["valid"].all() and tasks["sao2"]["valid"].all()


def test_missing_grid_preserves_prior_value_reason(tmp_path):
    path=tmp_path/"tasks.csv"
    write_csv(path,[{"Epoch":1,"Stage":"L","SaO2 Min":0}])
    tasks=task_arrays(path,1)
    events={"event_map":json.dumps({str(i):v for i,v in enumerate(["Unknown","N3","N2","N1","REM","Wake"])}),
            "starts":[],"ends":[],"codes":[]}
    alignment=verify_source_alignment(events,tasks,1)
    assert alignment["time_unverified_epochs"]==[0]
    assert tasks["sleep_stage"]["qc_code"].tolist()==[3]
    assert tasks["sao2"]["field_qc_code"].tolist()==[[5,2]]


def test_atomic_replace_retry_and_nonfatal_progress(tmp_path,monkeypatch):
    from data_preprocess.i0002_v100 import full
    real_replace=full.os.replace
    calls=[]
    def flaky(source,dest):
        calls.append(1)
        if len(calls)<3:
            raise PermissionError("reader denies delete sharing")
        real_replace(source,dest)
    monkeypatch.setattr(full.os,"replace",flaky)
    monkeypatch.setattr(full.time,"sleep",lambda _:None)
    path=tmp_path/"progress.json"
    full.atomic_json(path,{"status":"RUNNING"})
    assert len(calls)==3 and json.loads(path.read_text())["status"]=="RUNNING"
    def blocked(*_):
        raise PermissionError("held open")
    monkeypatch.setattr(full.os,"replace",blocked)
    events=[]
    assert full.write_progress(path,{"status":"NEW"},lambda event,**kw:events.append(event)) is False
    assert events==["PROGRESS_WRITE_DEFERRED"]
    assert json.loads(path.read_text())["status"]=="RUNNING"
    with pytest.raises(PermissionError):
        full.atomic_json(tmp_path/"commit.json",{})
    assert not list(tmp_path.glob("*.tmp"))


def test_recovery_plan_preserves_published_and_requeues_failures():
    from data_preprocess.i0002_v100.full import recovery_plan
    rows=[{"session_key":str(i),"epoch_count":1,"estimated_bytes":360000} for i in range(8)]
    commits={0:{"records":[rows[0],rows[2]],"failed_sessions":[{"session_key":"1"}]}}
    plan=recovery_plan([rows[:3],rows[3:6],rows[6:]],commits,max_records=3)
    assert [r["session_key"] for r in plan[0]]==["0","2"]
    keys=[r["session_key"] for batch in plan[1:] for r in batch]
    assert keys==["1","3","4","5","6","7"]
    assert len(set(keys))==len(keys)


def test_recovered_manifest_does_not_double_count_old_failures(tmp_path):
    import pyarrow.parquet as pq
    from data_preprocess.i0002_v100.full import refresh_manifests
    (tmp_path/"reports").mkdir()
    def record(key):
        return {"session_key":key,"recording_id":key,"subject_id":key,"session_id":"ses-1",
                "epoch_count":1,"duration_sec":30.,"night_grade":1,
                "source_h5":"test.h5","source_h5_sha256":"test","source_task_sha256":"test"}
    old={"records":[record("a")],"shard_id":"00000","artifacts":[],"summary":[],
         "failed_sessions":[{"session_key":"b","error":"old conflict"}]}
    refresh_manifests(tmp_path,{0:old},[],2,2,"RUNNING",{},recovery_baseline=1)
    release=json.loads((tmp_path/"manifests/release.json").read_text())
    assert release["published_records"]==1 and release["failed_records"]==0
    new={"records":[record("b")],"shard_id":"00001","artifacts":[],"summary":[],"failed_sessions":[]}
    refresh_manifests(tmp_path,{0:old,1:new},[],2,2,"COMPLETE",{},recovery_baseline=1)
    rows=pq.read_table(tmp_path/"manifests/records.parquet").to_pylist()
    assert [r["session_key"] for r in rows]==["a","b"]
    assert all(r["status"]=="QC_PASSED" for r in rows)
    new["records"]=[]
    new["failed_sessions"]=[{"session_key":"b","error":"new error"}]
    refresh_manifests(tmp_path,{0:old,1:new},[],2,2,"COMPLETE_WITH_FAILURES",{},recovery_baseline=1)
    release=json.loads((tmp_path/"manifests/release.json").read_text())
    assert release["failed_records"]==1
