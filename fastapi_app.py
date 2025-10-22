import argparse
import asyncio
import shutil
import tempfile
from contextlib import asynccontextmanager
from copy import deepcopy
from pathlib import Path
from dataclasses import dataclass, field
import threading
from typing import Any, AsyncIterator, Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from omegaconf import DictConfig, OmegaConf

from scripts.inference import main as run_inference

PIPELINE_CONFIG: Optional[DictConfig] = None
PIPELINE_ARGS: Optional[argparse.Namespace] = None
JOB_STATES: Dict[str, "JobState"] = {}
JOB_LOCK = threading.Lock()


@dataclass
class JobState:
    temp_dir: Path
    output_path: Path
    status: str = "queued"
    detail: Optional[str] = None
    step_total: int = 0
    current_step: int = 0
    events: List[Dict[str, Any]] = field(default_factory=list)
    result_ready: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global PIPELINE_CONFIG, PIPELINE_ARGS
    PIPELINE_CONFIG = OmegaConf.load("configs/unet/stage2_512.yaml")
    PIPELINE_ARGS = argparse.Namespace(
        inference_ckpt_path="checkpoints/latentsync_unet.pt",
        temp_dir="temp",
        enable_deepcache=True,
        inference_steps=20,
        guidance_scale=1.5,
        seed=1247,
        video_path=None,
        audio_path=None,
        video_out_path=None,
    )
    with JOB_LOCK:
        JOB_STATES.clear()
    try:
        yield
    finally:
        PIPELINE_CONFIG = None
        PIPELINE_ARGS = None


app = FastAPI(lifespan=lifespan)


def _cleanup_job(job_id: str) -> None:
    with JOB_LOCK:
        state = JOB_STATES.pop(job_id, None)
    if state is not None:
        shutil.rmtree(state.temp_dir, ignore_errors=True)


def _get_job_state(job_id: str) -> Optional[JobState]:
    with JOB_LOCK:
        return JOB_STATES.get(job_id)


@app.post("/sync")
async def sync_video(
    video: UploadFile = File(...),
    audio: UploadFile = File(...),
    guidance_scale: float = Form(1.5),
    inference_steps: int = Form(20),
    seed: int = Form(1247),
) -> FileResponse:
    if PIPELINE_CONFIG is None or PIPELINE_ARGS is None:
        raise HTTPException(status_code=503, detail="Inference pipeline is not ready.")

    tmp_root = Path(tempfile.mkdtemp(prefix="latentsync_"))
    cleanup_immediately = True
    try:
        video_name = video.filename or "input_video"
        audio_name = audio.filename or "input_audio"
        video_path = tmp_root / video_name
        audio_path = tmp_root / audio_name
        out_path = tmp_root / "synced.mp4"
        video_path.write_bytes(await video.read())
        audio_path.write_bytes(await audio.read())

        base_args = deepcopy(PIPELINE_ARGS)
        base_args.video_path = str(video_path)
        base_args.audio_path = str(audio_path)
        base_args.video_out_path = str(out_path)
        base_args.guidance_scale = guidance_scale
        base_args.inference_steps = inference_steps
        base_args.seed = seed
        base_args.temp_dir = str(tmp_root / "cache")

        run_inference(config=PIPELINE_CONFIG, args=base_args)
        if not out_path.exists():
            raise HTTPException(status_code=500, detail="Generation failed.")
        cleanup_immediately = False
        background = BackgroundTask(shutil.rmtree, tmp_root, ignore_errors=True)
        return FileResponse(out_path, media_type="video/mp4", filename=out_path.name, background=background)
    finally:
        if cleanup_immediately:
            shutil.rmtree(tmp_root, ignore_errors=True)


@app.post("/jobs")
async def create_job(
    video: UploadFile = File(...),
    audio: UploadFile = File(...),
    guidance_scale: float = Form(1.5),
    inference_steps: int = Form(20),
    seed: int = Form(1247),
) -> Dict[str, str]:
    if PIPELINE_CONFIG is None or PIPELINE_ARGS is None:
        raise HTTPException(status_code=503, detail="Inference pipeline is not ready.")

    job_id = uuid4().hex
    tmp_root = Path(tempfile.mkdtemp(prefix=f"latentsync_{job_id}_"))
    try:
        video_name = video.filename or "input_video"
        audio_name = audio.filename or "input_audio"
        video_path = tmp_root / video_name
        audio_path = tmp_root / audio_name
        out_path = tmp_root / "synced.mp4"

        video_path.write_bytes(await video.read())
        audio_path.write_bytes(await audio.read())

        base_args = deepcopy(PIPELINE_ARGS)
        base_args.video_path = str(video_path)
        base_args.audio_path = str(audio_path)
        base_args.video_out_path = str(out_path)
        base_args.guidance_scale = guidance_scale
        base_args.inference_steps = inference_steps
        base_args.seed = seed
        base_args.temp_dir = str(tmp_root / "cache")

        job_state = JobState(
            temp_dir=tmp_root,
            output_path=out_path,
            step_total=0,
        )
        with JOB_LOCK:
            JOB_STATES[job_id] = job_state

        loop = asyncio.get_running_loop()

        def runner() -> None:
            event_counter = 0
            reported_total_steps = 0

            def init_callback(total_steps: int) -> None:
                nonlocal reported_total_steps
                reported_total_steps = total_steps
                with JOB_LOCK:
                    state = JOB_STATES.get(job_id)
                    if state is not None:
                        state.status = "running"
                        state.step_total = total_steps

            def update_callback(step_index: int) -> None:
                nonlocal event_counter
                event_counter += 1
                with JOB_LOCK:
                    state = JOB_STATES.get(job_id)
                    if state is not None:
                        state.status = "running"
                        state.current_step = step_index
                        if state.step_total == 0:
                            state.step_total = reported_total_steps or step_index
                        event = {
                            "event_index": event_counter,
                            "step": step_index,
                            "step_total": state.step_total,
                        }
                        state.events.append(event)
                        if len(state.events) > 200:
                            del state.events[: len(state.events) - 200]

            try:
                run_inference(
                    config=PIPELINE_CONFIG,
                    args=base_args,
                    progress_total_callback=init_callback,
                    progress_update_callback=update_callback,
                )
                if not out_path.exists():
                    raise RuntimeError("Generation failed.")
                with JOB_LOCK:
                    state = JOB_STATES.get(job_id)
                    if state is not None:
                        state.status = "completed"
                        state.result_ready = True
                        if state.step_total == 0:
                            state.step_total = reported_total_steps or state.current_step
                        if state.current_step < state.step_total:
                            state.current_step = state.step_total
            except Exception as exc:  # pylint: disable=broad-except
                with JOB_LOCK:
                    state = JOB_STATES.get(job_id)
                    if state is not None:
                        state.status = "failed"
                        state.detail = str(exc)
                        if state.step_total == 0:
                            state.step_total = reported_total_steps
                shutil.rmtree(tmp_root, ignore_errors=True)

        loop.run_in_executor(None, runner)
    except Exception:
        shutil.rmtree(tmp_root, ignore_errors=True)
        raise
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> Dict[str, Any]:
    with JOB_LOCK:
        state = JOB_STATES.get(job_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        state_snapshot = {
            "job_id": job_id,
            "status": state.status,
            "detail": state.detail,
            "current_step": state.current_step,
            "step_total": state.step_total,
            "result_ready": state.result_ready,
        }
    return state_snapshot


@app.get("/jobs/{job_id}/result")
def download_job_result(job_id: str) -> FileResponse:
    state = _get_job_state(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if not state.result_ready:
        raise HTTPException(status_code=409, detail="Job not complete.")
    if not state.output_path.exists():
        raise HTTPException(status_code=500, detail="Output file missing.")

    background = BackgroundTask(_cleanup_job, job_id)
    return FileResponse(
        state.output_path,
        media_type="video/mp4",
        filename=state.output_path.name,
        background=background,
    )


@app.get("/healthz")
def health() -> Dict[str, str]:
    return {"status": "ok"}
