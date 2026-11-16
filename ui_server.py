"""Local browser interface for the existing TinyAgent demo."""
import argparse
import asyncio
import codecs
import datetime
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import signal
import socket
import sys
import urllib.error
import urllib.request
import uuid
import webbrowser

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware
from typing import Literal
from demo_config import ENABLED_APPS, ENABLED_TOOLS, select_tool_names

ROOT = Path(__file__).resolve().parent
MODELS = {"7b": "TinyAgent 7B · 4-bit", "1.1b": "TinyAgent 1.1B · 4-bit",
          "1.1b-f16": "TinyAgent 1.1B · 16-bit"}
RUNS = ROOT / "runtime" / "ui-runs"
RUNS.mkdir(parents=True, exist_ok=True)


def served_models():
    try:
        with urllib.request.urlopen("http://127.0.0.1:8080/v1/models", timeout=2) as response:
            return {item["id"] for item in json.load(response)["data"]}
    except (OSError, ValueError, KeyError):
        return set()


async def stop_process(process):
    if process is None or process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        await asyncio.wait_for(process.wait(), timeout=5)
    except ProcessLookupError:
        pass
    except asyncio.TimeoutError:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()


class ModelServer:
    def __init__(self):
        self.process = None
        self.model = None

    async def ensure(self, model):
        alias = f"tinyagent-{model}"
        existing = await asyncio.to_thread(served_models)
        if alias in existing:
            return
        if self.process is not None:
            await self.stop()
        elif existing:
            raise RuntimeError("Another model server is using port 8080. Stop it before switching models here.")
        with socket.socket() as connection:
            if connection.connect_ex(("127.0.0.1", 8080)) == 0:
                raise RuntimeError("Port 8080 is in use by another app. Stop that app's server and try again.")
        with (ROOT / "runtime" / "ui-model-server.log").open("a") as log:
            self.process = await asyncio.create_subprocess_exec(
                str(ROOT / "scripts/serve.sh"), model, cwd=ROOT,
                stdout=log, stderr=asyncio.subprocess.STDOUT, start_new_session=True)
        self.model = model
        for _ in range(120):
            if self.process.returncode is not None:
                raise RuntimeError("The model could not start. Details are in runtime/ui-model-server.log.")
            if alias in await asyncio.to_thread(served_models):
                return
            await asyncio.sleep(0.5)
        await self.stop()
        raise RuntimeError("The model took too long to load. Try again.")

    async def stop(self):
        await stop_process(self.process)
        self.process = None
        self.model = None


model_server = ModelServer()
job_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app):
    yield
    await model_server.stop()


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])


@app.middleware("http")
async def local_requests(request: Request, call_next):
    if request.method == "POST":
        origin = request.headers.get("origin")
        if origin and origin not in {f"http://127.0.0.1:{UI_PORT}", f"http://localhost:{UI_PORT}"}:
            return JSONResponse({"detail": "Requests must come from the local TinyAgent window."}, status_code=403)
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Frame-Options"] = "DENY"
    return response


@app.get("/")
async def index():
    return FileResponse(ROOT / "ui/index.html")


@app.get("/api/status")
async def status():
    return {"app": "tinyagent-local", "models": MODELS, "busy": job_lock.locked(),
            "loaded_model": model_server.model,
            "enabled_apps": ENABLED_APPS, "enabled_tools": sorted(ENABLED_TOOLS),
            "demo_files": {"workshop": str(ROOT / "demo-files/Workshop brief.pdf"),
                           "itinerary": str(ROOT / "demo-files/Museum itinerary.pdf")}}


class Prompt(BaseModel):
    prompt: str = Field(min_length=1, max_length=16000)
    model: Literal["7b", "1.1b", "1.1b-f16"] = "7b"
    mode: Literal["preview", "execute"] = "preview"


@app.post("/api/run")
async def run_task(task: Prompt):
    if not task.prompt.strip():
        raise HTTPException(422, "Enter a prompt first.")
    if job_lock.locked():
        raise HTTPException(409, "An agent is already working. Wait for it to finish.")
    await job_lock.acquire()

    async def events():
        process = None
        run_id = uuid.uuid4().hex
        metadata = {"run_id": run_id, "model": task.model, "mode": task.mode,
                    "selected_tools": sorted(select_tool_names(task.prompt)),
                    "prompt": task.prompt, "started_at": datetime.datetime.now().astimezone().isoformat(),
                    "status": "running"}
        metadata_path = RUNS / f"{run_id}.json"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        def event(kind, **values):
            return json.dumps({"type": kind, **values}) + "\n"
        try:
            yield event("status", text=f"Loading {MODELS[task.model]}…")
            await model_server.ensure(task.model)
            command = [str(ROOT / "scripts/demo.sh"), "--model", task.model, "--retrieve-examples"]
            if task.mode == "execute":
                command.append("--execute")
            command.extend(["--", task.prompt])
            env = dict(os.environ, PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       TINYAGENT_TRACE_PATH=str(RUNS / f"{run_id}.execution.json"),
                       TINYAGENT_PLAN_PATH=str(RUNS / f"{run_id}.plan.log"))
            yield event("status", text="Running tools…" if task.mode == "execute" else "Generating a plan…")
            with (RUNS / f"{run_id}.stderr.log").open("w") as errors, (RUNS / f"{run_id}.log").open("w") as transcript:
                process = await asyncio.create_subprocess_exec(*command, cwd=ROOT, env=env,
                    stdout=asyncio.subprocess.PIPE, stderr=errors, start_new_session=True)
                decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
                async with asyncio.timeout(240):
                    while chunk := await process.stdout.read(4096):
                        text = decoder.decode(chunk)
                        transcript.write(text)
                        transcript.flush()
                        yield event("output", text=text)
                    result = await process.wait()
                metadata.update(status="completed" if result == 0 else "failed", exit_code=result)
                if result:
                    detail = (RUNS / f"{run_id}.stderr.log").read_text()
                    lines = [line for line in detail.splitlines() if line.startswith("Demo error:")]
                    error = lines[-1] if lines else (detail.strip().splitlines()[-1] if "Traceback" in detail else "The task failed. The app saved its error log in runtime/ui-runs.")
                    metadata["error"] = error
                    yield event("error", text=error)
                yield event("done", ok=result == 0, run_id=run_id)
        except asyncio.TimeoutError:
            metadata.update(status="timed_out", error="Task timed out")
            yield event("error", text="The task timed out. Some earlier actions may already have completed.")
            yield event("done", ok=False, run_id=run_id)
        except Exception as error:
            metadata.update(status="failed", error=str(error))
            yield event("error", text=str(error))
            yield event("done", ok=False, run_id=run_id)
        finally:
            try:
                await stop_process(process)
            finally:
                job_lock.release()
                if metadata["status"] == "running":
                    metadata["status"] = "interrupted"
                metadata["finished_at"] = datetime.datetime.now().astimezone().isoformat()
                metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    return StreamingResponse(events(), media_type="application/x-ndjson")


UI_PORT = 8765


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--open", action="store_true", help="Open the interface in your browser")
    args = parser.parse_args()
    url = f"http://127.0.0.1:{UI_PORT}"
    # Reopening the launcher reuses the app that's already running.
    try:
        with urllib.request.urlopen(url + "/api/status", timeout=1) as response:
            existing = json.load(response)
        if existing.get("app") == "tinyagent-local":
            if args.open:
                webbrowser.open(url)
            raise SystemExit(0)
    except (OSError, ValueError):
        pass
    if args.open:
        import threading
        threading.Timer(1, lambda: webbrowser.open(url)).start()
    print(f"TinyAgent is available at {url}. Close this window or press Ctrl-C to stop it.", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=UI_PORT, log_level="warning", timeout_graceful_shutdown=5)
