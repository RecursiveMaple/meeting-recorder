import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from whisperlivekit import AudioProcessor, TranscriptionEngine, get_inline_ui_html, parse_args
from whisperlivekit.session_store import session_store
from whisperlivekit.summary.templates import get_template, list_templates

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger().setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.getLogger("whisperlivekit.qwen3_asr").setLevel(logging.DEBUG)

config = parse_args()
transcription_engine = None
summary_runtime_config: Dict[str, str] = {
    "template_id": getattr(config, "summary_template", "meeting_minutes"),
    "system_prompt": getattr(config, "summary_system_prompt", "") or "",
    "user_prompt": getattr(config, "summary_user_prompt", "") or "",
    "api_url": getattr(config, "llm_api_url", "http://localhost:11434/v1"),
    "api_key": getattr(config, "llm_api_key", ""),
    "model": getattr(config, "llm_model", "llama3.2"),
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global transcription_engine
    transcription_engine = TranscriptionEngine(config=config)
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def get():
    return HTMLResponse(get_inline_ui_html())


@app.get("/health")
async def health():
    """Health check endpoint."""
    global transcription_engine
    backend = getattr(transcription_engine.config, "backend", "whisper") if transcription_engine else None
    return JSONResponse(
        {
            "status": "ok",
            "backend": backend,
            "ready": transcription_engine is not None,
        }
    )


async def handle_websocket_results(websocket, results_generator, diff_tracker=None):
    """Consumes results from the audio processor and sends them via WebSocket."""
    try:
        async for response in results_generator:
            if diff_tracker is not None:
                await websocket.send_json(diff_tracker.to_message(response))
            else:
                await websocket.send_json(response.to_dict())
        # when the results_generator finishes it means all audio has been processed
        logger.info("Results generator finished. Sending 'ready_to_stop' to client.")
        await websocket.send_json({"type": "ready_to_stop"})
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected while handling results (client likely closed connection).")
    except Exception as e:
        logger.exception(f"Error in WebSocket results handler: {e}")


@app.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket):
    global transcription_engine

    # Read per-session options from query parameters
    session_language = websocket.query_params.get("language", None)
    mode = websocket.query_params.get("mode", "full")

    # Create a session for this WebSocket connection
    summary_template = getattr(config, "summary_template", "meeting_minutes") if config else "meeting_minutes"
    session_id = session_store.create_session(summary_template=summary_template)

    audio_processor = AudioProcessor(
        transcription_engine=transcription_engine,
        language=session_language,
        session_id=session_id,
    )

    # Register the processor for retry functionality
    session_store.register_processor(session_id, audio_processor)

    await websocket.accept()
    logger.info(
        "WebSocket connection opened.%s session_id=%s",
        f" language={session_language}" if session_language else "",
        session_id,
    )
    diff_tracker = None
    if mode == "diff":
        from whisperlivekit.diff_protocol import DiffTracker

        diff_tracker = DiffTracker()
        logger.info("Client requested diff mode")

    try:
        await websocket.send_json(
            {"type": "config", "useAudioWorklet": bool(config.pcm_input), "mode": mode, "session_id": session_id}
        )
    except Exception as e:
        logger.warning(f"Failed to send config to client: {e}")

    results_generator = await audio_processor.create_tasks()
    websocket_task = asyncio.create_task(handle_websocket_results(websocket, results_generator, diff_tracker))

    try:
        while True:
            message = await websocket.receive_bytes()
            await audio_processor.process_audio(message)
    except KeyError as e:
        if "bytes" in str(e):
            logger.warning("Client has closed the connection.")
        else:
            logger.error(f"Unexpected KeyError in websocket_endpoint: {e}", exc_info=True)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client during message receiving loop.")
    except Exception as e:
        logger.error(f"Unexpected error in websocket_endpoint main loop: {e}", exc_info=True)
    finally:
        logger.info("Cleaning up WebSocket endpoint...")
        if not websocket_task.done():
            websocket_task.cancel()
        try:
            await websocket_task
        except asyncio.CancelledError:
            logger.info("WebSocket results handler task was cancelled.")
        except Exception as e:
            logger.warning(f"Exception while awaiting websocket_task completion: {e}")

        await audio_processor.cleanup()
        session_store.unregister_processor(session_id)
        logger.info("WebSocket endpoint cleaned up successfully.")


# ---------------------------------------------------------------------------
# Deepgram-compatible WebSocket API  (/v1/listen)
# ---------------------------------------------------------------------------


@app.websocket("/v1/listen")
async def deepgram_websocket_endpoint(websocket: WebSocket):
    """Deepgram-compatible live transcription WebSocket."""
    global transcription_engine
    from whisperlivekit.deepgram_compat import handle_deepgram_websocket

    await handle_deepgram_websocket(websocket, transcription_engine, config)


# ---------------------------------------------------------------------------
# OpenAI-compatible REST API  (/v1/audio/transcriptions)
# ---------------------------------------------------------------------------


async def _convert_to_pcm(audio_bytes: bytes) -> bytes:
    """Convert any audio format to PCM s16le mono 16kHz using ffmpeg."""
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-i",
        "pipe:0",
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-loglevel",
        "error",
        "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate(input=audio_bytes)
    if proc.returncode != 0:
        from fastapi import HTTPException

        raise HTTPException(status_code=400, detail=f"Audio conversion failed: {stderr.decode().strip()}")
    return stdout


def _parse_time_str(time_str: str) -> float:
    """Parse 'H:MM:SS.cc' to seconds."""
    parts = time_str.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(parts[0])


def _format_openai_response(front_data, response_format: str, language: Optional[str], duration: float) -> dict:
    """Convert FrontData to OpenAI-compatible response."""
    d = front_data.to_dict()
    lines = d.get("lines", [])

    # Combine all speech text (exclude silence segments)
    text_parts = [l["text"] for l in lines if l.get("text") and l.get("speaker", 0) != -2]
    full_text = " ".join(text_parts).strip()

    if response_format == "text":
        return full_text

    # Build segments and words for verbose_json
    segments = []
    words = []
    for i, line in enumerate(lines):
        if line.get("speaker") == -2 or not line.get("text"):
            continue
        start = _parse_time_str(line.get("start", "0:00:00"))
        end = _parse_time_str(line.get("end", "0:00:00"))
        segments.append(
            {
                "id": len(segments),
                "start": round(start, 2),
                "end": round(end, 2),
                "text": line["text"],
            }
        )
        # Split segment text into approximate words with estimated timestamps
        seg_words = line["text"].split()
        if seg_words:
            word_duration = (end - start) / max(len(seg_words), 1)
            for j, word in enumerate(seg_words):
                words.append(
                    {
                        "word": word,
                        "start": round(start + j * word_duration, 2),
                        "end": round(start + (j + 1) * word_duration, 2),
                    }
                )

    if response_format == "verbose_json":
        return {
            "task": "transcribe",
            "language": language or "unknown",
            "duration": round(duration, 2),
            "text": full_text,
            "words": words,
            "segments": segments,
        }

    if response_format in ("srt", "vtt"):
        lines_out = []
        if response_format == "vtt":
            lines_out.append("WEBVTT\n")
        for i, seg in enumerate(segments):
            start_ts = _srt_timestamp(seg["start"], response_format)
            end_ts = _srt_timestamp(seg["end"], response_format)
            if response_format == "srt":
                lines_out.append(f"{i + 1}")
            lines_out.append(f"{start_ts} --> {end_ts}")
            lines_out.append(seg["text"])
            lines_out.append("")
        return "\n".join(lines_out)

    # Default: json
    return {"text": full_text}


def _srt_timestamp(seconds: float, fmt: str) -> str:
    """Format seconds as SRT (HH:MM:SS,mmm) or VTT (HH:MM:SS.mmm) timestamp."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    sep = "," if fmt == "srt" else "."
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


@app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form(default=""),
    language: Optional[str] = Form(default=None),
    prompt: str = Form(default=""),
    response_format: str = Form(default="json"),
    timestamp_granularities: Optional[List[str]] = Form(default=None),
):
    """OpenAI-compatible audio transcription endpoint.

    Accepts the same parameters as OpenAI's /v1/audio/transcriptions API.
    The `model` parameter is accepted but ignored (uses the server's configured backend).
    """
    global transcription_engine

    audio_bytes = await file.read()
    if not audio_bytes:
        from fastapi import HTTPException

        raise HTTPException(status_code=400, detail="Empty audio file")

    # Convert to PCM for pipeline processing
    pcm_data = await _convert_to_pcm(audio_bytes)
    duration = len(pcm_data) / (16000 * 2)  # 16kHz, 16-bit

    # Process through the full pipeline
    processor = AudioProcessor(
        transcription_engine=transcription_engine,
        language=language,
    )
    # Force PCM input regardless of server config
    processor.is_pcm_input = True

    results_gen = await processor.create_tasks()

    # Collect results in background while feeding audio
    final_result = None

    async def collect():
        nonlocal final_result
        async for result in results_gen:
            final_result = result

    collect_task = asyncio.create_task(collect())

    # Feed audio in chunks (1 second each)
    chunk_size = 16000 * 2  # 1 second of PCM
    for i in range(0, len(pcm_data), chunk_size):
        await processor.process_audio(pcm_data[i : i + chunk_size])

    # Signal end of audio
    await processor.process_audio(b"")

    # Wait for pipeline to finish
    try:
        await asyncio.wait_for(collect_task, timeout=120.0)
    except asyncio.TimeoutError:
        logger.warning("Transcription timed out after 120s")
    finally:
        await processor.cleanup()

    if final_result is None:
        return JSONResponse({"text": ""})

    result = _format_openai_response(final_result, response_format, language, duration)

    if isinstance(result, str):
        return PlainTextResponse(result)
    return JSONResponse(result)


@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible model listing endpoint."""
    global transcription_engine
    backend = getattr(transcription_engine.config, "backend", "whisper") if transcription_engine else "whisper"
    model_size = getattr(transcription_engine.config, "model_size", "base") if transcription_engine else "base"
    return JSONResponse(
        {
            "object": "list",
            "data": [
                {
                    "id": f"{backend}/{model_size}" if backend != "whisper" else f"whisper-{model_size}",
                    "object": "model",
                    "owned_by": "whisperlivekit",
                }
            ],
        }
    )


# ---------------------------------------------------------------------------
# JSONL Export API  (/v1/export/jsonl)
# ---------------------------------------------------------------------------


@app.get("/v1/export/jsonl")
async def export_jsonl(session_id: Optional[str] = None):
    """Export transcription data as JSONL format.

    Each line is a JSON object:
    - Session metadata: {"type": "session", ...}
    - Segment: {"type": "segment", ...}
    - Summary: {"type": "summary", ...}

    Args:
        session_id: Optional session ID to export. If not provided, exports all sessions.

    Returns:
        Plain text response with JSONL content.
    """
    jsonl_content = session_store.to_jsonl(session_id)
    return PlainTextResponse(jsonl_content, media_type="application/x-ndjson")


@app.get("/v1/templates")
async def get_templates():
    templates = [template.to_dict() for template in list_templates()]
    return JSONResponse({"templates": templates, "count": len(templates)})


@app.get("/v1/summary/config")
async def get_summary_config():
    template_id = summary_runtime_config.get("template_id") or getattr(config, "summary_template", "meeting_minutes")
    template = get_template(template_id) or get_template("meeting_minutes")
    return JSONResponse(
        {
            "enabled": getattr(config, "llm_summary_enabled", False),
            "template_id": template_id,
            "system_prompt": summary_runtime_config.get("system_prompt")
            or (template.system_prompt if template else ""),
            "user_prompt": summary_runtime_config.get("user_prompt")
            or (template.user_prompt if template else "{{text}}"),
            "api_url": summary_runtime_config.get("api_url", getattr(config, "llm_api_url", "")),
            "api_key": summary_runtime_config.get("api_key", getattr(config, "llm_api_key", "")),
            "model": summary_runtime_config.get("model", getattr(config, "llm_model", "")),
        }
    )


@app.post("/v1/summary/config")
async def save_summary_config(payload: dict = Body(...)):
    template_id = payload.get("template_id") or getattr(config, "summary_template", "meeting_minutes")
    template = get_template(template_id) or get_template("meeting_minutes")

    summary_runtime_config["template_id"] = template_id
    summary_runtime_config["system_prompt"] = payload.get("system_prompt") or (
        template.system_prompt if template else ""
    )
    summary_runtime_config["user_prompt"] = payload.get("user_prompt") or (
        template.user_prompt if template else "{{text}}"
    )
    summary_runtime_config["api_url"] = payload.get("api_url") or getattr(
        config, "llm_api_url", "http://localhost:11434/v1"
    )
    summary_runtime_config["api_key"] = payload.get("api_key") or ""
    summary_runtime_config["model"] = payload.get("model") or getattr(config, "llm_model", "llama3.2")

    config.summary_template = summary_runtime_config["template_id"]
    config.summary_system_prompt = summary_runtime_config["system_prompt"]
    config.summary_user_prompt = summary_runtime_config["user_prompt"]
    config.llm_api_url = summary_runtime_config["api_url"]
    config.llm_api_key = summary_runtime_config["api_key"]
    config.llm_model = summary_runtime_config["model"]
    config.llm_summary_enabled = True

    for session in session_store.get_all_sessions():
        session.summary_template = template_id
        processor = session_store.get_processor(session.id)
        if processor:
            processor.args.llm_summary_enabled = True
            processor.update_summary_runtime_config(
                api_url=summary_runtime_config["api_url"],
                api_key=summary_runtime_config["api_key"],
                model=summary_runtime_config["model"],
                template_id=summary_runtime_config["template_id"],
                system_prompt=summary_runtime_config["system_prompt"],
                user_prompt=summary_runtime_config["user_prompt"],
            )

    return JSONResponse({"status": "saved", **summary_runtime_config})


@app.get("/v1/sessions")
async def list_sessions():
    """List all active sessions."""
    sessions = session_store.get_all_sessions()
    return JSONResponse(
        {
            "sessions": [s.to_dict() for s in sessions],
            "count": len(sessions),
        }
    )


@app.get("/v1/sessions/{session_id}")
async def get_session(session_id: str):
    """Get details of a specific session."""
    session = session_store.get_session(session_id)
    if not session:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    segments = session_store.get_session_segments(session_id)
    return JSONResponse(
        {
            "session": session.to_dict(),
            "segments": [s.to_dict() for s in segments],
        }
    )


# ---------------------------------------------------------------------------
# Summary Retry API  (/v1/summary/retry)
# ---------------------------------------------------------------------------


@app.post("/v1/summary/retry")
async def retry_summary(payload: dict = Body(...)):
    """Retry summary generation for a specific segment.

    Args:
        session_id: The session ID.
        segment_id: The segment ID to retry.

    Returns:
        JSON response with the new summary status.
    """
    session_id = payload.get("session_id")
    segment_id = payload.get("segment_id")
    if not session_id or segment_id is None:
        raise HTTPException(status_code=400, detail="session_id and segment_id are required")

    # Get the segment
    segment = session_store.get_segment(session_id, segment_id)
    if not segment:
        raise HTTPException(status_code=404, detail=f"Segment {segment_id} not found in session {session_id}")

    if not segment.text:
        raise HTTPException(status_code=400, detail="Segment has no text to summarize")

    # Get the processor for this session
    processor = session_store.get_processor(session_id)
    if not processor:
        raise HTTPException(status_code=404, detail=f"No active processor for session {session_id}")

    # Check if LLM is enabled
    if not processor.llm_client or not processor.summary_queue:
        raise HTTPException(status_code=400, detail="Summary generation is not enabled for this session")

    # Reset status and queue for retry
    session_store.update_segment_summary(session_id, segment_id, status="pending")

    # Queue for retry
    await processor.summary_queue.put((segment_id, segment.text))

    return JSONResponse(
        {
            "status": "queued",
            "session_id": session_id,
            "segment_id": segment_id,
        }
    )


def main():
    """Entry point for the CLI command."""
    import uvicorn

    from whisperlivekit.cli import print_banner

    ssl = bool(config.ssl_certfile and config.ssl_keyfile)
    print_banner(config, config.host, config.port, ssl=ssl)

    uvicorn_kwargs = {
        "app": "whisperlivekit.basic_server:app",
        "host": config.host,
        "port": config.port,
        "reload": False,
        "log_level": "info",
        "lifespan": "on",
    }

    ssl_kwargs = {}
    if config.ssl_certfile or config.ssl_keyfile:
        if not (config.ssl_certfile and config.ssl_keyfile):
            raise ValueError("Both --ssl-certfile and --ssl-keyfile must be specified together.")
        ssl_kwargs = {
            "ssl_certfile": config.ssl_certfile,
            "ssl_keyfile": config.ssl_keyfile,
        }

    if ssl_kwargs:
        uvicorn_kwargs = {**uvicorn_kwargs, **ssl_kwargs}
    if config.forwarded_allow_ips:
        uvicorn_kwargs = {**uvicorn_kwargs, "forwarded_allow_ips": config.forwarded_allow_ips}

    uvicorn.run(**uvicorn_kwargs)


if __name__ == "__main__":
    main()
