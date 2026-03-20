import os
from argparse import ArgumentParser
from typing import Any, Optional


def _flatten_config(config: dict, parent_key: str = "", sep: str = "_") -> dict:
    """Flatten nested config dict into flat dict with underscore-separated keys."""
    items = {}
    for k, v in config.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_config(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def _load_config_file(config_path: Optional[str]) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, looks for default locations.

    Returns:
        Flattened config dict with CLI-compatible keys.
    """
    if config_path is None:
        # Look for default config locations
        default_paths = [
            "config.yaml",
            "config.yml",
            os.path.expanduser("~/.config/wlk/config.yaml"),
            os.path.expanduser("~/.config/wlk/config.yml"),
        ]
        for path in default_paths:
            if os.path.isfile(path):
                config_path = path
                break

    if config_path is None or not os.path.isfile(config_path):
        return {}

    try:
        import yaml

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if config is None:
            return {}
        return _flatten_config(config)
    except ImportError:
        print("Warning: PyYAML is not installed. Config file will be ignored.")
        print("Install it with: pip install pyyaml")
        return {}
    except Exception as e:
        print(f"Warning: Failed to load config file '{config_path}': {e}")
        return {}


def _apply_config_defaults(parser: ArgumentParser, config: dict) -> ArgumentParser:
    """Apply config values as defaults for arguments not specified on CLI.

    This modifies the parser's defaults in-place.
    """
    for action in parser._actions:
        if action.dest in config:
            # Only set default if it's not the help action
            if action.dest != "help":
                action.default = config[action.dest]
    return parser


def parse_args():
    parser = ArgumentParser(
        description="Whisper FastAPI Online Server",
        epilog="""
Configuration file:
  Use --config to specify a YAML configuration file.
  Priority: CLI arguments > config file > built-in defaults
  
  Example config.yaml:
    server:
      host: "0.0.0.0"
      port: 8080
    transcription:
      backend: "faster-whisper"
      model_size: "large-v3"
        """,
    )

    # Config file argument (processed first)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file. CLI arguments override config file values.",
    )

    # Server arguments
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="The host address to bind the server to.",
    )
    parser.add_argument("--port", type=int, default=8000, help="The port number to bind the server to.")
    parser.add_argument(
        "--warmup-file",
        type=str,
        default=None,
        dest="warmup_file",
        help="""
        The path to a speech audio wav file to warm up Whisper so that the very first chunk processing is fast.
        If not set, uses https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav.
        If empty, no warmup is performed.
        """,
    )

    parser.add_argument(
        "--confidence-validation",
        action="store_true",
        help="Accelerates validation of tokens using confidence scores. Transcription will be faster but punctuation might be less accurate.",
    )

    parser.add_argument(
        "--diarization",
        action="store_true",
        default=False,
        help="Enable speaker diarization.",
    )

    parser.add_argument(
        "--punctuation-split",
        action="store_true",
        default=False,
        help="Use punctuation marks from transcription to improve speaker boundary detection. Requires both transcription and diarization to be enabled.",
    )

    parser.add_argument(
        "--segmentation-model",
        type=str,
        default="pyannote/segmentation-3.0",
        help="Hugging Face model ID for pyannote.audio segmentation model.",
    )

    parser.add_argument(
        "--embedding-model",
        type=str,
        default="pyannote/embedding",
        help="Hugging Face model ID for pyannote.audio embedding model.",
    )

    parser.add_argument(
        "--diarization-backend",
        type=str,
        default="sortformer",
        choices=["sortformer", "diart"],
        help="The diarization backend to use.",
    )

    parser.add_argument(
        "--no-transcription",
        action="store_true",
        help="Disable transcription to only see live diarization results.",
    )

    parser.add_argument(
        "--disable-punctuation-split",
        action="store_true",
        help="Disable the split parameter.",
    )

    parser.add_argument(
        "--min-chunk-size",
        type=float,
        default=0.1,
        help="Minimum audio chunk size in seconds. It waits up to this time to do processing. If the processing takes shorter time, it waits, otherwise it processes the whole segment that was received by this time.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="base",
        dest="model_size",
        help="Name size of the Whisper model to use (default: tiny). Suggested values: tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v1,large-v2,large-v3,large,large-v3-turbo. The model is automatically downloaded from the model hub if not present in model cache dir.",
    )

    parser.add_argument(
        "--model_cache_dir",
        type=str,
        default=None,
        help="Overriding the default model cache dir where models downloaded from the hub are saved",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Dir where Whisper model.bin and other files are saved. This option overrides --model and --model_cache_dir parameter.",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        dest="lora_path",
        help="Path or Hugging Face repo ID for LoRA adapter weights (e.g., QuentinFuxa/whisper-base-french-lora). Only works with native Whisper backend.",
    )
    parser.add_argument(
        "--lan",
        "--language",
        type=str,
        default="auto",
        dest="lan",
        help="Source language code, e.g. en,de,cs, or 'auto' for language detection.",
    )
    parser.add_argument(
        "--direct-english-translation",
        action="store_true",
        default=False,
        help="Use Whisper to directly translate to english.",
    )

    parser.add_argument(
        "--target-language",
        type=str,
        default="",
        dest="target_language",
        help="Target language for translation. Not functional yet.",
    )

    parser.add_argument(
        "--backend-policy",
        type=str,
        default="simulstreaming",
        choices=["1", "2", "simulstreaming", "localagreement"],
        help="Select the streaming policy: 1 or 'simulstreaming' for AlignAtt, 2 or 'localagreement' for LocalAgreement.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=[
            "auto",
            "mlx-whisper",
            "faster-whisper",
            "whisper",
            "openai-api",
            "voxtral",
            "voxtral-mlx",
            "qwen3",
            "qwen3-mlx",
            "qwen3-mlx-simul",
            "qwen3-simul",
            "vllm-realtime",
        ],
        help="Select the ASR backend implementation. Use 'qwen3-mlx-simul' for Qwen3-ASR SimulStreaming on Apple Silicon (MLX). Use 'qwen3-mlx' for Qwen3-ASR LocalAgreement on MLX. Use 'qwen3-simul' for Qwen3-ASR SimulStreaming (PyTorch). Use 'vllm-realtime' for vLLM Realtime WebSocket.",
    )
    parser.add_argument(
        "--no-vac",
        action="store_true",
        default=False,
        help="Disable VAC = voice activity controller.",
    )
    parser.add_argument("--vac-chunk-size", type=float, default=0.04, help="VAC sample size in seconds.")

    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable VAD (voice activity detection).",
    )

    parser.add_argument(
        "--buffer_trimming",
        type=str,
        default="segment",
        choices=["sentence", "segment"],
        help='Buffer trimming strategy -- trim completed sentences marked with punctuation mark and detected by sentence segmenter, or the completed segments returned by Whisper. Sentence segmenter must be installed for "sentence" option.',
    )
    parser.add_argument(
        "--buffer_trimming_sec",
        type=float,
        default=15,
        help="Buffer trimming length threshold in seconds. If buffer length is longer, trimming sentence/segment is triggered.",
    )
    parser.add_argument(
        "-l",
        "--log-level",
        dest="log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the log level",
        default="DEBUG",
    )
    parser.add_argument("--ssl-certfile", type=str, help="Path to the SSL certificate file.", default=None)
    parser.add_argument("--ssl-keyfile", type=str, help="Path to the SSL private key file.", default=None)
    parser.add_argument("--forwarded-allow-ips", type=str, help="Allowed ips for reverse proxying.", default=None)
    parser.add_argument(
        "--pcm-input",
        action="store_true",
        default=False,
        help="If set, raw PCM (s16le) data is expected as input and FFmpeg will be bypassed. Frontend will use AudioWorklet instead of MediaRecorder.",
    )
    # vLLM Realtime backend arguments
    parser.add_argument(
        "--vllm-url",
        type=str,
        default="ws://localhost:8000/v1/realtime",
        dest="vllm_url",
        help="URL of the vLLM realtime WebSocket endpoint.",
    )
    parser.add_argument(
        "--vllm-model",
        type=str,
        default="",
        dest="vllm_model",
        help="Model name to use with vLLM (e.g. Qwen/Qwen3-ASR-1.7B).",
    )

    # SimulStreaming-specific arguments
    simulstreaming_group = parser.add_argument_group(
        "SimulStreaming arguments (only used with --backend simulstreaming)"
    )

    simulstreaming_group.add_argument(
        "--disable-fast-encoder",
        action="store_true",
        default=False,
        dest="disable_fast_encoder",
        help="Disable Faster Whisper or MLX Whisper backends for encoding (if installed). Slower but helpful when GPU memory is limited",
    )

    simulstreaming_group.add_argument(
        "--custom-alignment-heads",
        type=str,
        default=None,
        help="Use your own alignment heads, useful when `--model-dir` is used",
    )

    simulstreaming_group.add_argument(
        "--frame-threshold",
        type=int,
        default=25,
        dest="frame_threshold",
        help="Threshold for the attention-guided decoding. The AlignAtt policy will decode only until this number of frames from the end of audio. In frames: one frame is 0.02 seconds for large-v3 model.",
    )

    simulstreaming_group.add_argument(
        "--beams",
        "-b",
        type=int,
        default=1,
        help="Number of beams for beam search decoding. If 1, GreedyDecoder is used.",
    )

    simulstreaming_group.add_argument(
        "--decoder",
        type=str,
        default=None,
        dest="decoder_type",
        choices=["beam", "greedy"],
        help="Override automatic selection of beam or greedy decoder. If beams > 1 and greedy: invalid.",
    )

    simulstreaming_group.add_argument(
        "--audio-max-len",
        type=float,
        default=30.0,
        dest="audio_max_len",
        help="Max length of the audio buffer, in seconds.",
    )

    simulstreaming_group.add_argument(
        "--audio-min-len",
        type=float,
        default=0.0,
        dest="audio_min_len",
        help="Skip processing if the audio buffer is shorter than this length, in seconds. Useful when the --min-chunk-size is small.",
    )

    simulstreaming_group.add_argument(
        "--cif-ckpt-path",
        type=str,
        default=None,
        dest="cif_ckpt_path",
        help="The file path to the Simul-Whisper's CIF model checkpoint that detects whether there is end of word at the end of the chunk. If not, the last decoded space-separated word is truncated because it is often wrong -- transcribing a word in the middle. The CIF model adapted for the Whisper model version should be used. Find the models in https://github.com/backspacetg/simul_whisper/tree/main/cif_models . Note that there is no model for large-v3.",
    )

    simulstreaming_group.add_argument(
        "--never-fire",
        action="store_true",
        default=False,
        dest="never_fire",
        help="Override the CIF model. If True, the last word is NEVER truncated, no matter what the CIF model detects. If False: if CIF model path is set, the last word is SOMETIMES truncated, depending on the CIF detection. Otherwise, if the CIF model path is not set, the last word is ALWAYS trimmed.",
    )

    simulstreaming_group.add_argument(
        "--init-prompt",
        type=str,
        default=None,
        dest="init_prompt",
        help="Init prompt for the model. It should be in the target language.",
    )

    simulstreaming_group.add_argument(
        "--static-init-prompt",
        type=str,
        default=None,
        dest="static_init_prompt",
        help="Do not scroll over this text. It can contain terminology that should be relevant over all document.",
    )

    simulstreaming_group.add_argument(
        "--max-context-tokens",
        type=int,
        default=None,
        dest="max_context_tokens",
        help="Max context tokens for the model. Default is 0.",
    )

    simulstreaming_group.add_argument(
        "--model-path",
        type=str,
        default=None,
        dest="model_path",
        help="Direct path to the SimulStreaming Whisper .pt model file. Overrides --model for SimulStreaming backend.",
    )

    simulstreaming_group.add_argument(
        "--nllb-backend",
        type=str,
        default="transformers",
        help="transformers or ctranslate2",
    )

    simulstreaming_group.add_argument(
        "--nllb-size",
        type=str,
        default="600M",
        help="600M or 1.3B",
    )

    # LLM Summary arguments (for meeting recorder)
    llm_group = parser.add_argument_group("LLM Summary arguments (for meeting recorder)")

    llm_group.add_argument(
        "--llm-summary-enabled",
        action="store_true",
        default=False,
        dest="llm_summary_enabled",
        help="Enable LLM-based summarization for meeting recorder.",
    )

    llm_group.add_argument(
        "--llm-api-url",
        type=str,
        default="http://localhost:11434/v1",
        dest="llm_api_url",
        help="LLM API URL (Ollama, OpenAI, DeepSeek, etc.).",
    )

    llm_group.add_argument(
        "--llm-api-key",
        type=str,
        default="",
        dest="llm_api_key",
        help="API key for LLM authentication.",
    )

    llm_group.add_argument(
        "--llm-model",
        type=str,
        default="llama3.2",
        dest="llm_model",
        help="Model name for summarization.",
    )

    llm_group.add_argument(
        "--llm-timeout",
        type=float,
        default=5.0,
        dest="llm_timeout",
        help="Request timeout in seconds.",
    )

    llm_group.add_argument(
        "--llm-max-tokens",
        type=int,
        default=100,
        dest="llm_max_tokens",
        help="Maximum tokens for summary.",
    )

    llm_group.add_argument(
        "--llm-temperature",
        type=float,
        default=0.3,
        dest="llm_temperature",
        help="Temperature for generation.",
    )

    llm_group.add_argument(
        "--summary-template",
        type=str,
        default="meeting_minutes",
        dest="summary_template",
        help="Summary template ID (meeting_minutes, interview, general).",
    )

    llm_group.add_argument(
        "--summary-min-tokens",
        type=int,
        default=5,
        dest="summary_min_tokens",
        help="Minimum tokens before summarizing.",
    )

    # Parse known args first to get --config
    args, unknown = parser.parse_known_args()

    # Load config file
    config_path = args.config
    config = _load_config_file(config_path)

    # Apply config defaults to parser
    if config:
        _apply_config_defaults(parser, config)

    # Parse all args again with config defaults applied
    args = parser.parse_args()

    # Handle boolean flags from config
    # Config file may have "enabled: true" but argparse expects --flag
    # We need to map config boolean keys to argparse flag names
    bool_flags = {
        "diarization": "diarization",
        "punctuation_split": "punctuation_split",
        "confidence_validation": "confidence_validation",
        "no_transcription": "no_transcription",
        "disable_punctuation_split": "disable_punctuation_split",
        "no_vac": "no_vac",
        "no_vad": "no_vad",
        "pcm_input": "pcm_input",
        "disable_fast_encoder": "disable_fast_encoder",
        "never_fire": "never_fire",
        "direct_english_translation": "direct_english_translation",
        "llm_summary_enabled": "llm_summary_enabled",
    }

    # For boolean flags in config, convert to argparse format
    for config_key, arg_attr in bool_flags.items():
        if config_key in config:
            val = config[config_key]
            if isinstance(val, bool):
                setattr(args, arg_attr, val)

    args.transcription = not args.no_transcription
    args.vad = not args.no_vad
    args.vac = not args.no_vac
    delattr(args, "no_transcription")
    delattr(args, "no_vad")
    delattr(args, "no_vac")

    from whisperlivekit.config import WhisperLiveKitConfig

    return WhisperLiveKitConfig.from_namespace(args)
