# Meeting Recorder

基于 [WhisperLiveKit](https://github.com/QuentinFuxa/WhisperLiveKit) 的实时会议记录员工具。

## 功能

- **实时语音转文字**：超低延迟转录，支持多语言
- **说话人区分**：实时识别不同说话人（Speaker 1、Speaker 2...）
- **每句总结**：每句话自动生成摘要，支持自定义模板
- **JSONL 导出**：转录和总结数据导出为 JSONL 格式
- **开箱即用**：无需配置，点击录音即可开始

## 安装

### 前置要求

- Python 3.11-3.13
- FFmpeg（音频处理）

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# 从 https://ffmpeg.org/download.html 下载并添加到 PATH
```

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/RecursiveMaple/meeting-recorder.git
cd meeting-recorder

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或 venv\Scripts\activate  # Windows

# 3. 安装基础依赖
pip install -e .

# 4. 安装说话人分离依赖（可选但推荐）
pip install -e ".[diarization-sortformer]"

# 或者使用更强但更旧的 diart 后端
# pip install -e ".[diarization-diart]"
```

### CPU 环境（无 GPU）

```bash
# 安装 CPU 版本的 PyTorch
pip install -e ".[cpu]"
```

### 验证安装

```bash
# 检查是否安装成功
wlk --help

# 如果提示找不到命令，尝试：
python -m whisperlivekit.cli --help
```

## 快速开始

### 1. 启动服务

```bash
# 基本启动（仅转录）
wlk serve

# 启用说话人分离
wlk serve --diarization

# 启用说话人分离 + LLM 总结
wlk serve --diarization --llm-summary-enabled --llm-api-url http://localhost:11434/v1 --llm-model llama3.2
```

### 2. 打开浏览器

访问 http://localhost:8000

### 3. 开始录音

1. 选择麦克风设备
2. 点击录音按钮
3. 实时查看转录和总结

## 配置选项

### 配置文件方式

支持使用 YAML 配置文件管理所有配置项。配置优先级：**命令行参数 > 配置文件 > 默认值**

```bash
# 使用默认配置文件 (config.yaml)
wlk serve

# 使用自定义配置文件
wlk serve --config /path/to/config.yaml

# 命令行参数会覆盖配置文件中的值
wlk serve --config config.yaml --port 9000
```

配置文件示例 (`config.yaml`)：

```yaml
# 服务器设置
server:
  host: "localhost"
  port: 8000

# 转录设置
transcription:
  backend: "faster-whisper"
  model_size: "base"
  lan: "auto"

# 说话人分离
diarization:
  enabled: true
  backend: "sortformer"

# LLM 总结
llm_summary:
  enabled: true
  api_url: "http://localhost:11434/v1"
  model: "llama3.2"
```

默认配置文件位置（按优先级）：
1. `./config.yaml` 或 `./config.yml`（当前目录）
2. `~/.config/wlk/config.yaml`（用户配置目录）

### 命令行参数

### LLM 总结配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--llm-summary-enabled` | 启用 LLM 总结 | `false` |
| `--llm-api-url` | LLM API 地址 | `http://localhost:11434/v1` |
| `--llm-api-key` | API 密钥 | `""` |
| `--llm-model` | 模型名称 | `llama3.2` |
| `--llm-timeout` | 超时时间（秒） | `5.0` |
| `--llm-max-tokens` | 最大 token 数 | `100` |
| `--llm-temperature` | 温度参数 | `0.3` |
| `--summary-template` | 总结模板 ID | `meeting_minutes` |
| `--summary-min-tokens` | 最小 token 数才触发总结 | `5` |

### 说话人分离配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--diarization` | 启用说话人分离 | `false` |
| `--diarization-backend` | 分离后端 | `sortformer` |

### 转录配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 转录模型大小 | `base` |
| `--backend` | 转录后端 | `whisper` |
| `--language` | 语言代码 | `auto` |

## API 端点

### WebSocket: `/asr`

实时转录流。

### GET `/v1/export/jsonl`

导出转录数据为 JSONL 格式。

```
GET /v1/export/jsonl?session_id=<session_id>
```

每行是一个 JSON 对象：
```json
{"type": "segment", "speaker": 1, "text": "...", "start": "0:00:01.23", "end": "0:00:05.67", "summary": "..."}
```

### GET `/v1/sessions`

列出所有活跃会话。

### POST `/v1/summary/retry`

重试总结生成。

```
POST /v1/summary/retry?session_id=<session_id>&segment_id=<segment_id>
```

## 总结模板

内置模板：
- `meeting_minutes` - 会议纪要（默认）
- `interview` - 面试留档
- `general` - 通用总结

## 使用 Ollama

```bash
# 安装 Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 下载模型
ollama pull llama3.2

# 启动服务
ollama serve

# 另一个终端启动 meeting-recorder
wlk serve --diarization --llm-summary-enabled
```

## 使用云端 API

```bash
# 使用 OpenAI
wlk serve --diarization --llm-summary-enabled \
  --llm-api-url https://api.openai.com/v1 \
  --llm-api-key sk-xxx \
  --llm-model gpt-4o-mini

# 使用 DeepSeek
wlk serve --diarization --llm-summary-enabled \
  --llm-api-url https://api.deepseek.com/v1 \
  --llm-api-key sk-xxx \
  --llm-model deepseek-chat
```

## 故障排除

### 常见问题

**1. `ModuleNotFoundError: No module named 'whisperlivekit'`**
```bash
# 确保在虚拟环境中安装
pip install -e .
```

**2. `ModuleNotFoundError: No module named 'torch'`**
```bash
# 安装 PyTorch
pip install torch torchaudio
# 或使用 CPU 版本
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**3. `ModuleNotFoundError: No module named 'nemo'`（说话人分离）**
```bash
# 安装 NeMo 工具包
pip install -e ".[diarization-sortformer]"
```

**4. `ffmpeg not found`**
```bash
# 安装 FFmpeg（见前置要求）
```

**5. `wlk: command not found`**
```bash
# 使用 Python 模块方式运行
python -m whisperlivekit.cli serve --diarization
```

### 清洁环境测试

```bash
# 创建全新虚拟环境测试
python -m venv test_env
source test_env/bin/activate
pip install -e .
pip install -e ".[diarization-sortformer]"
wlk serve --diarization
```

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest
```

## 许可证

Apache 2.0（继承自 WhisperLiveKit）

## 致谢

- [WhisperLiveKit](https://github.com/QuentinFuxa/WhisperLiveKit) - 基础框架
- [Sortformer](https://arxiv.org/abs/2507.18446) - 说话人分离
- [Whisper](https://github.com/openai/whisper) - 转录模型