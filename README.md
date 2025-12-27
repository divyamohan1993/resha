# Resha (The Fine Line)

**Brought to you by [dmj.one](https://dmj.one)**

**Version 3.0.0** - Production-Ready Hybrid Multi-Model Architecture

"Resha" (Hindi for "Fiber" or "Fine Line") serves as the fine line between a candidate's acceptance and rejection. It checks resumes dynamically, ensuring precise and fair shortlisting.

A production-grade, AI-powered microservice for analyzing resumes against job descriptions using advanced NLP techniques and modern lightweight LLMs.

## 🌟 Key Features

### Resha AI Architecture
*   **Hybrid Multi-Model**: 3-tier inference (Onboard AI → Local LLM → Cloud LLM)
*   **Real Chain-of-Thought**: Genuine streaming from LLM, not simulated steps
*   **Offline Capable**: Local LLM via Ollama (CPU-only, no GPU required)
*   **Multi-Model Consensus**: Weighted voting for higher accuracy

### Technical Capabilities
*   **Semantic Analysis**: SBERT (all-MiniLM-L6-v2) for semantic matching
*   **Entity Extraction**: Spacy-based extraction of Skills, Experience, Education
*   **Local LLM**: Ollama integration (Phi-3, Qwen, Gemma models)
*   **Cloud LLM**: Google Gemini fallback
*   **File Support**: PDF, DOCX, TXT

### Security & Compliance
*   **Data Privacy**: Local inference keeps data on-device
*   **API Key Auth**: Production-grade authentication
*   **Audit Trail**: SQLite-based decision logging
*   **Strict Security Headers**: CSP, HSTS, X-Frame-Options

---

## 🚀 One-Click Deployment (Production)

### Deploy to VM (reas.dmj.one/task2/)

```bash
# 1. Clone the repository
git clone https://github.com/divyamohan1993/resha.git
cd resha

# 2. Run the deployment script
sudo bash run_project.sh
```

This will:
- ✅ Install all system dependencies (Python, nginx, etc.)
- ✅ Create Python virtual environment
- ✅ Install Python dependencies (FastAPI, PyTorch CPU, spaCy, etc.)
- ✅ Configure environment with secure auto-generated credentials
- ✅ Create and start systemd service (port 22000)
- ✅ Configure nginx for `/task2/` path routing
- ✅ NOT interfere with existing `/task1/` service

### After Deployment

| Command | Description |
|---------|-------------|
| `systemctl status resha` | Check service status |
| `journalctl -u resha -f` | View live logs |
| `systemctl restart resha` | Restart service |
| `sudo bash stop_service.sh` | Stop service |

### Access Points

| URL | Description |
|-----|-------------|
| `https://reas.dmj.one/task2/` | Main Application |
| `https://reas.dmj.one/task2/dev.html` | Development Mode (Chain-of-Thought) |
| `https://reas.dmj.one/task2/api/health` | Health Check |

---

## 🛠️ Local Development

### Prerequisites
*   **Python 3.10+**
*   **Ollama** (for local LLM) - [Download](https://ollama.ai)

### Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/divyamohan1993/resha.git
cd resha

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 4. Configure environment
cp .env.example .env
# Edit .env: Set your GEMINI_API_KEY

# 5. Start the server
python run_server.py --port 22000

# 6. Open http://localhost:22000
```

### With Local LLM (Optional)

```bash
# Install Ollama from https://ollama.ai
ollama serve                  # Start Ollama service
ollama pull phi3:mini         # Pull recommended model (~2GB)
```

---

## 📡 API Endpoints

### Core Analysis
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/analyze` | POST | SBERT-based semantic analysis |
| `/api/shortlist` | POST | LLM-based shortlisting (Gemini) |
| `/api/analyze-file` | POST | File upload (PDF/DOCX/TXT) |
| `/api/health` | GET | Health check |
| `/api/history` | GET | Analysis history |

### Development Mode Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/dev/config` | GET | Development mode status |
| `/api/dev/analyze-stream` | POST | Real-time streaming CoT |
| `/api/dev/hybrid-analyze` | POST | Multi-model consensus |
| `/api/dev/status` | GET | System health check |

### Example Request

```bash
curl -X POST https://reas.dmj.one/task2/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "resume_text": "John Smith, Senior Python Developer...",
    "job_description": "Looking for 5+ years Python..."
  }'
```

---

## 🤖 Supported Local Models

| Model | Size | Speed | Quality | Command |
|-------|------|-------|---------| --------|
| **Phi-3 Mini** | ~2GB | ★★★★☆ | ★★★★★ | `ollama pull phi3:mini` |
| **Qwen2.5** | ~2GB | ★★★★☆ | ★★★★☆ | `ollama pull qwen2.5:3b` |
| **Gemma 2B** | ~1.5GB | ★★★★★ | ★★★☆☆ | `ollama pull gemma:2b` |
| **TinyLlama** | ~700MB | ★★★★★ | ★★☆☆☆ | `ollama pull tinyllama` |

### ⚡ Performance Optimizations

Resha includes several optimizations to prevent timeout issues with slow CPU-based Ollama inference:

| Feature | Description |
|---------|-------------|
| **Model Warmup** | Automatically preloads model into RAM on startup |
| **Response Caching** | LRU cache for repeated analyses (instant response) |
| **Extended Timeouts** | 5-minute timeout for slow CPU inference |
| **SSE Streaming** | Disabled buffering for real-time chain-of-thought |

**Warmup & Cache Endpoints:**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/dev/warmup` | POST | Manually warmup LLM model |
| `/api/dev/cache` | GET | View cache status |
| `/api/dev/cache` | DELETE | Clear cache |

**Troubleshooting Slow Responses:**
```bash
# If experiencing timeouts, run the fix script:
sudo bash fix_ollama_timeout.sh

# Or manually warmup the model:
curl -X POST https://reas.dmj.one/task2/api/dev/warmup
```

## 📊 Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                   HYBRID AI ORCHESTRATOR                       │
├───────────────────────────────────────────────────────────────┤
│   TIER 1          TIER 2           TIER 3                     │
│   ┌─────────┐     ┌──────────┐     ┌──────────────┐          │
│   │ Onboard │────▶│ Local    │────▶│ Cloud LLM    │          │
│   │   AI    │     │ LLM      │     │ (Gemini)     │          │
│   │(TF-IDF) │     │(Ollama)  │     │              │          │
│   └─────────┘     └──────────┘     └──────────────┘          │
│       │               │                   │                   │
│       └───────────────┴───────────────────┘                   │
│                       │                                       │
│               ┌───────▼───────┐                              │
│               │   CONSENSUS   │                              │
│               │   (Weighted)  │                              │
│               └───────┬───────┘                              │
│                       │                                       │
│               SHORTLISTED / REJECTED                          │
└───────────────────────────────────────────────────────────────┘
```

---

## � Environment Variables

```env
# Application
APP_NAME="Resha"
PORT=22000
DEVELOPMENT_MODE=true

# Security (auto-generated by run_project.sh)
SECRET_KEY=your-secure-key
API_KEY=your-api-key

# Cloud LLM (Optional)
GEMINI_API_KEY=your-gemini-key

# Local LLM (Ollama)
OLLAMA_BASE_URL=http://localhost:11434
LOCAL_LLM_ENABLED=true
```

---

## 📈 Performance

| Mode | Latency | Accuracy | GPU Required |
|------|---------|----------|--------------|
| Onboard Only | ~50ms | ~80% | ❌ |
| Local LLM | ~3-5s | ~90% | ❌ |
| Cloud LLM | ~1-3s | ~95% | ❌ |
| Consensus | ~4-6s | ~95%+ | ❌ |

---

## 📁 Deliverables

All Task B deliverables are in the `deliverables/` folder:

| File | Description |
|------|-------------|
| `deliverables/Full_Stack/resume_shortlisting_agent.py` | Complete standalone agent code |
| `deliverables/Full_Stack/agent_reasoning_loop_output.json` | Sample reasoning loop output |
| `deliverables/Full_Stack/AGENT_ARCHITECTURE.md` | Detailed architecture documentation |

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Test Task B functionality
python tests/test_task_b_verification.py
```

---

## 📄 License

MIT License - See LICENSE file for details.

---

**Built with ❤️ by [dmj.one](https://dmj.one)**
