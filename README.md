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

## 🚀 Getting Started

### Prerequisites
*   **Python 3.11+**
*   **Ollama** (for local LLM) - [Download](https://ollama.ai)
*   **Docker Desktop** (optional, for containerized deployment)

### Quick Start (Development Mode)

```bash
# 1. Clone and setup
git clone https://github.com/divyamohan1993/resha.git
cd resha
pip install -r requirements.txt

# 2. Install Ollama and pull a model
# Download Ollama from https://ollama.ai
ollama serve                  # Start Ollama service
ollama pull phi3:mini         # Pull recommended model (~2GB)

# 3. Configure environment
cp .env.example .env
# Edit .env: Set DEVELOPMENT_MODE=true

# 4. Start the server
python run_server.py

# 5. Open http://localhost:8000
```

### Docker Deployment (Production)

```powershell
# Windows (PowerShell)
.\start.ps1
```

```bash
# Linux / WSL
./start.sh
```

## 📡 API Endpoints

### Core Analysis
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/analyze` | POST | SBERT-based semantic analysis |
| `/api/shortlist` | POST | LLM-based shortlisting (Gemini) |
| `/api/analyze-file` | POST | File upload (PDF/DOCX/TXT) |

### Hybrid AI (Development Mode)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/dev/hybrid-analyze` | POST | Multi-model consensus |
| `/api/dev/hybrid-stream` | POST | Real-time streaming CoT |
| `/api/dev/local-llm` | POST | Local LLM only |
| `/api/dev/status` | GET | System health check |

### Example Request

```bash
curl -X POST http://localhost:8000/api/dev/hybrid-analyze \
  -H "Content-Type: application/json" \
  -d '{
    "resume_text": "John Smith, Senior Python Developer...",
    "job_description": "Looking for 5+ years Python..."
  }'
```

## 🤖 Supported Local Models

| Model | Size | Speed | Quality | Command |
|-------|------|-------|---------|---------|
| **Phi-3 Mini** | ~2GB | ★★★★☆ | ★★★★★ | `ollama pull phi3:mini` |
| **Qwen2.5** | ~2GB | ★★★★☆ | ★★★★☆ | `ollama pull qwen2.5:3b` |
| **Gemma 2B** | ~1.5GB | ★★★★★ | ★★★☆☆ | `ollama pull gemma:2b` |
| **TinyLlama** | ~700MB | ★★★★★ | ★★☆☆☆ | `ollama pull tinyllama` |

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

## 📁 Deliverables

All Task B deliverables are in the `deliverables/` folder:

| File | Description |
|------|-------------|
| `deliverables/Full_Stack/resume_shortlisting_agent.py` | Complete standalone agent code |
| `deliverables/Full_Stack/agent_reasoning_loop_output.json` | Sample reasoning loop output |
| `deliverables/Full_Stack/AGENT_ARCHITECTURE.md` | Detailed architecture documentation |

## 🔐 Environment Variables

```env
# Required for production
API_KEY=your-secure-api-key

# Optional: Cloud LLM
GEMINI_API_KEY=your-gemini-key

# Local LLM (Ollama)
OLLAMA_BASE_URL=http://localhost:11434
LOCAL_LLM_ENABLED=true

# Development
DEVELOPMENT_MODE=true
DEBUG=false
```

## 📈 Performance

| Mode | Latency | Accuracy | GPU Required |
|------|---------|----------|--------------|
| Onboard Only | ~50ms | ~80% | ❌ |
| Local LLM | ~3-5s | ~90% | ❌ |
| Cloud LLM | ~1-3s | ~95% | ❌ |
| Consensus | ~4-6s | ~95%+ | ❌ |

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Test Task B functionality
python tests/test_task_b_verification.py
```

## 📄 License

MIT License - See LICENSE file for details.

---

**Built with ❤️ by [dmj.one](https://dmj.one)**
