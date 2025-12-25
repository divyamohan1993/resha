import pytest
import asyncio
from src.app.services.local_llm import LocalLLMService

@pytest.mark.asyncio
async def test_ollama_integration_verification():
    """
    Verification script to ensure Ollama is correctly integrated 
    and generating dynamic chain-of-thought tokens.
    
    This test verifies:
    1. Ollama service connectivity
    2. Model availability
    3. Real-time streaming token generation (Chain of Thoughts)
    """
    print("\n[VERIFICATION] Starting Ollama Integration Check...")
    
    # 1. Initialize Service
    service = LocalLLMService()
    is_available = await service.initialize()
    
    if not is_available:
        pytest.skip("Ollama service not running or no models available. Skipping verification.")
        return

    print(f"[VERIFICATION] Service connected. Model: {service.model_name}")
    assert service.is_available, "Service should be marked available after initialization"

    # 2. Test Streaming (Chain of Thoughts)
    # We use a very simple prompt to ensure quick execution
    resume_text = "Experienced Python Developer with 5 years in AWS."
    jd_text = "Looking for Python expert with cloud experience."
    
    print("[VERIFICATION] Triggering Chain-of-Thought Stream...")
    
    token_count = 0
    received_content = ""
    phases_received = set()
    
    # Consume the stream
    async for event in service.stream_chain_of_thought(resume_text, jd_text):
        phase = event.get("phase")
        phases_received.add(phase)
        
        if phase == "streaming":
            token = event.get("token", "")
            if token:
                token_count += 1
                received_content += token
                # Print first few tokens to prove potential dynamic generation to console logs
                if token_count <= 5:
                    print(f"   -> Token {token_count}: {repr(token)}")
        
        # Stop early if we have enough proof of life to save time
        if token_count > 20:
             break

    # 3. Verify Results
    print(f"[VERIFICATION] Stream stopped. Total tokens received: {token_count}")
    print(f"[VERIFICATION] Phases seen: {phases_received}")
    
    # Assertions
    assert "init" in phases_received, "Must receive initialization phase"
    assert "streaming" in phases_received, "Must receive streaming phase (proof of CoT)"
    assert token_count > 5, "Should successfully stream multiple tokens"
    assert len(received_content) > 0, "Content should be generated"
    
    print("[SUCCESS] Ollama Chain-of-Thought verified successfully.")
