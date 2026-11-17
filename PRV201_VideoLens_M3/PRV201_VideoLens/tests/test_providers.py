"""Contract tests with fake SDK clients; never evidence of live provider execution."""
import sys
from types import SimpleNamespace as NS
import pytest
from videolens.providers import summarize

@pytest.mark.parametrize("provider", ["claude", "openai", "azure", "bedrock"])
def test_provider_contract(monkeypatch, provider):
    for name in ["ANTHROPIC_API_KEY", "CLAUDE_MODEL", "OPENAI_API_KEY", "OPENAI_MODEL",
                 "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_BASE_URL", "AZURE_OPENAI_DEPLOYMENT",
                 "AWS_REGION", "BEDROCK_MODEL_ID"]:
        monkeypatch.setenv(name, "TEST_ONLY")
    calls = []
    def responses(**kwargs):
        calls.append(kwargs)
        return NS(output_text="Synthetic data; association only.")
    def messages(**kwargs):
        calls.append(kwargs)
        return NS(content=[NS(type="text", text="Synthetic data; association only.")])
    monkeypatch.setitem(sys.modules, "openai", NS(OpenAI=lambda **k:NS(responses=NS(create=responses))))
    monkeypatch.setitem(sys.modules, "anthropic", NS(Anthropic=lambda **k:NS(messages=NS(create=messages))))
    def converse(**kwargs):
        calls.append(kwargs)
        return {"output":{"message":{"content":[{"text":"Synthetic data; association only."}]}}}
    monkeypatch.setitem(sys.modules, "boto3", NS(Session=lambda **k:NS(client=lambda *a,**k:NS(converse=converse))))
    monkeypatch.setitem(sys.modules, "botocore", NS())
    monkeypatch.setitem(sys.modules, "botocore.config", NS(Config=lambda **k:k))
    result = summarize(provider, {"contains_synthetic_data":True, "video_count":60})
    assert result["status"] == "executed"
    assert result["provider"] == provider
    assert "Synthetic" in result["text"]
    assert len(calls) == 1

def test_missing_key_is_explicit(monkeypatch):
    from videolens.providers import required
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        required("OPENAI_API_KEY")
