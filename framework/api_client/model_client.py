"""
Lightweight API client for agentic misalignment experiments.
Supports Claude, GPT, and Gemini with minimal dependencies.
"""

import asyncio
import os
import time
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import json
import hashlib  # local provider (Agentic-Misalignment-Qwen patch)
import urllib.error  # local provider
import urllib.request  # local provider

# Set up logging
logger = logging.getLogger(__name__)

# Try importing clients, fail gracefully
try:
    from anthropic import AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

try:
    from together import AsyncTogether
    TOGETHER_AVAILABLE = True
except ImportError:
    TOGETHER_AVAILABLE = False

class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

# Model provider mappings
ANTHROPIC_MODELS = {
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229", 
    "claude-3-haiku-20240307",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-latest",
    "claude-3-5-haiku-20241022",
    "claude-3-5-haiku-latest",
    "claude-3-7-sonnet-20250219",
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514"
}

OPENAI_MODELS = {
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0125", 
    "gpt-4",
    "gpt-4-0125-preview",
    "gpt-4-turbo",
    "gpt-4-turbo-2024-04-09",
    "gpt-4o",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-11-20",
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "gpt-4.1",
    "o3",
    "o4-mini-2025-04-16"
}

GEMINI_MODELS = {
    # Gemini 1.5 models
    "gemini-1.5-pro",
    "gemini-1.5-pro-latest",
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash-8b",
    
    # Gemini 2.0 models
    "gemini-2.0-flash",
    "gemini-2.0-flash-exp",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-preview-image-generation",
    "gemini-2.0-flash-live-001",
    
    # Gemini 2.5 models
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.5-flash-preview-native-audio-dialog",
    "gemini-2.5-flash-exp-native-audio-thinking-dialog",
    "gemini-2.5-flash-preview-tts",
    "gemini-2.5-pro-preview-05-06",
    "gemini-2.5-pro-preview-tts"
}

# Together API models
TOGETHER_MODELS = {
    "deepseek-ai/DeepSeek-R1"
}

# OpenRouter API models (includes former Together models)
OPENROUTER_MODELS = {
    # Original OpenRouter models
    "x-ai/grok-3-beta",
    "mistralai/mistral-large-2411",
    "mistral/ministral-8b",
    
    # Gemini models via OpenRouter
    "google/gemini-2.0-flash-exp",
    "google/gemini-2.0-flash",
    "google/gemini-1.5-pro",
    "google/gemini-1.5-flash",
    "google/gemini-1.5-flash-8b",
    "google/gemini-2.5-pro-preview",
    "google/gemini-2.5-pro-experimental",
    "google/gemini-2.5-flash-preview-05-20",
    "google/gemini-2.5-flash-preview-05-20:thinking",
    "google/gemini-2.5-flash-preview-04-17",
    "google/gemini-2.5-flash-preview-04-17:thinking",
    
    # Migrated Llama models (from Together)
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-3.1-70b-instruct",
    "meta-llama/llama-3.1-405b-instruct",
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",  # Together's special version
    "meta-llama/llama-3.3-70b-instruct",
    
    # Additional Llama models (free tier)
    "meta-llama/llama-3.3-8b-instruct:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "meta-llama/llama-4-maverick:free",
    "meta-llama/llama-4-maverick",
    "meta-llama/llama-4-scout:free",
    
    # Migrated Qwen models (from Together)
    "qwen/qwen-2.5-72b-instruct",
    "qwen/qwen-2.5-coder-32b-instruct",
    "qwen/qwq-32b-preview",
    
    # Additional Qwen models (free tier)
    "qwen/qwen3-32b:free",
    "qwen/qwen3-14b:free",
    "qwen/qwen3-8b:free",
    "qwen/qwen3-235b-a22b:free",
    "qwen/qwen3-235b-a22b",
    
    # DeepSeek models on OpenRouter
    "deepseek/deepseek-r1",
    "deepseek/deepseek-r1-zero",
    "deepseek/deepseek-r1-zero:free",
    "deepseek/deepseek-v3",
    "deepseek/deepseek-v3-base",
    "deepseek/deepseek-v3-base:free",
    "deepseek/deepseek-v3-0324",
    "deepseek/deepseek-v2.5",
    "deepseek/deepseek-coder-v2",
    "deepseek/deepseek-prover-v2",
    "deepseek/deepseek-r1-distill-llama-8b",
    "deepseek/deepseek-r1-distill-llama-70b",
    "deepseek/deepseek-r1-distill-qwen-1.5b",
    "deepseek/deepseek-r1-distill-qwen-14b",
    "deepseek/deepseek-r1-distill-qwen-32b",
    "deepseek/deepseek-r1-0528",
}

# Placeholder for future providers
DEEPSEEK_MODELS = set()  # DeepSeek now via Together

# --- Agentic-Misalignment-Qwen patch: local llama.cpp provider -------------
# Model ids "local/<name>" go to a local llama-server through its
# OpenAI-compatible /v1/chat/completions endpoint. Endpoint, token file, the
# fixed sampling values and the GGUF each id must be served from are read from
# the JSON file named by AM_LOCAL_PROVIDER_CONFIG. See the benchmark README.
LOCAL_MODEL_PREFIX = "local/"
LOCAL_SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repeat_penalty")


def is_local_model(model_id: str) -> bool:
    return model_id.startswith(LOCAL_MODEL_PREFIX)
# ---------------------------------------------------------------------------

def get_provider_for_model(model_id: str) -> str:
    """
    Get the API provider for a given model ID.
    
    Args:
        model_id: The model identifier (e.g., "claude-3-5-sonnet-latest", "gpt-4o")
        
    Returns:
        Provider name: "anthropic", "openai", "google", or "together"
        
    Raises:
        ValueError: If model provider cannot be determined
    """
    if is_local_model(model_id):  # local provider patch: before the "/" -> openrouter fallback
        return "local"
    if model_id in ANTHROPIC_MODELS:
        return "anthropic"
    elif model_id in OPENAI_MODELS:
        return "openai"
    elif model_id in GEMINI_MODELS:
        return "google"
    elif model_id in TOGETHER_MODELS:
        return "together"
    elif model_id in OPENROUTER_MODELS:
        return "openrouter"
    else:
        # Check for provider prefixes first (e.g., "google/gemini-*" should go to OpenRouter)
        if "/" in model_id:
            # Models with provider prefixes are typically OpenRouter models
            return "openrouter"
        
        # Fallback to string matching for unknown models
        model_lower = model_id.lower()
        if "claude" in model_lower:
            return "anthropic"
        elif "gpt" in model_lower:
            return "openai"
        elif "gemini" in model_lower:
            return "google"
        elif "grok" in model_lower or "x-ai" in model_lower or "mistral" in model_lower:
            return "openrouter"
        elif "llama" in model_lower or "meta-" in model_lower or "qwen" in model_lower or "deepseek" in model_lower:
            # These are now on OpenRouter
            return "openrouter"
        else:
            raise ValueError(f"Unknown provider for model: {model_id}")

@dataclass
class ChatMessage:
    role: MessageRole
    content: str

@dataclass
class LLMResponse:
    model_id: str
    completion: str
    stop_reason: Optional[str] = None
    cost: Optional[float] = None
    duration: Optional[float] = None
    api_duration: Optional[float] = None
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
    reasoning: Optional[str] = None  # local provider: llama.cpp reasoning_content (thinking)
    extra: Optional[Dict[str, Any]] = None  # local provider: request, echo and identity record

class ModelClient:
    """Lightweight client for multi-model inference."""
    
    def __init__(self):
        self.anthropic_client = None
        self.openai_client = None
        self.together_client = None
        self.openrouter_client = None
        self._setup_clients()
    
    def _setup_clients(self):
        """Initialize API clients with environment variables."""
        if ANTHROPIC_AVAILABLE and os.getenv("ANTHROPIC_API_KEY"):
            self.anthropic_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        if GOOGLE_AVAILABLE and os.getenv("GOOGLE_API_KEY"):
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        
        if TOGETHER_AVAILABLE and os.getenv("TOGETHER_API_KEY"):
            self.together_client = AsyncTogether(api_key=os.getenv("TOGETHER_API_KEY"))
        
        if OPENAI_AVAILABLE and os.getenv("OPENROUTER_API_KEY"):
            self.openrouter_client = AsyncOpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1"
            )
    
    def _detect_provider(self, model_id: str) -> str:
        """Detect the appropriate provider for a given model ID."""
        if is_local_model(model_id):  # local provider patch
            return "local"
        # Check explicit model sets first (exact matches)
        if model_id in ANTHROPIC_MODELS:
            return "anthropic"
        elif model_id in OPENAI_MODELS:
            return "openai"
        elif model_id in GEMINI_MODELS:
            return "gemini"
        elif model_id in TOGETHER_MODELS:
            return "together"
        elif model_id in OPENROUTER_MODELS:
            return "openrouter"
        else:
            # Fallback to legacy string matching for unknown models
            model_lower = model_id.lower()
            if "claude" in model_lower:
                return "anthropic"
            elif "gpt" in model_lower:
                return "openai"
            elif "gemini" in model_lower:
                return "gemini"
            elif "llama" in model_lower or "qwen" in model_lower or "deepseek" in model_lower:
                return "openrouter"
            elif "grok" in model_lower or "mistral" in model_lower:
                return "openrouter"
            else:
                raise ValueError(f"Unsupported model: {model_id}")
    
    def get_supported_models(self) -> Dict[str, List[str]]:
        """Return all supported models organized by provider."""
        return {
            "anthropic": list(ANTHROPIC_MODELS),
            "openai": list(OPENAI_MODELS), 
            "gemini": list(GEMINI_MODELS),
            "together": list(TOGETHER_MODELS),
            "openrouter": list(OPENROUTER_MODELS),
            "deepseek": list(DEEPSEEK_MODELS)
        }
    
    async def __call__(
        self,
        model_id: str,
        messages: List[ChatMessage],
        max_tokens: int = 2000,
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        """Unified interface for all model providers."""
        start_time = time.time()
        
        provider = self._detect_provider(model_id)
        
        if provider == "anthropic":
            response = await self._call_anthropic(model_id, messages, max_tokens, temperature, **kwargs)
        elif provider == "openai":
            response = await self._call_openai(model_id, messages, max_tokens, temperature, **kwargs)
        elif provider == "gemini":
            response = await self._call_gemini(model_id, messages, max_tokens, temperature, **kwargs)
        elif provider == "together":
            response = await self._call_together(model_id, messages, max_tokens, temperature, **kwargs)
        elif provider == "openrouter":
            response = await self._call_openrouter(model_id, messages, max_tokens, temperature, **kwargs)
        elif provider == "local":  # local provider patch
            response = await self._call_local(model_id, messages, max_tokens, temperature, **kwargs)
        elif provider == "deepseek":
            raise NotImplementedError("DeepSeek API support moved to Together provider")
        else:
            raise ValueError(f"Unsupported provider: {provider} for model: {model_id}")
        
        response.duration = time.time() - start_time
        return response
    
    async def _call_anthropic(
        self,
        model_id: str,
        messages: List[ChatMessage],
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> LLMResponse:
        """Call Anthropic Claude API."""
        if not self.anthropic_client:
            raise RuntimeError("Anthropic client not available. Check ANTHROPIC_API_KEY.")
        
        # Separate system and user messages (Anthropic format)
        system_content = ""
        conversation_messages = []
        
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_content += msg.content + "\n"
            else:
                conversation_messages.append({
                    "role": msg.role.value,
                    "content": msg.content
                })
        
        api_start = time.time()
        # Handle timeout to avoid "streaming recommended" warning
        # The Anthropic SDK warns when non-streaming requests might take >10 minutes
        # Setting an explicit timeout <10 minutes disables this warning
        timeout_override = kwargs.pop('timeout', None)
        if timeout_override is None:
            # Use a 5-minute timeout which should be sufficient for most requests
            # and avoids the 10-minute streaming recommendation threshold
            timeout_override = 300.0  # 5 minutes
        
        response = await self.anthropic_client.messages.create(
            model=model_id,
            system=system_content.strip() if system_content else None,
            messages=conversation_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout_override,
            **kwargs
        )
        api_duration = time.time() - api_start
        
        return LLMResponse(
            model_id=model_id,
            completion=response.content[0].text,
            stop_reason=response.stop_reason,
            api_duration=api_duration,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            }
        )
    
    async def _call_openai(
        self,
        model_id: str,
        messages: List[ChatMessage],
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> LLMResponse:
        """Call OpenAI GPT API."""
        if not self.openai_client:
            raise RuntimeError("OpenAI client not available. Check OPENAI_API_KEY.")
        
        # Convert to OpenAI format
        openai_messages = [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
        ]
        
        # Handle O3 model's special parameter requirements
        api_params = {
            "model": model_id,
            "messages": openai_messages,
            **kwargs
        }
        
        # O3 and O4 models have special requirements
        if model_id in ["o3", "o4-mini-2025-04-16"]:
            api_params["max_completion_tokens"] = max_tokens
            # Both O3 and O4-mini only support temperature=1 (default), so omit temperature parameter
        else:
            api_params["max_tokens"] = max_tokens
            api_params["temperature"] = temperature
        
        api_start = time.time()
        response = await self.openai_client.chat.completions.create(**api_params)
        api_duration = time.time() - api_start
        
        choice = response.choices[0]
        return LLMResponse(
            model_id=model_id,
            completion=choice.message.content,
            finish_reason=choice.finish_reason,
            api_duration=api_duration,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        )
    
    async def _call_gemini(
        self,
        model_id: str,
        messages: List[ChatMessage],
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> LLMResponse:
        """Call Google Gemini API."""
        if not GOOGLE_AVAILABLE:
            raise RuntimeError("Google AI not available. Install google-generativeai package.")
        
        # Import safety enums
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        
        # Configure safety settings for research purposes using the working approach
        safety_settings = [
            {
                "category": cat,
                "threshold": HarmBlockThreshold.BLOCK_NONE,
            }
            for cat in HarmCategory
            if cat != HarmCategory.HARM_CATEGORY_UNSPECIFIED
        ]
        
        # Convert messages to Gemini format
        model = genai.GenerativeModel(model_id, safety_settings=safety_settings)
        
        # Combine system and user messages for Gemini
        full_prompt = ""
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                full_prompt += f"SYSTEM: {msg.content}\n\n"
            elif msg.role == MessageRole.USER:
                full_prompt += f"USER: {msg.content}\n\n"
        
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature
        )
        
        api_start = time.time()
        response = await model.generate_content_async(
            full_prompt,
            generation_config=generation_config
        )
        api_duration = time.time() - api_start
        
        # Check if response has text first
        try:
            response_text = response.text
        except ValueError as e:
            # Handle safety-filtered responses
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                finish_reason = getattr(candidate, 'finish_reason', 'Unknown')
                safety_ratings = getattr(candidate, 'safety_ratings', [])
                
                # Format safety ratings for debugging
                safety_info = []
                for rating in safety_ratings:
                    category = getattr(rating, 'category', 'Unknown')
                    probability = getattr(rating, 'probability', 'Unknown')
                    safety_info.append(f"{category}: {probability}")
                
                error_msg = f"Gemini blocked response. Finish reason: {finish_reason}"
                if safety_info:
                    error_msg += f"\nSafety ratings: {', '.join(safety_info)}"
                    
                # Check prompt feedback too
                if hasattr(response, 'prompt_feedback'):
                    block_reason = getattr(response.prompt_feedback, 'block_reason', None)
                    if block_reason:
                        error_msg += f"\nPrompt block reason: {block_reason}"
                
                raise RuntimeError(error_msg)
            else:
                raise RuntimeError(f"Gemini returned no response: {str(e)}")
        
        return LLMResponse(
            model_id=model_id,
            completion=response_text,
            api_duration=api_duration,
            usage={
                "prompt_token_count": response.usage_metadata.prompt_token_count if response.usage_metadata else None,
                "candidates_token_count": response.usage_metadata.candidates_token_count if response.usage_metadata else None
            }
        )
    
    async def _call_together(
        self,
        model_id: str,
        messages: List[ChatMessage],
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> LLMResponse:
        """Call Together AI API."""
        if not self.together_client:
            raise RuntimeError("Together client not available. Check TOGETHER_API_KEY.")
        
        # Convert to Together format (same as OpenAI)
        together_messages = [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
        ]
        
        api_start = time.time()
        response = await self.together_client.chat.completions.create(
            model=model_id,
            messages=together_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        api_duration = time.time() - api_start
        
        choice = response.choices[0]
        return LLMResponse(
            model_id=model_id,
            completion=choice.message.content,
            finish_reason=choice.finish_reason,
            api_duration=api_duration,
            usage={
                "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0) if response.usage else 0,
                "completion_tokens": getattr(response.usage, 'completion_tokens', 0) if response.usage else 0,
                "total_tokens": getattr(response.usage, 'total_tokens', 0) if response.usage else 0
            }
        )
    
    async def _call_openrouter(
        self,
        model_id: str,
        messages: List[ChatMessage],
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> LLMResponse:
        """Call OpenRouter API."""
        if not self.openrouter_client:
            raise RuntimeError("OpenRouter client not available. Check OPENROUTER_API_KEY.")
        
        # Convert to OpenRouter format (same as OpenAI)
        openrouter_messages = [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
        ]
        
        api_start = time.time()
        response = await self.openrouter_client.chat.completions.create(
            model=model_id,
            messages=openrouter_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        api_duration = time.time() - api_start
        
        # Handle None response or missing choices
        if response is None or not hasattr(response, 'choices') or not response.choices:
            logger.warning(f"OpenRouter returned invalid response for {model_id}: {response}")
            return LLMResponse(
                model_id=model_id,
                completion="",
                finish_reason="error",
                api_duration=api_duration,
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            )
        
        choice = response.choices[0]
        # Handle empty content
        if choice.message.content is None:
            logger.warning(f"OpenRouter returned empty content for {model_id}")
            return LLMResponse(
                model_id=model_id,
                completion="",
                finish_reason=choice.finish_reason or "empty",
                api_duration=api_duration,
                usage={
                    "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0) if response.usage else 0,
                    "completion_tokens": getattr(response.usage, 'completion_tokens', 0) if response.usage else 0,
                    "total_tokens": getattr(response.usage, 'total_tokens', 0) if response.usage else 0
                }
            )
        
        return LLMResponse(
            model_id=model_id,
            completion=choice.message.content,
            finish_reason=choice.finish_reason,
            api_duration=api_duration,
            usage={
                "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0) if response.usage else 0,
                "completion_tokens": getattr(response.usage, 'completion_tokens', 0) if response.usage else 0,
                "total_tokens": getattr(response.usage, 'total_tokens', 0) if response.usage else 0
            }
        )

    # --- Agentic-Misalignment-Qwen patch: local llama.cpp provider ----------
    #
    # Contract, enforced on every request:
    #   * the server must have the GGUF loaded that this model id names
    #     (/props model_path), with the expected chat template and slot count;
    #   * all six sampling values come from the provider config, the caller's
    #     temperature must agree with it, and a per-request seed is mandatory;
    #   * the server echo (__verbose.generation_settings) must show exactly
    #     those values and that seed, otherwise the request counts as failed.
    # Uses the standard library only, so no extra dependency is needed.

    _local_config_cache: Optional[Dict[str, Any]] = None

    @classmethod
    def _local_config(cls) -> Dict[str, Any]:
        if cls._local_config_cache is None:
            path = os.getenv("AM_LOCAL_PROVIDER_CONFIG")
            if not path:
                raise RuntimeError("AM_LOCAL_PROVIDER_CONFIG is not set; local/ models need the provider config file")
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            missing = [k for k in LOCAL_SAMPLING_KEYS if k not in cfg.get("sampling", {})]
            if missing:
                raise RuntimeError(f"local provider config lacks sampling values: {missing}")
            token_file = cfg.get("token_file")
            cfg["_token"] = ""
            if token_file:
                with open(os.path.expanduser(token_file), "r", encoding="utf-8") as tf:
                    cfg["_token"] = tf.read().strip()
            cls._local_config_cache = cfg
        return cls._local_config_cache

    @staticmethod
    def _local_http(method: str, url: str, token: str, body: Optional[Dict[str, Any]] = None,
                    timeout: float = 60.0) -> Dict[str, Any]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, method=method)
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:2000]
            raise RuntimeError(f"llama-server {method} {url} -> HTTP {e.code}: {detail}") from None

    def _local_identity(self, cfg: Dict[str, Any], model_id: str) -> Dict[str, Any]:
        """Prove from /props that the running server serves the GGUF this model id names."""
        if model_id not in cfg.get("models", {}):
            raise RuntimeError(f"{model_id} is not listed in the local provider config")
        expected = cfg["models"][model_id]
        props = self._local_http("GET", cfg["base_url"].rstrip("/") + "/props", cfg["_token"], timeout=30.0)
        model_path = props.get("model_path", "")
        if os.path.basename(model_path) != expected["gguf"]:
            raise RuntimeError(
                f"wrong model loaded: {model_id} needs {expected['gguf']}, the server has {model_path!r}")
        template_sha = hashlib.sha256((props.get("chat_template") or "").encode("utf-8")).hexdigest()
        if cfg.get("chat_template_sha256") and template_sha != cfg["chat_template_sha256"]:
            raise RuntimeError(f"chat template sha256 {template_sha} != expected {cfg['chat_template_sha256']}")
        slots = props.get("total_slots")
        if cfg.get("total_slots") and slots != cfg["total_slots"]:
            raise RuntimeError(f"server has {slots} slots, the provider config expects {cfg['total_slots']}")
        dgs = props.get("default_generation_settings") or {}
        return {
            "model_path": model_path,
            "gguf_sha256_expected": expected.get("sha256"),
            "chat_template_sha256": template_sha,
            "total_slots": slots,
            "n_ctx_slot": dgs.get("n_ctx"),
            "build_info": props.get("build_info"),
        }

    async def _call_local(
        self,
        model_id: str,
        messages: List[ChatMessage],
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> LLMResponse:
        """Call a local llama-server (OpenAI-compatible /v1/chat/completions)."""
        cfg = self._local_config()
        sampling = cfg["sampling"]
        if abs(float(temperature) - float(sampling["temperature"])) > 1e-9:
            raise RuntimeError(f"temperature {temperature} != provider config {sampling['temperature']}")
        seed = kwargs.pop("seed", None)
        if seed is None or not (0 <= int(seed) < 2**31):
            raise RuntimeError("local provider requires an explicit per-request seed in [0, 2^31)")
        seed = int(seed)
        timeout = float(kwargs.pop("timeout", None) or cfg.get("request_timeout_seconds", 3600))
        if kwargs:
            raise RuntimeError(f"unsupported arguments for the local provider: {sorted(kwargs)}")

        identity = await asyncio.to_thread(self._local_identity, cfg, model_id)

        body = {
            "model": cfg.get("server_alias", "local"),
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
            "max_tokens": int(max_tokens),
            "seed": seed,
            "stream": False,
            "verbose": True,  # ask for the __verbose echo of the settings actually used
        }
        for key in LOCAL_SAMPLING_KEYS:
            body[key] = sampling[key]

        api_start = time.time()
        resp = await asyncio.to_thread(
            self._local_http, "POST", cfg["base_url"].rstrip("/") + "/v1/chat/completions",
            cfg["_token"], body, timeout)
        api_duration = time.time() - api_start

        choice = (resp.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or ""
        verbose = resp.get("__verbose") or {}
        settings = verbose.get("generation_settings") or {}

        # The server must echo exactly the seed, sampling values and budget we asked for.
        # Every key must be present, numeric, finite and equal (review R2: a missing key
        # used to become NaN and slip through; a different n_predict was only a warning).
        mismatches, warnings = [], []

        def _num(value):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
            value = float(value)
            return value if value == value and value not in (float("inf"), float("-inf")) else None

        if not settings:
            mismatches.append("no __verbose.generation_settings in the response")
        else:
            expected = {"seed": seed, "n_predict": int(max_tokens), **{k: sampling[k] for k in LOCAL_SAMPLING_KEYS}}
            for key, want in expected.items():
                got = _num(settings.get(key))
                if got is None:
                    mismatches.append(f"{key} missing or not a finite number ({settings.get(key)!r})")
                elif abs(got - float(want)) > 1e-6:
                    mismatches.append(f"{key} {settings.get(key)} != {want}")
        prompt_text = verbose.get("prompt") or ""
        if isinstance(prompt_text, list):
            prompt_text = json.dumps(prompt_text)
        if not prompt_text:
            mismatches.append("no __verbose.prompt (rendered prompt) in the response")
        usage_check = resp.get("usage") or {}
        if _num(usage_check.get("prompt_tokens")) is None or _num(usage_check.get("completion_tokens")) is None:
            mismatches.append("usage.prompt_tokens / completion_tokens missing")
        if mismatches:
            raise RuntimeError("llama-server did not use the requested settings: " + "; ".join(mismatches))
        usage = resp.get("usage") or {}
        extra = {
            "request": {k: v for k, v in body.items() if k != "messages"},
            "identity": identity,
            "response_model": resp.get("model"),
            "system_fingerprint": resp.get("system_fingerprint"),
            "usage": usage,
            "timings": resp.get("timings"),
            "echo": {
                "generation_settings": {k: settings.get(k) for k in
                                        ("seed", "n_predict", "chat_format", "reasoning_format",
                                         "samplers", *LOCAL_SAMPLING_KEYS)},
                "tokens_predicted": verbose.get("tokens_predicted"),
                "tokens_evaluated": verbose.get("tokens_evaluated"),
                "tokens_cached": verbose.get("tokens_cached"),
                "truncated": verbose.get("truncated"),
                "stop_type": verbose.get("stop_type"),
                "stopping_word": verbose.get("stopping_word"),
                "id_slot": verbose.get("id_slot"),
                "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest() if prompt_text else None,
                "prompt_chars": len(prompt_text),
            },
            "raw_generated_text": verbose.get("content"),
            "settings_verified": True,
            "warnings": warnings,
        }
        return LLMResponse(
            model_id=model_id,
            completion=content,
            finish_reason=choice.get("finish_reason"),
            api_duration=api_duration,
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
            reasoning=reasoning,
            extra=extra,
        )
    # --- end local provider patch -------------------------------------------

# Convenience functions
def create_message(role: str, content: str) -> ChatMessage:
    """Create a ChatMessage with proper role enum."""
    return ChatMessage(role=MessageRole(role), content=content)

def create_prompt(system_prompt: str, user_prompt: str) -> List[ChatMessage]:
    """Create a standard system + user prompt."""
    return [
        create_message("system", system_prompt),
        create_message("user", user_prompt)
    ]

# Default client instance
_default_client = None

async def get_client() -> ModelClient:
    """Get or create the default client instance."""
    global _default_client
    if _default_client is None:
        _default_client = ModelClient()
    return _default_client

async def call_model(
    model_id: str,
    messages: Union[List[ChatMessage], List[Dict[str, str]]],
    **kwargs
) -> LLMResponse:
    """Convenience function for model calls."""
    client = await get_client()
    
    # Convert dict messages to ChatMessage if needed
    if messages and isinstance(messages[0], dict):
        messages = [ChatMessage(role=MessageRole(msg["role"]), content=msg["content"]) 
                   for msg in messages]
    
    return await client(model_id, messages, **kwargs)