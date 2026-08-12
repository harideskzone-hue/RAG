import asyncio
import logging
from typing import List, Dict, Any

class VLMResponse:
    def __init__(self, content: str):
        self.content = content

class LocalReasoningClient:
    """
    A lightweight, text-only instruction LLM client specifically for reasoning.
    Used for structured JSON extraction on structured Evidence and Knowledge Graphs.
    """
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct"):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._device = None
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def _initialize_model(self):
        if self._model is not None:
            return
            
        self.logger.info(f"Lazy loading reasoning model {self.model_name}...")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Determine best device
            if torch.backends.mps.is_available():
                self._device = "mps"
            elif torch.cuda.is_available():
                self._device = "cuda"
            else:
                self._device = "cpu"
                
            self.logger.info(f"Using device: {self._device}")
            
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16 if self._device != "cpu" else torch.float32
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map=self._device
            )
            self.logger.info("Reasoning model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load text reasoning model: {e}")
            raise e

    async def ainvoke(self, messages: List[Dict[str, Any]]) -> VLMResponse:
        return await asyncio.to_thread(self._invoke_sync, messages)
        
    def _invoke_sync(self, messages: List[Dict[str, Any]]) -> VLMResponse:
        self._initialize_model()
        
        # Ensure we just have simple role/content pairs of strings
        formatted_messages = []
        for msg in messages:
            content = msg.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            formatted_messages.append({"role": msg["role"], "content": content})
                
        text = self._tokenizer.apply_chat_template(formatted_messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer([text], return_tensors="pt").to(self._device)
        
        # Generate with low temperature for strict structural extraction
        generated_ids = self._model.generate(**inputs, max_new_tokens=1024, temperature=0.1, do_sample=True)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self._tokenizer.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return VLMResponse(content=output_text)
