import asyncio
from typing import List, Dict, Any
from app.domain.vlm.client import VLMClient, VLMResponse
import logging

class SmolVLM256Client(VLMClient):
    def __init__(self, model_name="HuggingFaceTB/SmolVLM2-256M-Video-Instruct"):
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._device = None
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def _initialize_model(self):
        if self._model is not None:
            return
            
        self.logger.info(f"Lazy loading {self.model_name}...")
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForImageTextToText
            
            # Determine best device
            if torch.backends.mps.is_available():
                self._device = "mps"
            elif torch.cuda.is_available():
                self._device = "cuda"
            else:
                self._device = "cpu"
                
            self.logger.info(f"Using device: {self._device}")
            
            # We load in bfloat16 or float16 if possible to save memory on MPS/CUDA
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16 if self._device != "cpu" else torch.float32
            
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
            ).to(self._device)
            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self.logger.info("Model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load SmolVLM model: {e}")
            raise e

    async def ainvoke(self, messages: List[Dict[str, Any]]) -> VLMResponse:
        return await asyncio.to_thread(self._invoke_sync, messages)
        
    def _invoke_sync(self, messages: List[Dict[str, Any]]) -> VLMResponse:
        self._initialize_model()
        
        # Translate Langchain-style messages to HF format
        formatted_messages = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                formatted_messages.append({"role": msg["role"], "content": [{"type": "text", "text": content}]})
            else:
                formatted_messages.append({"role": msg["role"], "content": content})
                
        prompt = self._processor.apply_chat_template(formatted_messages, add_generation_prompt=True)
        
        # For purely text intents, we don't pass images to the processor
        inputs = self._processor(text=prompt, images=None, return_tensors="pt")
        inputs = inputs.to(self._device)
        
        generated_ids = self._model.generate(**inputs, max_new_tokens=1024, temperature=0.1)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self._processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return VLMResponse(content=output_text[0])
