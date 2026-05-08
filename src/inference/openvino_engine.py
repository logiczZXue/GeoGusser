"""Unified inference engine with OpenVINO backend.

Supports multi-device execution (CPU/GPU/NPU) with image embedding caching
for the 3-stage GeoCoT pipeline.
"""

import logging
from typing import Optional

from PIL import Image

from .device_manager import DeviceConfig, get_device_config

log = logging.getLogger(__name__)


class OpenVINOEngine:
    """OpenVINO-based inference engine for VLM geolocation."""

    def __init__(self, model_dir: str, device_config: Optional[DeviceConfig] = None):
        import openvino as ov

        self._core = ov.Core()
        self._device_config = device_config or get_device_config()
        self._cached_image_embeddings = None
        self._cached_image_id = None

        log.info(f"Loading models from {model_dir}")
        log.info(f"Device config: ViT={self._device_config.vit_device}, "
                 f"LLM={self._device_config.llm_generate_device}")

        # Load and compile vision encoder
        vit_path = f"{model_dir}/openvino_vision_encoder_model.xml"
        try:
            vit_model = self._core.read_model(vit_path)
            self._vit_compiled = self._core.compile_model(vit_model, self._device_config.vit_device)
            log.info("Vision encoder loaded on %s", self._device_config.vit_device)
        except Exception as e:
            log.warning(f"Could not load vision encoder: {e}")
            self._vit_compiled = None

        # Load and compile language model
        llm_path = f"{model_dir}/openvino_language_model.xml"
        try:
            llm_model = self._core.read_model(llm_path)
            self._llm_compiled = self._core.compile_model(llm_model, self._device_config.llm_generate_device)
            log.info("Language model loaded on %s", self._device_config.llm_generate_device)
        except Exception as e:
            log.warning(f"Could not load language model: {e}")
            self._llm_compiled = None

    def encode_image(self, image: Image.Image, image_id: str = "") -> object:
        """Encode image using vision encoder, with caching for reuse across CoT stages."""
        if self._cached_image_id == image_id and self._cached_image_embeddings is not None:
            log.debug("Reusing cached image embeddings")
            return self._cached_image_embeddings

        if self._vit_compiled is None:
            raise RuntimeError("Vision encoder not loaded")

        # Preprocess image to model input format
        import numpy as np
        img_array = np.array(image.convert("RGB").resize((840, 840)))
        img_array = img_array.transpose(2, 0, 1).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        result = self._vit_compiled(img_array)
        embeddings = next(iter(result.values()))

        self._cached_image_embeddings = embeddings
        self._cached_image_id = image_id
        return embeddings

    def generate(self, prompt_tokens, image_embeddings=None, max_new_tokens=300):
        """Generate text tokens using the language model."""
        if self._llm_compiled is None:
            raise RuntimeError("Language model not loaded")

        # Token-by-token generation loop
        generated_tokens = []
        current_input = prompt_tokens

        for _ in range(max_new_tokens):
            if image_embeddings is not None:
                result = self._llm_compiled({"input_ids": current_input, "image_embeddings": image_embeddings})
            else:
                result = self._llm_compiled({"input_ids": current_input})

            next_token = next(iter(result.values()))[:, -1:]
            generated_tokens.append(next_token)
            current_input = np.concatenate([current_input, next_token], axis=-1)

            # Check for end-of-sequence token (simplified)
            if next_token.item() in [2, 128001, 128009]:
                break

        return generated_tokens

    def create_model_fn(self):
        """Create a callable compatible with GeoCoTPipeline's model_fn interface."""
        from src.Geocot.Geocot import GeoCoTPipeline

        def model_fn(image: Image.Image, prompt_text: str, prev_outputs: dict) -> str:
            image_id = str(id(image))
            image_embeddings = self.encode_image(image, image_id)

            # Tokenize prompt (simplified — real implementation needs the processor)
            # This is a placeholder that will be refined with actual tokenization
            import numpy as np
            prompt_tokens = np.array([[1]])  # BOS token placeholder

            tokens = self.generate(prompt_tokens, image_embeddings, max_new_tokens=300)
            # Decode tokens (placeholder — needs actual tokenizer)
            return " ".join(str(t) for t in tokens[:20]) + "..."

        return model_fn
