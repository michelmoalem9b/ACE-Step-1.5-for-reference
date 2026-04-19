import torch
import torchaudio
import re
import os

class AceStepAudioToCodes_Custom:
    """
    ComfyUI node to convert an audio waveform into AceStep audio codes.
    Requires initialized DiT and VAE models from AceStep.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dit_model": ("MODEL",),
                "vae_model": ("VAE",),
                "silence_latent": ("LATENT",), # Shape [seq_len, dim]
                "audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("audio_codes",)
    FUNCTION = "convert"
    CATEGORY = "AceStep/Understanding"

    def convert(self, dit_model, vae_model, silence_latent, audio):
        # Unwrap ComfyUI standard wrappers
        if hasattr(vae_model, 'first_stage_model'):
            vae_model = vae_model.first_stage_model

        if hasattr(dit_model, 'model'):
            dit_model = dit_model.model

        # 1. Extract waveform and sample rate from ComfyUI AUDIO tuple
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]

        # Squeeze batch if present
        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)

        # 2. Normalize to stereo 48kHz (AceStep standard)
        device = next(vae_model.parameters()).device
        dtype = next(vae_model.parameters()).dtype
        waveform = waveform.to(device)

        if sample_rate != 48000:
            resampler = torchaudio.transforms.Resample(sample_rate, 48000).to(device)
            waveform = resampler(waveform)

        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2, :]

        waveform = waveform.to(dtype)

        # 3. VAE Encoding
        with torch.inference_mode():
            # Add batch dimension for VAE
            audio_batch = waveform.unsqueeze(0)
            if hasattr(vae_model, 'tiled_encode'):
                latents = vae_model.tiled_encode(audio_batch, offload_latent_to_cpu=False)
            else:
                latents = vae_model.encode(audio_batch).latent_dist.sample()

            latents = latents.to(device).to(dtype)

            # CRITICAL: Transpose latents to [Batch, Seq_Len, Dim]
            latents = latents.transpose(1, 2)

        # 4. DiT Tokenization (Audio to Codes)
        # Unwrap ComfyUI LATENT dictionary
        if isinstance(silence_latent, dict) and "samples" in silence_latent:
            silence_latent = silence_latent["samples"]

        # Ensure it has the shape [seq_len, dim]
        if silence_latent.dim() == 3:
            if silence_latent.shape[1] < silence_latent.shape[2]:
                silence_latent = silence_latent.transpose(1, 2)
            silence_latent = silence_latent[0]
        elif silence_latent.dim() == 4:
            silence_latent = silence_latent.squeeze(2).squeeze(2)
            if silence_latent.shape[0] == 1:
                silence_latent = silence_latent[0]
            if silence_latent.shape[0] < silence_latent.shape[1]:
                silence_latent = silence_latent.transpose(0, 1)

        silence_latent = silence_latent.to(device).to(dtype)

        hidden_states = latents
        attention_mask = torch.ones(hidden_states.shape[0], hidden_states.shape[1], dtype=torch.bool, device=device)

        pool_window_size = dit_model.config.pool_window_size
        if hidden_states.shape[1] % pool_window_size != 0:
            pad_len = pool_window_size - (hidden_states.shape[1] % pool_window_size)
            hidden_states = torch.cat([hidden_states, silence_latent[:1, :pad_len].repeat(hidden_states.shape[0], 1, 1)], dim=1)
            attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_len), mode='constant', value=False)

        from einops import rearrange
        hidden_states = rearrange(hidden_states, 'n (t_patch p) d -> n t_patch p d', p=pool_window_size)

        with torch.inference_mode():
            # Use the official logic to tokenize the rearranged hidden states.
            _, indices = dit_model.tokenizer(hidden_states)

        # Flatten and format into <|audio_code_X|> strings
        indices_flat = indices.flatten().cpu().tolist()
        codes_string = "".join([f"<|audio_code_{idx}|>" for idx in indices_flat])

        return (codes_string,)


class AceStepUnderstandMusic_Custom:
    """
    ComfyUI node to use the 5Hz LLM to understand audio codes and generate metadata and lyrics.
    Standalone version that relies on raw HuggingFace models rather than AceStep's LLMHandler.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "llm_model": ("LLM_MODEL",),
                "llm_tokenizer": ("LLM_TOKENIZER",),
                "audio_codes": ("STRING", {"multiline": True, "forceInput": True}),
                "temperature": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_new_tokens": ("INT", {"default": 1024, "min": 128, "max": 4096}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "FLOAT", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("caption", "lyrics", "bpm", "duration", "keyscale", "language", "timesignature")
    FUNCTION = "understand"
    CATEGORY = "AceStep/Understanding"

    def understand(self, llm_model, llm_tokenizer, audio_codes, temperature, top_p, max_new_tokens):
        if not audio_codes or not audio_codes.strip():
            audio_codes = "NO USER INPUT"

        DEFAULT_LM_UNDERSTAND_INSTRUCTION = "Understand the given musical conditions and describe the audio semantics accordingly:"

        # 1. Format prompt exactly as AceStep expects using the tokenizer's chat template
        formatted_prompt = llm_tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": f"# Instruction\n{DEFAULT_LM_UNDERSTAND_INSTRUCTION}\n\n"
                },
                {
                    "role": "user",
                    "content": audio_codes
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

        # 2. Tokenize prompt
        inputs = llm_tokenizer(formatted_prompt, return_tensors="pt")

        # Get the device the model is currently on
        device = next(llm_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 3. Generate output text
        # Since we don't have AceStep's FSM ConstrainedLogitsProcessor, we rely on the
        # model's native instruction tuning to follow the `<think>` format properly.
        # The 5Hz model is heavily fine-tuned to do this, so it will work well 99% of the time.
        with torch.inference_mode():
            outputs = llm_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=(temperature > 0),
                repetition_penalty=1.0,
            )

        # 4. Decode the generated tokens
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        output_text = llm_tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # 5. Regex Parser to extract metadata from the <think> tag
        # The expected output format is:
        # <think>
        # bpm: 120
        # caption: A cool song
        # ...
        # </think>
        # [Intro] ...lyrics...

        metadata = {
            "bpm": 0,
            "caption": "",
            "duration": 0.0,
            "keyscale": "",
            "language": "",
            "timesignature": "",
            "lyrics": ""
        }

        think_match = re.search(r'<think>(.*?)</think>', output_text, re.DOTALL)
        if think_match:
            think_content = think_match.group(1).strip()

            # Extract fields
            for line in think_content.split('\n'):
                line = line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()

                    if key in metadata:
                        if key == 'bpm':
                            try: metadata['bpm'] = int(value)
                            except: pass
                        elif key == 'duration':
                            try: metadata['duration'] = float(value)
                            except: pass
                        else:
                            metadata[key] = value

            # Extract lyrics (everything after </think>)
            lyrics_text = output_text[think_match.end():].strip()
            if lyrics_text.startswith("# Lyric"):
                lyrics_text = lyrics_text[len("# Lyric"):].strip()
            metadata['lyrics'] = lyrics_text
        else:
            # Fallback if the model hallucinated and didn't generate <think>
            metadata['caption'] = output_text.strip()

        return (
            metadata["caption"],
            metadata["lyrics"],
            metadata["bpm"],
            metadata["duration"],
            metadata["keyscale"],
            metadata["language"],
            metadata["timesignature"]
        )

import folder_paths

# Case-insensitive search for LLM folder in extra_model_paths config
LLM_KEY = "LLM" if "LLM" in folder_paths.folder_names_and_paths else "llm"
if LLM_KEY not in folder_paths.folder_names_and_paths:
    llm_dir = os.path.join(folder_paths.models_dir, "llm")
    os.makedirs(llm_dir, exist_ok=True)
    supported_extensions = getattr(
        folder_paths,
        "supported_pt_extensions",
        {".safetensors", ".pt", ".bin", ".ckpt"}
    )
    folder_paths.folder_names_and_paths[LLM_KEY] = ([llm_dir], supported_extensions)


class AceStepHuggingFaceLoader_Custom:
    """
    ComfyUI node to load a HuggingFace LLM directly without AceStep's library.
    Returns standard LLM_MODEL and LLM_TOKENIZER.
    """
    @classmethod
    def INPUT_TYPES(s):
        import folder_paths
        try:
            llm_models = folder_paths.get_filename_list(LLM_KEY)
        except Exception:
            llm_models = []

        if not llm_models:
            llm_models = ["AceStep-5Hz-LM (Put model in models/llm)"]

        return {
            "required": {
                "model_name": (llm_models, ),
                "device": (["cuda", "cpu", "mps"], {"default": "cuda"}),
                "dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16"})
            }
        }

    RETURN_TYPES = ("LLM_MODEL", "LLM_TOKENIZER")
    RETURN_NAMES = ("llm_model", "llm_tokenizer")
    FUNCTION = "load_llm"
    CATEGORY = "AceStep/Loaders"

    def load_llm(self, model_name, device, dtype):
        if LLM_KEY in folder_paths.folder_names_and_paths:
            model_path = folder_paths.get_full_path(LLM_KEY, model_name)
        else:
            model_path = os.path.join(folder_paths.models_dir, "llm", model_name)

        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model {model_name} not found at {model_path}")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("The 'transformers' python package is required. Please install it in your ComfyUI environment.")

        # Resolve dtype
        if dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "float16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        # Load the model directly using HuggingFace
        print(f"Loading HuggingFace LLM: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True
        )
        model.eval()

        return (model, tokenizer)


NODE_CLASS_MAPPINGS = {
    "AceStepAudioToCodes_Custom": AceStepAudioToCodes_Custom,
    "AceStepUnderstandMusic_Custom": AceStepUnderstandMusic_Custom,
    "AceStepHuggingFaceLoader_Custom": AceStepHuggingFaceLoader_Custom
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepAudioToCodes_Custom": "AceStep Audio to Codes (Custom)",
    "AceStepUnderstandMusic_Custom": "AceStep Understand Music (Codes to Prompt) (Custom)",
    "AceStepHuggingFaceLoader_Custom": "AceStep HuggingFace LLM Loader (Custom)"
}
