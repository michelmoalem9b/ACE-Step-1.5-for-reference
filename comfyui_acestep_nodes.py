import torch
import torchaudio

class AceStepAudioToCodes:
    """
    ComfyUI node to convert an audio waveform into AceStep audio codes.
    Requires initialized DiT and VAE models from AceStep.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dit_model": ("ACESTEP_DIT_MODEL",),
                "vae_model": ("ACESTEP_VAE_MODEL",),
                "silence_latent": ("TENSOR",), # Shape [seq_len, dim]
                "audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("audio_codes",)
    FUNCTION = "convert"
    CATEGORY = "AceStep/Understanding"

    def convert(self, dit_model, vae_model, silence_latent, audio):
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
            # The model's forward or quantize methods might need to be called depending on ComfyUI integration,
            # but we use the tokenizer to match the official logic exactly.
            if hasattr(dit_model, "tokenize") and callable(dit_model.tokenize):
                # Try passing raw latents if tokenize handles it internally
                try:
                    _, indices, _ = dit_model.tokenize(latents, silence_latent, torch.ones(latents.shape[0], latents.shape[1], dtype=torch.bool, device=device).unsqueeze(0))
                except TypeError:
                    # Fallback to direct quantization
                    _, indices = dit_model.tokenizer(hidden_states)
            else:
                _, indices = dit_model.tokenizer(hidden_states)

        # Flatten and format into <|audio_code_X|> strings
        indices_flat = indices.flatten().cpu().tolist()
        codes_string = "".join([f"<|audio_code_{idx}|>" for idx in indices_flat])

        return (codes_string,)


class AceStepUnderstandMusic:
    """
    ComfyUI node to use the 5Hz LLM to understand audio codes and generate metadata and lyrics.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "llm_handler": ("ACESTEP_LLM_HANDLER",),
                "audio_codes": ("STRING", {"multiline": True, "forceInput": True}),
                "temperature": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "FLOAT", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("caption", "lyrics", "bpm", "duration", "keyscale", "language", "timesignature")
    FUNCTION = "understand"
    CATEGORY = "AceStep/Understanding"

    def understand(self, llm_handler, audio_codes, temperature, top_p):
        if not getattr(llm_handler, "llm_initialized", False):
            raise ValueError("AceStep LLM Handler is not initialized.")

        if not audio_codes or not audio_codes.strip():
            audio_codes = "NO USER INPUT"

        # Replicate the logic of understand_audio_from_codes directly in case
        # llm_handler lacks it or for strict alignment with the provided explanation.

        DEFAULT_LM_UNDERSTAND_INSTRUCTION = "Understand the given musical conditions and describe the audio semantics accordingly:"

        if hasattr(llm_handler, "understand_audio_from_codes"):
            metadata, status = llm_handler.understand_audio_from_codes(
                audio_codes=audio_codes,
                temperature=temperature,
                top_p=top_p,
                use_constrained_decoding=True
            )
        else:
            # Fallback direct implementation mirroring the exact code format
            formatted_prompt = llm_handler.llm_tokenizer.apply_chat_template(
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

            output_text, status = llm_handler.generate_from_formatted_prompt(
                formatted_prompt=formatted_prompt,
                cfg={
                    "temperature": temperature,
                    "top_p": top_p,
                    "generation_phase": "understand"
                },
                use_constrained_decoding=True
            )

            metadata, _ = llm_handler.parse_lm_output(output_text)

            # Extract lyrics section
            import re
            think_end_pattern = r'</think>'
            match = re.search(think_end_pattern, output_text)
            if match:
                lyrics = output_text[match.end():].strip()
                if lyrics.startswith("# Lyric"):
                    lyrics = lyrics[len("# Lyric"):].strip()
                metadata['lyrics'] = lyrics


        if not metadata:
            raise RuntimeError(f"Failed to understand audio codes: {status}")

        # Extract fields
        caption = metadata.get('caption', '')
        lyrics = metadata.get('lyrics', '')
        keyscale = metadata.get('keyscale', '')
        language = metadata.get('language', metadata.get('vocal_language', ''))
        timesignature = metadata.get('timesignature', '')

        bpm = metadata.get('bpm')
        if bpm == 'N/A' or not bpm:
            bpm = 0
        else:
            try:
                bpm = int(bpm)
            except:
                bpm = 0

        duration = metadata.get('duration')
        if duration == 'N/A' or not duration:
            duration = 0.0
        else:
            try:
                duration = float(duration)
            except:
                duration = 0.0

        return (caption, lyrics, bpm, duration, keyscale, language, timesignature)


NODE_CLASS_MAPPINGS = {
    "AceStepAudioToCodes": AceStepAudioToCodes,
    "AceStepUnderstandMusic": AceStepUnderstandMusic
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepAudioToCodes": "AceStep Audio to Codes",
    "AceStepUnderstandMusic": "AceStep Understand Music (Codes to Prompt)"
}
