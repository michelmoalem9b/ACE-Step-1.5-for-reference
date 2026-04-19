def patch_comfyui_acestep_nodes():
    with open('comfyui_acestep_nodes.py', 'r') as f:
        content = f.read()

    new_content = content.replace('''        # Call the llm_handler's understand function
        # The official implementation has an understand_audio_from_codes method
        # that internally handles applying the chat template with DEFAULT_LM_UNDERSTAND_INSTRUCTION
        # and parsing the metadata + lyrics.
        metadata, status = llm_handler.understand_audio_from_codes(
            audio_codes=audio_codes,
            temperature=temperature,
            top_p=top_p,
            use_constrained_decoding=True
        )''',
'''        # Replicate the logic of understand_audio_from_codes directly in case
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
                        "content": f"# Instruction\\n{DEFAULT_LM_UNDERSTAND_INSTRUCTION}\\n\\n"
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
''')

    with open('comfyui_acestep_nodes.py', 'w') as f:
        f.write(new_content)

patch_comfyui_acestep_nodes()
