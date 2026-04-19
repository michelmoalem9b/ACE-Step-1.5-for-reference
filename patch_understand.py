import torch

def patch_comfyui_acestep_nodes():
    with open('comfyui_acestep_nodes.py', 'r') as f:
        content = f.read()

    new_content = content.replace('''        # Call the llm_handler's understand function
        # This function internally handles the DEFAULT_LM_UNDERSTAND_INSTRUCTION
        # and the constrained decoding to force the <think> metadata format.
        metadata, status = llm_handler.understand_audio_from_codes(
            audio_codes=audio_codes,
            temperature=temperature,
            top_p=top_p,
            use_constrained_decoding=True
        )''',
'''        # Call the llm_handler's understand function
        # The official implementation has an understand_audio_from_codes method
        # that internally handles applying the chat template with DEFAULT_LM_UNDERSTAND_INSTRUCTION
        # and parsing the metadata + lyrics.
        metadata, status = llm_handler.understand_audio_from_codes(
            audio_codes=audio_codes,
            temperature=temperature,
            top_p=top_p,
            use_constrained_decoding=True
        )''')

    with open('comfyui_acestep_nodes.py', 'w') as f:
        f.write(new_content)

patch_comfyui_acestep_nodes()
