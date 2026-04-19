def patch_comfyui_acestep_nodes():
    with open('comfyui_acestep_nodes.py', 'r') as f:
        content = f.read()

    new_content = content.replace('''        hidden_states = latents
        attention_mask = torch.ones(hidden_states.shape[0], hidden_states.shape[1], dtype=torch.bool, device=device)

        with torch.inference_mode():
            # The official tokenize function handles pool_window_size padding and rearranging if called with
            # the signature (x, silence_latent, attention_mask).
            # We call it directly:
            _, indices, _ = dit_model.tokenize(hidden_states, silence_latent, attention_mask.unsqueeze(0))''',
'''        hidden_states = latents
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
                _, indices = dit_model.tokenizer(hidden_states)''')

    with open('comfyui_acestep_nodes.py', 'w') as f:
        f.write(new_content)

patch_comfyui_acestep_nodes()
