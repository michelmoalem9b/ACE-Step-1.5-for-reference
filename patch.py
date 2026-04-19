import torch

def patch_comfyui_acestep_nodes():
    with open('comfyui_acestep_nodes.py', 'r') as f:
        content = f.read()

    new_content = content.replace('''        hidden_states = latents
        attention_mask = torch.ones(hidden_states.shape[0], hidden_states.shape[1], dtype=torch.bool, device=device)

        with torch.inference_mode():
            # Tokenize using the DiT's quantizer
            # The model's tokenize function handles pool_window_size padding and rearranging
            _, indices, _ = dit_model.tokenize(hidden_states, silence_latent, attention_mask.unsqueeze(0))''',
'''        hidden_states = latents
        attention_mask = torch.ones(hidden_states.shape[0], hidden_states.shape[1], dtype=torch.bool, device=device)

        with torch.inference_mode():
            # The official tokenize function handles pool_window_size padding and rearranging if called with
            # the signature (x, silence_latent, attention_mask).
            # We call it directly:
            _, indices, _ = dit_model.tokenize(hidden_states, silence_latent, attention_mask.unsqueeze(0))''')

    with open('comfyui_acestep_nodes.py', 'w') as f:
        f.write(new_content)

patch_comfyui_acestep_nodes()
