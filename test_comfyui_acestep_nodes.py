import sys
from unittest.mock import MagicMock

# Mock torchaudio and its transforms so it won't crash when imported
sys.modules['torchaudio'] = MagicMock()
sys.modules['torchaudio.transforms'] = MagicMock()

import pytest
from comfyui_acestep_nodes import AceStepAudioToCodes, AceStepUnderstandMusic

def test_nodes_can_be_imported():
    assert AceStepAudioToCodes is not None
    assert AceStepUnderstandMusic is not None

def test_node_class_mappings():
    from comfyui_acestep_nodes import NODE_CLASS_MAPPINGS
    assert "AceStepAudioToCodes" in NODE_CLASS_MAPPINGS
    assert "AceStepUnderstandMusic" in NODE_CLASS_MAPPINGS

def test_acestep_understand_music_input_types():
    types = AceStepUnderstandMusic.INPUT_TYPES()
    assert "audio_codes" in types["required"]
    assert "temperature" in types["required"]
    assert "top_p" in types["required"]
    assert "llm_handler" in types["required"]

def test_acestep_audio_to_codes_input_types():
    types = AceStepAudioToCodes.INPUT_TYPES()
    assert "audio" in types["required"]
    assert "dit_model" in types["required"]
    assert "vae_model" in types["required"]
    assert "silence_latent" in types["required"]
