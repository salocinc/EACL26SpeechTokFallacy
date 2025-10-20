import os
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, LlamaForCausalLM
import json
from typing import Dict, List, Optional, Tuple
base_dir = os.path.dirname(__file__)

class AudioTokenManager:
    """
    Manages audio tokens and embeddings separately from the base LLaMA model.
    This allows for flexible vocabulary management without saving multiple model copies.
    """
    
    def __init__(self, base_model_path: str = "meta-llama/Llama-3.2-3B"):
        self.base_model_path = base_model_path
        self.base_tokenizer = None
        self.base_model = None
        self.original_vocab_size = None

    def save_base_model_once(self, save_path: str = os.path.join(base_dir, "base_llama_model")):
        """
        Save the base LLaMA model and tokenizer once for reuse.
        """
        print("Saving base LLaMA model and tokenizer...")
        
        # Load and save tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_path, use_fast=True)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.save_pretrained(save_path)
        
        # Load and save model
        model = LlamaForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float32,
        )
        model.save_pretrained(save_path)
        
        # Save metadata
        metadata = {
            "original_vocab_size": len(tokenizer),
            "base_model_path": self.base_model_path,
            "embedding_dim": model.config.hidden_size
        }
        
        with open(f"{save_path}/base_model_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Base model saved to {save_path}")
        print(f"Original vocabulary size: {len(tokenizer)}")
        
        return len(tokenizer), model.config.hidden_size
    
    def create_audio_token_config(self, selected_features: np.ndarray, n_layers_rvq: int, 
                                config_name: str) -> Dict:
        """
        Create and save audio token configuration.
        """
        # Create mapping from selected features to tokens
        selected_features_dict = {}
        for i, feature in enumerate(selected_features):
            selected_features_dict[int(feature)] = f"<audio_token_{i}>"
        
        # Special tokens for multimodal format
        special_tokens = ["<|text|>", "<|text_end|>", "<|audio|>", "<|audio_end|>"]
        
        config = {
            "config_name": config_name,
            "n_layers_rvq": n_layers_rvq,
            "selected_features": selected_features.tolist(),
            "feature_to_token_map": selected_features_dict,
            "special_tokens": special_tokens,
            "total_new_tokens": len(selected_features_dict) + len(special_tokens),
            "audio_tokens": list(selected_features_dict.values()),
        }
        
        # Save configuration
        config_path = os.path.join(base_dir, f"audio_token_configs/{config_name}.json")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Audio token configuration saved to {config_path}")
        return config
    
    def load_model_with_audio_tokens(self, base_model_path: str, audio_config_name: str, 
                                   embeddings_path: Optional[str] = None) -> Tuple[LlamaForCausalLM, AutoTokenizer, Dict]:
        """
        Load base model and dynamically add audio tokens.
        Optionally load pre-trained audio embeddings.
        """
        print(f"Loading model with audio configuration: {audio_config_name}")
        
        # Load base model metadata
        with open(f"{base_model_path}/base_model_metadata.json", 'r') as f:
            base_metadata = json.load(f)
        
        # Load audio token configuration
        config_path = os.path.join(base_dir, f"audio_token_configs/{audio_config_name}.json")
        with open(config_path, 'r') as f:
            audio_config = json.load(f)
        
        # Load base tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
        model = LlamaForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float32,
        )
        
        # Add audio tokens to tokenizer
        new_tokens = audio_config["audio_tokens"] + audio_config["special_tokens"]
        tokenizer.add_tokens(new_tokens)
        tokenizer.add_special_tokens({"additional_special_tokens": audio_config["special_tokens"]})
        
        # Resize model embeddings
        original_vocab_size = base_metadata["original_vocab_size"]
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        
        # Load pre-trained audio embeddings if available
        if embeddings_path and os.path.exists(embeddings_path):
            print(f"Loading pre-trained audio embeddings from {embeddings_path}")
            embedding_data = torch.load(embeddings_path, map_location='cpu')
            
            # Get the embedding layer
            embedding_layer = model.get_input_embeddings()
            
            # Load the trained audio embeddings
            trained_embeddings = embedding_data['audio_embeddings']
            
            # Place them in the correct positions
            with torch.no_grad():
                embedding_layer.weight[original_vocab_size:original_vocab_size + len(trained_embeddings)] = trained_embeddings
            
            print(f"Loaded {len(trained_embeddings)} pre-trained audio embeddings")
        else:
            print("No pre-trained embeddings found. Using \"randomly\" initialized embeddings.")
        
        # Create combined metadata
        combined_metadata = {
            **base_metadata,
            **audio_config,
            "current_vocab_size": len(tokenizer),
            "embeddings_loaded": embeddings_path is not None
        }
        
        return model, tokenizer, combined_metadata
    
    def save_audio_embeddings_only(self, model: LlamaForCausalLM, original_vocab_size: int, 
                                 config_name: str, training_metadata: Optional[Dict] = None):
        """
        Save only the audio token embeddings separately.
        """
        embedding_layer = model.get_input_embeddings()
        audio_embeddings = embedding_layer.weight[original_vocab_size:].detach().cpu()
        
        # Prepare embedding data
        embedding_data = {
            'audio_embeddings': audio_embeddings,
            'config_name': config_name,
            'original_vocab_size': original_vocab_size,
            'audio_vocab_size': audio_embeddings.size(0),
            'embedding_dim': audio_embeddings.size(1),
            'creation_timestamp': torch.tensor(torch.get_rng_state()).sum().item(),  # Simple timestamp
        }
        
        # Add training metadata if provided
        if training_metadata:
            embedding_data.update(training_metadata)
        
        # Save embeddings
        embeddings_dir = os.path.join(base_dir,"audio_embeddings")
        os.makedirs(embeddings_dir, exist_ok=True)
        embeddings_path = os.path.join(embeddings_dir, f"trained_audio_embeddings_{config_name}.pt")

        torch.save(embedding_data, embeddings_path)
        print(f"Audio embeddings saved to {embeddings_path}")
        
        return embeddings_path