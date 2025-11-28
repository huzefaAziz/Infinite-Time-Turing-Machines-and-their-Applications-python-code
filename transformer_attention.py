"""
Transformer Attention Mechanisms
Reinterpreted through the ITTM/USM framework as described in the paper.

The paper discusses how Transformers can be understood as ITTM implementations,
with attention mechanisms mapping to information routing in the knowledge graph.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from universal_state_machine import UniversalStateMachine, KnowledgeGraph


class AttentionMechanism:
    """
    Attention mechanism reinterpreted through the ITTM/USM framework.
    Maps to information routing in the knowledge graph.
    """
    
    def __init__(self, d_model: int, d_k: Optional[int] = None):
        self.d_model = d_model
        self.d_k = d_k or d_model
        
        # Initialize weight matrices
        self.W_q = np.random.randn(d_model, self.d_k) * 0.1
        self.W_k = np.random.randn(d_model, self.d_k) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1
    
    def scaled_dot_product_attention(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scaled dot-product attention.
        In the ITTM framework, this represents information routing.
        """
        # Compute attention scores
        scores = np.dot(Q, K.T) / np.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        
        # Apply softmax
        attention_weights = self._softmax(scores, axis=-1)
        
        # Apply attention to values
        output = np.dot(attention_weights, V)
        
        return output, attention_weights
    
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Softmax function"""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def forward(
        self,
        queries: np.ndarray,
        keys: np.ndarray,
        values: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through attention mechanism.
        """
        # Project to query, key, value spaces
        Q = np.dot(queries, self.W_q)
        K = np.dot(keys, self.W_k)
        V = np.dot(values, self.W_v)
        
        # Apply attention
        output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        return output, attention_weights


class TransformerBlock:
    """
    Transformer block as an ITTM implementation.
    Represents a computational step in the transfinite computation.
    """
    
    def __init__(self, d_model: int, d_ff: int, num_heads: int = 1):
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        
        # Multi-head attention (simplified to single head for clarity)
        self.attention = AttentionMechanism(d_model)
        
        # Feed-forward network
        self.W_ff1 = np.random.randn(d_model, d_ff) * 0.1
        self.W_ff2 = np.random.randn(d_ff, d_model) * 0.1
        self.bias1 = np.zeros(d_ff)
        self.bias2 = np.zeros(d_model)
        
        # Layer normalization parameters
        self.gamma1 = np.ones(d_model)
        self.beta1 = np.zeros(d_model)
        self.gamma2 = np.ones(d_model)
        self.beta2 = np.zeros(d_model)
    
    def layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Layer normalization"""
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True) + 1e-6
        return gamma * (x - mean) / std + beta
    
    def feed_forward(self, x: np.ndarray) -> np.ndarray:
        """Feed-forward network"""
        # First layer with ReLU
        h = np.dot(x, self.W_ff1) + self.bias1
        h = np.maximum(0, h)  # ReLU
        
        # Second layer
        output = np.dot(h, self.W_ff2) + self.bias2
        return output
    
    def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through transformer block.
        In ITTM terms, this is a successor ordinal step.
        """
        # Self-attention with residual connection and layer norm
        attn_output, attention_weights = self.attention.forward(x, x, x, mask)
        x = self.layer_norm(x + attn_output, self.gamma1, self.beta1)
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.layer_norm(x + ff_output, self.gamma2, self.beta2)
        
        return x, attention_weights


class TransformerAsITTM:
    """
    Transformer architecture reinterpreted as an ITTM implementation.
    
    As described in the paper:
    - Each layer represents a computational step
    - Attention mechanisms map to information routing
    - The architecture can be understood through the ITTM framework
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_ff: int,
        num_layers: int,
        num_heads: int = 1
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = np.random.randn(vocab_size, d_model) * 0.1
        
        # Transformer blocks (each is a computational step)
        self.blocks = [
            TransformerBlock(d_model, d_ff, num_heads)
            for _ in range(num_layers)
        ]
        
        # Output projection
        self.output_proj = np.random.randn(d_model, vocab_size) * 0.1
    
    def encode(self, token_ids: List[int], max_length: Optional[int] = None) -> np.ndarray:
        """Encode tokens to embeddings"""
        if max_length:
            token_ids = token_ids[:max_length]
        
        embeddings = self.embedding[token_ids]
        return embeddings
    
    def forward(
        self,
        token_ids: List[int],
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Forward pass through the transformer.
        Each layer represents a step in the ITTM computation.
        """
        # Encode tokens
        x = self.encode(token_ids)
        
        all_attention_weights = []
        
        # Process through each transformer block
        for block in self.blocks:
            x, attention_weights = block.forward(x, mask)
            all_attention_weights.append(attention_weights)
        
        # Project to vocabulary
        logits = np.dot(x, self.output_proj)
        
        return logits, all_attention_weights
    
    def to_usm_representation(self, usm: UniversalStateMachine) -> Dict[str, List[str]]:
        """
        Convert transformer structure to USM knowledge graph representation.
        Each transformer block becomes a set of nodes in the knowledge graph.
        """
        node_mapping = {}
        
        for i, block in enumerate(self.blocks):
            # Create nodes for each block
            block_id = f"transformer_block_{i}"
            attention_id = f"attention_{i}"
            ff_id = f"feedforward_{i}"
            
            # Add to USM
            usm.add_knowledge(block_id, {"type": "transformer_block", "layer": i})
            usm.add_knowledge(attention_id, {"type": "attention", "layer": i})
            usm.add_knowledge(ff_id, {"type": "feedforward", "layer": i})
            
            # Connect components
            usm.knowledge_graph.add_edge(block_id, attention_id)
            usm.knowledge_graph.add_edge(attention_id, ff_id)
            
            # Connect layers
            if i > 0:
                prev_block_id = f"transformer_block_{i-1}"
                usm.knowledge_graph.add_edge(prev_block_id, block_id)
            
            node_mapping[f"layer_{i}"] = [block_id, attention_id, ff_id]
        
        return node_mapping


def create_causal_mask(sequence_length: int) -> np.ndarray:
    """Create causal mask for autoregressive generation"""
    mask = np.tril(np.ones((sequence_length, sequence_length)))
    return mask

