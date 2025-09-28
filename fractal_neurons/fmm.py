import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any

import time # Import time module

class FMMNode:
    """Represents a single node in the Fractal Memory Matrix."""
    def __init__(self, semantic_vector: torch.Tensor, context_anchor: Optional[torch.Tensor] = None, parent_id: Optional[str] = None):
        self.id = str(torch.randint(0, 1000000000, (1,)).item()) # Unique ID for the node
        self.semantic_vector = semantic_vector # [dim]
        self.context_anchor = context_anchor # [dim], e.g., mean of input that created it
        self.parent_id = parent_id
        self.children_ids: List[str] = []
        self.last_accessed: float = 0.0 # For entropy-adaptive refresh
        self.entropy: float = 0.0 # Information entropy of content

    def add_child(self, child_id: str):
        self.children_ids.append(child_id)

    def update_access_time(self):
        self.last_accessed = time.time()

    def update_entropy(self, new_entropy: float):
        self.entropy = new_entropy

class FractalMemoryMatrix(nn.Module):
    """A hierarchical, recursive storage lattice for reasoning chains."""
    def __init__(self, dim: int, max_nodes: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_nodes = max_nodes
        self.nodes: Dict[str, FMMNode] = {}
        self.root_nodes: List[str] = [] # Top-level nodes without parents
        self.node_vectors: Dict[str, torch.Tensor] = {}

    def add_node(self, semantic_vector: torch.Tensor, context_anchor: Optional[torch.Tensor] = None, parent_id: Optional[str] = None) -> FMMNode:
        if len(self.nodes) >= self.max_nodes:
            # Implement a pruning strategy here (e.g., least recently used, lowest entropy)
            pass # For now, just don't add if full

        node = FMMNode(semantic_vector, context_anchor, parent_id)
        self.nodes[node.id] = node
        self.node_vectors[node.id] = semantic_vector

        if parent_id and parent_id in self.nodes:
            self.nodes[parent_id].add_child(node.id)
        else:
            self.root_nodes.append(node.id)
        return node

    def get_node(self, node_id: str) -> Optional[FMMNode]:
        return self.nodes.get(node_id)

    def get_semantic_vector(self, node_id: str) -> Optional[torch.Tensor]:
        return self.node_vectors.get(node_id)

    def retrieve(self, query_vector: torch.Tensor, top_k: int = 5) -> List[Tuple[FMMNode, float]]:
        """Retrieves top_k most similar nodes based on semantic vector similarity."""
        if not self.node_vectors:
            return []

        all_vectors = torch.stack(list(self.node_vectors.values()))
        similarities = F.cosine_similarity(query_vector, all_vectors)
        top_similarities, top_indices = torch.topk(similarities, min(top_k, len(self.node_vectors)))

        results = []
        node_ids = list(self.node_vectors.keys())
        for sim, idx in zip(top_similarities, top_indices):
            node_id = node_ids[idx]
            node = self.nodes[node_id]
            node.update_access_time()
            results.append((node, sim.item()))
        return results

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """FMM itself doesn't have a traditional forward pass, but this can be a placeholder for future integration."""
        return x

    def state_dict(self) -> Dict[str, Any]:
        # Convert FMMNode objects to a serializable format
        serializable_nodes = {
            node_id: {
                "semantic_vector": node.semantic_vector.cpu().tolist(),
                "context_anchor": node.context_anchor.cpu().tolist() if node.context_anchor is not None else None,
                "parent_id": node.parent_id,
                "children_ids": node.children_ids,
                "last_accessed": node.last_accessed.item(),
                "entropy": node.entropy.item(),
            }
            for node_id, node in self.nodes.items()
        }
        return {
            "dim": self.dim,
            "max_nodes": self.max_nodes,
            "nodes": serializable_nodes,
            "root_nodes": self.root_nodes,
        }

    def load_state_dict(self, state_dict: Dict[str, Any], device: torch.device):
        self.dim = state_dict["dim"]
        self.max_nodes = state_dict["max_nodes"]
        self.nodes = {}
        self.node_vectors = {}
        for node_id, node_data in state_dict["nodes"].items():
            semantic_vector = torch.tensor(node_data["semantic_vector"], device=device)
            context_anchor = torch.tensor(node_data["context_anchor"], device=device) if node_data["context_anchor"] is not None else None
            node = FMMNode(semantic_vector, context_anchor, node_data["parent_id"])
            node.id = node_id # Ensure ID is preserved
            node.children_ids = node_data["children_ids"]
            node.last_accessed = torch.tensor(node_data["last_accessed"])
            node.entropy = torch.tensor(node_data["entropy"])
            self.nodes[node_id] = node
            self.node_vectors[node_id] = semantic_vector
        self.root_nodes = state_dict["root_nodes"]


