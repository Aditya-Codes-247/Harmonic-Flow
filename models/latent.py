import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class DynamicLatentSpace(nn.Module):
    """
    Dynamic latent space that adapts based on feedback using reinforcement learning.
    This is a simplified version for demonstration purposes.
    """
    def __init__(
        self,
        input_dim=config.LATENT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        latent_dim=config.LATENT_DIM,
        adaptation_rate=0.1,
        dropout=config.DROPOUT
    ):
        super().__init__()
        
        self.adaptation_rate = adaptation_rate
        self.latent_dim = latent_dim
        
        # Encoder-decoder architecture for latent space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim * 2)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Value network for RL
        self.value_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # Feedback memory
        self.feedback_memory = None
        
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick for VAE.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, feedback=None):
        """
        Forward pass.
        
        Args:
            x: Input features
            feedback: Optional feedback signal for adaptation
            
        Returns:
            z: Latent representation
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            value: Value prediction
        """
        # Encode to latent space
        hidden = self.encoder(x)
        mu, logvar = torch.chunk(hidden, 2, dim=1)
        
        # Sample latent vector
        z = self.reparameterize(mu, logvar)
        
        # Compute state value (for RL)
        value = self.value_net(z)
        
        # Update based on feedback (if provided)
        if feedback is not None:
            self.adapt(z, feedback)
        
        return z, mu, logvar, value
    
    def decode(self, z):
        """
        Decode from latent space.
        
        Args:
            z: Latent representation
            
        Returns:
            reconstruction: Reconstructed input
        """
        return self.decoder(z)
    
    def adapt(self, z, feedback, update_memory=True):
        """
        Adapt latent space based on feedback.
        
        Args:
            z: Current latent representation
            feedback: Feedback signal (positive/negative)
            update_memory: Whether to update feedback memory
        """
        # Simple adaptive mechanism: move latent representation toward 
        # positive feedback examples and away from negative ones
        if update_memory:
            if self.feedback_memory is None:
                self.feedback_memory = {
                    'positive': [],
                    'negative': []
                }
            
            # Store feedback in memory (limited to 100 examples)
            if feedback > 0 and len(self.feedback_memory['positive']) < 100:
                self.feedback_memory['positive'].append(z.detach())
            elif feedback < 0 and len(self.feedback_memory['negative']) < 100:
                self.feedback_memory['negative'].append(z.detach())
        
        # No adaptation needed if no memory
        if self.feedback_memory is None:
            return
        
        # Compute adaptation signal
        adaptation_signal = torch.zeros_like(z)
        
        # Attract toward positive examples
        if self.feedback_memory['positive']:
            positive_examples = torch.stack(self.feedback_memory['positive'], dim=0)
            positive_centroid = positive_examples.mean(dim=0)
            attraction = positive_centroid - z
            adaptation_signal += attraction
        
        # Repel from negative examples
        if self.feedback_memory['negative']:
            negative_examples = torch.stack(self.feedback_memory['negative'], dim=0)
            negative_centroid = negative_examples.mean(dim=0)
            repulsion = z - negative_centroid
            adaptation_signal += repulsion
        
        # Apply adaptation signal
        z_adapted = z + self.adaptation_rate * adaptation_signal
        
        return z_adapted


class NeuralAudioGraph(nn.Module):
    """
    Graph Neural Network for modeling relationships between instruments,
    harmonics, and temporal patterns.
    """
    def __init__(
        self,
        node_dim=config.LATENT_DIM,
        edge_dim=config.LATENT_DIM // 4,
        hidden_dim=config.HIDDEN_DIM,
        num_instruments=len(config.INSTRUMENTS),
        num_temporal_nodes=16,
        num_layers=3,
        dropout=config.DROPOUT
    ):
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_instruments = num_instruments
        self.num_temporal_nodes = num_temporal_nodes
        self.total_nodes = num_instruments + num_temporal_nodes
        
        # Node embeddings for instruments and temporal nodes
        self.node_embeddings = nn.Parameter(
            torch.randn(self.total_nodes, node_dim)
        )
        
        # Edge prediction network (determines connectivity)
        self.edge_predictor = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Edge feature network
        self.edge_feature_net = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, edge_dim)
        )
        
        # Graph convolution layers
        self.conv_layers = nn.ModuleList([
            GATConv(
                in_channels=node_dim if i == 0 else hidden_dim,
                out_channels=hidden_dim,
                heads=4,
                dropout=dropout,
                concat=False
            ) for i in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_dim)
        )
        
    def _construct_graph(self, features=None):
        """
        Construct the graph structure.
        
        Args:
            features: Optional node features to use instead of learned embeddings
            
        Returns:
            node_features: Node features
            edge_index: Edge indices
            edge_features: Edge features
        """
        # Use provided features or learned embeddings
        node_features = features if features is not None else self.node_embeddings
        batch_size = 1 if features is None else features.size(0)
        
        # For each pair of nodes, compute edge probability
        edges = []
        edge_features_list = []
        
        for i in range(self.total_nodes):
            for j in range(self.total_nodes):
                if i != j:  # No self-loops
                    # Concatenate node features
                    node_i = node_features[i].unsqueeze(0) if features is None else node_features[:, i, :]
                    node_j = node_features[j].unsqueeze(0) if features is None else node_features[:, j, :]
                    pair_features = torch.cat([node_i, node_j], dim=-1)
                    
                    # Predict edge probability
                    edge_prob = self.edge_predictor(pair_features)
                    
                    # Add edge if probability is high enough
                    if edge_prob.item() > 0.5:
                        edges.append([i, j])
                        
                        # Compute edge features
                        edge_feats = self.edge_feature_net(pair_features)
                        edge_features_list.append(edge_feats)
        
        # Convert to tensor
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_features = torch.cat(edge_features_list, dim=0)
        else:
            # Fallback: create a fully connected graph
            edge_index = torch.tensor([[i, j] for i in range(self.total_nodes) for j in range(self.total_nodes) if i != j],
                                     dtype=torch.long).t().contiguous()
            
            # Create dummy edge features
            node_pairs = torch.cat([
                node_features.repeat_interleave(self.total_nodes - 1, dim=0),
                node_features.repeat(self.total_nodes - 1, 1)
            ], dim=-1)
            edge_features = self.edge_feature_net(node_pairs)
        
        return node_features, edge_index, edge_features
    
    def forward(self, node_features=None):
        """
        Forward pass.
        
        Args:
            node_features: Optional node features (batch_size, num_nodes, node_dim)
            
        Returns:
            output_features: Updated node features
            edge_index: Edge indices
            edge_features: Edge features
        """
        # Construct graph
        node_feats, edge_index, edge_features = self._construct_graph(node_features)
        
        # Apply graph convolutions
        x = node_feats
        
        for conv in self.conv_layers:
            x = conv(x, edge_index)
            x = F.leaky_relu(x, 0.2)
        
        # Output projection
        output_features = self.output_proj(x)
        
        return output_features, edge_index, edge_features
    
    def extract_instrument_features(self, output_features):
        """
        Extract instrument-specific features from output features.
        
        Args:
            output_features: Updated node features
            
        Returns:
            instrument_features: Dictionary of instrument features
        """
        instrument_features = {}
        
        for i, instrument in enumerate(config.INSTRUMENTS):
            instrument_features[instrument] = output_features[i]
        
        return instrument_features


class LatentSpaceModule(nn.Module):
    """
    Complete latent space module integrating dynamic latent space and neural audio graph.
    """
    def __init__(
        self,
        dynamic_latent=None,
        neural_graph=None,
        input_dim=config.LATENT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        latent_dim=config.LATENT_DIM,
        dropout=config.DROPOUT
    ):
        super().__init__()
        
        self.dynamic_latent = dynamic_latent if dynamic_latent else DynamicLatentSpace(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout
        )
        
        self.neural_graph = neural_graph if neural_graph else NeuralAudioGraph(
            node_dim=latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # Integration layer
        self.integration_layer = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim)
        )
    
    def forward(self, features, feedback=None):
        """
        Forward pass.
        
        Args:
            features: Input features
            feedback: Optional feedback signal
            
        Returns:
            integrated_latent: Integrated latent representation
            outputs: Dictionary of intermediate outputs
        """
        # Pass through dynamic latent space
        z, mu, logvar, value = self.dynamic_latent(features, feedback)
        
        # Create node features for neural graph
        # For simplicity, we'll use the same latent representation for all nodes
        node_features = z.unsqueeze(1).repeat(1, self.neural_graph.total_nodes, 1)
        
        # Pass through neural audio graph
        graph_features, edge_index, edge_features = self.neural_graph(node_features)
        
        # Extract instrument features
        instrument_graph_features = self.neural_graph.extract_instrument_features(graph_features)
        
        # Average instrument features to get graph latent
        graph_latent = torch.stack(list(instrument_graph_features.values()), dim=0).mean(dim=0)
        
        # Integrate dynamic latent and graph latent
        combined_latent = torch.cat([z, graph_latent], dim=1)
        integrated_latent = self.integration_layer(combined_latent)
        
        outputs = {
            'latent': z,
            'mu': mu,
            'logvar': logvar,
            'value': value,
            'graph_features': graph_features,
            'edge_index': edge_index,
            'edge_features': edge_features,
            'instrument_graph_features': instrument_graph_features,
            'graph_latent': graph_latent,
            'integrated_latent': integrated_latent
        }
        
        return integrated_latent, outputs 