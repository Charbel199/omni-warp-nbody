import torch
import torch.nn as nn
from torch_cluster import radius_graph
from torch_geometric.nn import MessagePassing


class _MLP(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _GNSLayer(MessagePassing):

    def __init__(self, latent_dim: int = 128):
        super().__init__(aggr="add")
        self.edge_mlp = nn.Sequential(
            nn.Linear(latent_dim * 2 + latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.node_mlp(torch.cat([x, out], dim=-1))
        return x + out  # residual

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        return self.edge_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))


class NBodyGNN(nn.Module):

    def __init__(self, latent_dim: int = 128, num_layers: int = 3, cutoff: float = 5.0):
        super().__init__()
        self.cutoff = cutoff

        self.node_encoder = _MLP(4, latent_dim, latent_dim)
        self.edge_encoder = _MLP(7, latent_dim, latent_dim)

        self.gns_layers = nn.ModuleList(
            [_GNSLayer(latent_dim) for _ in range(num_layers)]
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 3),
        )

    def forward(
        self,
        pos: torch.Tensor,
        vel: torch.Tensor,
        mass: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)

        node_feat = torch.cat([vel, mass], dim=-1)
        x = self.node_encoder(node_feat)

        row, col = edge_index
        rel_pos = pos[col] - pos[row]
        dist = rel_pos.norm(dim=-1, keepdim=True)
        rel_vel = vel[col] - vel[row]
        edge_feat = torch.cat([rel_pos, dist, rel_vel], dim=-1)
        edge_attr = self.edge_encoder(edge_feat)

        for layer in self.gns_layers:
            x = layer(x, edge_index, edge_attr)

        return self.decoder(x)
