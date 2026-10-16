"""Topology-conditioned residual refinement for existing lane queries."""

import torch
import torch.nn as nn


class TopologyGeometryRefinementHead(nn.Module):
    """Refine lane geometry without adding nodes to the predicted graph."""

    def __init__(self,
                 embed_dims=256,
                 num_points=11,
                 residual_scale=(6.0, 6.0, 1.0),
                 detach_inputs=True,
                 detach_graph=True,
                 graph_threshold=0.5,
                 gate_bias=-2.0):
        super().__init__()
        self.num_points = num_points
        self.detach_inputs = detach_inputs
        self.detach_graph = detach_graph
        self.graph_threshold = graph_threshold
        self.register_buffer(
            'residual_scale', torch.tensor(residual_scale).view(1, 1, 1, 3))

        self.geometry_embedding = nn.Sequential(
            nn.Linear(8, embed_dims), nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims))
        self.fusion = nn.Sequential(
            nn.Linear(embed_dims * 4, embed_dims),
            nn.LayerNorm(embed_dims), nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims), nn.ReLU(inplace=True))
        self.residual = nn.Linear(embed_dims, num_points * 3)
        self.gate = nn.Linear(embed_dims, 1)

        # Loading the baseline starts as an exact identity mapping.
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_bias)

    def forward(self, lane_feats, lane_points, lclc_logits):
        if self.detach_inputs:
            lane_feats = lane_feats.detach()
            lane_points = lane_points.detach()
            lclc_logits = lclc_logits.detach()

        adjacency = lclc_logits.squeeze(-1).sigmoid()
        if self.detach_graph:
            adjacency = adjacency.detach()
        if self.graph_threshold is not None:
            adjacency = adjacency * (adjacency > self.graph_threshold).to(
                adjacency.dtype)
        incoming = adjacency.transpose(1, 2)

        out_degree = adjacency.sum(-1, keepdim=True)
        in_degree = incoming.sum(-1, keepdim=True)
        out_context = torch.bmm(adjacency, lane_feats) / (out_degree + 1e-6)
        in_context = torch.bmm(incoming, lane_feats) / (in_degree + 1e-6)

        start = lane_points[:, :, 0]
        end = lane_points[:, :, -1]
        expected_start = torch.bmm(incoming, end) / (in_degree + 1e-6)
        expected_end = torch.bmm(adjacency, start) / (out_degree + 1e-6)
        has_in = (in_degree > 0.5).to(lane_points.dtype)
        has_out = (out_degree > 0.5).to(lane_points.dtype)
        start_offset = (expected_start - start) * has_in
        end_offset = (expected_end - end) * has_out
        geometry = self.geometry_embedding(torch.cat(
            [start_offset, end_offset, has_in, has_out], dim=-1))

        fused = self.fusion(torch.cat(
            [lane_feats, in_context, out_context, geometry], dim=-1))
        raw_residual = self.residual(fused).reshape(
            lane_points.shape[0], lane_points.shape[1], self.num_points, 3)
        bounded_residual = raw_residual.tanh() * self.residual_scale.to(
            raw_residual)
        gate = self.gate(fused).sigmoid()
        residual = gate.unsqueeze(-1) * bounded_residual
        return dict(
            lane_points=lane_points + residual,
            residual=residual,
            gate=gate,
            graph_support=torch.cat([has_in, has_out], dim=-1))
