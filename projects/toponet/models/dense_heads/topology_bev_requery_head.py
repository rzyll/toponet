"""Topology-guided BEV re-query refinement for existing lane queries."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopologyBEVRequeryHead(nn.Module):
    """Gather fresh BEV evidence around an E5-refined lane."""

    def __init__(self,
                 embed_dims=256,
                 num_points=11,
                 num_samples=5,
                 sample_radius=1.5,
                 learned_offset_scale=2.0,
                 residual_scale=(2.0, 2.0, 0.5),
                 graph_threshold=0.5,
                 detach_inputs=True,
                 detach_graph=True,
                 num_heads=8,
                 ffn_dims=512,
                 dropout=0.1):
        super().__init__()
        if num_samples != 5:
            raise ValueError('E9 currently expects five BEV sampling points')
        self.num_points = num_points
        self.num_samples = num_samples
        self.learned_offset_scale = learned_offset_scale
        self.graph_threshold = graph_threshold
        self.detach_inputs = detach_inputs
        self.detach_graph = detach_graph
        self.register_buffer(
            'residual_scale', torch.tensor(residual_scale).view(1, 1, 1, 3))
        self.register_buffer('fixed_offsets', torch.tensor([
            [0.0, 0.0], [sample_radius, 0.0], [-sample_radius, 0.0],
            [0.0, sample_radius], [0.0, -sample_radius]
        ]).view(1, 1, 1, num_samples, 2))

        self.geometry_embedding = nn.Sequential(
            nn.Linear(8, embed_dims), nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims))
        self.reference_fusion = nn.Sequential(
            nn.Linear(embed_dims * 4, embed_dims),
            nn.LayerNorm(embed_dims), nn.ReLU(inplace=True))
        self.sampling_offsets = nn.Linear(
            embed_dims, num_points * num_samples * 2)
        self.attention_weights = nn.Linear(
            embed_dims, num_points * num_samples)
        self.point_fusion = nn.Sequential(
            nn.Linear(embed_dims * 6, embed_dims),
            nn.LayerNorm(embed_dims), nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims), nn.ReLU(inplace=True))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dims, nhead=num_heads,
            dim_feedforward=ffn_dims, dropout=dropout)
        self.point_encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.residual = nn.Linear(embed_dims, 3)

        nn.init.zeros_(self.sampling_offsets.weight)
        nn.init.zeros_(self.sampling_offsets.bias)
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    @staticmethod
    def _sample_bev(bev_feats, points, bev_h, bev_w, pc_range):
        bs, _, channels = bev_feats.shape
        bev = bev_feats.reshape(bs, bev_h, bev_w, channels).permute(0, 3, 1, 2)
        original_shape = points.shape[1:-1]
        flat = points.reshape(bs, -1, 3)
        x = 2 * (flat[..., 0] - pc_range[0]) / (
            pc_range[3] - pc_range[0]) - 1
        y = 2 * (flat[..., 1] - pc_range[1]) / (
            pc_range[4] - pc_range[1]) - 1
        grid = torch.stack([x, y], dim=-1).unsqueeze(2)
        sampled = F.grid_sample(
            bev, grid, mode='bilinear', padding_mode='zeros',
            align_corners=False)
        sampled = sampled.squeeze(-1).permute(0, 2, 1)
        return sampled.reshape(bs, *original_shape, channels)

    def forward(self, lane_feats, lane_points, lclc_logits, bev_feats,
                bev_h, bev_w, pc_range):
        if self.detach_inputs:
            lane_feats = lane_feats.detach()
            lane_points = lane_points.detach()
            lclc_logits = lclc_logits.detach()
            bev_feats = bev_feats.detach()

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
        has_in = (in_degree > 0.5).to(lane_points.dtype)
        has_out = (out_degree > 0.5).to(lane_points.dtype)
        expected_start = torch.bmm(incoming, end) / (in_degree + 1e-6)
        expected_end = torch.bmm(adjacency, start) / (out_degree + 1e-6)
        start_offset = (expected_start - start) * has_in
        end_offset = (expected_end - end) * has_out
        geometry = self.geometry_embedding(torch.cat(
            [start_offset, end_offset, has_in, has_out], dim=-1))
        reference_query = self.reference_fusion(torch.cat(
            [lane_feats, in_context, out_context, geometry], dim=-1))

        steps = torch.linspace(
            0, 1, self.num_points, dtype=lane_points.dtype,
            device=lane_points.device).view(1, 1, self.num_points, 1)
        topology_offset = ((1 - steps) * start_offset.unsqueeze(2) +
                           steps * end_offset.unsqueeze(2))
        guided_points = lane_points + topology_offset

        bs, num_lane = lane_points.shape[:2]
        learned_offsets = self.sampling_offsets(reference_query).reshape(
            bs, num_lane, self.num_points, self.num_samples, 2)
        learned_offsets = learned_offsets.tanh() * self.learned_offset_scale
        sample_offsets = self.fixed_offsets.to(learned_offsets) + learned_offsets
        sample_points = guided_points.unsqueeze(3).expand(
            -1, -1, -1, self.num_samples, -1).clone()
        sample_points[..., :2] = sample_points[..., :2] + sample_offsets
        sampled = self._sample_bev(
            bev_feats, sample_points, bev_h, bev_w, pc_range)
        attention = self.attention_weights(reference_query).reshape(
            bs, num_lane, self.num_points, self.num_samples).softmax(-1)
        attended_bev = (sampled * attention.unsqueeze(-1)).sum(3)
        current_bev = self._sample_bev(
            bev_feats, lane_points, bev_h, bev_w, pc_range)

        def expand(value):
            return value.unsqueeze(2).expand(-1, -1, self.num_points, -1)

        point_features = self.point_fusion(torch.cat([
            expand(lane_feats), expand(in_context), expand(out_context),
            expand(geometry), current_bev, attended_bev], dim=-1))
        encoded = self.point_encoder(
            point_features.reshape(
                bs * num_lane, self.num_points, -1).transpose(0, 1)
        ).transpose(0, 1).reshape(bs, num_lane, self.num_points, -1)
        residual = self.residual(encoded).tanh() * self.residual_scale.to(encoded)
        return dict(
            lane_points=lane_points + residual,
            original_lane_points=lane_points,
            residual=residual,
            guided_offset=topology_offset,
            sampling_offset=sample_offsets,
            graph_support=torch.cat([has_in, has_out], dim=-1))
