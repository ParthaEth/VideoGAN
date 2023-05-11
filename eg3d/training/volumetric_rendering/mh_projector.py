import torch
from positional_encodings.torch_encodings import PositionalEncodingPermute2D

class MHprojector(torch.nn.Module):
    def __init__(self, proj_dim, pos_enc_embed_dim, num_heads, query_dim):
        super().__init__()
        self.proj_dim = proj_dim
        self.planes_pos_encoder = PositionalEncodingPermute2D(proj_dim)
        self.mha = torch.nn.MultiheadAttention(proj_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False,
                                               add_zero_attn=False, kdim=proj_dim + self.proj_dim,
                                               vdim=proj_dim + self.proj_dim,
                                               batch_first=True, device=None, dtype=None)
        self.q_map = torch.nn.Linear(query_dim, self.proj_dim)

        self.feed_forward = torch.nn.Sequential(torch.nn.Linear(self.proj_dim, self.proj_dim),
                                                torch.nn.ReLU(self.proj_dim),
                                                torch.nn.Linear(self.proj_dim, self.proj_dim))
        self.layer_norm = torch.nn.LayerNorm(self.proj_dim)

    def forward(self, plane_features, query_pt):
        """
            :param plane_features: (batch, ch, h, w).
            :param query_pt: (batch, n_points, 3).
        """
        batch, ch, h, w = plane_features.shape
        assert self.proj_dim == ch
        key_val_feature_planes = torch.cat((plane_features, self.planes_pos_encoder(plane_features)), dim=1)
        key_val_feature_planes = key_val_feature_planes.permute(0, 2, 3, 1)  # batch, h, w, ch

        key_val_feature_planes = key_val_feature_planes.reshape(batch * h, w, self.proj_dim + self.proj_dim)

        # import ipdb; ipdb.set_trace()
        batch, num_pts, pt_dim = query_pt.shape
        assert pt_dim == 3
        query_pt = query_pt.reshape(batch * h, -1, 3)
        attn_output, _ = self.mha(query=self.q_map(query_pt), key=key_val_feature_planes, value=key_val_feature_planes)
        attn_output = attn_output + self.feed_forward(attn_output)
        return self.layer_norm(attn_output).reshape(batch, num_pts, -1) # dim: batch, num_pts, features

