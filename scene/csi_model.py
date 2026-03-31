
import torch
import torch.nn as nn


# ---- CSI encoder: uplink CSI -> TX position ----
class CSIEncoder(nn.Module):

    def __init__(self, n_antennas=8, n_subcarriers=26, hidden_dim=256, latent_dim=3, pos_scale=2.0):
        super().__init__()
        input_dim = n_antennas * n_subcarriers * 2
        self.latent_dim = latent_dim
        self.pos_scale = pos_scale
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )

    def forward(self, uplink_re, uplink_im):
        squeeze = uplink_re.dim() == 2
        if squeeze:
            uplink_re = uplink_re.unsqueeze(0)
            uplink_im = uplink_im.unsqueeze(0)

        x = torch.cat([uplink_re, uplink_im], dim=-1)
        x = x.view(x.shape[0], -1)
        pos = self.net(x)
        pos = torch.tanh(pos) * self.pos_scale

        if squeeze:
            pos = pos.squeeze(0)
        return pos


# ---- CSI decoder: TX position -> downlink CSI ----
class CSIAutoDecoder(nn.Module):

    def __init__(self, n_antennas=8, n_subcarriers=26, hidden_dim=256, latent_dim=3):
        super().__init__()
        self.n_antennas = n_antennas
        self.n_subcarriers = n_subcarriers
        output_dim = n_antennas * n_subcarriers * 2

        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, position):
        squeeze = position.dim() == 1
        if squeeze:
            position = position.unsqueeze(0)

        x = self.net(position)
        x = x.view(-1, self.n_antennas, self.n_subcarriers * 2)
        re = x[..., :self.n_subcarriers]
        im = x[..., self.n_subcarriers:]

        if squeeze:
            re = re.squeeze(0)
            im = im.squeeze(0)
        return re, im
