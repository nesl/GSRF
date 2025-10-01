import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedder(nn.Module):
    def __init__(self, **kwargs):
        super(Embedder, self).__init__()

        self.kwargs = kwargs

        self.create_embedding_fn()


    def create_embedding_fn(self):
        embed_fns = []

        d = self.kwargs["input_dims"]

        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]

        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)

        for freq in freq_bands:
            
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim


    def forward(self, inputs):

        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class Update_SH_Coeffs(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Update_SH_Coeffs, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, output_size)
        )


    def forward(self, tvec_tx, shs_coeffs):
        N = shs_coeffs.size(0)
        
        tvec_tx = tvec_tx.repeat(N, 1)  
        
        sig_amp, sig_pha = shs_coeffs.split(1, dim=-1)  

        sig_amp = torch.cat((tvec_tx, sig_amp.squeeze(dim=-1)), dim=-1)
        sig_pha = torch.cat((tvec_tx, sig_pha.squeeze(dim=-1)), dim=-1)

        sig_amp = self.model(sig_amp)
        sig_pha = self.model(sig_pha)

        shs_coeffs_updated = torch.stack((sig_amp, sig_pha), dim=-1)

        return shs_coeffs_updated



