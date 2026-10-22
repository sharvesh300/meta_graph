"""Crystal Graph Convolutional Neural Network, vendored from txie-93/cgcnn.

Source: https://github.com/txie-93/cgcnn/blob/f42ab233c4ee0c416879d6bc2d22a264418413ad/cgcnn/model.py

    MIT License
    Copyright (c) 2018 Tian Xie

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

Changes from the original, none of which alter parameter names (the
pre-trained state_dicts load unchanged) or the computed values:
- crystal_atom_idx is a single LongTensor mapping each atom to its crystal,
  and pooling is a vectorised scatter-mean instead of a per-crystal loop;
- forward(..., return_embedding=True) also returns the pooled crystal vector
  (atom_fea_len dims, before conv_to_fc) for structural retrieval;
- torch.autograd.Variable and Python 2 compatibility imports are dropped.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    """Gated graph convolution over each atom's M neighbours."""

    def __init__(self, atom_fea_len: int, nbr_fea_len: int):
        super().__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2 * self.atom_fea_len + self.nbr_fea_len, 2 * self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2 * self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea: torch.Tensor, nbr_fea: torch.Tensor, nbr_fea_idx: torch.Tensor) -> torch.Tensor:
        """atom_in_fea (N, atom_fea_len), nbr_fea (N, M, nbr_fea_len), nbr_fea_idx (N, M)."""
        N, M = nbr_fea_idx.shape
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len), atom_nbr_fea, nbr_fea], dim=2
        )
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(-1, self.atom_fea_len * 2)).view(N, M, self.atom_fea_len * 2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        return self.softplus2(atom_in_fea + nbr_sumed)


class CrystalGraphConvNet(nn.Module):
    """Embedding -> n_conv ConvLayers -> mean pooling -> MLP head."""

    def __init__(
        self,
        orig_atom_fea_len: int,
        nbr_fea_len: int,
        atom_fea_len: int = 64,
        n_conv: int = 3,
        h_fea_len: int = 128,
        n_h: int = 1,
        classification: bool = False,
    ):
        super().__init__()
        self.classification = classification
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList(
            [ConvLayer(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len) for _ in range(n_conv)]
        )
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h - 1)])
            self.softpluses = nn.ModuleList([nn.Softplus() for _ in range(n_h - 1)])
        self.fc_out = nn.Linear(h_fea_len, 2 if self.classification else 1)
        if self.classification:
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()

    def forward(
        self,
        atom_fea: torch.Tensor,
        nbr_fea: torch.Tensor,
        nbr_fea_idx: torch.Tensor,
        crystal_atom_idx: torch.Tensor,
        return_embedding: bool = False,
    ):
        """crystal_atom_idx (N,) gives each atom's crystal index (0..N0-1).

        Returns the output (N0, 1) - or (N0, 2) log-probabilities for
        classification - and, with return_embedding, also the pooled crystal
        vectors (N0, atom_fea_len).
        """
        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        embedding = self.pooling(atom_fea, crystal_atom_idx)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(embedding))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        if self.classification:
            crys_fea = self.dropout(crys_fea)
        if hasattr(self, "fcs") and hasattr(self, "softpluses"):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        out = self.fc_out(crys_fea)
        if self.classification:
            out = self.logsoftmax(out)
        return (out, embedding) if return_embedding else out

    @staticmethod
    def pooling(atom_fea: torch.Tensor, crystal_atom_idx: torch.Tensor) -> torch.Tensor:
        """Mean of each crystal's atom features."""
        n_crystals = int(crystal_atom_idx.max()) + 1
        summed = atom_fea.new_zeros(n_crystals, atom_fea.shape[1]).index_add_(0, crystal_atom_idx, atom_fea)
        counts = torch.bincount(crystal_atom_idx, minlength=n_crystals).clamp(min=1).to(atom_fea.dtype)
        return summed / counts.unsqueeze(1)
