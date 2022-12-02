from functools import partial
import math
from typing import Union, Dict
import os
import pytorch_lightning as pl
from e3nn import o3
from e3nn.math import soft_unit_step
from e3nn.util.jit import compile_mode
import torch
from torch.nn import functional as F
from torch import nn
from einops import rearrange
from cg import NUM_CG_TYPES
from utils_data import NUM_EDGE_TYPE, MAX_DIST
from utils import (compute_X_uv, compute_X_uv_pred, compute_X_v_pred,
                   compute_FAPE_uv, quaternion_slerp, compute_d_ijab,
                   compute_d_ijab_pred, compute_x_pdb, compute_struct_loss,
                   R_from_quaternion_u, compose_rotations)


def compute_init_struct(init_scheme, resnum, dtype):
    device = resnum.device
    N_cg = len(resnum)
    T_size = (N_cg, 3)
    if init_scheme == "blackhole":
        R_pred = torch.eye(3, device=device, dtype=dtype).repeat((N_cg, 1, 1))
        T_pred = torch.zeros(T_size, device=device, dtype=dtype)
    elif init_scheme == "random":
        R_pred = o3.rand_matrix(N_cg, device=device, dtype=dtype)
        T_init_sig = 1.0
        T_pred = T_init_sig * torch.randn(T_size, device=device, dtype=dtype)
    else:
        raise NotImplementedError

    return R_pred, T_pred


def compute_weight_cutoff(edge_length, rc):
    return soft_unit_step(10 * (1 - edge_length.unsqueeze(-1) / rc))


@compile_mode('script')
class MLP(torch.nn.Module):
    def __init__(self, num_neurons, activation, apply_layer_norm=False):
        super(MLP, self).__init__()
        self.activation = activation
        self.layers = torch.nn.ModuleList()
        self.apply_layer_norm = apply_layer_norm
        if apply_layer_norm:
            self.layer_norms = torch.nn.ModuleList()
            idx = 0
        for nin, nout in zip(num_neurons[:-1], num_neurons[1:]):
            self.layers.append(torch.nn.Linear(nin, nout, bias=True))
            if self.apply_layer_norm:
                if idx < len(num_neurons) - 1:
                    self.layer_norms.append(torch.nn.LayerNorm(nout))
                idx += 1

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x))
            if self.apply_layer_norm:
                x = layer_norm[i](x)
        x = self.layers[-1](x)

        return x


@compile_mode('script')
class BesselBasis(torch.nn.Module):
    """This would be more aptly called sinusoidal radial basis given the implementation
    """
    def __init__(
        self,
        rc,
        radial_num_basis=16
    ) -> None:
        super().__init__()
        self.rc = rc
        self.radial_num_basis = radial_num_basis
        self.prefactor = 2.0 / self.rc

        bessel_weights = (
            torch.linspace(start=1.0, end=self.radial_num_basis, steps=self.radial_num_basis) * math.pi
        )
        self.bessel_weights = torch.nn.Parameter(bessel_weights)

    def forward(self, x):
        return self.prefactor * torch.sin(self.bessel_weights[None, None, :] *  x[:, :, None] / self.rc) # / x.unsqueeze(-1)


@compile_mode('script')
class RadialNN(torch.nn.Module):
    def __init__(
        self,
        num_out_features,
        rc,
        radial_num_basis=16,
        radial_num_hidden=16,
        radial_num_layers=2,
        include_edge_features=False,
        num_edge_features=None
    ) -> None:
        super().__init__()
        self.num_out_features = num_out_features
        self.rc = rc
        self.radial_num_basis = radial_num_basis
        self.radial_num_hidden = radial_num_hidden
        self.radial_num_layers = radial_num_layers
        self.include_edge_features = include_edge_features
        self.num_edge_features = num_edge_features
        if self.include_edge_features:
            assert type(num_edge_features) is int

        # ---- MLP
        self.mlp = MLP([self.radial_num_basis + self.num_edge_features if self.include_edge_features else self.radial_num_basis] + \
                      [self.radial_num_hidden] * self.radial_num_layers + \
                      [self.num_out_features],
                       F.silu)
        # ---- bassel basis
        self.bessels = BesselBasis(self.rc, self.radial_num_basis)

    def forward(self, r_ij, edges_ij, weight_cutoff=None):
        # compute basis
        inputs = self.bessels(r_ij) 
        if weight_cutoff is not None:
            inputs = inputs * weight_cutoff

        # combine edge features
        if self.include_edge_features:
            inputs = torch.cat([inputs, edges_ij], dim=-1)

        weight = self.mlp(inputs)

        return weight


@compile_mode('script')
class LayerNorm(torch.nn.Module):
    def __init__(self, irreps):
        super().__init__()
        """implement layernorm in the equiformer"""
        self.irreps = irreps
        self.gamma_s = torch.nn.Parameter(torch.ones(self.irreps[0]))
        self.beta_s = torch.nn.Parameter(torch.zeros(self.irreps[0]))
        self.gamma_v = torch.nn.Parameter(torch.ones(self.irreps[1]))
    
    def forward(self, s, v):
        # -- scalar
        x = s
        # subtact mean
        mu = x.mean(dim=1, keepdim=True)
        x = x - mu
        # normalize and rms
        square_norm_x = x.square()
        rms = (square_norm_x.mean(dim=1) + 1e-6).sqrt() # [N]
        # apply params
        s = self.gamma_s[None, :] * x / rms[:, None] + self.beta_s[None, :]

        # -- vector
        x = v
        # normalize and rms
        square_norm_x = x.square()
        rms = (square_norm_x.sum(dim=[1, 2]) / self.irreps[1] + 1e-6).sqrt() # [N]
        # apply params
        v = self.gamma_v[None, :, None] * x / rms[:, None, None]

        return s, v


@compile_mode('script')
class Emb(torch.nn.Module):
    def __init__(
        self,
        num_node_types,
        nc
    ) -> None:
        super().__init__()

        # input node embedding
        self.nc = nc
        self.d_s = nc
        self.d_v = 3 * nc
        self.num_node_types = num_node_types
        self.embed_s = torch.nn.Embedding(num_node_types, self.d_s, padding_idx=-1, max_norm=1, norm_type=2.0,
                                          scale_grad_by_freq=False, sparse=False)
        self.embed_v = torch.nn.Embedding(num_node_types, self.d_v, padding_idx=-1, max_norm=1, norm_type=2.0,
                                          scale_grad_by_freq=False, sparse=False)

    def forward(self, nodes, R):
        s = self.embed_s(nodes)
        v = self.embed_v(nodes).reshape(len(nodes), self.nc, 3) # [N_CG, nc, 3]

        return s, rotate_embedding(v, R)


def rotate_embedding(v, R):
    return torch.einsum("rij,rkj->rki", R, v)


@compile_mode('script')
class Linear(torch.nn.Module):
    def __init__(
        self,
        nc_s_in,
        nc_s_out,
        nc_v_in,
        nc_v_out,
        add_bias=False
    ) -> None:
        super().__init__()
        self.nc_s_in=nc_s_in
        self.nc_s_out=nc_s_out
        self.nc_v_in=nc_v_in
        self.nc_v_out=nc_v_out

        assert (nc_v_out > 0) or (nc_s_out > 0)

        if nc_s_out > 0:
            w_s = torch.empty((nc_s_out, nc_s_in))
            nn.init.xavier_uniform_(w_s, gain=1)
            self.w_s = torch.nn.Parameter(w_s)
            self.add_bias = add_bias
            if self.add_bias:
                self.b_s = torch.nn.Parameter(torch.zeros(nc_s_out))

        if nc_v_out > 0:
            w_v = torch.empty((nc_v_out, nc_v_in))
            nn.init.xavier_uniform_(w_v, gain=1)
            self.w_v = torch.nn.Parameter(w_v)

    def forward(self, s, v):
        if self.nc_s_out > 0:
            s = torch.einsum("ij,...j->...i", self.w_s, s)
            if self.add_bias:
                if len(s.size()) == 2:
                    s = s + self.b_s[None, :]
                elif len(s.size()) == 3:
                    s = s + self.b_s[None, None, :]
                else:
                    raise NotImplementedError
        else: 
            s = None
        v = torch.einsum("ij,...jk->...ik", self.w_v, v) if self.nc_v_out > 0 else None

        return s, v


class NN(pl.LightningModule):
    def __init__(self,
                 lr=1e-5,
                 wd=1e-8,
                 slerp_warmup=True,
                 lr_warmup=False,
                 lr_anneal=False,
                 lr_anneal_final_step=200000,
                 # number of channels
                 nc=32,              
                 # interaction type
                 interaction_type="attn-direct",
                 attn_num_heads=1,
                 # interaction module params
                 distinct_blocks=False,
                 distinct_embeddings=False,       
                 num_blocks=4,
                 num_layers=3,
                 rc=100., # cutoff
                 d_edge=32,
                 radial_num_basis=32,
                 radial_num_hidden=32,
                 # for CG initialization during training
                 warmup_steps=1,
                 init_scheme="blackhole",
                 apply_layer_norm=False,
                 attend_to_self=False,
                 disable_cutoff=False,
                 accumulate_grad_batches=1,
                 gradient_clip_val=5.0,
                 fape_clip_val=10.,
                 weight_struct_loss=1.0,
                 weight_struct_loss_scale="constant"
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.lr=lr
        self.lr_anneal=lr_anneal
        self.lr_anneal_final_step=lr_anneal_final_step
        self.slerp_warmup=slerp_warmup
        self.lr_warmup=lr_warmup
        self.wd=wd
        self.nc=nc
        self.num_blocks=num_blocks
        self.num_layers=num_layers
        self.rc=rc
        self.attn_num_heads=attn_num_heads
        self.interaction_type=interaction_type
        self.d_edge=d_edge
        self.radial_num_basis=radial_num_basis
        self.radial_num_hidden=radial_num_hidden
        self.warmup_steps=warmup_steps
        self.init_scheme=init_scheme
        self.distinct_blocks=distinct_blocks
        self.distinct_embeddings=distinct_embeddings
        self.apply_layer_norm=apply_layer_norm
        self.disable_cutoff=disable_cutoff
        self.accumulate_grad_batches=accumulate_grad_batches
        self.gradient_clip_val=gradient_clip_val
        self.fape_clip_val=fape_clip_val
        self.weight_struct_loss=weight_struct_loss
        self.weight_struct_loss_scale=weight_struct_loss_scale

        if self.distinct_embeddings:
            self.embs = torch.nn.ModuleList([Emb(NUM_CG_TYPES+1, self.nc) for _ in range(self.num_blocks)])
        else:
            self.emb = Emb(NUM_CG_TYPES+1, self.nc)

        self.attend_to_self=attend_to_self

        if distinct_blocks:
            self.enns = torch.nn.ModuleList([self.make_block() for _ in range(self.num_blocks)])
        else:        
            self.enn = self.make_block()

    def make_block(self):
        block = E3NN(nc=self.nc,
                     num_layers=self.num_layers,
                     rc=self.rc,
                     nonlinearity="gated",
                     include_edge_features=True,
                     d_embed_edge=self.d_edge,
                     num_edge_types=NUM_EDGE_TYPE,
                     radial_num_basis=self.radial_num_basis,
                     radial_num_hidden=self.radial_num_hidden,
                     radial_num_layers=2,
                     interaction_type=self.interaction_type,
                     attn_num_heads=self.attn_num_heads,
                     apply_layer_norm=self.apply_layer_norm,
                     attend_to_self=self.attend_to_self,
                     disable_cutoff=self.disable_cutoff)
        return block

    def forward(self, batch, compute_loss=False, return_struct=False, is_train=False,
                set_RT_to_ground_truth=False, skip_first=False):

        return_dict = {"losses_fape": [[] for _ in range(self.num_blocks + 1)], # eventually, list of floats; averaged over block; list comprehension necessary
                       "losses_bond": [[] for _ in range(self.num_blocks + 1)],
                       "losses_angle": [[] for _ in range(self.num_blocks + 1)],
                       "losses_clash": [[] for _ in range(self.num_blocks + 1)],
                       "R_pred": [], # list of lists
                       "T_pred": [], # list of lists
                       "X_pred": [], # list of lists
                       "x_pred": [], # list of lists; pdb
                       "loss_total": 0} # scalar

        N = len(batch)
        for b, data in enumerate(batch): 
            # ---- initialize predicted R,T 
            X0 = data["cg_X0"] # initial coordinates
            resnum = data["cg_resnum"]
            dtype = X0.dtype            
            R_pred, T_pred = compute_init_struct(self.init_scheme, resnum, dtype)

            if compute_loss:
                # ---- ground truth for FAPE loss
                mask = data["cg_mask"]
                # truth
                R = data["cg_R"]
                T = data["cg_T"]
                X = data["cg_X"]
                mask_atom = data["cg_atom_mask"]
                mask_amb = data["cg_amb"]
                # alt truth
                R_alt = data["cg_R_alt"]
                T_alt = data["cg_T_alt"]
                X_alt = data["cg_X_alt"]
                mask_atom_alt = data["cg_atom_mask_alt"]
                mask_amb_alt = data["cg_amb_alt"]

                # used for resolving ambiguity due to symmetry
                d_ijab, mask_ijab = compute_d_ijab(X, mask_atom, mask_amb)
                d_ijab_alt, mask_ijab_alt = compute_d_ijab(X_alt, mask_atom_alt, mask_amb_alt)

                # ---- attempt to help with initial training
                tau = min(1., self.trainer.global_step / self.warmup_steps) if is_train else 1.0
                if is_train and not set_RT_to_ground_truth:
                    # tau: 0 to 1 --> ground truth (minus centroid) to initial scheme
                    # for masked nodes, apply the initial scheme
                    if self.slerp_warmup:
                        if tau < 1.0:
                            unmasked = mask == 1.
                            T_pred[unmasked] = tau * T_pred[unmasked] + (1 - tau) * T[unmasked]
                            R_pred[unmasked] = quaternion_slerp(R[unmasked], R_pred[unmasked], tau)

                if set_RT_to_ground_truth:
                    R_pred = R
                    T_pred = T

            # ---- embed the nodes and steer
            # residue distance with max
            # [N_cg, N_cg]; i is the dest/query for attention
            with torch.no_grad():
                edge_type_ij = torch.clamp(resnum[None, :] - resnum[:, None],
                                           min=-MAX_DIST, max=MAX_DIST) + MAX_DIST
                edge_type_ij[edge_type_ij == (2 * MAX_DIST)] = 0 # treat min / max dist as equal "far away"

            # ---- iteratively update structure
            if return_struct:            
                return_dict["R_pred"].append([])
                return_dict["T_pred"].append([])
                return_dict["X_pred"].append([])
                return_dict["x_pred"].append([])

            for i in range(self.num_blocks+1):
                if i == 0 and skip_first:
                    # to avoid unnecessary comp during training
                    continue 

                # 0-th step is to store initial structure
                if i > 0:
                    # this has to be here since otherwise gradients won't be accumulated
                    if self.distinct_embeddings:
                        emb = self.embs[i-1]
                    else:
                        emb = self.emb

                    s, v = emb(data["cg_cgidx"], R_pred)

                    if self.distinct_blocks:
                        block = self.enns[i-1]
                    else:
                        block = self.enn

                    # predict updates
                    R_pred, T_pred = block(s, v, R_pred, T_pred, edge_type_ij)

                # compute current structure
                if return_struct or compute_loss:
                    X_v_pred = compute_X_v_pred(X0, R_pred, T_pred)
                    x_pred = compute_x_pdb(X_v_pred, data["scatter_index"], data["scatter_w"], len(data["dst_resnum"]))

                if return_struct:
                    return_dict["R_pred"][b].append(R_pred.detach())
                    return_dict["T_pred"][b].append(T_pred.detach())
                    return_dict["X_pred"][b].append(X_v_pred.detach())
                    return_dict["x_pred"][b].append(x_pred.detach())
                
                if compute_loss:
                    d_ijab_pred = compute_d_ijab_pred(X_v_pred)
                    X_uv, mask_atom_uv = compute_X_uv(mask, X, R, T, mask_atom,
                                                      X_alt, R_alt, T_alt, mask_atom_alt,
                                                      d_ijab, mask_ijab, 
                                                      d_ijab_alt, mask_ijab_alt,
                                                      d_ijab_pred)
                    X_uv_pred = compute_X_uv_pred(X_v_pred, R_pred, T_pred)
                    loss = compute_FAPE_uv(X_uv, mask_atom_uv, X_uv_pred, eps=1e-4, d_max=self.fape_clip_val)
                    return_dict["losses_fape"][i].append(loss.detach())

                    # does take that long
                    loss_bond, loss_angle, loss_clash = compute_struct_loss(x_pred, data)
                    return_dict["losses_bond"][i].append(loss_bond.detach())
                    return_dict["losses_angle"][i].append(loss_angle.detach())
                    return_dict["losses_clash"][i].append(loss_clash.detach())
                    loss_struct = loss_bond + loss_angle + loss_clash
                    if self.weight_struct_loss > 0.:
                        if self.weight_struct_loss_scale == "constant":
                            scale = 1.
                        elif self.weight_struct_loss_scale == "linear":
                            scale = i / self.num_blocks
                        elif self.weight_struct_loss_scale == "quadratic":
                            scale = (i / self.num_blocks)**2
                        else:
                            raise ValueError
                        loss = loss + tau * self.weight_struct_loss * loss_struct * scale

                    if i > 0:
                        if is_train:
                            loss_for_grad = loss / self.accumulate_grad_batches / N # normalization added # grad acc does appear here correctly
                            if not self.distinct_blocks:
                                assert not self.distinct_embeddings
                                loss_for_grad = loss_for_grad / self.num_blocks

                            # https://github.com/Lightning-AI/lightning/discussions/10792#discussioncomment-1712526
                            # only sync after last block of last sample?
                            if i == self.num_blocks and b == (N - 1):
                                self.manual_backward(loss_for_grad)
                            else:
                                with self.trainer.model.no_sync():
                                    self.manual_backward(loss_for_grad)

                        return_dict["loss_total"] = return_dict["loss_total"] + loss.detach()

                        # detach after loss calculation
                        R_pred = R_pred.detach()
                        T_pred = T_pred.detach()

        # final averages
        if compute_loss:
            for c in ["fape", "angle", "bond", "clash"]:
                return_dict[f"losses_{c}"] = [sum(x) / N for x in return_dict[f"losses_{c}"]] # grad acc does not appear here correctly
            return_dict["loss_total"] = return_dict["loss_total"] / N / self.num_blocks # mean over blocks

        return return_dict

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)

        return optimizer

    def training_step(self, train_batch, batch_idx):
        opt = self.optimizers()
        return_dict = self(train_batch, compute_loss=True, return_struct=False, is_train=True,
                           skip_first=True)
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip_val)
            opt.step()
            opt.zero_grad()
            if self.lr_warmup or self.lr_anneal:
                if self.trainer.global_step < self.warmup_steps:
                    lr_scale = min(1., float(self.trainer.global_step + 1) / self.warmup_steps)
                elif self.lr_anneal and self.trainer.global_step >= self.warmup_steps: # anneal
                    step_after_warmup = self.trainer.global_step - self.warmup_steps
                    if step_after_warmup < self.lr_anneal_final_step:
                        arg = (math.pi / 2.) * (step_after_warmup / self.lr_anneal_final_step)
                        lr_scale = 0.9 * math.cos(arg) + 0.1
                    else:
                        lr_scale = 0.1
                else: # only after warm up and if no aneal
                    lr_scale = 1.

                for pg in opt.param_groups:
                    pg['lr'] = lr_scale * self.lr

        self.log('train_loss', return_dict["loss_total"], batch_size=len(train_batch), sync_dist=True)
        for c in ["fape", "angle", "bond", "clash"]:    
            self.log(f'train_loss_{c}_final', return_dict[f"losses_{c}"][-1], batch_size=len(train_batch), sync_dist=True)
        return return_dict["loss_total"]

    def validation_step(self, val_batch, batch_idx):
        return_dict = self(val_batch, compute_loss=True, return_struct=False, is_train=False)
        self.log('val_loss', return_dict["loss_total"], batch_size=len(val_batch), sync_dist=True)
        for c in ["fape", "angle", "bond", "clash"]:    
            self.log(f'val_loss_{c}_final', return_dict[f"losses_{c}"][-1], batch_size=len(val_batch), sync_dist=True)
        return return_dict["loss_total"]

    def test_step(self, test_batch, batch_idx):
        return_dict = self(test_batch, compute_loss=True, return_struct=False, is_train=False)
        self.log('test_loss', return_dict["loss_total"], batch_size=len(test_batch), sync_dist=True)
        for c in ["fape", "angle", "bond", "clash"]:    
            self.log(f'test_loss_{c}_final', return_dict[f"losses_{c}"][-1], batch_size=len(test_batch), sync_dist=True)


@compile_mode('script')
class DTPByHead(torch.nn.Module):
    def __init__(
        self, 
        nc_s_in, nc_v_in, 
        nc_s_out, nc_v_out,
        num_heads) -> None:
        """
        depth-wise tensor product with SHs

        performs
        - DTP w/ provided weights or internal (uvu)
        - apply linear w/ bias
        """
        super().__init__()
        assert nc_s_in == nc_v_in
        self.nc_s_in = nc_s_in
        self.nc_v_in = nc_v_in
        self.num_heads = num_heads
        self.dim_post_dtp = 2 * nc_s_in
        self.weight_numel = 4 * nc_s_in * num_heads

        # determine tp out shapes
        self.nc_s_out = nc_s_out
        self.nc_v_out = nc_v_out

        # weights for linear
        # scalar
        w_s = torch.empty((num_heads, nc_s_out, self.dim_post_dtp))
        nn.init.xavier_uniform_(w_s, gain=1)
        self.w_s = torch.nn.Parameter(w_s)
        self.b_s = torch.nn.Parameter(torch.zeros((num_heads, nc_s_out)))
        # vector
        w_v = torch.empty((num_heads, nc_v_out, self.dim_post_dtp))
        nn.init.xavier_uniform_(w_v, gain=1)
        self.w_v = torch.nn.Parameter(w_v)

    def forward(self, s, v, r_ij_vec, weights):
        """reshaping is done at the input"""
        w_ss, w_sv, w_vs, w_vv = rearrange(weights, 'i j (c h m) -> c h i j m', c=4, h=self.num_heads)

        # tp
        ss = w_ss * s
        sv = w_sv.unsqueeze(-1) * s.unsqueeze(-1) * r_ij_vec.unsqueeze(-2)
        vs = w_vs.unsqueeze(-1) * v
        vv = w_vv * (v * r_ij_vec.unsqueeze(-2)).sum(-1)
        s = rearrange([ss, vv], 'c h i j m -> h i j (c m)')
        v = rearrange([sv, vs], 'c h i j m k -> h i j (c m) k')

        # apply linear
        s = torch.einsum("h m n, h i j n -> h i j m", self.w_s, s) + self.b_s[:, None, None, :]
        v = torch.einsum("h m n, h i j n k -> h i j m k", self.w_v, v)

        return s, v


@compile_mode('script')
class Equiformer(torch.nn.Module):
    """Implements Fig.1b of Equiformer, with several modifications"""
    def __init__(
        self,
        irreps_in, # (nc scalar, nc vector)
        irreps_out, # (nc scalar, nc vector)
        rc,
        radial_num_basis=16,
        radial_num_hidden=16,
        radial_num_layers=2,
        include_edge_features=False,
        num_edge_features=None,
        num_heads=1,
        apply_layer_norm=True, # for both attn and ff
        apply_resnet=True, # only concerns ff block
        apply_nonlinear=False, # this is a dummy param; no effect
        ff_mul=3,
        attend_to_self=False,
        interaction_type="attn-direct"
    ) -> None:
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out

        self.rc = rc
        self.radial_num_basis = radial_num_basis
        self.radial_num_hidden = radial_num_hidden
        self.radial_num_layers = radial_num_layers
        self.include_edge_features = include_edge_features
        self.num_edge_features = num_edge_features
        self.num_heads = num_heads
        self.apply_layer_norm = apply_layer_norm
        if apply_layer_norm:
            self.layer_norm_attn = LayerNorm(self.irreps_in)
            self.layer_norm_ff = LayerNorm(self.irreps_in)
        self.apply_resnet = apply_resnet
        self.interaction_type = interaction_type

        self.nc_s_in = nc_s_in = self.irreps_in[0]
        self.nc_v_in = nc_v_in = self.irreps_in[1]

        # ---- initial mixing
        self.linear_src = Linear(nc_s_in, nc_s_in, nc_v_in, nc_v_in, add_bias=True)
        self.linear_dst = Linear(nc_s_in, nc_s_in, nc_v_in, nc_v_in, add_bias=True)
        assert nc_v_in == nc_s_in

        # -- linear after tp
        self.nc_by_head = nc_s_in // num_heads  # ex: 8 = 32 / 4        
        nc_middle = self.nc_by_head # ex: 8
        nc_s_in_by_head = nc_v_in_by_head = 2 * self.nc_by_head
        # scalar
        w_s = torch.empty((num_heads, nc_middle, nc_s_in_by_head))
        nn.init.xavier_uniform_(w_s, gain=1)
        self.w_s_init = torch.nn.Parameter(w_s)
        self.b_s_init = torch.nn.Parameter(torch.zeros((num_heads, nc_middle)))
        # vector
        w_v = torch.empty((num_heads, nc_middle, nc_s_in_by_head))
        nn.init.xavier_uniform_(w_v, gain=1)
        self.w_v_init = torch.nn.Parameter(w_v)

        # ---- pre-attn dtp with sh
        nc_s_out_by_head = 3 * self.nc_by_head 
        nc_v_out_by_head = self.nc_by_head
        self.pre_attn_dtp_linear = DTPByHead(nc_middle,
                                             nc_middle, 
                                             nc_s_out_by_head,
                                             nc_v_out_by_head,
                                             num_heads)
        self.radialnn = RadialNN(self.pre_attn_dtp_linear.weight_numel,
                                 self.rc,
                                 self.radial_num_basis,
                                 self.radial_num_hidden,
                                 self.radial_num_layers,
                                 self.include_edge_features,
                                 self.num_edge_features)

        # ---- attn linear
        w_s = torch.empty((num_heads, self.nc_by_head, 2 * self.nc_by_head))
        nn.init.xavier_uniform_(w_s, gain=1)
        self.attn_msg_w_s = torch.nn.Parameter(w_s)
        self.attn_msg_b_s = torch.nn.Parameter(torch.zeros((num_heads, self.nc_by_head)))
        # vector
        w_v = torch.empty((num_heads, self.nc_by_head, 2 * self.nc_by_head))
        nn.init.xavier_uniform_(w_v, gain=1)
        self.attn_msg_w_v = torch.nn.Parameter(w_v)

        # ---- attn weight
        self.attend_to_self = attend_to_self
        self.attn_weight_relu = torch.nn.LeakyReLU(0.1)
        w = torch.empty((num_heads, self.nc_by_head))
        nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('leaky_relu', 0.1))
        self.attn_w = torch.nn.Parameter(w)

        # ---- attn final linear
        self.linear_attn_final = Linear(nc_s_in, nc_s_in, nc_v_in, nc_v_in, add_bias=True)

        # ---- feed-forward
        # ff1 -> gate -> ff2
        self.ff_mul = ff_mul
        self.nc_s_out = nc_s_out = self.irreps_out[0]
        self.nc_v_out = nc_v_out = self.irreps_out[1]
        assert nc_v_out > 0, "assume there will always be at least one vector output"
        # -- comput v norms: (nc_s, nc_v) -> (nc_s + nc_v, nc_v)
        # -- ff1: (nc_s + nc_v, nc_v) -> (m * nc_s out + m * nc_v out, m * nc_v out)
        self.ff1 = Linear(nc_s_in, #  + nc_v_in, 
                          ff_mul * nc_s_out + ff_mul * nc_v_out,
                          nc_v_in,
                          ff_mul * nc_v_out,
                          add_bias=True)
        # -- gate: (m * nc_s out + m * nc_v out, m * nc_v out) -> (m * nc_s out, m * nc_v out)
        # -- ff2: (m * nc_s out, m * nc_v out) -> (nc_s out, nc_v out)
        self.ff2 = Linear(ff_mul * nc_s_out, nc_s_out, ff_mul * nc_v_out, nc_v_out,
                          add_bias=True)

    def forward(self, s, v, edges_ij, r_ij, r_ij_vec, weight_cutoff=None):
        """
        args:
            edges [N, N]: precomputed residue num diff embedding
            r_ij [N, N]
            r_ij_vec [N, N, 3] 
        """
        # ---- attn module
        s0, v0 = s, v # for skip

        if self.apply_layer_norm:
            s, v = self.layer_norm_attn(s, v)

        # ---- initial mixing
        # i is the dst/query, which gets first dim
        s_i, v_i = self.linear_dst(s, v)
        s_j, v_j = self.linear_src(s, v)
        s_i = rearrange(s_i, "i (h m)   -> h  i ()  m", h=self.num_heads)
        v_i = rearrange(v_i, "i (h m) k -> h  i ()  m k", h=self.num_heads)
        s_j = rearrange(s_j, "j (h m)   -> h ()  j  m", h=self.num_heads)
        v_j = rearrange(v_j, "j (h m) k -> h ()  j  m k", h=self.num_heads)
        # channel wise tp
        ss = s_i * s_j
        sv = s_i.unsqueeze(-1) * v_j
        vs = v_i * s_j.unsqueeze(-1)
        vv = (v_i * v_j).sum(-1)
        # concat 
        s_ij = torch.cat([ss, vv], dim=-1)
        v_ij = torch.cat([sv, vs], dim=-2)        
        # post tp linear
        s_ij = torch.einsum("h m n, h i j n -> h i j m", self.w_s_init, s_ij) + self.b_s_init[:, None, None, :]
        v_ij = torch.einsum("h m n, h i j n k -> h i j m k", self.w_v_init, v_ij)

        # ---- pre attn dtp with sh
        weights = self.radialnn(r_ij, edges_ij, weight_cutoff)
        s_ij, v_ij = self.pre_attn_dtp_linear(s_ij, v_ij, r_ij_vec, weights)

        # split (grouped by head)
        s_ij0, gate_v, s_ij = rearrange(s_ij, 'h i j (c m) -> c h i j m', c=3)

        # -- compute messages
        # gate
        s_ij = F.silu(s_ij)
        v_ij = torch.sigmoid(gate_v).unsqueeze(-1) * v_ij
        # tp; r_ij_vec (i j k)
        ss = s_ij
        sv = s_ij.unsqueeze(-1) * r_ij_vec[None, :, :, None, :]
        vs = v_ij
        vv = torch.einsum('h i j m k, i j k -> h i j m', [v_ij, r_ij_vec])
        s = rearrange([ss, vv], 'c h i j m -> h i j (c m)')
        v = rearrange([sv, vs], 'c h i j m k -> h i j (c m) k')
        # apply linear
        s_ij = torch.einsum("h m n, h i j n -> h i j m", self.attn_msg_w_s, s) + self.attn_msg_b_s[:, None, None, :]
        v_ij = torch.einsum("h m n, h i j n k -> h i j m k", self.attn_msg_w_v, v)

        # -- compute attn score
        z_ij = torch.einsum("h n, h i j n -> h i j", self.attn_w, s_ij0)
        if not self.attend_to_self:
            nn = z_ij.size()[1]
            z_ij = z_ij - 1e9 * torch.eye(nn, device=z_ij.device).unsqueeze(0)
        a_ij = F.softmax(z_ij, dim=-1) # over dst

        # -- combine
        s = torch.einsum('h i j, h i j m -> h i m', [a_ij, s_ij])
        s = rearrange(s, 'h i m -> i (h m)')
        v = torch.einsum('h i j, h i j m k -> h i m k', [a_ij, v_ij])
        v = rearrange(v, 'h i m k -> i (h m) k')
        s, v = self.linear_attn_final(s, v)
        
        # skip
        s = s0 + s
        v = v0 + v

        # ---- ff module
        if self.apply_resnet:
            s0, v0 = s, v # for skip

        if self.apply_layer_norm:
            s, v = self.layer_norm_ff(s, v)

        # -- norm
        # todo: eliminate this?
        # v_norm = (nodes["v"].square().sum(-1) + 1e-6).sqrt() # [N, nc_v]
        
        # -- ff1
        # s = torch.cat([s, v_norm], dim=1)
        s, v = self.ff1(s, v)
        
        # -- gate
        if self.nc_s_out > 0:
            offset = self.ff_mul * self.nc_s_out # for scalar
            gate_v = s[:, offset:]
            s = F.silu(s[:, :offset])
        else:
            gate_v = s
            s = None
        v = torch.sigmoid(gate_v).unsqueeze(-1) * v            

        # -- ff2
        s, v = self.ff2(s, v)

        if self.apply_resnet:
            s = s0 + s
            v = v0 + v

        return s, v


@compile_mode('script')
class E3NN(torch.nn.Module):
    def __init__(
        self,
        nc=None,
        num_layers=None,
        rc=None,
        nonlinearity=None,
        include_edge_features=None,
        d_embed_edge=None,
        num_edge_types=None,
        radial_num_basis=None,
        radial_num_hidden=None,
        radial_num_layers=None,
        interaction_type=None,
        attn_num_heads=None,
        attend_to_self=False,
        apply_layer_norm=False,
        disable_cutoff=False
    ) -> None:
        super().__init__()
        self.nc = nc
        self.irreps = (nc, nc)
        self.disable_cutoff = disable_cutoff

        # ---- edge embedding
        self.d_embed_edge = d_embed_edge
        self.num_edge_types = num_edge_types
        self.include_edge_features = include_edge_features
        self.embed_edge = torch.nn.Embedding(num_edge_types, self.d_embed_edge, padding_idx=-1, max_norm=1, norm_type=2.0,
                                             scale_grad_by_freq=False, sparse=False)

        # ---- spherical
        self.num_layers = num_layers
        assert num_layers >= 1

        # ---- radial components
        self.radial_num_basis = radial_num_basis
        self.radial_num_hidden = radial_num_hidden
        self.radial_num_layers = radial_num_layers

        # ---- interaction block
        self.interaction_type=interaction_type
        self.apply_layer_norm=apply_layer_norm
        self.attend_to_self=attend_to_self
        self.attn_num_heads=attn_num_heads
        self.rc = rc
        assert interaction_type in ["attn-direct"]
        UpdateModule = partial(Equiformer,
                               radial_num_basis=self.radial_num_basis,
                               radial_num_hidden=self.radial_num_hidden,
                               radial_num_layers=self.radial_num_layers,
                               num_edge_features=self.d_embed_edge,
                               num_heads=self.attn_num_heads,
                               rc=self.rc,
                               include_edge_features=self.include_edge_features,
                               apply_layer_norm=self.apply_layer_norm,
                               attend_to_self=self.attend_to_self,
                               interaction_type=self.interaction_type
                               )

        # ---- define the update net
        self.nonlinearity = nonlinearity
        assert nonlinearity == "gated"
        self.layers = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(UpdateModule(self.irreps, # in
                                            self.irreps, # out
                                            apply_resnet=True) 
                               )
        # final layer predicts transformation
        self.layer_euclidean = UpdateModule(self.irreps,
                                            (0, 2), # out
                                            apply_resnet=False)

    def forward(self, s, v, R, T, edge_type_ij):
        # embed edges
        edges_ij = self.embed_edge(edge_type_ij)

        # compute dist and spherical harmonics
        # no grad flow here with the heuristic that the network should focus on 
        # the "next move" given specified geometry
        with torch.no_grad():
            r_ij_vec = T[None, :] - T[:, None]
            r_ij = (r_ij_vec.square().sum(-1) + 1e-6).sqrt()
            r_ij_vec = r_ij_vec / r_ij.unsqueeze(-1)
            weight_cutoff = compute_weight_cutoff(r_ij, self.rc) if not self.disable_cutoff else None

        # update node embeddings
        for layer in self.layers:
            s, v = layer(s, v, edges_ij, r_ij, r_ij_vec, weight_cutoff)

        # predict delta euclidean
        _, out = self.layer_euclidean(s, v, edges_ij, r_ij, r_ij_vec, weight_cutoff)
        dT = out[:, 0]
        u = out[:, 1]
        dR = R_from_quaternion_u(u)

        # -- update euclidean
        # cannot detach dR, T here to get the loss
        T = T + dT
        # steer embedding
        R = compose_rotations(dR, R)

        return R, T
