from collections import defaultdict
import os
import numpy as np
import torch
from openfold_light.residue_constants import (atom_types, residue_atoms, 
        restype_3to1, restype_1to3, resnames, load_stereo_chemical_props,
        van_der_waals_radius, ca_ca, between_res_bond_length_c_n, between_res_bond_length_stddev_c_n,
        between_res_cos_angles_c_n_ca, between_res_cos_angles_ca_c_n)

from cg import (cg_dict, resname_to_idx, idx_to_resname, cg_to_np, cg_to_idx,
                idx_to_cg, N_CG_MAX, cg_atom_rename_np, cg_atom_ambiguous_np,
                cgidx_to_atomidx)
from utils import get_euclidean, get_euclidean_kabsch
MAX_DIST = 32 # max residue distance
NUM_EDGE_TYPE = MAX_DIST * 2 + 2
# ---- template coords
template_coords = "cg_X0.npz"
if os.path.exists(template_coords):
    cg_X0 = torch.from_numpy(np.load(template_coords)["x"].astype(np.float32))
else:
    cg_X0 = None
# ---- precompute arrs to be used for struct violation loss 
residue_bonds, residue_virtual_bonds, residue_bond_angles = load_stereo_chemical_props()
tol_factor = 3
ambiguous_atoms = defaultdict(set)
for resname, cgs in cg_dict.items():
    for atom in residue_atoms[resname]:
        # atom appears in more than 1 cg
        if sum([atom in cg for cg in cgs]) > 1:
            ambiguous_atoms[resname].add(atom)

# -- bond length
# (idx1, idx2, length, tol_factor * stddev)
def cross_cg_bond(resname, bond):
    cgs = cg_dict[resname]
    for cg in cgs:
        if bond.atom1_name in cg and bond.atom2_name in cg:
            return False
    return True

bonds_np = dict()
for resname, bonds in residue_bonds.items():
    if resname == "UNK":
        continue 
    bonds_virtual = residue_virtual_bonds[resname]
    bonds = bonds + bonds_virtual
    # nprev = len(bonds)
    
    # need to keep all bonds to inform clash loss
    bonds_ = []
    for bond in bonds:
        length_mask = cross_cg_bond(resname, bond) or bond.atom1_name in ambiguous_atoms[resname] or bond.atom2_name in ambiguous_atoms[resname]
        bonds_.append((
             residue_atoms[resname].index(bond.atom1_name),
             residue_atoms[resname].index(bond.atom2_name),
             bond.length, tol_factor * bond.stddev,
             length_mask # whether to mask bond loss since unambiguous
             ))
    # print(resname, len(bonds_), nprev)
    bonds_np[resname] = tuple([np.asarray(x) for x in zip(*bonds_)])

# -- bond angle
# (idx1, idx2, idx3, mid, tol)
# idx2 is the middle atom
def cross_cg_angle(resname, bond):
    cgs = cg_dict[resname]
    for cg in cgs:
        if bond.atom1_name in cg and bond.atom2_name in cg and bond.atom3name in cg:
            return False
    return True

bond_angles_np = dict()
for resname, bond_angles in residue_bond_angles.items():
    if resname == "UNK":
        continue        
    bond_angles_ = []
    # nprev = len(bond_angles)
    for bond in bond_angles:
        # could skip angles
        if not (cross_cg_angle(resname, bond) or bond.atom1_name in ambiguous_atoms[resname] or bond.atom2_name in ambiguous_atoms[resname] or bond.atom3name in ambiguous_atoms[resname]):
            continue

        cosa = np.cos(bond.angle_rad)
        cosap = np.cos(bond.angle_rad + tol_factor * bond.stddev)
        cosan = np.cos(bond.angle_rad - tol_factor * bond.stddev)
        mid = (cosan + cosap) / 2.
        tol = np.abs(cosan - mid)
        bond_angles_.append(
            (
             residue_atoms[resname].index(bond.atom1_name),
             residue_atoms[resname].index(bond.atom2_name),
             residue_atoms[resname].index(bond.atom3name),
             mid,
             tol
             )
            )
    # print(resname, len(bond_angles_), nprev)
    bond_angles_np[resname] = tuple([np.asarray(x) for x in zip(*bond_angles_)])
# clash
atom_width_np = {resname: np.asarray([van_der_waals_radius[atom[0]] for atom in atoms]) for resname, atoms in residue_atoms.items()}


def get_peptide_bond_lengths(resname):
    c_n = between_res_bond_length_c_n[0] if resname != "PRO" else between_res_bond_length_c_n[1]
    c_n_stddev = between_res_bond_length_stddev_c_n[0] if resname != "PRO" else between_res_bond_length_stddev_c_n[1]

    return c_n, c_n_stddev


def subtract_centroid_and_mask(cg_X, cg_T, cg_R, cg_mask, cg_atom_mask):
    T_centroid = (cg_mask.reshape(-1, 1) * cg_T).sum(0) / cg_mask.sum()
    cg_T = cg_T - T_centroid
    cg_X = cg_X - T_centroid
    cg_X[cg_atom_mask == 0] = 0 # zero properly
    masked = cg_mask == 0.
    cg_T[masked] = 0. # so it's not flung off somewhere
    cg_R[masked] = torch.eye(3).repeat(masked.sum(), 1, 1) # to keep it uniform

    return cg_X, cg_T, cg_R


def get_cg_RT(cg_cgidx, cg_X, cg_mask, cg_atom_mask, use_kabsch):
    # get transformation
    if not use_kabsch:
        cg_T, cg_R = get_euclidean(torch.from_numpy(cg_X[:, :3]))
    else:
        cg_T, cg_R = get_euclidean_kabsch(torch.from_numpy(cg_X), cg_X0[cg_cgidx], torch.from_numpy(cg_atom_mask))
    cg_T, cg_R = cg_T.numpy(), cg_R.numpy()

    return cg_X, cg_T, cg_R


def collate_fn(x):
    """returns the input (list)"""
    return ListData(x)


class ListData:
    def __init__(self, data):
        super(ListData, self).__init__()
        self.data = data

    def to(self, device):
        self.data = [x.to(device) for x in self.data]
        return self

    def __getitem__(self, idx): 
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def x_to_pdb(x, resnum, resname, atoms, b_factors=None) -> str:
    """Converts a cg protein to a PDB string.

    Args:
      cg_prot: The cg protein to convert to PDB.

    Returns:
      PDB string.
    """
    pdb_lines = []
    pdb_lines.append("MODEL     1")
    atom_index = 1
    chain_id = "A"
    # Add all atom sites.
    if b_factors is None:
        b_factors = [0] * len(x)
    for pos, resnum_, resname_, atom_name, b_factor in zip(x, resnum, resname, atoms, b_factors):
        record_type = "ATOM"
        name = atom_name if len(atom_name) == 4 else f" {atom_name}"
        alt_loc = ""
        insertion_code = ""
        occupancy = 1.00
        element = atom_name[
            0
        ]  # Protein supports only C, N, O, S, this works.
        charge = ""
        # PDB is a columnar format, every space matters here!
        atom_line = (
            f"{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}"
            f"{resname_:>3} {chain_id:>1}"
            f"{resnum_:>4}{insertion_code:>1}   "
            f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
            f"{occupancy:>6.2f}{b_factor:>6.2f}          "
            f"{element:>2}{charge:>2}"
        )
        pdb_lines.append(atom_line)
        atom_index += 1

    # Close the chain.
    chain_end = "TER"
    # chain_termination_line = (
    #     f"{chain_end:<6}{atom_index:>5}      {resname_:>3} "
    #     f"{chain_id:>1}{resnum_:>4}"
    # )
    chain_termination_line = (
        f"{chain_end:<6}"
    )
    pdb_lines.append(chain_termination_line)
    pdb_lines.append("ENDMDL")

    pdb_lines.append("END")
    pdb_lines.append("")
    return "\n".join(pdb_lines)


def pdb_feats_to_data(pdb_feats, use_kabsch, real_pdb=False, dst_idx_offset=0):
    try:
        sequence = pdb_feats["sequence"][0].decode()
    except:
        sequence = pdb_feats["sequence"].decode()

    # residue type. should match restypes
    aatype = np.where(pdb_feats["aatype"] == 1)[1].astype(np.int64)

    # items to save
    pos = pdb_feats["all_atom_positions"] # [N, 37, 3]
    mask = pdb_feats["all_atom_mask"] # [N, 37]

    # -- all atoms to cg mapping
    # - ground truth
    # Ncg := total number of CG nodes
    resnum = np.arange(len(sequence), dtype=np.int64)
    cg_resnums = [] # [Ncg]; for edge attributes
    cg_cgidxs = [] # [Ncg]; for node attributes
    cg_Xs = [] # [Ncg, N_CG_MAX, 3]
    cg_atom_masks = [] # [Ncg, N_CG_MAX]; atom level mask; 1.0 if both atom experimentally present and belongs to the CG node else 0.0
    cg_masks = [] # [Ncg]; CG level mask; 1.0 if the first three atoms present else 0.0
    for res, cgs in cg_dict.items():
        res_idx = resname_to_idx[res]
        
        # get all residues of the type "res"
        ii = np.where(aatype == res_idx)[0]

        # do the mapping
        for j in range(len(cgs)):
            cg = (res_idx, j)

            # relevant atoms among 37 heavy atoms
            icg_atoms = cg_to_np[cg]

            # get CG mask
            atom_mask = np.zeros((len(ii), N_CG_MAX), dtype=mask.dtype)            
            if real_pdb: 
                mask_ = mask[ii][:, icg_atoms]
                cg_mask = ~(mask_[:, :3] == 0).any(axis=1) # true here means good
                atom_mask[cg_mask, :len(icg_atoms)] = mask_[cg_mask]
                atom_mask[~cg_mask, :len(icg_atoms)] = 1. # fill with dummy values
            else:
                # since these are de novo all atoms are assumed to be present (except for H)
                cg_mask = np.ones(len(ii), dtype=bool) # true here means good
                atom_mask[:, :len(icg_atoms)] = 1.

            cg_idxs = np.full(len(ii), cg_to_idx[cg], dtype=int)

            # get pos
            pos_ = pos[ii][:, icg_atoms]

            if use_kabsch:
                pos_[~cg_mask] = cg_X0[cg_idxs[~cg_mask],:len(icg_atoms)]
            else:
                # filling in dummy values for first three if not present
                pos_[~cg_mask, :3] = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

            # CG frame
            cg_X = np.zeros((len(pos_), N_CG_MAX, 3), dtype=mask.dtype)
            cg_X[:, :len(icg_atoms)] = pos_

            # collect info
            cg_Xs.append(cg_X)
            cg_resnums.append(ii)
            cg_cgidxs.append(cg_idxs)
            cg_masks.append(cg_mask)
            cg_atom_masks.append(atom_mask)

    cg_resnum = np.concatenate(cg_resnums, axis=0)
    cg_cgidx = np.concatenate(cg_cgidxs, axis=0)
    cg_mask = np.concatenate(cg_masks, axis=0).astype(np.float32)
    cg_atom_mask = np.concatenate(cg_atom_masks, axis=0).astype(np.float32)
    cg_X = np.concatenate(cg_Xs, axis=0)

    if cg_mask.sum() < 30:
        assert False

    cg_X, cg_T, cg_R = get_cg_RT(cg_cgidx, cg_X, cg_mask, cg_atom_mask, use_kabsch)
    cg_amb = cg_atom_ambiguous_np[cg_cgidx]
    if real_pdb:
        # reject CG nodes that have too big of rmsd from template coords
        cg_X_fit = torch.einsum("rij,rkj->rki", torch.from_numpy(cg_R), cg_X0[cg_cgidx]) + torch.from_numpy(cg_T).unsqueeze(1)
        d = ((cg_X_fit - cg_X).square().sum(-1) + 1e-6).sqrt()
        ireject = ((d * cg_atom_mask).sum(-1) / cg_atom_mask.sum(-1)) > 1**2
        cg_mask[ireject] = 0.

    # -- alternative truth to account for possible 180 deg symmetry
    # permute atoms
    permut = cg_atom_rename_np[cg_cgidx]
    cg_X_alt = np.transpose(cg_X[np.arange(len(permut)), permut.T], (1, 0, 2))
    cg_amb_alt = cg_amb[np.arange(len(permut)), permut.T].T
    cg_atom_mask_alt = cg_atom_mask[np.arange(len(permut)), permut.T].T
    cg_X_alt, cg_T_alt, cg_R_alt = get_cg_RT(cg_cgidx, cg_X_alt, cg_mask, cg_atom_mask_alt, use_kabsch)

    # -- indices for scatter reduction and structure violation calculation
    # compute residue based offsets
    dst_bonds = []
    dst_angles = []
    dst_atom_widths = []
    resnum_to_offset = {}
    offset = 0
    for i, aa in enumerate(sequence):
        resnum_to_offset[i] = offset
        resname = restype_1to3[aa]
        
        # precompute arrs for struct violation loss
        i1, i2, l, tol, mask = bonds_np[resname]
        dst_bonds.append((i1 + offset, i2 + offset, l, tol, mask))
        i1, i2, i3, mid, tol = bond_angles_np[resname]
        dst_angles.append((i1 + offset, i2 + offset, i3 + offset, mid, tol))
        dst_atom_widths.append(atom_width_np[resname])

        offset_increment = len(residue_atoms[resname])

        # add peptide bond constraints
        if i < len(sequence) - 1:
            resname_next = restype_1to3[sequence[i+1]]
            ca_i = residue_atoms[resname].index("CA")
            c_i = residue_atoms[resname].index("C")
            n_ip1 = residue_atoms[resname_next].index("N")
            ca_ip1 = residue_atoms[resname_next].index("CA")
            i1 = [ca_i, c_i]
            i2 = [ca_ip1, n_ip1]
            c_n, c_n_stddev = get_peptide_bond_lengths(resname)
            # ca-ca / C[i] - N[i+1] bond
            dst_bonds.append((np.asarray(i1) + offset, 
                              np.asarray(i2) + offset + offset_increment, 
                              np.asarray([ca_ca, c_n]),
                              np.asarray([0.05, c_n_stddev * tol_factor]), # first element is handpicked
                              np.asarray([1.0, 1.0])
                              ))
            # inter-residue angles
            i1 = [c_i, ca_i]
            i2 = [n_ip1 + offset_increment, c_i]
            i3 = [ca_ip1 + offset_increment, n_ip1 + offset_increment]
            mid = [between_res_cos_angles_c_n_ca[0], between_res_cos_angles_ca_c_n[0]]
            tol = [between_res_cos_angles_c_n_ca[1], between_res_cos_angles_ca_c_n[1]]
            dst_angles.append((np.asarray(i1) + offset, 
                              np.asarray(i2) + offset,
                              np.asarray(i3) + offset, 
                              np.asarray(mid),
                              np.asarray(tol),
                              ))
        offset += offset_increment

    dst_bonds = [np.concatenate(x) for x in zip(*dst_bonds)]
    dst_angles = [np.concatenate(x) for x in zip(*dst_angles)]
    dst_atom_widths = np.concatenate(dst_atom_widths)

    # reduction index
    N_CG = len(cg_cgidx)
    scatter_index = np.zeros(N_CG * N_CG_MAX, dtype=int)
    scatter_w = np.zeros(N_CG * N_CG_MAX, dtype=float)
    dst_resnum = np.zeros(offset, dtype=int)
    dst_atom = np.zeros(offset, dtype='>U3')
    dst_resname = np.zeros(offset, dtype='>U3')
    dst_atom_mask = np.ones(offset, dtype=float) # for rmsd calc against gt
    for i, (cgidx, resnum_) in enumerate(zip(cg_cgidx, cg_resnum)):
        atomidxs = cgidx_to_atomidx[cgidx]
        for k, (resname_, atom, atomidx, w) in enumerate(atomidxs):
            src_idx = i * N_CG_MAX + k
            dst_idx = resnum_to_offset[resnum_] + atomidx
            scatter_index[src_idx] = dst_idx + dst_idx_offset
            scatter_w[src_idx] = 1 / w
            dst_resnum[dst_idx] = resnum_
            dst_atom[dst_idx] = atom
            dst_resname[dst_idx] = resname_
            dst_atom_mask[dst_idx] = cg_atom_mask[i][k] * cg_mask[i] # necessary due to dummy value filling

    # save
    feats = {
            # res/full atoms level
            'sequence': sequence,
            'resnum': resnum,
            # cg
            'cg_resnum': cg_resnum,
            'cg_cgidx': cg_cgidx,
            'cg_mask': cg_mask,
            # 0
            'cg_T': cg_T,
            'cg_R': cg_R,
            'cg_atom_mask': cg_atom_mask,
            'cg_X': cg_X,
            'cg_amb': cg_amb,
            # alt
            'cg_T_alt': cg_T_alt, # should always be identical to cg_T
            'cg_R_alt': cg_R_alt,
            'cg_atom_mask_alt': cg_atom_mask_alt,
            'cg_X_alt': cg_X_alt,
            'cg_amb_alt': cg_amb_alt,
            # CG to "PDB" mapping
            "scatter_index": scatter_index,
            "scatter_w": scatter_w,
            "dst_resnum": dst_resnum,
            "dst_atom": dst_atom,
            "dst_resname": dst_resname,
            "final_offset": offset,
            # struct
            "dst_bonds_i1": dst_bonds[0] + dst_idx_offset, # w/o the offset, incorrect violation loss
            "dst_bonds_i2": dst_bonds[1] + dst_idx_offset, # w/o the offset, incorrect violation loss
            "dst_bonds_l": dst_bonds[2],
            "dst_bonds_tol": dst_bonds[3],
            "dst_bonds_mask": dst_bonds[4],
            "dst_angles_i1": dst_angles[0] + dst_idx_offset, # w/o the offset, incorrect violation loss
            "dst_angles_i2": dst_angles[1] + dst_idx_offset, # w/o the offset, incorrect violation loss
            "dst_angles_i3": dst_angles[2] + dst_idx_offset, # w/o the offset, incorrect violation loss
            "dst_angles_mid": dst_angles[3],
            "dst_angles_tol": dst_angles[4],
            "dst_atom_widths": dst_atom_widths,
            "dst_atom_mask": dst_atom_mask
            }

    return feats


def sequence_to_feats(sequence, dst_idx_offset=0):
    seq = np.asarray(list(sequence))
    cg_resnums = [] # [Ncg]; for edge attributes
    cg_cgidxs = [] # [Ncg]; for node attributes
    for res, cgs in cg_dict.items():
        # get all residues of the type "res"
        res_idx = resname_to_idx[res]
        ii = np.where(seq == restype_3to1[res])[0]
        # do the mapping
        for j in range(len(cgs)):
            cg = (res_idx, j)
            cg_idxs = np.full(len(ii), cg_to_idx[cg], dtype=int)
            cg_resnums.append(ii)
            cg_cgidxs.append(cg_idxs)
    cg_resnum = np.concatenate(cg_resnums)
    cg_cgidx = np.concatenate(cg_cgidxs)

    # compute residue based offsets
    resnum_to_offset = {}
    offset = 0
    for i, aa in enumerate(sequence):
        resnum_to_offset[i] = offset
        resname = restype_1to3[aa]
        offset += len(residue_atoms[resname])

    # reduction index
    N_CG = len(cg_cgidx)
    scatter_index = np.zeros(N_CG * N_CG_MAX, dtype=int)
    scatter_w = np.zeros(N_CG * N_CG_MAX, dtype=float)
    dst_resnum = np.zeros(offset, dtype=int)
    dst_atom = np.zeros(offset, dtype='>U3')
    dst_resname = np.zeros(offset, dtype='>U3')
    dst_atom_mask = np.ones(offset, dtype=float) # for rmsd calc against gt
    for i, (cgidx, resnum_) in enumerate(zip(cg_cgidx, cg_resnum)):
        atomidxs = cgidx_to_atomidx[cgidx]
        for k, (resname_, atom, atomidx, w) in enumerate(atomidxs):
            src_idx = i * N_CG_MAX + k
            dst_idx = resnum_to_offset[resnum_] + atomidx
            scatter_index[src_idx] = dst_idx + dst_idx_offset
            scatter_w[src_idx] = 1 / w
            dst_resnum[dst_idx] = resnum_
            dst_atom[dst_idx] = atom
            dst_resname[dst_idx] = resname_
            # dst_atom_mask[dst_idx] = cg_atom_mask[i][k] * cg_mask[i] # not needed for inference only

    return cg_cgidx, cg_resnum, scatter_index, scatter_w, dst_resnum, dst_atom, dst_resname, offset
