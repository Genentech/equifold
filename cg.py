"""
defines a coarse graining scheme
"""
from collections import Counter
from copy import deepcopy
import numpy as np
from openfold_light.residue_constants import (
    atom_order, residue_atoms, resname_to_idx, resnames)

cg_dict = {
    "ALA": [("C", "CA", "CB", "N"), ("C", "CA", "O")],
    "ARG": [("C", "CA", "CB", "N"), ("C", "CA", "O"), ("CB", "CG", "CD"), ("NE", "NH1", "NH2", "CZ")],
    "ASP": [("C", "CA", "CB", "N"), ("C", "CA", "O"), ("CG", "OD1", "OD2")],
    "ASN": [("C", "CA", "CB", "N"), ("C", "CA", "O"), ("CG", "ND2", "OD1")],
    "CYS": [("C", "CA", "CB", "N"), ("C", "CA", "O"), ("CA", "CB", "SG")],
    "GLU": [("C", "CA", "CB", "N"), ("C", "CA", "O"), ("CG", "CD", "OE1", "OE2")],
    "GLN": [("C", "CA", "CB", "N"), ("C", "CA", "O"), ("CG", "CD", "OE1", "NE2")],
    "GLY": [("C", "CA", "N"), ("C", "CA", "O")],
    "HIS": [("C", "CA", "CB", "N"), ("C", "CA", "O"), ("CG", "CD2", "CE1", "ND1", "NE2")],
    "ILE": [("C", "CA", "CB", "N"), ("C", "CA", "O"), ("CB", "CG1", "CG2"), ("CB", "CG1", "CD1")],
    "LEU": [("C", "CA", "CB", "N"), ("C", "CA", "O"), ("CG", "CD1", "CD2")],
    "LYS": [("C", "CA", "CB", "N"), ("C", "CA", "O"), ("CB", "CG", "CD"), ("CD", "CE", "NZ")],
    "MET": [("C", "CA", "CB", "N"), ("C", "CA", "O"), ("CG", "CE", "SD")],
    "PHE": [("C", "CA", "CB", "N"), ("C", "CA", "O"), ("CG", "CD1", "CD2", "CE1", "CE2", "CZ")],
    "PRO": [("C", "CA", "CB", "N"), ("C", "CA", "O"), ("CB", "CG", "CD")],
    "SER": [("C", "CA", "CB", "N"), ("C", "CA", "O"), ("CA", "CB", "OG")],
    "THR": [("C", "CA", "CB", "N"), ("C", "CA", "O"), ("CB", "CG2", "OG1")],
    "TRP": [("C", "CA", "CB", "N"), ("C", "CA", "O"), ("CG", "CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2", "NE1")],
    "TYR": [("C", "CA", "CB", "N"), ("C", "CA", "O"), ("CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH")],
    "VAL": [("C", "CA", "CB", "N"), ("C", "CA", "O"), ("CB", "CG1", "CG2")]
}

# max num atoms in CG
N_CG_MAX = max([max(len(y) for y in x) for x in cg_dict.values()])

# check for completeness
for res, atoms in residue_atoms.items():
    assert set([atom for cg in cg_dict[res] for atom in cg]) == set(atoms)

# assign a unique id to each CG, identified by (residue index, CG number)
cg_to_idx = {}
idx = 0
for res in resnames[:-1]: # get rid of UNK
    for j in range(len(cg_dict[res])):
        cg_to_idx[(resname_to_idx[res], j)] = idx
        idx += 1
idx_to_cg = {i: cg for cg, i in cg_to_idx.items()}
NUM_CG_TYPES = len(cg_to_idx)

# for each CG node, get corresponding atom indices in atom_order
idx_to_resname = {i: resname for resname, i in resname_to_idx.items()}
cg_to_np = {}
for res_idx, j in cg_to_idx.keys():
    atoms = cg_dict[idx_to_resname[res_idx]][j]
    cg_to_np[(res_idx, j)] = np.asarray([atom_order[x] for x in atoms])

# lists permutations due to 180 deg symmetry
cg_dict_rename = {
    "ALA": [None, None],
    "ARG": [None, None, None, None],
    "ASP": [None, None, (0, 2, 1)], # in af2
    "ASN": [None, None, None],
    "CYS": [None, None, None],
    "GLU": [None, None, (0, 1, 3, 2)], # in af2
    "GLN": [None, None, None],
    "GLY": [None, None],
    "HIS": [None, None, None],
    "ILE": [None, None, None, None],
    "LEU": [None, None, None],
    "LYS": [None, None, None, None],
    "MET": [None, None, None],
    "PHE": [None, None, (0, 2, 1, 4, 3, 5)], # in af2
    "PRO": [None, None, None],
    "SER": [None, None, None],
    "THR": [None, None, None],
    "TRP": [None, None, None],
    "TYR": [None, None, (0, 2, 1, 4, 3, 5, 6)], # in af2
    "VAL": [None, None, None]
}

# used for resolving naming ambiguity
iden = np.arange(N_CG_MAX, dtype=int).reshape(1, -1).repeat(len(idx_to_cg), axis=0)
cg_atom_rename_np = deepcopy(iden)
for cgidx, (res_idx, j) in idx_to_cg.items():
    permut = cg_dict_rename[idx_to_resname[res_idx]][j]
    if permut is not None:
        permut = np.asarray(permut)
        cg_atom_rename_np[cgidx, :len(permut)] = permut
cg_atom_ambiguous_np = (cg_atom_rename_np != iden).astype(np.float32)

# -- for reverse mapping
# weight factors
cg_atom_weight = {}
for resn, atomss in cg_dict.items():
    cg_atom_weight[resn] = Counter([y for x in atomss for y in x])

cgidx_to_atomidx = {}
for idx, cg in idx_to_cg.items():
    residx, j = cg
    resn = idx_to_resname[residx]
    atoms_in_cg = cg_dict[resn][j]
    cgidx_to_atomidx[idx] = [(resn, atom, residue_atoms[resn].index(atom), cg_atom_weight[resn][atom]) for atom in atoms_in_cg]
