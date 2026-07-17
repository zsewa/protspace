"""Extract mean pLDDT confidence from AF2-predicted PDB structures.

AF2 tooling (ColabFold, AlphaFold2) writes per-residue pLDDT (0-100) into
the PDB B-factor field, uniformly across every atom of a residue.
"""

_ATOM_NAME_START, _ATOM_NAME_END = 12, 16
_B_FACTOR_START, _B_FACTOR_END = 60, 66


def mean_plddt_from_pdb(pdb_text: str) -> float | None:
    """Computes the mean per-residue pLDDT from a PDB file's B-factor column.

    Reads the CA atom's B-factor per residue (one value per residue, since
    AF2 duplicates the same score across every atom in a residue) and
    averages them. Returns None when no CA atoms are found (e.g. the
    structure isn't an AF2 prediction, or is empty).
    """
    values = []
    for line in pdb_text.splitlines():
        if line[:6].rstrip() != "ATOM":
            continue
        if line[_ATOM_NAME_START:_ATOM_NAME_END].strip() != "CA":
            continue
        try:
            values.append(float(line[_B_FACTOR_START:_B_FACTOR_END]))
        except ValueError:
            continue
    if not values:
        return None
    return sum(values) / len(values)
