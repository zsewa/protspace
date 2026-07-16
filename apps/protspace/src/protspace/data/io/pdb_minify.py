"""Minify PDB structure text for smaller bundle sizes.

Cartoon rendering (the default view in the web structure viewer) only needs
the protein backbone trace -- N, CA, C, O per residue -- so stripping side
chains, waters, and other heteroatoms shrinks bundled structures by roughly
70% with no visible difference in cartoon mode.
"""

_BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O"})

# Structural bookkeeping and lightweight header metadata kept verbatim;
# everything else (HETATM, ANISOU, REMARK, SEQRES, connectivity records, ...)
# is dropped.
_KEEP_RECORD_TYPES = frozenset(
    {
        "HEADER",
        "TITLE",
        "COMPND",
        "SOURCE",
        "CRYST1",
        "MODEL",
        "ENDMDL",
        "TER",
        "END",
    }
)


def minify_pdb_backbone(pdb_text: str) -> str:
    """Strip a PDB file's text down to backbone atoms (N, CA, C, O).

    Drops HETATM records (waters, ligands, ions), ANISOU records, and all
    non-backbone side-chain ATOM records. Structural bookkeeping records
    (MODEL/ENDMDL/TER/END) and header metadata are preserved.
    """
    kept_lines = []
    for line in pdb_text.splitlines(keepends=True):
        record = line[:6]
        if record.startswith("ATOM"):
            atom_name = line[12:16].strip()
            if atom_name in _BACKBONE_ATOMS:
                kept_lines.append(line)
        elif record.rstrip() in _KEEP_RECORD_TYPES:
            kept_lines.append(line)
    return "".join(kept_lines)
