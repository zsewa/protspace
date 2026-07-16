"""Tests for backbone-only PDB minification."""

from protspace.data.io.pdb_minify import minify_pdb_backbone

_SAMPLE_PDB = """\
HEADER    TEST STRUCTURE
TITLE     A FAKE RESIDUE
COMPND    MOL_ID: 1;
MODEL        1
ATOM      1  N   MET A   1      11.104  13.207   2.100  1.00 20.00           N
ATOM      2  CA  MET A   1      12.560  13.207   2.100  1.00 20.00           C
ATOM      3  CB  MET A   1      13.100  11.900   2.100  1.00 20.00           C
ATOM      4  C   MET A   1      13.100  14.620   2.100  1.00 20.00           C
ATOM      5  O   MET A   1      12.400  15.630   2.100  1.00 20.00           O
ANISOU    1  N   MET A   1     2406   1892   1614    198    159   -297       N
HETATM    6  O   HOH A 101      20.000  20.000  20.000  1.00 30.00           O
TER       7      MET A   1
ENDMDL
END
"""


class TestMinifyPdbBackbone:
    def test_keeps_only_backbone_atoms(self):
        result = minify_pdb_backbone(_SAMPLE_PDB)
        atom_names = [
            line[12:16].strip() for line in result.splitlines() if line.startswith("ATOM")
        ]
        assert atom_names == ["N", "CA", "C", "O"]

    def test_drops_hetatm_and_anisou(self):
        result = minify_pdb_backbone(_SAMPLE_PDB)
        assert "HETATM" not in result
        assert "ANISOU" not in result

    def test_keeps_structural_and_header_records(self):
        result = minify_pdb_backbone(_SAMPLE_PDB)
        for record in ("HEADER", "TITLE", "COMPND", "MODEL", "TER", "ENDMDL", "END"):
            assert record in result

    def test_significantly_smaller(self):
        result = minify_pdb_backbone(_SAMPLE_PDB)
        assert len(result) < len(_SAMPLE_PDB)

    def test_empty_input(self):
        assert minify_pdb_backbone("") == ""

    def test_idempotent(self):
        once = minify_pdb_backbone(_SAMPLE_PDB)
        twice = minify_pdb_backbone(once)
        assert once == twice
