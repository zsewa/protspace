from protspace.data.io.pdb_plddt import mean_plddt_from_pdb

_TWO_RESIDUE_PDB = """\
ATOM      1  N   MET A   1      11.104  13.207   2.100  1.00 40.00           N
ATOM      2  CA  MET A   1      12.560  13.207   2.100  1.00 40.00           C
ATOM      3  C   MET A   1      13.100  14.620   2.100  1.00 40.00           C
ATOM      4  O   MET A   1      12.400  15.630   2.100  1.00 40.00           O
ATOM      5  N   ALA A   2      14.104  13.207   2.100  1.00 90.00           N
ATOM      6  CA  ALA A   2      15.560  13.207   2.100  1.00 90.00           C
ATOM      7  C   ALA A   2      16.100  14.620   2.100  1.00 90.00           C
ATOM      8  O   ALA A   2      15.400  15.630   2.100  1.00 90.00           O
HETATM    9  O   HOH A 101      20.000  20.000  20.000  1.00 30.00           O
TER      10      ALA A   2
END
"""


def test_mean_plddt_averages_ca_b_factors():
    assert mean_plddt_from_pdb(_TWO_RESIDUE_PDB) == 65.0


def test_mean_plddt_ignores_hetatm():
    # HOH's B-factor (30.00) must not pull the mean toward 30.
    assert mean_plddt_from_pdb(_TWO_RESIDUE_PDB) != 30.0


def test_mean_plddt_returns_none_for_no_ca_atoms():
    no_ca = "HETATM    1  O   HOH A 101      20.000  20.000  20.000  1.00 30.00           O\n"
    assert mean_plddt_from_pdb(no_ca) is None


def test_mean_plddt_returns_none_for_empty_text():
    assert mean_plddt_from_pdb("") is None
