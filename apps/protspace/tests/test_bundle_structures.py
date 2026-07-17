"""Round-trip tests for the optional sixth (structures) bundle part."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

from protspace.data.io.bundle import (
    PARQUET_BUNDLE_DELIMITER,
    extract_bundle_to_dir,
    read_bundle,
    read_statistics_from_bundle,
    read_structures_from_bundle,
    replace_settings_in_bundle,
    write_bundle,
)


def _core() -> list[pa.Table]:
    return [
        pa.table({"protein_id": ["a", "b"]}),
        pa.table({"projection_name": ["PCA_2"]}),
        pa.table({"projection_name": ["PCA_2", "PCA_2"], "identifier": ["a", "b"]}),
    ]


def _stats() -> pa.Table:
    return pa.table({"space_name": ["PCA_2"], "metric": ["silhouette"], "value": [0.5]})


def _structures() -> pa.Table:
    return pa.table(
        {
            "protein_id": ["a", "b"],
            "pdb_data": ["ATOM      1  N   MET A   1\n", "ATOM      1  N   GLY B   1\n"],
        }
    )


def _ndelims(path) -> int:
    return path.read_bytes().count(PARQUET_BUNDLE_DELIMITER)


def test_six_part_full_bundle(tmp_path):
    p = tmp_path / "b.parquetbundle"
    write_bundle(_core(), p, settings={"k": 1}, statistics=_stats(), structures=_structures())
    assert _ndelims(p) == 5
    _, settings = read_bundle(p)
    assert settings == {"k": 1}
    assert read_statistics_from_bundle(p) is not None
    structures_bytes = read_structures_from_bundle(p)
    assert structures_bytes is not None
    table = pq.read_table(pa.BufferReader(structures_bytes))
    assert table.column("protein_id").to_pylist() == ["a", "b"]
    assert "ATOM" in table.column("pdb_data")[0].as_py()


def test_six_part_structures_only_empty_settings_and_stats(tmp_path):
    """structures given alone -> zero-byte settings AND statistics sentinels."""
    p = tmp_path / "b.parquetbundle"
    write_bundle(_core(), p, structures=_structures())
    assert _ndelims(p) == 5  # zero-byte settings + zero-byte statistics slots
    core, settings = read_bundle(p)
    assert len(core) == 3 and settings is None
    assert read_statistics_from_bundle(p) is None
    structures_bytes = read_structures_from_bundle(p)
    assert structures_bytes is not None
    table = pq.read_table(pa.BufferReader(structures_bytes))
    assert table.column("protein_id").to_pylist() == ["a", "b"]


def test_six_part_settings_and_structures_no_stats(tmp_path):
    """settings + structures given, statistics absent -> zero-byte stats sentinel."""
    p = tmp_path / "b.parquetbundle"
    write_bundle(_core(), p, settings={"k": 1}, structures=_structures())
    assert _ndelims(p) == 5
    _, settings = read_bundle(p)
    assert settings == {"k": 1}
    assert read_statistics_from_bundle(p) is None
    assert read_structures_from_bundle(p) is not None


def test_five_part_bundle_has_no_structures(tmp_path):
    p = tmp_path / "b.parquetbundle"
    write_bundle(_core(), p, settings={"k": 1}, statistics=_stats())
    assert _ndelims(p) == 4
    assert read_structures_from_bundle(p) is None


def test_extract_to_dir_writes_structures(tmp_path):
    p = tmp_path / "b.parquetbundle"
    write_bundle(_core(), p, statistics=_stats(), structures=_structures())
    out = extract_bundle_to_dir(p, tmp_path / "out")
    assert (tmp_path / "out" / "structures.parquet").exists()
    assert not (tmp_path / "out" / "settings.parquet").exists()
    table = pq.read_table(str(tmp_path / "out" / "structures.parquet"))
    assert table.column("protein_id").to_pylist() == ["a", "b"]
    assert out


def test_style_preserves_structures_and_statistics(tmp_path):
    src = tmp_path / "b.parquetbundle"
    write_bundle(_core(), src, settings={"old": 1}, statistics=_stats(), structures=_structures())
    out = tmp_path / "styled.parquetbundle"
    replace_settings_in_bundle(src, out, {"new": 2})
    _, settings = read_bundle(out)
    assert settings == {"new": 2}
    assert read_statistics_from_bundle(out) is not None
    assert read_structures_from_bundle(out) is not None


def test_style_preserves_structures_on_structures_only_input(tmp_path):
    src = tmp_path / "b.parquetbundle"
    write_bundle(_core(), src, structures=_structures())  # empty settings+stats slots
    out = tmp_path / "styled.parquetbundle"
    replace_settings_in_bundle(src, out, {"new": 2})
    _, settings = read_bundle(out)
    assert settings == {"new": 2}
    assert read_structures_from_bundle(out) is not None


_PDB_WITH_SIDECHAIN = """\
ATOM      1  N   MET A   1      11.104  13.207   2.100  1.00 20.00           N
ATOM      2  CA  MET A   1      12.560  13.207   2.100  1.00 20.00           C
ATOM      3  CB  MET A   1      13.100  11.900   2.100  1.00 20.00           C
ATOM      4  C   MET A   1      13.100  14.620   2.100  1.00 20.00           C
ATOM      5  O   MET A   1      12.400  15.630   2.100  1.00 20.00           O
HETATM    6  O   HOH A 101      20.000  20.000  20.000  1.00 30.00           O
TER       7      MET A   1
END
"""


def _make_bundle_inputs(tmp_path):
    proj = tmp_path / "project"
    proj.mkdir()
    pq.write_table(
        pa.table({"projection_name": ["PCA_2"], "dimensions": [2], "info_json": ["{}"]}),
        str(proj / "projections_metadata.parquet"),
    )
    pq.write_table(
        pa.table(
            {
                "projection_name": ["PCA_2", "PCA_2"],
                "identifier": ["P1", "P2"],
                "x": [0.1, 0.2],
                "y": [0.3, 0.4],
                "z": [None, None],
            }
        ),
        str(proj / "projections_data.parquet"),
    )
    ann_path = tmp_path / "annotations.parquet"
    pq.write_table(pa.table({"identifier": ["P1", "P2"]}), str(ann_path))

    structures_dir = tmp_path / "structures"
    structures_dir.mkdir()
    (structures_dir / "P1.pdb").write_text(_PDB_WITH_SIDECHAIN)
    (structures_dir / "P2.pdb").write_text(_PDB_WITH_SIDECHAIN.replace("MET", "GLY"))

    return proj, ann_path, structures_dir


def test_bundle_cli_structures_flag(tmp_path):
    from typer.testing import CliRunner

    from protspace.cli.app import app

    proj, ann_path, structures_dir = _make_bundle_inputs(tmp_path)

    out = tmp_path / "data.parquetbundle"
    result = CliRunner().invoke(
        app,
        [
            "bundle",
            "-p",
            str(proj),
            "-a",
            str(ann_path),
            "-o",
            str(out),
            "-t",
            str(structures_dir),
        ],
    )
    assert result.exit_code == 0, result.output

    structures_bytes = read_structures_from_bundle(out)
    assert structures_bytes is not None
    table = pq.read_table(pa.BufferReader(structures_bytes))
    assert set(table.column("protein_id").to_pylist()) == {"P1", "P2"}


def test_bundle_cli_minifies_structures_by_default(tmp_path):
    """Default behavior: bundled structures are stripped to backbone atoms."""
    from typer.testing import CliRunner

    from protspace.cli.app import app

    proj, ann_path, structures_dir = _make_bundle_inputs(tmp_path)

    out = tmp_path / "data.parquetbundle"
    result = CliRunner().invoke(
        app,
        ["bundle", "-p", str(proj), "-a", str(ann_path), "-o", str(out), "-t", str(structures_dir)],
    )
    assert result.exit_code == 0, result.output

    table = pq.read_table(pa.BufferReader(read_structures_from_bundle(out)))
    for pdb_data in table.column("pdb_data").to_pylist():
        assert "HETATM" not in pdb_data
        assert " CB " not in pdb_data
        assert " CA " in pdb_data  # backbone kept
        assert len(pdb_data) < len(_PDB_WITH_SIDECHAIN)


def test_bundle_cli_no_minify_structures_keeps_full_atoms(tmp_path):
    """--no-minify-structures preserves side chains and heteroatoms verbatim."""
    from typer.testing import CliRunner

    from protspace.cli.app import app

    proj, ann_path, structures_dir = _make_bundle_inputs(tmp_path)

    out = tmp_path / "data.parquetbundle"
    result = CliRunner().invoke(
        app,
        [
            "bundle",
            "-p",
            str(proj),
            "-a",
            str(ann_path),
            "-o",
            str(out),
            "-t",
            str(structures_dir),
            "--no-minify-structures",
        ],
    )
    assert result.exit_code == 0, result.output

    table = pq.read_table(pa.BufferReader(read_structures_from_bundle(out)))
    pdb_by_id = dict(
        zip(
            table.column("protein_id").to_pylist(),
            table.column("pdb_data").to_pylist(),
            strict=True,
        )
    )
    assert "HETATM" in pdb_by_id["P1"]
    assert " CB " in pdb_by_id["P1"]
    assert pdb_by_id["P1"] == _PDB_WITH_SIDECHAIN


def test_bundle_cli_structures_flag_rejects_empty_dir(tmp_path):
    from typer.testing import CliRunner

    from protspace.cli.app import app

    proj = tmp_path / "project"
    proj.mkdir()
    pq.write_table(
        pa.table({"projection_name": ["PCA_2"], "dimensions": [2], "info_json": ["{}"]}),
        str(proj / "projections_metadata.parquet"),
    )
    pq.write_table(
        pa.table(
            {
                "projection_name": ["PCA_2"],
                "identifier": ["P1"],
                "x": [0.1],
                "y": [0.3],
                "z": [None],
            }
        ),
        str(proj / "projections_data.parquet"),
    )
    ann_path = tmp_path / "annotations.parquet"
    pq.write_table(pa.table({"identifier": ["P1"]}), str(ann_path))

    empty_dir = tmp_path / "empty_structures"
    empty_dir.mkdir()

    out = tmp_path / "data.parquetbundle"
    result = CliRunner().invoke(
        app,
        [
            "bundle",
            "-p",
            str(proj),
            "-a",
            str(ann_path),
            "-o",
            str(out),
            "-t",
            str(empty_dir),
        ],
    )
    assert result.exit_code != 0


def test_bundle_cli_accepts_raw_colabfold_output(tmp_path):
    """--structures accepts unrenamed ColabFold output, picking the best-ranked model."""
    from typer.testing import CliRunner

    from protspace.cli.app import app

    proj = tmp_path / "project"
    proj.mkdir()
    pq.write_table(
        pa.table({"projection_name": ["PCA_2"], "dimensions": [2], "info_json": ["{}"]}),
        str(proj / "projections_metadata.parquet"),
    )
    pq.write_table(
        pa.table(
            {
                "projection_name": ["PCA_2", "PCA_2"],
                "identifier": ["P1", "P2"],
                "x": [0.1, 0.2],
                "y": [0.3, 0.4],
                "z": [None, None],
            }
        ),
        str(proj / "projections_data.parquet"),
    )
    ann_path = tmp_path / "annotations.parquet"
    pq.write_table(pa.table({"identifier": ["P1", "P2"]}), str(ann_path))

    structures_dir = tmp_path / "structures"
    structures_dir.mkdir()
    # Raw ColabFold output: multiple ranked models per target, unrenamed.
    (structures_dir / "P1_relaxed_rank_001_alphafold2_ptm_model_4_seed_000.pdb").write_text(
        _PDB_WITH_SIDECHAIN
    )
    (structures_dir / "P1_relaxed_rank_002_alphafold2_ptm_model_1_seed_000.pdb").write_text(
        _PDB_WITH_SIDECHAIN.replace("MET", "ALA")
    )
    (structures_dir / "P2_relaxed_rank_001_alphafold2_ptm_model_3_seed_000.pdb").write_text(
        _PDB_WITH_SIDECHAIN.replace("MET", "GLY")
    )

    out = tmp_path / "data.parquetbundle"
    result = CliRunner().invoke(
        app,
        [
            "bundle",
            "-p",
            str(proj),
            "-a",
            str(ann_path),
            "-o",
            str(out),
            "-t",
            str(structures_dir),
            "--no-minify-structures",
        ],
    )
    assert result.exit_code == 0, result.output

    table = pq.read_table(pa.BufferReader(read_structures_from_bundle(out)))
    assert set(table.column("protein_id").to_pylist()) == {"P1", "P2"}
    pdb_by_id = dict(
        zip(
            table.column("protein_id").to_pylist(),
            table.column("pdb_data").to_pylist(),
            strict=True,
        )
    )
    # rank_001 (best model) wins over rank_002 for P1
    assert pdb_by_id["P1"] == _PDB_WITH_SIDECHAIN
    assert pdb_by_id["P2"] == _PDB_WITH_SIDECHAIN.replace("MET", "GLY")


def test_bundle_cli_structures_adds_plddt_annotation(tmp_path):
    """--structures adds a numeric 'plddt' annotation from each PDB's B-factor column."""
    from typer.testing import CliRunner

    from protspace.cli.app import app

    proj, ann_path, structures_dir = _make_bundle_inputs(tmp_path)

    out = tmp_path / "data.parquetbundle"
    result = CliRunner().invoke(
        app,
        ["bundle", "-p", str(proj), "-a", str(ann_path), "-o", str(out), "-t", str(structures_dir)],
    )
    assert result.exit_code == 0, result.output

    core, _ = read_bundle(out)
    annotations_table = pq.read_table(pa.BufferReader(core[0]))
    assert "plddt" in annotations_table.column_names
    plddt_by_id = dict(
        zip(
            annotations_table.column("protein_id").to_pylist(),
            annotations_table.column("plddt").to_pylist(),
            strict=True,
        )
    )
    # _PDB_WITH_SIDECHAIN's single CA atom has B-factor 20.00 for both P1 and P2.
    assert plddt_by_id == {"P1": 20.0, "P2": 20.0}


def test_bundle_cli_structures_missing_ca_leaves_plddt_null(tmp_path):
    """A protein whose PDB has no CA atoms gets a null plddt, not an error."""
    from typer.testing import CliRunner

    from protspace.cli.app import app

    proj, ann_path, structures_dir = _make_bundle_inputs(tmp_path)
    (structures_dir / "P2.pdb").write_text(
        "HETATM    1  O   HOH A 101      20.000  20.000  20.000  1.00 30.00           O\n"
    )

    out = tmp_path / "data.parquetbundle"
    result = CliRunner().invoke(
        app,
        ["bundle", "-p", str(proj), "-a", str(ann_path), "-o", str(out), "-t", str(structures_dir)],
    )
    assert result.exit_code == 0, result.output

    core, _ = read_bundle(out)
    annotations_table = pq.read_table(pa.BufferReader(core[0]))
    plddt_by_id = dict(
        zip(
            annotations_table.column("protein_id").to_pylist(),
            annotations_table.column("plddt").to_pylist(),
            strict=True,
        )
    )
    assert plddt_by_id == {"P1": 20.0, "P2": None}


def test_bundle_cli_structures_preserves_existing_plddt_column(tmp_path):
    """An annotations file that already has 'plddt' is not clobbered."""
    from typer.testing import CliRunner

    from protspace.cli.app import app

    proj, ann_path, structures_dir = _make_bundle_inputs(tmp_path)
    pq.write_table(
        pa.table({"identifier": ["P1", "P2"], "plddt": [1.0, 2.0]}),
        str(ann_path),
    )

    out = tmp_path / "data.parquetbundle"
    result = CliRunner().invoke(
        app,
        ["bundle", "-p", str(proj), "-a", str(ann_path), "-o", str(out), "-t", str(structures_dir)],
    )
    assert result.exit_code == 0, result.output

    core, _ = read_bundle(out)
    annotations_table = pq.read_table(pa.BufferReader(core[0]))
    plddt_by_id = dict(
        zip(
            annotations_table.column("protein_id").to_pylist(),
            annotations_table.column("plddt").to_pylist(),
            strict=True,
        )
    )
    assert plddt_by_id == {"P1": 1.0, "P2": 2.0}
