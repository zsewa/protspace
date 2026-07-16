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


def test_bundle_cli_structures_flag(tmp_path):
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
    (structures_dir / "P1.pdb").write_text("ATOM      1  N   MET A   1\n")
    (structures_dir / "P2.pdb").write_text("ATOM      1  N   GLY A   1\n")

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
