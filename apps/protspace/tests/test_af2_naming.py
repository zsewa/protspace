from protspace.data.io.af2_naming import parse_af2_filename, resolve_structure_files


def test_parse_plain_id_falls_back_to_whole_stem():
    assert parse_af2_filename("P12345") == ("P12345", None)


def test_parse_colabfold_ranked_name():
    assert parse_af2_filename(
        "P12345_relaxed_rank_001_alphafold2_ptm_model_4_seed_000"
    ) == ("P12345", 1)


def test_parse_colabfold_ranked_name_unrelaxed_no_leading_zeros():
    assert parse_af2_filename("P12345_unrelaxed_rank_2_model_3") == ("P12345", 2)


def test_parse_af2_unranked_name():
    assert parse_af2_filename("P12345_unrelaxed_model_1_pred_0") == ("P12345", None)


def test_parse_id_containing_underscores_is_preserved():
    assert parse_af2_filename(
        "sp_P12345_relaxed_rank_001_alphafold2_ptm_model_1_seed_000"
    ) == ("sp_P12345", 1)


def test_resolve_plain_filenames(tmp_path):
    (tmp_path / "P1.pdb").write_text("ATOM\n")
    (tmp_path / "P2.pdb").write_text("ATOM\n")
    resolved = resolve_structure_files(tmp_path)
    assert set(resolved) == {"P1", "P2"}
    assert resolved["P1"].name == "P1.pdb"


def test_resolve_picks_best_ranked_colabfold_model(tmp_path):
    (tmp_path / "P1_relaxed_rank_001_alphafold2_ptm_model_4_seed_000.pdb").write_text("best\n")
    (tmp_path / "P1_relaxed_rank_002_alphafold2_ptm_model_1_seed_000.pdb").write_text("worse\n")
    (tmp_path / "P1_relaxed_rank_003_alphafold2_ptm_model_2_seed_000.pdb").write_text("worst\n")
    resolved = resolve_structure_files(tmp_path)
    assert set(resolved) == {"P1"}
    assert resolved["P1"].read_text() == "best\n"


def test_resolve_multiple_targets_colabfold(tmp_path):
    (tmp_path / "P1_relaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb").write_text("p1\n")
    (tmp_path / "P2_relaxed_rank_001_alphafold2_ptm_model_3_seed_000.pdb").write_text("p2\n")
    resolved = resolve_structure_files(tmp_path)
    assert resolved["P1"].read_text() == "p1\n"
    assert resolved["P2"].read_text() == "p2\n"


def test_resolve_unranked_duplicates_pick_first_sorted_and_warn(tmp_path, caplog):
    (tmp_path / "P1_unrelaxed_model_1_pred_0.pdb").write_text("first\n")
    (tmp_path / "P1_unrelaxed_model_2_pred_0.pdb").write_text("second\n")
    with caplog.at_level("WARNING"):
        resolved = resolve_structure_files(tmp_path)
    assert resolved["P1"].read_text() == "first\n"
    assert "P1" in caplog.text


def test_resolve_empty_dir_returns_empty(tmp_path):
    assert resolve_structure_files(tmp_path) == {}
