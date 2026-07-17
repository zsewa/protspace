"""protspace bundle — combine projections + annotations into a .parquetbundle."""

import logging
from pathlib import Path
from typing import Annotated

import typer

from protspace.cli.app import PANEL_STAGES, app, setup_logging
from protspace.cli.common_options import Opt_Verbose

logger = logging.getLogger(__name__)


@app.command(rich_help_panel=PANEL_STAGES)
def bundle(
    projections: Annotated[
        Path,
        typer.Option(
            "-p",
            "--projections",
            help="Directory containing projections_metadata.parquet and projections_data.parquet.",
            exists=True,
        ),
    ],
    annotations: Annotated[
        Path,
        typer.Option(
            "-a",
            "--annotations",
            help="Annotations parquet file.",
            exists=True,
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("-o", "--output", help="Output .parquetbundle file path."),
    ],
    statistics: Annotated[
        Path | None,
        typer.Option(
            "-s",
            "--statistics",
            help="Optional projection-statistics parquet file → 5th bundle part.",
            exists=True,
        ),
    ] = None,
    settings: Annotated[
        Path | None,
        typer.Option(
            "--settings",
            help="Optional settings JSON (e.g. auto-generated cluster styles) → 4th bundle part.",
            exists=True,
        ),
    ] = None,
    structures: Annotated[
        Path | None,
        typer.Option(
            "-t",
            "--structures",
            help=(
                "Optional directory of .pdb structure files → 6th bundle part "
                "(bundled structures, shown alongside AlphaFold DB in the viewer). "
                "Accepts plain <protein_id>.pdb files or raw ColabFold/AlphaFold2 "
                "output (e.g. <protein_id>_relaxed_rank_001_alphafold2_ptm_model_"
                "1_seed_000.pdb) — the protein id and best-ranked model are "
                "inferred automatically. Each structure's mean pLDDT (read from "
                "the PDB B-factor column) is also added to the annotations as a "
                "numeric 'plddt' column, unless one already exists."
            ),
            exists=True,
            file_okay=False,
            dir_okay=True,
        ),
    ] = None,
    minify_structures: Annotated[
        bool,
        typer.Option(
            "--minify-structures/--no-minify-structures",
            help=(
                "Strip bundled PDB structures (--structures) to backbone atoms only "
                "(N, CA, C, O) for smaller bundles — enough for cartoon rendering. "
                "Use --no-minify-structures to keep full atom detail."
            ),
        ),
    ] = True,
    verbose: Opt_Verbose = 0,
) -> None:
    """Merge projections + annotations → .parquetbundle.

    \b
    Reads projections_metadata.parquet, projections_data.parquet from the
    projections directory and an annotations parquet file, then writes a
    single .parquetbundle file.
    """
    setup_logging(verbose)

    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    from protspace.data.annotations.encoding import stamp_format_version
    from protspace.data.io.bundle import write_bundle

    settings_obj = json.loads(settings.read_text()) if settings is not None else None

    metadata_path = projections / "projections_metadata.parquet"
    data_path = projections / "projections_data.parquet"

    if not metadata_path.exists():
        raise typer.BadParameter(f"Missing: {metadata_path}")
    if not data_path.exists():
        raise typer.BadParameter(f"Missing: {data_path}")

    annotations_table = pq.read_table(str(annotations))
    metadata_table = pq.read_table(str(metadata_path))
    data_table = pq.read_table(str(data_path))

    # Rename identifier column to protein_id if needed (bundle format).
    # Note: pa.Table.rename_columns() drops schema metadata, so the
    # format-version stamp below must happen *after* this rename.
    col_names = annotations_table.column_names
    if "identifier" in col_names and "protein_id" not in col_names:
        annotations_table = annotations_table.rename_columns(
            [("protein_id" if c == "identifier" else c) for c in col_names]
        )

    # Trust boundary: the -a annotations input is ASSUMED to be produced by the
    # same-version annotate/prepare pipeline (i.e. already percent-encoded).
    # We don't inspect its contents, so it's unconditionally stamped as v2 --
    # there is currently no other producer of this parquet to distrust.
    annotations_table = stamp_format_version(annotations_table)

    statistics_table = (
        pq.read_table(str(statistics)) if statistics is not None else None
    )

    structures_table = None
    if structures is not None:
        from protspace.data.io.af2_naming import resolve_structure_files
        from protspace.data.io.pdb_plddt import mean_plddt_from_pdb

        resolved_structures = resolve_structure_files(structures)
        if not resolved_structures:
            raise typer.BadParameter(f"No .pdb files found in {structures}")

        protein_ids = sorted(resolved_structures)
        raw_pdb_texts = [resolved_structures[protein_id].read_text() for protein_id in protein_ids]

        pdb_texts = raw_pdb_texts
        if minify_structures:
            from protspace.data.io.pdb_minify import minify_pdb_backbone

            pdb_texts = [minify_pdb_backbone(text) for text in pdb_texts]

        structures_table = pa.table(
            {
                "protein_id": protein_ids,
                "pdb_data": pdb_texts,
            }
        )

        plddt_by_id = {
            protein_id: plddt
            for protein_id, text in zip(protein_ids, raw_pdb_texts, strict=True)
            if (plddt := mean_plddt_from_pdb(text)) is not None
        }
        if plddt_by_id:
            id_column = "protein_id" if "protein_id" in annotations_table.column_names else "identifier"
            if "plddt" in annotations_table.column_names:
                logger.warning(
                    "Annotations already have a 'plddt' column; not overwriting it with "
                    "pLDDT scores from --structures."
                )
            else:
                ids = annotations_table.column(id_column).to_pylist()
                annotations_table = annotations_table.append_column(
                    "plddt", pa.array([plddt_by_id.get(pid) for pid in ids], type=pa.float64())
                )

    output_path = output.with_suffix(".parquetbundle")
    write_bundle(
        [annotations_table, metadata_table, data_table],
        output_path,
        settings=settings_obj,
        statistics=statistics_table,
        structures=structures_table,
    )

    typer.echo(f"Saved: {output_path}")
