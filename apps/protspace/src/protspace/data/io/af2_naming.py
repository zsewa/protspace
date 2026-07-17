"""Derive protein ids from AF2/ColabFold-style prediction filenames.

ColabFold and AlphaFold2 write multiple ranked models per target with verbose
filenames such as ``P12345_relaxed_rank_001_alphafold2_ptm_model_4_seed_000.pdb``.
This lets ``protspace bundle --structures`` accept a raw folder of such
predictions directly, picking the best-ranked model per target, instead of
requiring files to be pre-renamed to ``<protein_id>.pdb``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# ColabFold / AlphaFold2 filenames embed the target id as a prefix, followed
# by relaxation state, an optional rank, and model/seed details, e.g.:
#   P12345_relaxed_rank_001_alphafold2_ptm_model_4_seed_000.pdb
#   P12345_unrelaxed_rank_1_model_3.pdb
#   P12345_unrelaxed_model_1_pred_0.pdb
_RANKED_RE = re.compile(r"^(?P<id>.+?)_(?:relaxed|unrelaxed)_rank_0*(?P<rank>\d+)_.*$")
_UNRANKED_RE = re.compile(r"^(?P<id>.+?)_(?:relaxed|unrelaxed)_model_\d+.*$")


def parse_af2_filename(stem: str) -> tuple[str, int | None]:
    """Parses (protein_id, rank) out of an AF2/ColabFold-style filename stem.

    Falls back to treating the whole stem as the protein id (rank ``None``)
    when no known AF2 naming pattern matches — preserves the original
    ``<protein_id>.pdb`` convention.
    """
    if m := _RANKED_RE.match(stem):
        return m.group("id"), int(m.group("rank"))
    if m := _UNRANKED_RE.match(stem):
        return m.group("id"), None
    return stem, None


def resolve_structure_files(directory: Path) -> dict[str, Path]:
    """Maps protein_id -> the best-ranked ``.pdb`` file for it in `directory`.

    Accepts plain ``<protein_id>.pdb`` files as well as raw ColabFold/AlphaFold2
    output (multiple ranked models per target). When several files resolve to
    the same protein id, the lowest rank number (best model) is kept; ties or
    unranked duplicates keep the first file in sorted order and log a warning.
    """
    grouped: dict[str, list[tuple[int, Path]]] = {}
    for path in sorted(directory.glob("*.pdb")):
        protein_id, rank = parse_af2_filename(path.stem)
        grouped.setdefault(protein_id, []).append((rank if rank is not None else 0, path))

    resolved: dict[str, Path] = {}
    for protein_id, candidates in grouped.items():
        candidates.sort(key=lambda candidate: candidate[0])
        resolved[protein_id] = candidates[0][1]
        if len(candidates) > 1:
            logger.warning(
                "%d structure files resolve to protein id '%s'; using best-ranked '%s'",
                len(candidates),
                protein_id,
                candidates[0][1].name,
            )
    return resolved
