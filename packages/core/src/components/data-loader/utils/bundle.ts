import { parquetReadObjects, parquetMetadata, type FileMetaData } from 'hyparquet';
import {
  BUNDLE_DELIMITER_BYTES,
  findBundleDelimiterPositions,
  normalizeBundleSettings,
  type BundleSettings,
} from '@protspace/utils';
import type { Rows, GenericRow } from './types';
import { assertValidParquetMagic, validateProjectionRows } from './validation';
import { sanitizePublishState } from '../../publish/publish-state-validator';

/** Key-value metadata key the Python writer stamps with the bundle's annotation format version. */
const FORMAT_VERSION_KEY = 'protspace_format_version';

/**
 * Result of extracting data from a parquetbundle.
 */
export interface BundleExtractionResult {
  /** Projection rows (x/y/z/projection_name/identifier) — annotation fields NOT spread in. */
  projections: Rows;
  /** Annotation rows keyed by protein id. */
  annotationsById: Map<string, GenericRow>;
  /** Column name in `projections` that carries the protein id. */
  projectionIdColumn: string;
  /** Column name in annotation rows that carries the protein id. */
  annotationIdColumn: string;
  projectionsMetadata: Rows;
  /** Settings loaded from bundle (null if not present) */
  settings: BundleSettings | null;
  /**
   * Bundled protein structures (part 6), keyed by `protein_id` → raw PDB file
   * text. `null` when the bundle has no structures part.
   */
  structuresById: Map<string, string> | null;
  /**
   * Bundle annotation format version, read from the `protspace_format_version`
   * parquet key-value metadata on the annotations part (part 1). `1` when the
   * key is absent, unparsable, or the part isn't a bundle at all (defaults to
   * legacy v1 behavior — plain-string labels, raw `;`-delimited multi-hit cells).
   */
  formatVersion: number;
}

/**
 * Reads the `protspace_format_version` key-value metadata entry from an
 * already-parsed parquet footer (part1's `FileMetaData`, produced once by
 * `parquetMetadata` and reused for the subsequent `parquetReadObjects` call —
 * avoids re-parsing the same footer twice).
 *
 * Returns `1` (legacy default) when the key is missing, non-numeric, or
 * lookup otherwise fails — this keeps v1/absent bundles rendering exactly as
 * before Task H2.
 */
function readFormatVersion(metadata: FileMetaData): number {
  const kv = metadata.key_value_metadata ?? [];
  const entry = kv.find((k) => k.key === FORMAT_VERSION_KEY);
  const v = entry?.value ? Number(entry.value) : 1;
  return Number.isFinite(v) ? v : 1;
}

/**
 * Splits the raw bundle bytes into parts on every delimiter occurrence.
 * `positions.length + 1` parts are returned, in order.
 */
function splitBundleParts(uint8Array: Uint8Array, positions: number[]): ArrayBuffer[] {
  const parts: ArrayBuffer[] = [];
  let start = 0;
  for (const pos of positions) {
    parts.push(uint8Array.subarray(start, pos).slice().buffer);
    start = pos + BUNDLE_DELIMITER_BYTES.length;
  }
  parts.push(uint8Array.subarray(start).slice().buffer);
  return parts;
}

/**
 * Extract rows and optional settings/structures from a parquetbundle.
 *
 * Supports 2–5 delimiters (3–6 parts), matching the positional layout the
 * Python writer produces: ``core(3) + settings? + statistics? + structures?``.
 * - 2 delimiters (3 parts): core only
 * - 3 delimiters (4 parts): core + settings
 * - 4 delimiters (5 parts): core + settings? + statistics (settings may be a
 *   zero-byte positional sentinel)
 * - 5 delimiters (6 parts): core + settings? + statistics? + structures
 *   (settings and/or statistics may be zero-byte positional sentinels)
 *
 * The statistics part (5th) has no web-app consumer today — it is skipped
 * over (not parsed) purely to keep byte offsets aligned for the structures
 * part that may follow it.
 */
export async function extractRowsFromParquetBundle(
  arrayBuffer: ArrayBuffer,
): Promise<BundleExtractionResult> {
  const uint8Array = new Uint8Array(arrayBuffer);
  const delimiterPositions = findBundleDelimiterPositions(uint8Array);

  if (delimiterPositions.length < 2 || delimiterPositions.length > 5) {
    throw new Error(
      `Expected 2 to 5 delimiters in parquetbundle, found ${delimiterPositions.length}`,
    );
  }

  const parts = splitBundleParts(uint8Array, delimiterPositions);

  let part1: ArrayBuffer | null = parts[0];
  let part2: ArrayBuffer | null = parts[1];
  let part3: ArrayBuffer | null = parts[2];
  // Settings (4th part): present when >=4 parts; a zero-length part is the
  // positional sentinel used when settings is absent but a later part exists.
  const part4: ArrayBuffer | null = parts.length >= 4 && parts[3].byteLength > 0 ? parts[3] : null;
  // Statistics (5th part): no TS consumer — intentionally not parsed.
  // Structures (6th part): present only in the full 6-part shape.
  const part6: ArrayBuffer | null = parts.length === 6 && parts[5].byteLength > 0 ? parts[5] : null;

  // Validate parquet magic for each part before parsing
  assertValidParquetMagic(part1);
  assertValidParquetMagic(part2);
  assertValidParquetMagic(part3);

  // Parse part1's footer once (the annotations part), before it's decoded, and reuse
  // the result both to read the format_version and as the `metadata` option below —
  // hyparquet re-derives metadata from the buffer when `metadata` is omitted, so
  // passing it explicitly avoids parsing the same footer twice. On parse failure,
  // fall back to `formatVersion = 1` and let `parquetReadObjects` (without `metadata`)
  // re-attempt the parse itself, surfacing the same error it would have before.
  let part1Metadata: FileMetaData | null = null;
  let formatVersion = 1;
  try {
    part1Metadata = parquetMetadata(part1);
    formatVersion = readFormatVersion(part1Metadata);
  } catch {
    formatVersion = 1;
  }

  // Decode sequentially and release each sliced buffer immediately after its decode completes.
  // hyparquet is CPU-bound on the single JS thread — Promise.all gives no real parallelism, only
  // interleaved async continuations that keep all three buffers + decode scratch live simultaneously.
  // Sequential decode ensures only one part's buffer is live at a time, cutting the transient
  // load-peak (critical for large datasets such as SwissProt 573 K where peak heap reached ~2.3 GB).
  const selectedAnnotationsData = part1Metadata
    ? await parquetReadObjects({ file: part1, metadata: part1Metadata })
    : await parquetReadObjects({ file: part1 });
  part1 = null;
  const projectionsMetadataData = await parquetReadObjects({ file: part2 });
  part2 = null;
  const projectionsData = await parquetReadObjects({ file: part3! });
  part3 = null;

  // Parse settings if present
  let settings: BundleSettings | null = null;
  if (part4) {
    settings = await extractSettings(part4);
  }

  // Parse bundled structures if present
  let structuresById: Map<string, string> | null = null;
  if (part6) {
    structuresById = await extractStructures(part6);
  }

  // Validate projection rows for expected bundle shape
  validateProjectionRows(projectionsData);

  // Find the ID column in annotation data
  const annotationIdColumn = findColumn(
    selectedAnnotationsData.length > 0 ? Object.keys(selectedAnnotationsData[0]) : [],
    ['protein_id', 'identifier', 'id', 'uniprot', 'entry'],
  );

  const finalAnnotationIdColumn =
    annotationIdColumn ||
    (selectedAnnotationsData.length > 0 ? Object.keys(selectedAnnotationsData[0])[0] : undefined) ||
    'identifier';

  // Build annotations map keyed by protein id
  const annotationsById = new Map<string, GenericRow>();
  for (const annotation of selectedAnnotationsData) {
    const proteinId = annotation[finalAnnotationIdColumn];
    if (proteinId != null) {
      annotationsById.set(String(proteinId), annotation);
    }
  }

  // Find the ID column in projection data
  const projectionIdColumn =
    findColumn(projectionsData.length > 0 ? Object.keys(projectionsData[0]) : [], [
      'identifier',
      'protein_id',
      'id',
      'uniprot',
      'entry',
    ]) || 'identifier';

  return {
    projections: projectionsData,
    annotationsById,
    projectionIdColumn,
    annotationIdColumn: finalAnnotationIdColumn,
    projectionsMetadata: projectionsMetadataData,
    settings,
    structuresById,
    formatVersion,
  };
}

/**
 * Extract and parse settings from the 4th part of the bundle.
 * Returns null if parsing fails (graceful degradation).
 */
async function extractSettings(settingsBuffer: ArrayBuffer): Promise<BundleSettings | null> {
  try {
    // Validate parquet magic
    assertValidParquetMagic(settingsBuffer);

    const settingsData = await parquetReadObjects({ file: settingsBuffer });

    if (!settingsData || settingsData.length === 0) {
      console.warn('Settings parquet is empty, using defaults');
      return null;
    }

    // Extract the settings_json column from the first row
    const firstRow = settingsData[0] as { settings_json?: string };
    const settingsJson = firstRow.settings_json;

    if (typeof settingsJson !== 'string') {
      console.warn('Settings JSON is not a string, using defaults');
      return null;
    }

    const parsed = JSON.parse(settingsJson);
    const normalized = normalizeBundleSettings(parsed, { sanitizePublishState });

    if (!normalized) {
      console.warn('Settings JSON does not match expected schema, using defaults');
      return null;
    }

    return normalized;
  } catch (error) {
    console.warn('Failed to parse settings from bundle, using defaults:', error);
    return null;
  }
}

/**
 * Extract and parse bundled structures from the 6th part of the bundle.
 * Returns null if parsing fails (graceful degradation — falls back to
 * AlphaFold-only structure loading).
 */
async function extractStructures(
  structuresBuffer: ArrayBuffer,
): Promise<Map<string, string> | null> {
  try {
    assertValidParquetMagic(structuresBuffer);

    const structuresData = await parquetReadObjects({ file: structuresBuffer });
    if (!structuresData || structuresData.length === 0) {
      return null;
    }

    const structuresById = new Map<string, string>();
    for (const row of structuresData as { protein_id?: unknown; pdb_data?: unknown }[]) {
      const proteinId = row.protein_id;
      const pdbData = row.pdb_data;
      if (proteinId != null && typeof pdbData === 'string') {
        structuresById.set(String(proteinId), pdbData);
      }
    }
    return structuresById.size > 0 ? structuresById : null;
  } catch (error) {
    console.warn('Failed to parse structures from bundle:', error);
    return null;
  }
}

export function findColumn(columnNames: string[], candidates: string[]): string | null {
  for (const candidate of candidates) {
    const found = columnNames.find((col) => col.toLowerCase().includes(candidate.toLowerCase()));
    if (found) return found;
  }
  return null;
}

/**
 * Materializes a single merged row per protein by spreading annotation fields
 * into projection rows. Used by:
 *  - the small-dataset path of `convertParquetToVisualizationData` (where the
 *    O(N) spread cost is acceptable), and
 *  - the legacy-format fallback in `convertLargeDatasetOptimized`.
 *
 * The large-bundle hot path stays on the separated shape and never calls this.
 */
export function materializeMergedRows(extraction: BundleExtractionResult): Rows {
  const { projections, annotationsById, projectionIdColumn } = extraction;
  const merged: Rows = new Array(projections.length);
  for (let i = 0; i < projections.length; i++) {
    const projection = projections[i];
    const proteinId = projection[projectionIdColumn];
    const annotation = proteinId != null ? annotationsById.get(String(proteinId)) : undefined;
    merged[i] = annotation ? { ...projection, ...annotation } : { ...projection };
  }
  return merged;
}
