import { describe, it, expect } from 'vitest';
import {
  BUNDLE_DELIMITER,
  isParquetBundle,
  findBundleDelimiterPositions,
  createParquetBundle,
  type BundleSettings,
  type VisualizationData,
} from '@protspace/utils';
import { extractRowsFromParquetBundle } from './bundle';

// Helper to create a mock parquet-like buffer with PAR1 magic bytes
function createMockParquetBuffer(content: string = 'test'): ArrayBuffer {
  const encoder = new TextEncoder();
  const contentBytes = encoder.encode(content);
  // PAR1 magic at start and end
  const magic = encoder.encode('PAR1');
  const buffer = new Uint8Array(magic.length + contentBytes.length + magic.length);
  buffer.set(magic, 0);
  buffer.set(contentBytes, magic.length);
  buffer.set(magic, magic.length + contentBytes.length);
  return buffer.buffer;
}

// Helper to create a mock bundle with the specified number of parts
function createMockBundle(numParts: number): ArrayBuffer {
  const encoder = new TextEncoder();
  const delimiterBytes = encoder.encode(BUNDLE_DELIMITER);

  const parts: Uint8Array[] = [];
  for (let i = 0; i < numParts; i++) {
    const partContent = new Uint8Array(createMockParquetBuffer(`part${i + 1}`));
    parts.push(partContent);
  }

  // Calculate total size
  let totalSize = 0;
  for (let i = 0; i < parts.length; i++) {
    totalSize += parts[i].length;
    if (i < parts.length - 1) {
      totalSize += delimiterBytes.length;
    }
  }

  // Concatenate
  const result = new Uint8Array(totalSize);
  let offset = 0;
  for (let i = 0; i < parts.length; i++) {
    result.set(parts[i], offset);
    offset += parts[i].length;
    if (i < parts.length - 1) {
      result.set(delimiterBytes, offset);
      offset += delimiterBytes.length;
    }
  }

  return result.buffer;
}

function minimalVisualizationData(): VisualizationData {
  return {
    protein_ids: ['P1', 'P2'],
    projections: [
      {
        name: 'PCA_2',
        dimension: 2,
        data: new Float32Array([0, 0, 1, 1]),
      },
    ],
    annotations: {},
    annotation_data: {},
  };
}

/**
 * Manually splices a zero-byte statistics sentinel into a real 3-part bundle,
 * producing a 5-part / 4-delimiter shape (core + zero-byte settings + real
 * statistics) — the shape `protspace bundle -s statistics.parquet` (no
 * `--settings`) produces, which the TS writer itself never emits but the
 * reader must still accept and skip over.
 */
function inject5PartStatisticsBundle(threePartBundle: ArrayBuffer): ArrayBuffer {
  const encoder = new TextEncoder();
  const delimiterBytes = encoder.encode(BUNDLE_DELIMITER);
  const fakeStatsBytes = new Uint8Array(createMockParquetBuffer('fake-statistics-content'));
  const zeroByteSettings = new Uint8Array(0);

  const parts = [new Uint8Array(threePartBundle), zeroByteSettings, fakeStatsBytes];

  let totalSize = 0;
  for (const p of parts) totalSize += p.length;
  totalSize += delimiterBytes.length * 2;

  const result = new Uint8Array(totalSize);
  let offset = 0;
  result.set(parts[0], offset);
  offset += parts[0].length;
  result.set(delimiterBytes, offset);
  offset += delimiterBytes.length;
  result.set(parts[1], offset);
  offset += parts[1].length;
  result.set(delimiterBytes, offset);
  offset += delimiterBytes.length;
  result.set(parts[2], offset);

  return result.buffer;
}

describe('bundle utilities', () => {
  describe('isParquetBundle', () => {
    it('should return true for buffer containing delimiter', () => {
      const bundle = createMockBundle(3);
      expect(isParquetBundle(bundle)).toBe(true);
    });

    it('should return false for buffer without delimiter', () => {
      const buffer = createMockParquetBuffer('no delimiter');
      expect(isParquetBundle(buffer)).toBe(false);
    });

    it('should return false for empty buffer', () => {
      const buffer = new ArrayBuffer(0);
      expect(isParquetBundle(buffer)).toBe(false);
    });
  });

  describe('findBundleDelimiterPositions', () => {
    it('should find 2 delimiters in a 3-part bundle', () => {
      const bundle = createMockBundle(3);
      const uint8Array = new Uint8Array(bundle);
      const positions = findBundleDelimiterPositions(uint8Array);

      expect(positions.length).toBe(2);
      expect(positions[0]).toBeGreaterThan(0);
      expect(positions[1]).toBeGreaterThan(positions[0]);
    });

    it('should find 3 delimiters in a 4-part bundle', () => {
      const bundle = createMockBundle(4);
      const uint8Array = new Uint8Array(bundle);
      const positions = findBundleDelimiterPositions(uint8Array);

      expect(positions.length).toBe(3);
      expect(positions[0]).toBeGreaterThan(0);
      expect(positions[1]).toBeGreaterThan(positions[0]);
      expect(positions[2]).toBeGreaterThan(positions[1]);
    });

    it('should return empty array for buffer without delimiter', () => {
      const buffer = createMockParquetBuffer('no delimiter');
      const uint8Array = new Uint8Array(buffer);
      const positions = findBundleDelimiterPositions(uint8Array);

      expect(positions.length).toBe(0);
    });
  });

  describe('extractRowsFromParquetBundle — rejection boundaries', () => {
    it('should reject bundle with 1 delimiter (2 parts)', async () => {
      const bundle = createMockBundle(2);

      await expect(extractRowsFromParquetBundle(bundle)).rejects.toThrow(
        /Expected 2 to 5 delimiters/,
      );
    });

    it('should reject bundle with 6 delimiters (7 parts)', async () => {
      const bundle = createMockBundle(7);

      await expect(extractRowsFromParquetBundle(bundle)).rejects.toThrow(
        /Expected 2 to 5 delimiters/,
      );
    });

    it('should reject bundle with no delimiters', async () => {
      const buffer = createMockParquetBuffer('no delimiter');

      await expect(extractRowsFromParquetBundle(buffer)).rejects.toThrow(
        /Expected 2 to 5 delimiters/,
      );
    });
  });

  describe('extractRowsFromParquetBundle — real bundle shapes', () => {
    it('parses a 3-part bundle (core only) with no structures', async () => {
      const bundle = createParquetBundle(minimalVisualizationData());

      const result = await extractRowsFromParquetBundle(bundle);

      expect(result.projections).toHaveLength(2);
      expect(result.structuresById).toBeNull();
      expect(result.settings).toBeNull();
    });

    it('parses a 4-part bundle (core + settings), settings applied', async () => {
      const settings: BundleSettings = {
        legendSettings: {
          organism: {
            maxVisibleValues: 10,
            shapeSize: 24,
            sortMode: 'size-desc',
            hiddenValues: [],
            categories: {},
            enableDuplicateStackUI: false,
            selectedPaletteId: 'kellys',
          },
        },
        exportOptions: {},
      };
      const bundle = createParquetBundle(minimalVisualizationData(), {
        includeSettings: true,
        settings,
      });

      const result = await extractRowsFromParquetBundle(bundle);

      expect(result.settings?.legendSettings.organism.maxVisibleValues).toBe(10);
      expect(result.structuresById).toBeNull();
    });

    it('parses a 5-part bundle (core + zero-byte settings + statistics), skipping statistics', async () => {
      const threePartBundle = createParquetBundle(minimalVisualizationData());
      const fivePartBundle = inject5PartStatisticsBundle(threePartBundle);

      const result = await extractRowsFromParquetBundle(fivePartBundle);

      expect(result.projections).toHaveLength(2);
      expect(result.settings).toBeNull();
      expect(result.structuresById).toBeNull();
    });

    it('parses a 6-part bundle (core + zero-byte settings + zero-byte statistics + structures)', async () => {
      const data = minimalVisualizationData();
      data.structures = new Map([
        ['P1', 'ATOM      1  N   MET A   1\n'],
        ['P2', 'ATOM      1  N   GLY A   1\n'],
      ]);
      const bundle = createParquetBundle(data);

      const result = await extractRowsFromParquetBundle(bundle);

      expect(result.settings).toBeNull();
      expect(result.structuresById).not.toBeNull();
      expect(result.structuresById?.get('P1')).toContain('MET');
      expect(result.structuresById?.get('P2')).toContain('GLY');
    });

    it('parses a full 6-part bundle (core + settings + zero-byte statistics + structures)', async () => {
      const data = minimalVisualizationData();
      data.structures = new Map([['P1', 'ATOM      1  N   MET A   1\n']]);
      const settings: BundleSettings = {
        legendSettings: {},
        exportOptions: {},
      };
      // hasBundleSettings requires non-empty maps, so give it one entry
      settings.legendSettings.organism = {
        maxVisibleValues: 5,
        shapeSize: 16,
        sortMode: 'alpha-asc',
        hiddenValues: [],
        categories: {},
        enableDuplicateStackUI: false,
        selectedPaletteId: 'kellys',
      };
      const bundle = createParquetBundle(data, { includeSettings: true, settings });

      const result = await extractRowsFromParquetBundle(bundle);

      expect(result.settings?.legendSettings.organism.maxVisibleValues).toBe(5);
      expect(result.structuresById?.get('P1')).toContain('MET');
    });
  });
});

describe('BundleSettings type', () => {
  it('should have correct structure', () => {
    const settings: BundleSettings = {
      legendSettings: {
        organism: {
          maxVisibleValues: 10,
          // Legacy field — kept to verify backward-compat parsing.
          includeShapes: true,
          shapeSize: 24,
          sortMode: 'size-desc',
          hiddenValues: ['unknown'],
          categories: {
            human: { zOrder: 0, color: '#ff0000', shape: 'circle' },
          },
          enableDuplicateStackUI: false,
          selectedPaletteId: 'kellys',
        },
      },
      exportOptions: {
        organism: {
          imageWidth: 2048,
          imageHeight: 1024,
          lockAspectRatio: true,
          legendWidthPercent: 25,
          legendFontSizePx: 24,
          includeLegendSettings: true,
          includeExportOptions: true,
        },
      },
    };

    expect(settings.legendSettings.organism.maxVisibleValues).toBe(10);
    expect(settings.legendSettings.organism.sortMode).toBe('size-desc');
    expect(settings.legendSettings.organism.categories.human.color).toBe('#ff0000');
    expect(settings.exportOptions.organism.imageWidth).toBe(2048);
  });

  it('should accept settings with empty maps', () => {
    const settings: BundleSettings = {
      legendSettings: {},
      exportOptions: {},
    };

    expect(Object.keys(settings.legendSettings)).toHaveLength(0);
    expect(Object.keys(settings.exportOptions)).toHaveLength(0);
  });

  it('should accept settings with extra/unknown fields (forward compatibility)', () => {
    // This simulates loading settings from a newer version with additional fields
    const settingsWithExtras = {
      legendSettings: {
        organism: {
          maxVisibleValues: 10,
          shapeSize: 24,
          sortMode: 'size-desc',
          hiddenValues: [],
          categories: {},
          enableDuplicateStackUI: false,
          selectedPaletteId: 'kellys',
          unknownField: 'some value',
        },
      },
      exportOptions: {},
    };

    // Type-cast to BundleSettings - extra fields should be ignored
    const settings = settingsWithExtras as BundleSettings;
    expect(settings.legendSettings.organism.maxVisibleValues).toBe(10);
    expect(settings.legendSettings.organism.sortMode).toBe('size-desc');
  });

  it('should work with multiple annotations', () => {
    const settings: BundleSettings = {
      legendSettings: {
        organism: {
          maxVisibleValues: 10,
          shapeSize: 24,
          sortMode: 'size-desc',
          hiddenValues: ['unknown'],
          categories: {
            human: { zOrder: 0, color: '#ff0000', shape: 'circle' },
          },
          enableDuplicateStackUI: false,
          selectedPaletteId: 'kellys',
        },
        family: {
          maxVisibleValues: 5,
          shapeSize: 16,
          sortMode: 'alpha-asc',
          hiddenValues: [],
          categories: {
            kinase: { zOrder: 1, color: '#00ff00', shape: 'square' },
            phosphatase: { zOrder: 0, color: '#0000ff', shape: 'diamond' },
          },
          enableDuplicateStackUI: true,
          selectedPaletteId: 'kellys',
        },
      },
      exportOptions: {
        organism: {
          imageWidth: 2048,
          imageHeight: 1024,
          lockAspectRatio: true,
          legendWidthPercent: 25,
          legendFontSizePx: 24,
          includeLegendSettings: true,
          includeExportOptions: true,
        },
      },
    };

    expect(Object.keys(settings.legendSettings)).toHaveLength(2);
    expect(settings.legendSettings.family.categories.kinase.color).toBe('#00ff00');
  });
});
