#!/usr/bin/env node

/**
 * Restore the 2026-05-29 GPS log from a WPS .et (OOXML) backup, split it by
 * the timestamp ranges retained in test_pairs, and draw one GPS comparison
 * plot for every pair.
 *
 * Outputs:
 *   landmarkslam/implement/data/test/gps_record_20260529_113616.csv
 *   landmarkslam/implement/data/test_pairs/pair_XX_YY/seq1_gps.csv
 *   landmarkslam/implement/data/test_pairs/pair_XX_YY/seq2_gps.csv
 *   landmarkslam/implement/data/test_pairs/pair_XX_YY/seq1_gps_full.csv
 *   landmarkslam/implement/data/test_pairs/pair_XX_YY/seq2_gps_full.csv
 *   landmarkslam/implement/data/test_pairs/pair_XX_YY/gps_trajectory.svg
 *   landmarkslam/implement/data/test_pairs/pair_XX_YY/gps_trajectory.png
 */

const fs = require('fs');
const path = require('path');
const zlib = require('zlib');

const repoRoot = path.resolve(__dirname, '../../../..');
const dataDir = path.join(repoRoot, 'landmarkslam/implement/data');
const pairsDir = path.join(dataDir, 'test_pairs');
const restoredMaster = path.join(dataDir, 'test/gps_record_20260529_113616.csv');
const backupDir = '/home/ehuy/.local/share/Kingsoft/office6/data/backup';
const backupPrefix = 'gps_record_20260529_113616.csv.';

function xmlDecode(value) {
  return value
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'");
}

function unzipText(archive, member) {
  const zip = fs.readFileSync(archive);
  let eocd = -1;
  for (let i = zip.length - 22; i >= Math.max(0, zip.length - 65557); i--) {
    if (zip.readUInt32LE(i) === 0x06054b50) { eocd = i; break; }
  }
  if (eocd < 0) throw new Error(`Invalid ZIP/OOXML file: ${archive}`);
  const entries = zip.readUInt16LE(eocd + 10);
  let cursor = zip.readUInt32LE(eocd + 16);
  for (let i = 0; i < entries; i++) {
    if (zip.readUInt32LE(cursor) !== 0x02014b50) throw new Error('Invalid ZIP central directory');
    const method = zip.readUInt16LE(cursor + 10);
    const compressedSize = zip.readUInt32LE(cursor + 20);
    const nameLength = zip.readUInt16LE(cursor + 28);
    const extraLength = zip.readUInt16LE(cursor + 30);
    const commentLength = zip.readUInt16LE(cursor + 32);
    const localOffset = zip.readUInt32LE(cursor + 42);
    const name = zip.subarray(cursor + 46, cursor + 46 + nameLength).toString('utf8');
    if (name === member) {
      if (zip.readUInt32LE(localOffset) !== 0x04034b50) throw new Error('Invalid ZIP local header');
      const localNameLength = zip.readUInt16LE(localOffset + 26);
      const localExtraLength = zip.readUInt16LE(localOffset + 28);
      const start = localOffset + 30 + localNameLength + localExtraLength;
      const payload = zip.subarray(start, start + compressedSize);
      if (method === 0) return payload.toString('utf8');
      if (method === 8) return zlib.inflateRawSync(payload).toString('utf8');
      throw new Error(`Unsupported ZIP compression method ${method} for ${member}`);
    }
    cursor += 46 + nameLength + extraLength + commentLength;
  }
  throw new Error(`ZIP member not found: ${member}`);
}

function parseSharedStrings(xml) {
  const values = [];
  for (const match of xml.matchAll(/<si>([\s\S]*?)<\/si>/g)) {
    const parts = [...match[1].matchAll(/<t(?:\s[^>]*)?>([\s\S]*?)<\/t>/g)];
    values.push(xmlDecode(parts.map((item) => item[1]).join('')));
  }
  return values;
}

function parseSheet(xml, sharedStrings) {
  const rows = [];
  for (const rowMatch of xml.matchAll(/<row\b[^>]*>([\s\S]*?)<\/row>/g)) {
    const row = [];
    for (const cellMatch of rowMatch[1].matchAll(/<c\b([^>]*)>([\s\S]*?)<\/c>/g)) {
      const attrs = cellMatch[1];
      const body = cellMatch[2];
      const ref = /\br="([A-Z]+)\d+"/.exec(attrs);
      const valueMatch = /<v>([\s\S]*?)<\/v>/.exec(body);
      if (!ref || !valueMatch) continue;
      const column = ref[1].charCodeAt(0) - 65;
      let value = xmlDecode(valueMatch[1]);
      if (/\bt="s"/.test(attrs)) value = sharedStrings[Number(value)] ?? '';
      row[column] = value;
    }
    rows.push(row);
  }
  return rows;
}

function findBackup() {
  const candidates = fs.readdirSync(backupDir)
    .filter((name) => name.startsWith(backupPrefix) && name.endsWith('.et'))
    .map((name) => path.join(backupDir, name))
    .sort();
  if (!candidates.length) throw new Error(`No WPS GPS backup found in ${backupDir}`);
  return candidates[candidates.length - 1];
}

function loadGpsBackup(archive) {
  const shared = parseSharedStrings(unzipText(archive, 'xl/sharedStrings.xml'));
  const rows = parseSheet(unzipText(archive, 'xl/worksheets/sheet1.xml'), shared);
  if (rows.length < 2) throw new Error('GPS workbook contains no data rows');
  const header = rows[0].slice(0, 5);
  const expected = ['sys_timestamp', 'lat', 'lon', 'alt', 'fix_quality'];
  if (header.join(',') !== expected.join(',')) {
    throw new Error(`Unexpected GPS header: ${header.join(',')}`);
  }
  const records = rows.slice(1).map((row) => ({
    raw: row.slice(0, 5).map((value) => value ?? ''),
    timestamp: Number(row[0]),
    lat: Number(row[1]),
    lon: Number(row[2]),
    altitude: Number(row[3]),
    fixQuality: Number(row[4]),
  })).filter((row) => Number.isFinite(row.timestamp));
  records.sort((a, b) => a.timestamp - b.timestamp);
  return { header, records };
}

function csvEscape(value) {
  const text = String(value ?? '');
  return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function writeCsv(filePath, header, records) {
  const lines = [header.map(csvEscape).join(',')];
  for (const record of records) lines.push(record.raw.map(csvEscape).join(','));
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${lines.join('\n')}\n`, 'utf8');
}

function trajectoryRange(filePath) {
  const lines = fs.readFileSync(filePath, 'utf8').split(/\r?\n/).filter(Boolean);
  if (!lines.length) throw new Error(`Empty trajectory: ${filePath}`);
  const timestamps = lines.map((line) => Number(line.trim().split(/\s+/)[0]));
  return { start: Math.min(...timestamps), end: Math.max(...timestamps), count: lines.length };
}

function latestCompleteRun(pairDir) {
  const resultsDir = path.join(pairDir, 'lsq_results');
  const candidates = fs.readdirSync(resultsDir, { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && /^\d{8}_\d{6}$/.test(entry.name))
    .map((entry) => path.join(resultsDir, entry.name))
    .filter((runDir) => fs.existsSync(path.join(runDir, 'trajectory_03.txt'))
      && fs.existsSync(path.join(runDir, 'trajectory_19_aligned.txt')))
    .sort();
  if (!candidates.length) throw new Error(`No complete run found in ${resultsDir}`);
  return candidates[candidates.length - 1];
}

function selectGps(records, range) {
  return records.filter((record) => record.timestamp >= range.start && record.timestamp <= range.end);
}

function escapeXml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function makeGpsSvg(pairName, seq1, seq2, ranges, sourceRun) {
  const width = 1200;
  const height = 800;
  const plot = { x: 105, y: 125, w: 980, h: 555 };
  const valid = [...seq1, ...seq2].filter((p) => Number.isFinite(p.lat)
    && Number.isFinite(p.lon) && Math.abs(p.lat) > 1e-9 && Math.abs(p.lon) > 1e-9);
  if (!valid.length) throw new Error(`${pairName}: no valid latitude/longitude points`);
  const lat0 = valid.reduce((sum, p) => sum + p.lat, 0) / valid.length;
  const lon0 = valid.reduce((sum, p) => sum + p.lon, 0) / valid.length;
  const cosLat = Math.cos(lat0 * Math.PI / 180);
  const projectMeters = (p) => ({
    e: (p.lon - lon0) * 111320 * cosLat,
    n: (p.lat - lat0) * 110540,
  });
  const p1 = seq1.filter((p) => p.lat && p.lon).map(projectMeters);
  const p2 = seq2.filter((p) => p.lat && p.lon).map(projectMeters);
  const all = [...p1, ...p2];
  let minE = Math.min(...all.map((p) => p.e));
  let maxE = Math.max(...all.map((p) => p.e));
  let minN = Math.min(...all.map((p) => p.n));
  let maxN = Math.max(...all.map((p) => p.n));
  if (maxE - minE < 1) { minE -= 0.5; maxE += 0.5; }
  if (maxN - minN < 1) { minN -= 0.5; maxN += 0.5; }
  const pad = Math.max(maxE - minE, maxN - minN) * 0.08;
  minE -= pad; maxE += pad; minN -= pad; maxN += pad;
  const scale = Math.min(plot.w / (maxE - minE), plot.h / (maxN - minN));
  const usedW = (maxE - minE) * scale;
  const usedH = (maxN - minN) * scale;
  const offsetX = plot.x + (plot.w - usedW) / 2;
  const offsetY = plot.y + (plot.h - usedH) / 2;
  const xy = (p) => ({
    x: offsetX + (p.e - minE) * scale,
    y: offsetY + usedH - (p.n - minN) * scale,
  });
  const points = (items) => items.map((p) => {
    const q = xy(p);
    return `${q.x.toFixed(2)},${q.y.toFixed(2)}`;
  }).join(' ');
  const marker = (items, color) => {
    if (!items.length) return '';
    const start = xy(items[0]);
    const end = xy(items[items.length - 1]);
    return `<circle cx="${start.x}" cy="${start.y}" r="7" fill="white" stroke="${color}" stroke-width="4"/>`
      + `<rect x="${end.x - 6}" y="${end.y - 6}" width="12" height="12" fill="${color}"/>`;
  };
  const grid = [];
  for (let i = 0; i <= 5; i++) {
    const x = plot.x + (plot.w * i / 5);
    const y = plot.y + (plot.h * i / 5);
    grid.push(`<line x1="${x}" y1="${plot.y}" x2="${x}" y2="${plot.y + plot.h}" class="grid"/>`);
    grid.push(`<line x1="${plot.x}" y1="${y}" x2="${plot.x + plot.w}" y2="${y}" class="grid"/>`);
  }
  const fmtTime = (value) => new Date(value * 1000).toLocaleString('zh-CN', {
    timeZone: 'Asia/Shanghai', hour12: false,
  });
  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
  <style>
    text { font-family: "Noto Sans CJK SC", "Microsoft YaHei", sans-serif; fill: #14253b; }
    .grid { stroke: #dbe4ee; stroke-width: 1; }
    .small { font-size: 16px; fill: #52677f; }
  </style>
  <rect width="100%" height="100%" fill="#f7f9fc"/>
  <text x="60" y="58" font-size="31" font-weight="700">GPS trajectory comparison — ${escapeXml(pairName)}</text>
  <text x="60" y="91" class="small">Local ENU coordinates; circle = start, square = end</text>
  <rect x="${plot.x}" y="${plot.y}" width="${plot.w}" height="${plot.h}" fill="white" stroke="#b9c8d8" stroke-width="2"/>
  ${grid.join('\n  ')}
  <polyline points="${points(p1)}" fill="none" stroke="#246bd6" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
  <polyline points="${points(p2)}" fill="none" stroke="#e5524b" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
  ${marker(p1, '#246bd6')}
  ${marker(p2, '#e5524b')}
  <line x1="760" y1="62" x2="805" y2="62" stroke="#246bd6" stroke-width="5"/><text x="815" y="68" font-size="18">seq1 (${seq1.length} GPS)</text>
  <line x1="960" y1="62" x2="1005" y2="62" stroke="#e5524b" stroke-width="5"/><text x="1015" y="68" font-size="18">seq2 (${seq2.length} GPS)</text>
  <text x="595" y="716" text-anchor="middle" font-size="18">East (m)</text>
  <text x="32" y="405" text-anchor="middle" font-size="18" transform="rotate(-90 32 405)">North (m)</text>
  <text x="60" y="752" class="small">seq1: ${escapeXml(fmtTime(ranges.seq1.start))} – ${escapeXml(fmtTime(ranges.seq1.end))}</text>
  <text x="60" y="778" class="small">seq2: ${escapeXml(fmtTime(ranges.seq2.start))} – ${escapeXml(fmtTime(ranges.seq2.end))}</text>
  <text x="1140" y="778" text-anchor="end" class="small">source run: ${escapeXml(path.basename(sourceRun))}</text>
</svg>\n`;
}

function crc32(buffer) {
  let crc = 0xffffffff;
  for (const byte of buffer) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit++) crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function pngChunk(type, data) {
  const name = Buffer.from(type, 'ascii');
  const length = Buffer.alloc(4);
  length.writeUInt32BE(data.length);
  const checksum = Buffer.alloc(4);
  checksum.writeUInt32BE(crc32(Buffer.concat([name, data])));
  return Buffer.concat([length, name, data, checksum]);
}

function makeGpsPng(seq1, seq2) {
  const width = 1200;
  const height = 800;
  const plot = { x: 105, y: 100, w: 980, h: 610 };
  const pixels = Buffer.alloc(width * height * 4);
  const setPixel = (x, y, color) => {
    x = Math.round(x); y = Math.round(y);
    if (x < 0 || x >= width || y < 0 || y >= height) return;
    const index = (y * width + x) * 4;
    pixels[index] = color[0]; pixels[index + 1] = color[1];
    pixels[index + 2] = color[2]; pixels[index + 3] = 255;
  };
  const fillRect = (x, y, w, h, color) => {
    for (let yy = Math.max(0, y); yy < Math.min(height, y + h); yy++) {
      for (let xx = Math.max(0, x); xx < Math.min(width, x + w); xx++) setPixel(xx, yy, color);
    }
  };
  const fillCircle = (cx, cy, radius, color) => {
    for (let y = -radius; y <= radius; y++) {
      for (let x = -radius; x <= radius; x++) {
        if (x * x + y * y <= radius * radius) setPixel(cx + x, cy + y, color);
      }
    }
  };
  const drawLine = (a, b, color, thickness = 1) => {
    const dx = b.x - a.x; const dy = b.y - a.y;
    const steps = Math.max(1, Math.ceil(Math.max(Math.abs(dx), Math.abs(dy))));
    for (let i = 0; i <= steps; i++) {
      fillCircle(a.x + dx * i / steps, a.y + dy * i / steps, Math.max(1, Math.floor(thickness / 2)), color);
    }
  };
  const drawPath = (items, color) => {
    for (let i = 1; i < items.length; i++) drawLine(items[i - 1], items[i], color, 4);
    if (items.length) {
      fillCircle(items[0].x, items[0].y, 8, [255, 255, 255]);
      fillCircle(items[0].x, items[0].y, 5, color);
      fillRect(Math.round(items.at(-1).x) - 7, Math.round(items.at(-1).y) - 7, 14, 14, color);
    }
  };

  fillRect(0, 0, width, height, [247, 249, 252]);
  fillRect(plot.x, plot.y, plot.w, plot.h, [255, 255, 255]);
  for (let i = 0; i <= 5; i++) {
    const x = plot.x + Math.round(plot.w * i / 5);
    const y = plot.y + Math.round(plot.h * i / 5);
    drawLine({ x, y: plot.y }, { x, y: plot.y + plot.h }, [219, 228, 238]);
    drawLine({ x: plot.x, y }, { x: plot.x + plot.w, y }, [219, 228, 238]);
  }
  const valid = [...seq1, ...seq2].filter((p) => Number.isFinite(p.lat)
    && Number.isFinite(p.lon) && Math.abs(p.lat) > 1e-9 && Math.abs(p.lon) > 1e-9);
  const lat0 = valid.reduce((sum, p) => sum + p.lat, 0) / valid.length;
  const lon0 = valid.reduce((sum, p) => sum + p.lon, 0) / valid.length;
  const cosLat = Math.cos(lat0 * Math.PI / 180);
  const project = (p) => ({ e: (p.lon - lon0) * 111320 * cosLat, n: (p.lat - lat0) * 110540 });
  const p1m = seq1.filter((p) => p.lat && p.lon).map(project);
  const p2m = seq2.filter((p) => p.lat && p.lon).map(project);
  const all = [...p1m, ...p2m];
  let minE = Math.min(...all.map((p) => p.e)); let maxE = Math.max(...all.map((p) => p.e));
  let minN = Math.min(...all.map((p) => p.n)); let maxN = Math.max(...all.map((p) => p.n));
  if (maxE - minE < 1) { minE -= 0.5; maxE += 0.5; }
  if (maxN - minN < 1) { minN -= 0.5; maxN += 0.5; }
  const pad = Math.max(maxE - minE, maxN - minN) * 0.08;
  minE -= pad; maxE += pad; minN -= pad; maxN += pad;
  const scale = Math.min(plot.w / (maxE - minE), plot.h / (maxN - minN));
  const usedW = (maxE - minE) * scale; const usedH = (maxN - minN) * scale;
  const offsetX = plot.x + (plot.w - usedW) / 2; const offsetY = plot.y + (plot.h - usedH) / 2;
  const toPixel = (p) => ({
    x: offsetX + (p.e - minE) * scale,
    y: offsetY + usedH - (p.n - minN) * scale,
  });
  drawPath(p1m.map(toPixel), [36, 107, 214]);
  drawPath(p2m.map(toPixel), [229, 82, 75]);
  drawLine({ x: 385, y: 52 }, { x: 455, y: 52 }, [36, 107, 214], 6);
  drawLine({ x: 745, y: 52 }, { x: 815, y: 52 }, [229, 82, 75], 6);
  for (let x = plot.x; x <= plot.x + plot.w; x++) {
    setPixel(x, plot.y, [142, 160, 180]); setPixel(x, plot.y + plot.h, [142, 160, 180]);
  }
  for (let y = plot.y; y <= plot.y + plot.h; y++) {
    setPixel(plot.x, y, [142, 160, 180]); setPixel(plot.x + plot.w, y, [142, 160, 180]);
  }

  const raw = Buffer.alloc((width * 4 + 1) * height);
  for (let y = 0; y < height; y++) {
    const rowStart = y * (width * 4 + 1);
    raw[rowStart] = 0;
    pixels.copy(raw, rowStart + 1, y * width * 4, (y + 1) * width * 4);
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0); ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8; ihdr[9] = 6;
  return Buffer.concat([
    Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
    pngChunk('IHDR', ihdr),
    pngChunk('IDAT', zlib.deflateSync(raw, { level: 9 })),
    pngChunk('IEND', Buffer.alloc(0)),
  ]);
}

function main() {
  const backup = process.argv[2] || findBackup();
  const { header, records } = loadGpsBackup(backup);
  if (!records.length) throw new Error('No GPS records parsed from backup');
  writeCsv(restoredMaster, header, records);

  const pairNames = fs.readdirSync(pairsDir, { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && /^pair_\d+_\d+$/.test(entry.name))
    .map((entry) => entry.name)
    .sort();
  const pairInfos = pairNames.map((pairName) => {
    const pairDir = path.join(pairsDir, pairName);
    const runDir = latestCompleteRun(pairDir);
    return {
      pairName,
      pairDir,
      runDir,
      seq1Range: trajectoryRange(path.join(runDir, 'trajectory_03.txt')),
      seq2Range: trajectoryRange(path.join(runDir, 'trajectory_19_aligned.txt')),
    };
  });
  const fullRanges = {
    seq1: {
      start: Math.min(...pairInfos.map((info) => info.seq1Range.start)),
      end: Math.max(...pairInfos.map((info) => info.seq1Range.end)),
    },
    seq2: {
      start: Math.min(...pairInfos.map((info) => info.seq2Range.start)),
      end: Math.max(...pairInfos.map((info) => info.seq2Range.end)),
    },
  };
  const fullSeq1 = selectGps(records, fullRanges.seq1);
  const fullSeq2 = selectGps(records, fullRanges.seq2);
  const fullDir = path.join(pairsDir, 'full_gps');
  const fullSeq1Path = path.join(fullDir, 'seq1_gps.csv');
  const fullSeq2Path = path.join(fullDir, 'seq2_gps.csv');
  writeCsv(fullSeq1Path, header, fullSeq1);
  writeCsv(fullSeq2Path, header, fullSeq2);
  fs.writeFileSync(path.join(fullDir, 'gps_trajectory_full.svg'),
    makeGpsSvg('complete seq1 / seq2 journeys', fullSeq1, fullSeq2, fullRanges, 'all latest pair runs'), 'utf8');
  fs.writeFileSync(path.join(fullDir, 'gps_trajectory_full.png'), makeGpsPng(fullSeq1, fullSeq2));

  const summary = [];
  for (const { pairName, pairDir, runDir, seq1Range, seq2Range } of pairInfos) {
    const seq1 = selectGps(records, seq1Range);
    const seq2 = selectGps(records, seq2Range);
    if (!seq1.length || !seq2.length) {
      throw new Error(`${pairName}: GPS not found for ranges ${seq1Range.start}-${seq1Range.end}, ${seq2Range.start}-${seq2Range.end}`);
    }
    writeCsv(path.join(pairDir, 'seq1_gps.csv'), header, seq1);
    writeCsv(path.join(pairDir, 'seq2_gps.csv'), header, seq2);
    fs.copyFileSync(fullSeq1Path, path.join(pairDir, 'seq1_gps_full.csv'));
    fs.copyFileSync(fullSeq2Path, path.join(pairDir, 'seq2_gps_full.csv'));
    const svg = makeGpsSvg(pairName, seq1, seq2,
      { seq1: seq1Range, seq2: seq2Range }, runDir);
    fs.writeFileSync(path.join(pairDir, 'gps_trajectory.svg'), svg, 'utf8');
    fs.writeFileSync(path.join(pairDir, 'gps_trajectory.png'), makeGpsPng(seq1, seq2));
    summary.push({ pairName, run: path.basename(runDir), seq1: seq1.length, seq2: seq2.length });
  }

  console.log(`GPS backup: ${backup}`);
  console.log(`Restored master: ${restoredMaster} (${records.length} records)`);
  console.log(`Complete journeys: seq1=${fullSeq1.length}, seq2=${fullSeq2.length} GPS records`);
  console.log(`Complete journey copies: ${fullDir}`);
  for (const row of summary) {
    console.log(`${row.pairName}: ${row.seq1}/${row.seq2} GPS records, source ${row.run}`);
  }
}

main();
