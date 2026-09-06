/**
 * VoTT 2.x annotation export.
 *
 * VoTT stores one metadata file per image, named `<assetId>-asset.json`, in the
 * project's target-connection folder. The assetId is an MD5 of the image's
 * encoded file URI, so the export needs to know where the image will sit on
 * disk -- the same file under a different folder gets a different id and VoTT
 * will not match it up.
 *
 * Mirrors AssetService.createAssetFromFilePath / encodeFileURI from
 * microsoft/VoTT so the ids agree byte for byte.
 */
import type { Detection } from '../types';
import { md5Hex } from './md5';

/** Written into the `version` field; matches VoTT's last release. */
const VOTT_VERSION = '2.2.0';

/** VoTT's AssetState enum. */
const ASSET_STATE_TAGGED = 2;
/** VoTT's AssetType enum. */
const ASSET_TYPE_IMAGE = 1;

export interface VottRegion {
  id: string;
  type: 'RECTANGLE';
  tags: string[];
  boundingBox: { left: number; top: number; width: number; height: number };
  points: Array<{ x: number; y: number }>;
}

export interface VottAssetMetadata {
  asset: {
    id: string;
    format: string;
    state: number;
    type: number;
    name: string;
    path: string;
    size: { width: number; height: number };
  };
  regions: VottRegion[];
  version: string;
}

/**
 * Port of VoTT's encodeFileURI(path, true): backslashes become forward
 * slashes, the result is URI-encoded and prefixed with `file:`, then `#` and
 * `?` are escaped (encodeURI leaves those two alone).
 */
export function encodeFileURI(path: string): string {
  const encoded = `file:${encodeURI(path.replace(/\\/g, '/'))}`;
  return encoded.replace(/#/g, '%23').replace(/\?/g, '%3F');
}

/** Joins a folder and a file name with a single separator. */
export function joinPath(folder: string, name: string): string {
  return `${folder.replace(/[\\/]+$/, '')}/${name}`;
}

/**
 * Region ids only need to be unique within the file; VoTT generates them with
 * shortid, so anything of a similar shape is fine.
 */
function regionId(): string {
  const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  let out = '';
  for (let i = 0; i < 9; i++) {
    out += alphabet.charAt(Math.floor(Math.random() * alphabet.length));
  }
  return out;
}

const round2 = (n: number) => Math.round(n * 100) / 100;

/**
 * Builds the VoTT asset-metadata document for one detected image.
 *
 * @param folderPath Absolute path of the folder the image will live in when
 *   VoTT opens it, e.g. `C:\images` -- this feeds the asset id hash.
 * @param tagFor Maps a detection's class_name to the VoTT tag name.
 */
export function buildVottAssetMetadata(params: {
  imageName: string;
  folderPath: string;
  image: { width: number; height: number };
  detections: Detection[];
  tagFor: (className: string) => string;
}): VottAssetMetadata {
  const { imageName, folderPath, image, detections, tagFor } = params;

  const encodedPath = encodeFileURI(joinPath(folderPath, imageName));
  const id = md5Hex(encodedPath);

  // VoTT derives both from the already-encoded path, so we do too.
  const pathParts = encodedPath.split(/[\\/]/);
  const name = pathParts[pathParts.length - 1] ?? '';
  const nameParts = name.split('.');
  const format = (nameParts[nameParts.length - 1] ?? '').split(/[?#]/)[0] ?? '';

  const regions: VottRegion[] = detections.map((d) => {
    // Detections can spill a pixel or two past the edge; VoTT clamps on load
    // anyway, but keeping the file clean avoids surprises.
    const left = round2(Math.max(0, Math.min(d.bbox.x1, image.width)));
    const top = round2(Math.max(0, Math.min(d.bbox.y1, image.height)));
    const right = round2(Math.max(0, Math.min(d.bbox.x2, image.width)));
    const bottom = round2(Math.max(0, Math.min(d.bbox.y2, image.height)));
    const width = round2(Math.max(0, right - left));
    const height = round2(Math.max(0, bottom - top));

    return {
      id: regionId(),
      type: 'RECTANGLE',
      tags: [tagFor(d.class_name)],
      boundingBox: { left, top, width, height },
      points: [
        { x: left, y: top },
        { x: left + width, y: top },
        { x: left + width, y: top + height },
        { x: left, y: top + height },
      ],
    };
  });

  return {
    asset: {
      id,
      format,
      state: ASSET_STATE_TAGGED,
      type: ASSET_TYPE_IMAGE,
      name,
      path: encodedPath,
      size: { width: image.width, height: image.height },
    },
    regions,
    version: VOTT_VERSION,
  };
}

/** The file name VoTT expects for this asset's metadata. */
export function vottAssetFileName(metadata: VottAssetMetadata): string {
  return `${metadata.asset.id}-asset.json`;
}
