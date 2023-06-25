// Copyright 2023, University of Colorado Boulder

/**
 * An encoding of a Vello shader fragment. Encodings can be combined, and will be resolved to a packed strcuture
 * sent to the GPU.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { WorkgroupSize } from './WorkgroupSize.js';
import Vector2 from '../../../../dot/js/Vector2.js';
import Utils from '../../../../dot/js/Utils.js';
import { ByteBuffer } from './ByteBuffer.js';
import { Affine } from './Affine.js';
import { BufferImage } from './BufferImage.js';
import DeviceContext from './DeviceContext.js';
import { scenery } from '../../imports.js';
import { AtlasSubImage } from './Atlas.js';
import { Arc, Cubic, EllipticalArc, Line, Quadratic, Shape } from '../../../../kite/js/imports.js';
import { SourceImage } from './SourceImage.js';

const TILE_WIDTH = 16; // u32
const TILE_HEIGHT = 16; // u32
const PATH_REDUCE_WG = 256; // u32
const PATH_BBOX_WG = 256; // u32
const PATH_COARSE_WG = 256; // u32
const CLIP_REDUCE_WG = 256; // u32

const LAYOUT_BYTES = 10 * 4; // 10x u32
const CONFIG_UNIFORM_BYTES = 9 * 4 + LAYOUT_BYTES; // 9x u32 + Layout
const PATH_MONOID_BYTES = 5 * 4; // 5x u32
const PATH_BBOX_BYTES = 6 * 4; // 4x i32, f32, u32
const CUBIC_BYTES = 12 * 4; // 10x f32, 2x u32
const DRAW_MONOID_BYTES = 4 * 4; // 4x u32
const CLIP_BYTES = 2 * 4; // 2x u32
const CLIP_ELEMENT_BYTES = 4 + 12 + 4 * 4; // u32 + 12x u8 + 4x f32
const CLIP_BIC_BYTES = 2 * 4; // 2x u32
const CLIP_BBOX_BYTES = 4 * 4; // 4x f32
const DRAW_BBOX_BYTES = 4 * 4; // 4x f32
const BUMP_ALLOCATORS_BYTES = 6 * 4; // 6x u32
const BIN_HEADER_BYTES = 2 * 4; // 2x u32
const PATH_BYTES = 4 * 4 + 4 + 3 * 4; // 4x f32 + u32 + 3x u32
const TILE_BYTES = 2 * 4; // i32, u32
const PATH_SEGMENT_BYTES = 6 * 4; // 5x f32, u32

// e.g. 0xff0000ff is full-alpha red
export type ColorRGBA32 = number;

export type F32 = number;
export type U32 = number;
export type U8 = number;

// If you're looking here, prefer SourceImage. It can go directly from image sources, and you won't have to manually
// premultiply data (which is required for BufferImage). For HTMLImageElements, use createImageBitmap() with the
// premultiplyAlpha: 'premultiply' option
export type EncodableImage = SourceImage | BufferImage;

const size_to_words = ( byte_size: number ): number => byte_size / 4;

const align_up = ( len: number, alignment: number ): number => {
  return len + ( ( ( ~len ) + 1 ) & ( alignment - 1 ) );
};

const next_multiple_of = ( val: number, rhs: number ): number => {
  const r = val % rhs;
  return r === 0 ? val : val + ( rhs - r );
};

// Convert u32/f32 to 4 bytes in little endian order
const scratch_to_bytes = new Uint8Array( 4 );
export const f32_to_bytes = ( float: number ): U8[] => {
  const view = new DataView( scratch_to_bytes.buffer );
  view.setFloat32( 0, float );
  return [ ...scratch_to_bytes ].reverse();
};
export const u32_to_bytes = ( int: number ): U8[] => {
  const view = new DataView( scratch_to_bytes.buffer );
  view.setUint32( 0, int );
  return [ ...scratch_to_bytes ].reverse();
};

export const with_alpha_factor = ( color: ColorRGBA32, alpha: number ): ColorRGBA32 => {
  return ( color & 0xffffff00 ) | ( Utils.roundSymmetric( ( color & 0xff ) * alpha ) & 0xff );
};

export const to_premul_u32 = ( rgba8color: ColorRGBA32 ): ColorRGBA32 => {
  const a = ( rgba8color & 0xff ) / 255;
  const r = Utils.roundSymmetric( ( ( rgba8color >>> 24 ) & 0xff ) * a >>> 0 );
  const g = Utils.roundSymmetric( ( ( rgba8color >>> 16 ) & 0xff ) * a >>> 0 );
  const b = Utils.roundSymmetric( ( ( rgba8color >>> 8 ) & 0xff ) * a >>> 0 );
  return ( ( r << 24 ) | ( g << 16 ) | ( b << 8 ) | ( rgba8color & 0xff ) ) >>> 0;
};

const lerpChannel = ( x: number, y: number, a: number ) => {
  return Utils.roundSymmetric( x * ( 1 - a ) + y * a >>> 0 );
};

export const lerp_rgba8 = ( c1: ColorRGBA32, c2: ColorRGBA32, t: number ): ColorRGBA32 => {
  const r = lerpChannel( ( c1 >>> 24 ) & 0xff, ( c2 >>> 24 ) & 0xff, t );
  const g = lerpChannel( ( c1 >>> 16 ) & 0xff, ( c2 >>> 16 ) & 0xff, t );
  const b = lerpChannel( ( c1 >>> 8 ) & 0xff, ( c2 >>> 8 ) & 0xff, t );
  const a = lerpChannel( c1 & 0xff, c2 & 0xff, t );
  return ( ( r << 24 ) | ( g << 16 ) | ( b << 8 ) | a ) >>> 0;
};

let globalEncodingID = 0;
let globalPathID = 0;

export class VelloColorStop {
  public constructor( public offset: number, public color: number ) {}
}
scenery.register( 'VelloColorStop', VelloColorStop );

export enum Extend {
  // Extends the image by repeating the edge color of the brush.
  Pad = 0,
  // Extends the image by repeating the brush.
  Repeat = 1,
  // Extends the image by reflecting the brush.
  Reflect = 2
}

scenery.register( 'Extend', Extend );

export enum Mix {
  // Default attribute which specifies no blending. The blending formula simply selects the source color.
  Normal = 0,
  // Source color is multiplied by the destination color and replaces the destination.
  Multiply = 1,
  // Multiplies the complements of the backdrop and source color values, then complements the result.
  Screen = 2,
  // Multiplies or screens the colors, depending on the backdrop color value.
  Overlay = 3,
  // Selects the darker of the backdrop and source colors.
  Darken = 4,
  // Selects the lighter of the backdrop and source colors.
  Lighten = 5,
  // Brightens the backdrop color to reflect the source color. Painting with black produces no
  // change.
  ColorDodge = 6,
  // Darkens the backdrop color to reflect the source color. Painting with white produces no
  // change.
  ColorBurn = 7,
  // Multiplies or screens the colors, depending on the source color value. The effect is
  // similar to shining a harsh spotlight on the backdrop.
  HardLight = 8,
  // Darkens or lightens the colors, depending on the source color value. The effect is similar
  // to shining a diffused spotlight on the backdrop.
  SoftLight = 9,
  // Subtracts the darker of the two constituent colors from the lighter color.
  Difference = 10,
  // Produces an effect similar to that of the Difference mode but lower in contrast. Painting
  // with white inverts the backdrop color; painting with black produces no change.
  Exclusion = 11,
  // Creates a color with the hue of the source color and the saturation and luminosity of the
  // backdrop color.
  Hue = 12,
  // Creates a color with the saturation of the source color and the hue and luminosity of the
  // backdrop color. Painting with this mode in an area of the backdrop that is a pure gray
  // (no saturation) produces no change.
  Saturation = 13,
  // Creates a color with the hue and saturation of the source color and the luminosity of the
  // backdrop color. This preserves the gray levels of the backdrop and is useful for coloring
  // monochrome images or tinting color images.
  Color = 14,
  // Creates a color with the luminosity of the source color and the hue and saturation of the
  // backdrop color. This produces an inverse effect to that of the Color mode.
  Luminosity = 15,
  // Clip is the same as normal, but the latter always creates an isolated blend group and the
  // former can optimize that out.
  Clip = 128
}

scenery.register( 'Mix', Mix );

export enum Compose {
  // No regions are enabled.
  Clear = 0,
  // Only the source will be present.
  Copy = 1,
  // Only the destination will be present.
  Dest = 2,
  // The source is placed over the destination.
  SrcOver = 3,
  // The destination is placed over the source.
  DestOver = 4,
  // The parts of the source that overlap with the destination are placed.
  SrcIn = 5,
  // The parts of the destination that overlap with the source are placed.
  DestIn = 6,
  // The parts of the source that fall outside of the destination are placed.
  SrcOut = 7,
  // The parts of the destination that fall outside of the source are placed.
  DestOut = 8,
  // The parts of the source which overlap the destination replace the destination. The
  // destination is placed everywhere else.
  SrcAtop = 9,
  // The parts of the destination which overlaps the source replace the source. The source is
  // placed everywhere else.
  DestAtop = 10,
  // The non-overlapping regions of source and destination are combined.
  Xor = 11,
  // The sum of the source image and destination image is displayed.
  Plus = 12,
  // Allows two elements to cross fade by changing their opacities from 0 to 1 on one
  // element and 1 to 0 on the other element.
  PlusLighter = 13
}

scenery.register( 'Compose', Compose );

// u32
export class DrawTag {
  // No operation.
  public static readonly NOP = 0;

  // Color fill.
  public static readonly COLOR = 0x44;

  // Linear gradient fill.
  public static readonly LINEAR_GRADIENT = 0x114;

  // Radial gradient fill.
  public static readonly RADIAL_GRADIENT = 0x29c;

  // Image fill.
  public static readonly IMAGE = 0x248;

  // Begin layer/clip.
  public static readonly BEGIN_CLIP = 0x9;

  // End layer/clip.
  public static readonly END_CLIP = 0x21;

  // Returns the size of the info buffer (in u32s) used by this tag.
  public static info_size( drawTag: U32 ): U32 {
    return ( ( drawTag >>> 6 ) & 0xf ) >>> 0;
  }
}

// u8
export class PathTag {
  // 32-bit floating point line segment.
  ///
  // This is equivalent to (PathSegmentType::LINE_TO | PathTag::F32_BIT).
  public static readonly LINE_TO_F32 = 0x9;

  // 32-bit floating point quadratic segment.
  ///
  // This is equivalent to (PathSegmentType::QUAD_TO | PathTag::F32_BIT).
  public static readonly QUAD_TO_F32 = 0xa;

  // 32-bit floating point cubic segment.
  ///
  // This is equivalent to (PathSegmentType::CUBIC_TO | PathTag::F32_BIT).
  public static readonly CUBIC_TO_F32 = 0xb;

  // 16-bit integral line segment.
  public static readonly LINE_TO_I16 = 0x1;

  // 16-bit integral quadratic segment.
  public static readonly QUAD_TO_I16 = 0x2;

  // 16-bit integral cubic segment.
  public static readonly CUBIC_TO_I16 = 0x3;

  // Transform marker.
  public static readonly TRANSFORM = 0x20;

  // Path marker.
  public static readonly PATH = 0x10;

  // Line width setting.
  public static readonly LINEWIDTH = 0x40;

  // Bit for path segments that are represented as f32 values. If unset
  // they are represented as i16.
  public static readonly F32_BIT = 0x8;

  // Bit that marks a segment that is the end of a subpath.
  public static readonly SUBPATH_END_BIT = 0x4;

  // Mask for bottom 3 bits that contain the [PathSegmentType].
  public static readonly SEGMENT_MASK = 0x3;

  // Returns true if the tag is a segment.
  public static is_path_segment( pathTag: U8 ): boolean {
    return PathTag.path_segment_type( pathTag ) !== 0;
  }

  // Returns true if this is a 32-bit floating point segment.
  public static is_f32( pathTag: U8 ): boolean {
    return ( pathTag & PathTag.F32_BIT ) !== 0;
  }

  // Returns true if this segment ends a subpath.
  public static is_subpath_end( pathTag: U8 ): boolean {
    return ( pathTag & PathTag.SUBPATH_END_BIT ) !== 0;
  }

  // Sets the subpath end bit.
  public static with_subpath_end( pathTag: U8 ): U8 {
    return pathTag | PathTag.SUBPATH_END_BIT;
  }

  // Returns the segment type.
  public static path_segment_type( pathTag: U8 ): U8 {
    return pathTag & PathTag.SEGMENT_MASK;
  }
}

export class Layout {

  public n_draw_objects: U32 = 0; // Number of draw objects.
  public n_paths: U32 = 0; // Number of paths.
  public n_clips: U32 = 0; // Number of clips.
  public bin_data_start: U32 = 0; // Start of binning data.
  public path_tag_base: U32 = 0; // Start of path tag stream.
  public path_data_base: U32 = 0; // Start of path data stream.
  public draw_tag_base: U32 = 0; // Start of draw tag stream.
  public draw_data_base: U32 = 0; // Start of draw data stream.
  public transform_base: U32 = 0; // Start of transform stream.
  public linewidth_base: U32 = 0; // Start of linewidth stream.

  public path_tags_size(): number {
    const start = this.path_tag_base * 4;
    const end = this.path_data_base * 4;
    return end - start;
  }
}

export class SceneBufferSizes {

  // TODO: perhaps pooling objects like these?
  public readonly path_tag_padded: number;
  public readonly buffer_size: number;

  public constructor( encoding: Encoding ) {
    const n_path_tags = encoding.pathTagsBuf.byteLength + encoding.n_open_clips;

    // Padded length of the path tag stream in bytes.
    this.path_tag_padded = align_up( n_path_tags, 4 * PATH_REDUCE_WG );

    // Full size of the scene buffer in bytes.
    this.buffer_size = this.path_tag_padded
      + encoding.pathDataBuf.byteLength // u8
      + encoding.drawTagsBuf.byteLength + encoding.n_open_clips * 4 // u32 in rust
      + encoding.drawDataBuf.byteLength // u8
      + encoding.transforms.length * 6 * 4 // 6xf32
      + encoding.linewidths.length * 4; // f32

    // NOTE: because of not using the glyphs feature, our patch_sizes are effectively zero
  }
}

// Uniform render configuration data used by all GPU stages.
///
// This data structure must be kept in sync with the definition in
// shaders/shared/config.wgsl.
export class ConfigUniform {
  // Width of the scene in tiles.
  public width_in_tiles = 0;
  // Height of the scene in tiles.
  public height_in_tiles = 0;
  // Width of the target in pixels.
  public target_width = 0;
  // Height of the target in pixels.
  public target_height = 0;
  // The base background color applied to the target before any blends.
  public base_color: ColorRGBA32 = 0;
  // Layout of packed scene data.
  public layout: Layout;
  // Size of binning buffer allocation (in u32s).
  public binning_size = 0;
  // Size of tile buffer allocation (in Tiles).
  public tiles_size = 0;
  // Size of segment buffer allocation (in PathSegments).
  public segments_size = 0;
  // Size of per-tile command list buffer allocation (in u32s).
  public ptcl_size = 0;

  public constructor( layout: Layout ) {
    this.layout = layout;
  }

  public to_typed_array(): Uint8Array {
    const buf = new ByteBuffer( CONFIG_UNIFORM_BYTES );

    buf.pushU32( this.width_in_tiles );
    buf.pushU32( this.height_in_tiles );
    buf.pushU32( this.target_width );
    buf.pushU32( this.target_height );
    buf.pushU32( this.base_color );

    // Layout
    buf.pushU32( this.layout.n_draw_objects );
    buf.pushU32( this.layout.n_paths );
    buf.pushU32( this.layout.n_clips );
    buf.pushU32( this.layout.bin_data_start );
    buf.pushU32( this.layout.path_tag_base );
    buf.pushU32( this.layout.path_data_base );
    buf.pushU32( this.layout.draw_tag_base );
    buf.pushU32( this.layout.draw_data_base );
    buf.pushU32( this.layout.transform_base );
    buf.pushU32( this.layout.linewidth_base );

    buf.pushU32( this.binning_size );
    buf.pushU32( this.tiles_size );
    buf.pushU32( this.segments_size );
    buf.pushU32( this.ptcl_size );

    return buf.u8Array;
  }
}

export class WorkgroupCounts {

  // TODO: pooling
  public use_large_path_scan: boolean;
  public path_reduce: WorkgroupSize;
  public path_reduce2: WorkgroupSize;
  public path_scan1: WorkgroupSize;
  public path_scan: WorkgroupSize;
  public bbox_clear: WorkgroupSize;
  public path_seg: WorkgroupSize;
  public draw_reduce: WorkgroupSize;
  public draw_leaf: WorkgroupSize;
  public clip_reduce: WorkgroupSize;
  public clip_leaf: WorkgroupSize;
  public binning: WorkgroupSize;
  public tile_alloc: WorkgroupSize;
  public path_coarse: WorkgroupSize;
  public backdrop: WorkgroupSize;
  public coarse: WorkgroupSize;
  public fine: WorkgroupSize;

  public constructor( layout: Layout, width_in_tiles: number, height_in_tiles: number, n_path_tags: number ) {

    const n_paths = layout.n_paths;
    const n_draw_objects = layout.n_draw_objects;
    const n_clips = layout.n_clips;
    const path_tag_padded = align_up( n_path_tags, 4 * PATH_REDUCE_WG );
    const path_tag_wgs = Math.floor( path_tag_padded / ( 4 * PATH_REDUCE_WG ) );
    const use_large_path_scan = path_tag_wgs > PATH_REDUCE_WG;
    const reduced_size = use_large_path_scan ? align_up( path_tag_wgs, PATH_REDUCE_WG ) : path_tag_wgs;
    const draw_object_wgs = Math.floor( ( n_draw_objects + PATH_BBOX_WG - 1 ) / PATH_BBOX_WG );
    const path_coarse_wgs = Math.floor( ( n_path_tags + PATH_COARSE_WG - 1 ) / PATH_COARSE_WG );
    const clip_reduce_wgs = Math.floor( Math.max( 0, n_clips - 1 ) / CLIP_REDUCE_WG );
    const clip_wgs = Math.floor( ( n_clips + CLIP_REDUCE_WG - 1 ) / CLIP_REDUCE_WG );
    const path_wgs = Math.floor( ( n_paths + PATH_BBOX_WG - 1 ) / PATH_BBOX_WG );
    const width_in_bins = Math.floor( ( width_in_tiles + 15 ) / 16 );
    const height_in_bins = Math.floor( ( height_in_tiles + 15 ) / 16 );

    this.use_large_path_scan = use_large_path_scan;
    this.path_reduce = new WorkgroupSize( path_tag_wgs, 1, 1 );
    this.path_reduce2 = new WorkgroupSize( PATH_REDUCE_WG, 1, 1 );
    this.path_scan1 = new WorkgroupSize( Math.floor( reduced_size / PATH_REDUCE_WG ), 1, 1 );
    this.path_scan = new WorkgroupSize( path_tag_wgs, 1, 1 );
    this.bbox_clear = new WorkgroupSize( draw_object_wgs, 1, 1 );
    this.path_seg = new WorkgroupSize( path_coarse_wgs, 1, 1 );
    this.draw_reduce = new WorkgroupSize( draw_object_wgs, 1, 1 );
    this.draw_leaf = new WorkgroupSize( draw_object_wgs, 1, 1 );
    this.clip_reduce = new WorkgroupSize( clip_reduce_wgs, 1, 1 );
    this.clip_leaf = new WorkgroupSize( clip_wgs, 1, 1 );
    this.binning = new WorkgroupSize( draw_object_wgs, 1, 1 );
    this.tile_alloc = new WorkgroupSize( path_wgs, 1, 1 );
    this.path_coarse = new WorkgroupSize( path_coarse_wgs, 1, 1 );
    this.backdrop = new WorkgroupSize( path_wgs, 1, 1 );
    this.coarse = new WorkgroupSize( width_in_bins, height_in_bins, 1 );
    this.fine = new WorkgroupSize( width_in_tiles, height_in_tiles, 1 );
  }
}

export class BufferSize {
  public constructor( public length: number, public bytes_per_element: number ) {}

  // Creates a new buffer size from size in bytes (u32)
  public static from_size_in_bytes( size: number, bytes_per_element: number ): BufferSize {
    return new BufferSize( size / bytes_per_element, bytes_per_element );
  }

  // Returns the number of elements.
  public len(): number {
    return this.length;
  }

  // Returns the size in bytes.
  public size_in_bytes(): number {
    return this.bytes_per_element * this.length;
  }

  // Returns the size in bytes aligned up to the given value.
  public aligned_in_bytes( alignment: number ): number {
    return align_up( this.size_in_bytes(), alignment );
  }
}

export class BufferSizes {
  // TODO: this is a GREAT place to view documentation, go to each thing!
  // // Known size buffers
  // pub path_reduced: BufferSize<PathMonoid>,
  // pub path_reduced2: BufferSize<PathMonoid>,
  // pub path_reduced_scan: BufferSize<PathMonoid>,
  // pub path_monoids: BufferSize<PathMonoid>,
  // pub path_bboxes: BufferSize<PathBbox>,
  // pub cubics: BufferSize<Cubic>,
  // pub draw_reduced: BufferSize<DrawMonoid>,
  // pub draw_monoids: BufferSize<DrawMonoid>,
  // pub info: BufferSize<u32>,
  // pub clip_inps: BufferSize<Clip>,
  // pub clip_els: BufferSize<ClipElement>,
  // pub clip_bics: BufferSize<ClipBic>,
  // pub clip_bboxes: BufferSize<ClipBbox>,
  // pub draw_bboxes: BufferSize<DrawBbox>,
  // pub bump_alloc: BufferSize<BumpAllocators>, // 6x u32
  // pub bin_headers: BufferSize<BinHeader>,
  // pub paths: BufferSize<Path>,
  // // Bump allocated buffers
  // pub bin_data: BufferSize<u32>,
  // pub tiles: BufferSize<Tile>,
  // pub segments: BufferSize<PathSegment>,
  // pub ptcl: BufferSize<u32>,

  public path_reduced: BufferSize;
  public path_reduced2: BufferSize;
  public path_reduced_scan: BufferSize;
  public path_monoids: BufferSize;
  public path_bboxes: BufferSize;
  public cubics: BufferSize;
  public draw_reduced: BufferSize;
  public draw_monoids: BufferSize;
  public info: BufferSize;
  public clip_inps: BufferSize;
  public clip_els: BufferSize;
  public clip_bics: BufferSize;
  public clip_bboxes: BufferSize;
  public draw_bboxes: BufferSize;
  public bump_alloc: BufferSize;
  public bin_headers: BufferSize;
  public paths: BufferSize;

  // The following buffer sizes have been hand picked to accommodate the vello test scenes as
  // well as paris-30k. These should instead get derived from the scene layout using
  // reasonable heuristics.
  // TODO: derive from scene layout
  public bin_data: BufferSize;
  public tiles: BufferSize;
  public segments: BufferSize;
  public ptcl: BufferSize;

  // layout: &Layout, workgroups: &WorkgroupCounts, n_path_tags: u32
  public constructor( layout: Layout, workgroups: WorkgroupCounts, n_path_tags: number ) {

    const n_paths = layout.n_paths;
    const n_draw_objects = layout.n_draw_objects;
    const n_clips = layout.n_clips;
    const path_tag_wgs = workgroups.path_reduce.x;
    const reduced_size = workgroups.use_large_path_scan ? align_up( path_tag_wgs, PATH_REDUCE_WG ) : path_tag_wgs;
    this.path_reduced = new BufferSize( reduced_size, PATH_MONOID_BYTES );
    this.path_reduced2 = new BufferSize( PATH_REDUCE_WG, PATH_MONOID_BYTES );
    this.path_reduced_scan = new BufferSize( path_tag_wgs, PATH_MONOID_BYTES );
    this.path_monoids = new BufferSize( path_tag_wgs * PATH_REDUCE_WG, PATH_MONOID_BYTES );
    this.path_bboxes = new BufferSize( n_paths, PATH_BBOX_BYTES );
    this.cubics = new BufferSize( n_path_tags, CUBIC_BYTES );
    const draw_object_wgs = workgroups.draw_reduce.x;
    this.draw_reduced = new BufferSize( draw_object_wgs, DRAW_MONOID_BYTES );
    this.draw_monoids = new BufferSize( n_draw_objects, DRAW_MONOID_BYTES );
    this.info = new BufferSize( layout.bin_data_start, 4 );
    this.clip_inps = new BufferSize( n_clips, CLIP_BYTES );
    this.clip_els = new BufferSize( n_clips, CLIP_ELEMENT_BYTES );
    this.clip_bics = new BufferSize( Math.floor( n_clips / CLIP_REDUCE_WG ), CLIP_BIC_BYTES );
    this.clip_bboxes = new BufferSize( n_clips, CLIP_BBOX_BYTES );
    this.draw_bboxes = new BufferSize( n_paths, DRAW_BBOX_BYTES );
    this.bump_alloc = new BufferSize( 1, BUMP_ALLOCATORS_BYTES );
    this.bin_headers = new BufferSize( draw_object_wgs * 256, BIN_HEADER_BYTES );
    const n_paths_aligned = align_up( n_paths, 256 );
    this.paths = new BufferSize( n_paths_aligned, PATH_BYTES );

    // The following buffer sizes have been hand picked to accommodate the vello test scenes as
    // well as paris-30k. These should instead get derived from the scene layout using
    // reasonable heuristics.
    // TODO: derive from scene layout
    this.bin_data = new BufferSize( ( 1 << 18 ) >>> 0, 4 );
    this.tiles = new BufferSize( ( 1 << 21 ) >>> 0, TILE_BYTES );
    this.segments = new BufferSize( ( 1 << 21 ) >>> 0, PATH_SEGMENT_BYTES );
    this.ptcl = new BufferSize( ( 1 << 23 ) >>> 0, 4 );
  }
}

export class RenderConfig {

  // Workgroup counts for all compute pipelines.
  public workgroup_counts: WorkgroupCounts;

  // Sizes of all buffer resources.
  public buffer_sizes: BufferSizes;

  // TODO: rename
  public gpu: ConfigUniform;

  public config_bytes: Uint8Array;

  public constructor( layout: Layout, public width: number, public height: number, public base_color: ColorRGBA32 ) {
    const configUniform = new ConfigUniform( layout );

    const new_width = next_multiple_of( width, TILE_WIDTH );
    const new_height = next_multiple_of( height, TILE_HEIGHT );
    const width_in_tiles = new_width / TILE_WIDTH;
    const height_in_tiles = new_height / TILE_HEIGHT;
    const n_path_tags = layout.path_tags_size();
    const workgroup_counts = new WorkgroupCounts( layout, width_in_tiles, height_in_tiles, n_path_tags );
    const buffer_sizes = new BufferSizes( layout, workgroup_counts, n_path_tags );

    this.workgroup_counts = workgroup_counts;
    this.buffer_sizes = buffer_sizes;

    configUniform.width_in_tiles = width_in_tiles;
    configUniform.height_in_tiles = height_in_tiles;
    configUniform.target_width = width;
    configUniform.target_height = height;
    configUniform.base_color = to_premul_u32( base_color );
    configUniform.binning_size = buffer_sizes.bin_data.len() - layout.bin_data_start;
    configUniform.tiles_size = buffer_sizes.tiles.len();
    configUniform.segments_size = buffer_sizes.segments.len();
    configUniform.ptcl_size = buffer_sizes.ptcl.len();

    this.gpu = configUniform;

    this.config_bytes = this.gpu.to_typed_array();
  }
}

export const u8ToBase64 = ( u8array: Uint8Array ): string => {
  let string = '';

  for ( let i = 0; i < u8array.byteLength; i++ ) {
    string += String.fromCharCode( u8array[ i ] );
  }

  return window.btoa( string );
};

export const base64ToU8 = ( base64: string ): Uint8Array => {
  const string = window.atob( base64 );

  const bytes = new Uint8Array( string.length );
  for ( let i = 0; i < string.length; i++ ) {
    bytes[ i ] = string.charCodeAt( i );
  }

  return bytes;
};

export class RenderInfo {

  // generated with prepareRender
  public renderConfig: RenderConfig | null = null;

  public constructor( public packed: Uint8Array, public layout: Layout ) {}

  public prepareRender( width: number, height: number, base_color: ColorRGBA32 ): void {
    this.renderConfig = new RenderConfig( this.layout, width, height, base_color );
  }
}

export class VelloPatch {
  public constructor( public draw_data_offset: number ) {}
}

export class VelloImagePatch extends VelloPatch {

  public readonly type = 'image' as const;

  // Filled in by Atlas
  public atlasSubImage: AtlasSubImage | null = null;

  public constructor( draw_data_offset: number, public image: EncodableImage ) {
    super( draw_data_offset );
  }

  public withOffset( draw_data_offset: number ): VelloImagePatch {
    return new VelloImagePatch( draw_data_offset, this.image );
  }
}

export class VelloRampPatch extends VelloPatch {

  public readonly type = 'ramp' as const;

  // Filled in by Ramps
  public id = -1;

  public constructor( draw_data_offset: number, public stops: VelloColorStop[], public extend: Extend ) {
    super( draw_data_offset );
  }

  public withOffset( draw_data_offset: number ): VelloRampPatch {
    return new VelloRampPatch( draw_data_offset, this.stops, this.extend );
  }
}

const rustF32 = ( f: number ): string => {
  let str = '' + f;
  if ( !str.includes( '.' ) ) {
    str += '.0';
  }
  return str;
};
const rustTransform = ( t: Affine ): string => {
  return `Transform { matrix: [${rustF32( t.a00 )}, ${rustF32( t.a10 )}, ${rustF32( t.a01 )}, ${rustF32( t.a11 )}], translation: [${rustF32( t.a02 )}, ${rustF32( t.a12 )}] }`;
};
const rustDrawColor = ( color: ColorRGBA32 ): string => {
  return `DrawColor {rgba: 0x${( ( color & 0xffffffff ) >>> 0 ).toString( 16 )}}`;
};
const rustColorStops = ( stops: VelloColorStop[] ): string => {
  return `[${stops.map( stop => {
    return `ColorStop {offset: ${rustF32( stop.offset )}, color: peniko::Color {r: ${( ( stop.color >>> 24 ) & 0xff ).toString( 16 )}, g: ${( ( stop.color >>> 16 ) & 0xff ).toString( 16 )}, b: ${( ( stop.color >>> 8 ) & 0xff ).toString( 16 )}, a: ${( ( stop.color >>> 0 ) & 0xff ).toString( 16 )}}}`;
  } ).join( ', ' )}]`;
};

export default class Encoding {

  public id: number; // For things like output of drawing commands for validation

  public pathTagsBuf = new ByteBuffer(); // path_tags
  public pathDataBuf = new ByteBuffer(); // path_data
  public drawTagsBuf = new ByteBuffer(); // draw_tags // NOTE: was u32 array (effectively) in rust
  public drawDataBuf = new ByteBuffer(); // draw_data
  public transforms: Affine[] = []; // Vec<Transform> in rust, Affine[] in js
  public linewidths: number[] = []; // Vec<f32> in rust, number[] in js
  public n_paths = 0; // u32
  public n_path_segments = 0; // u32,
  public n_clips = 0; // u32
  public n_open_clips = 0; // u32
  public patches: ( VelloImagePatch | VelloRampPatch )[] = []; // Vec<Patch> in rust, Patch[] in js
  public color_stops: VelloColorStop[] = []; // Vec<ColorStop> in rust, VelloColorStop[] in js

  // Embedded PathEncoder
  public first_point: Vector2 = new Vector2( 0, 0 ); // mutated
  public state: ( 0x1 | 0x2 | 0x3 ) = Encoding.PATH_START;
  public n_encoded_segments = 0;
  public is_fill = true;

  // State
  public static readonly PATH_START = 0x1;
  public static readonly PATH_MOVE_TO = 0x2;
  public static readonly PATH_NONEMPTY_SUBPATH = 0x3;

  // For debugging, so we can dump rust commands out to execute
  public rustEncoding!: string;
  public rustLock!: number;

  public constructor() {
    this.id = globalEncodingID++;
    if ( sceneryLog && sceneryLog.Encoding ) {
      this.rustEncoding = '';
      this.rustLock = 0;
    }
    sceneryLog && sceneryLog.Encoding && this.rustLock === 0 && ( this.rustEncoding += `let mut encoding${this.id}: Encoding = Encoding::new();\n` );
  }

  public is_empty(): boolean {
    return this.pathTagsBuf.byteLength === 0;
  }

  // Clears the encoding.
  public reset( is_fragment: boolean ): void {
    sceneryLog && sceneryLog.Encoding && this.rustLock === 0 && ( this.rustEncoding += `encoding${this.id}.reset(${is_fragment});\n` );
    this.transforms.length = 0;
    this.pathTagsBuf.clear();
    this.pathDataBuf.clear();
    this.linewidths.length = 0;
    this.drawDataBuf.clear();
    this.drawTagsBuf.clear();
    this.n_paths = 0;
    this.n_path_segments = 0;
    this.n_clips = 0;
    this.n_open_clips = 0;
    this.patches.length = 0;
    this.color_stops.length = 0;
    if ( !is_fragment ) {
      this.transforms.push( Affine.IDENTITY );
      this.linewidths.push( -1.0 );
    }
  }

  // Appends another encoding to this one with an optional transform.
  public append( other: Encoding, transform: Affine | null = null ): void {
    const initial_draw_data_length = this.drawDataBuf.byteLength;

    this.pathTagsBuf.pushByteBuffer( other.pathTagsBuf );
    this.pathDataBuf.pushByteBuffer( other.pathDataBuf );
    this.drawTagsBuf.pushByteBuffer( other.drawTagsBuf );
    this.drawDataBuf.pushByteBuffer( other.drawDataBuf );
    this.n_paths += other.n_paths;
    this.n_path_segments += other.n_path_segments;
    this.n_clips += other.n_clips;
    this.n_open_clips += other.n_open_clips;
    if ( transform ) {
      this.transforms.push( ...other.transforms.map( t => transform.times( t ) ) );
    }
    else {
      this.transforms.push( ...other.transforms );
    }
    this.linewidths.push( ...other.linewidths );
    this.color_stops.push( ...other.color_stops );
    this.patches.push( ...other.patches.map( patch => patch.withOffset( patch.draw_data_offset + initial_draw_data_length ) ) );

    if ( sceneryLog && sceneryLog.Encoding && this.rustLock === 0 ) {
      if ( !this.rustEncoding?.includes( `let mut encoding${other.id} ` ) ) {
        this.rustEncoding = other.rustEncoding + this.rustEncoding;
      }
      this.rustEncoding += `encoding${this.id}.append(&mut encoding${other.id}, ${transform ? `&Some(${rustTransform( transform )})` : '&None'});\n`;
    }
  }

  // Encodes a linewidth.
  public encode_linewidth( linewidth: number ): void {
    if ( this.linewidths[ this.linewidths.length - 1 ] !== linewidth ) {
      sceneryLog && sceneryLog.Encoding && this.rustLock === 0 && ( this.rustEncoding += `encoding${this.id}.encode_linewidth(${rustF32( linewidth )});\n` );
      this.pathTagsBuf.pushU8( PathTag.LINEWIDTH );
      this.linewidths.push( linewidth );
    }
  }

  // Encodes a transform.
  ///
  // If the given transform is different from the current one, encodes it and
  // returns true. Otherwise, encodes nothing and returns false.
  public encode_transform( transform: Affine ): boolean {
    const last = this.transforms[ this.transforms.length - 1 ];
    if ( !last || !last.equals( transform ) ) {
      sceneryLog && sceneryLog.Encoding && this.rustLock === 0 && ( this.rustEncoding += `encoding${this.id}.encode_transform(${rustTransform( transform )});\n` );
      this.pathTagsBuf.pushU8( PathTag.TRANSFORM );
      this.transforms.push( transform );
      return true;
    }
    else {
      return false;
    }
  }

  // If `is_fill` is true, all subpaths will
  // be automatically closed.
  public encode_path( is_fill: boolean ): void {
    this.first_point.x = 0;
    this.first_point.y = 0;
    this.state = Encoding.PATH_START;
    this.n_encoded_segments = 0;
    this.is_fill = is_fill;

    if ( sceneryLog && sceneryLog.Encoding && this.rustLock === 0 ) {
      globalPathID++;

      this.rustEncoding += `let mut path_encoder${globalPathID} = encoding${this.id}.encode_path(${is_fill});\n`;
    }
  }


  // Encodes a move, starting a new subpath.
  public move_to( x: number, y: number ): void {
    sceneryLog && sceneryLog.Encoding && this.rustLock === 0 && ( this.rustEncoding += `path_encoder${globalPathID}.move_to(${rustF32( x )}, ${rustF32( y )});\n` );
    sceneryLog && sceneryLog.Encoding && this.rustLock++;

    if ( this.is_fill ) {
      this.close();
    }
    this.first_point.x = x;
    this.first_point.y = y;
    if ( this.state === Encoding.PATH_MOVE_TO ) {
      this.pathDataBuf.byteLength -= 8;
    }
    else if ( this.state === Encoding.PATH_NONEMPTY_SUBPATH ) {
      this.setSubpathEndTag();
    }
    this.pathDataBuf.pushF32( x );
    this.pathDataBuf.pushF32( y );
    this.state = Encoding.PATH_MOVE_TO;

    sceneryLog && sceneryLog.Encoding && this.rustLock--;
  }

  // Encodes a line.
  public line_to( x: number, y: number ): void {
    sceneryLog && sceneryLog.Encoding && this.rustLock === 0 && ( this.rustEncoding += `path_encoder${globalPathID}.line_to(${rustF32( x )}, ${rustF32( y )});\n` );
    sceneryLog && sceneryLog.Encoding && this.rustLock++;

    if ( this.state === Encoding.PATH_START ) {
      if ( this.n_encoded_segments === 0 ) {
        // This copies the behavior of kurbo which treats an initial line, quad
        // or curve as a move.
        this.move_to( x, y );
        sceneryLog && sceneryLog.Encoding && this.rustLock--;
        return;
      }
      this.move_to( this.first_point.x, this.first_point.y );
    }
    this.pathDataBuf.pushF32( x );
    this.pathDataBuf.pushF32( y );
    this.pathTagsBuf.pushU8( PathTag.LINE_TO_F32 );
    this.state = Encoding.PATH_NONEMPTY_SUBPATH;
    this.n_encoded_segments += 1;

    sceneryLog && sceneryLog.Encoding && this.rustLock--;
  }

  // Encodes a quadratic bezier.
  public quad_to( x1: number, y1: number, x2: number, y2: number ): void {
    sceneryLog && sceneryLog.Encoding && this.rustLock === 0 && ( this.rustEncoding += `path_encoder${globalPathID}.quad_to(${rustF32( x1 )}, ${rustF32( y1 )}, ${rustF32( x2 )}, ${rustF32( y2 )});\n` );
    sceneryLog && sceneryLog.Encoding && this.rustLock++;

    if ( this.state === Encoding.PATH_START ) {
      if ( this.n_encoded_segments === 0 ) {
        this.move_to( x2, y2 );
        sceneryLog && sceneryLog.Encoding && this.rustLock--;
        return;
      }
      this.move_to( this.first_point.x, this.first_point.y );
    }
    this.pathDataBuf.pushF32( x1 );
    this.pathDataBuf.pushF32( y1 );
    this.pathDataBuf.pushF32( x2 );
    this.pathDataBuf.pushF32( y2 );
    this.pathTagsBuf.pushU8( PathTag.QUAD_TO_F32 );
    this.state = Encoding.PATH_NONEMPTY_SUBPATH;
    this.n_encoded_segments += 1;

    sceneryLog && sceneryLog.Encoding && this.rustLock--;
  }

  // Encodes a cubic bezier.
  public cubic_to( x1: number, y1: number, x2: number, y2: number, x3: number, y3: number ): void {
    sceneryLog && sceneryLog.Encoding && this.rustLock === 0 && ( this.rustEncoding += `path_encoder${globalPathID}.cubic_to(${rustF32( x1 )}, ${rustF32( y1 )}, ${rustF32( x2 )}, ${rustF32( y2 )}, ${rustF32( x3 )}, ${rustF32( y3 )});\n` );
    sceneryLog && sceneryLog.Encoding && this.rustLock++;

    if ( this.state === Encoding.PATH_START ) {
      if ( this.n_encoded_segments === 0 ) {
        this.move_to( x3, y3 );
        sceneryLog && sceneryLog.Encoding && this.rustLock--;
        return;
      }
      this.move_to( this.first_point.x, this.first_point.y );
    }
    this.pathDataBuf.pushF32( x1 );
    this.pathDataBuf.pushF32( y1 );
    this.pathDataBuf.pushF32( x2 );
    this.pathDataBuf.pushF32( y2 );
    this.pathDataBuf.pushF32( x3 );
    this.pathDataBuf.pushF32( y3 );
    this.pathTagsBuf.pushU8( PathTag.CUBIC_TO_F32 );
    this.state = Encoding.PATH_NONEMPTY_SUBPATH;
    this.n_encoded_segments += 1;

    sceneryLog && sceneryLog.Encoding && this.rustLock--;
  }

  // Closes the current subpath.
  public close(): void {
    sceneryLog && sceneryLog.Encoding && this.rustLock === 0 && ( this.rustEncoding += `path_encoder${globalPathID}.close();\n` );

    if ( this.state === Encoding.PATH_START ) {
      return;
    }
    else if ( this.state === Encoding.PATH_MOVE_TO ) {
      this.pathDataBuf.byteLength -= 8;
      this.state = Encoding.PATH_START;
      return;
    }
    const len = this.pathDataBuf.byteLength / 4;
    if ( len < 8 ) {
      // can't happen
      return;
    }
    const lastX = this.pathDataBuf.fullF32Array[ len - 2 ];
    const lastY = this.pathDataBuf.fullF32Array[ len - 1 ];
    if ( Math.abs( lastX - this.first_point.x ) > 1e-8 || Math.abs( lastY - this.first_point.y ) > 1e-8 ) {
      this.pathDataBuf.pushF32( this.first_point.x );
      this.pathDataBuf.pushF32( this.first_point.y );
      this.pathTagsBuf.pushU8( PathTag.with_subpath_end( PathTag.LINE_TO_F32 ) );
      this.n_encoded_segments += 1;
    }
    else {
      this.setSubpathEndTag();
    }
    this.state = Encoding.PATH_START;
  }

  // Completes path encoding and returns the actual number of encoded segments.
  ///
  // If `insert_path_marker` is true, encodes the [PathTag::PATH] tag to signify
  // the end of a complete path object. Setting this to false allows encoding
  // multiple paths with differing transforms for a single draw object.
  public finish( insert_path_marker: boolean ): number {
    sceneryLog && sceneryLog.Encoding && this.rustLock === 0 && ( this.rustEncoding += `path_encoder${globalPathID}.finish(${insert_path_marker});\n` );
    sceneryLog && sceneryLog.Encoding && this.rustLock++;

    if ( this.is_fill ) {
      this.close();
    }
    if ( this.state === Encoding.PATH_MOVE_TO ) {
      this.pathDataBuf.byteLength -= 8;
    }
    if ( this.n_encoded_segments !== 0 ) {
      this.setSubpathEndTag();
      this.n_path_segments += this.n_encoded_segments;
      if ( insert_path_marker ) {
        this.insert_path_marker();
      }
    }

    sceneryLog && sceneryLog.Encoding && this.rustLock--;

    return this.n_encoded_segments;
  }

  public setSubpathEndTag(): void {
    if ( this.pathTagsBuf.byteLength ) {
      // In-place replace, add the "subpath end" flag
      const lastIndex = this.pathTagsBuf.byteLength - 1;

      this.pathTagsBuf.fullU8Array[ lastIndex ] = PathTag.with_subpath_end( this.pathTagsBuf.fullU8Array[ lastIndex ] );
    }
  }

  // Exposed for glyph handling
  public insert_path_marker(): void {
    sceneryLog && sceneryLog.Encoding && this.rustLock === 0 && ( this.rustEncoding += `encoding${this.id}.path_tags.push(PathTag::PATH);\n*encoding${this.id}.n_paths += 1;\n` );
    this.pathTagsBuf.pushU8( PathTag.PATH );
    this.n_paths += 1;
  }

  // Encodes a solid color brush.
  public encode_color( color: ColorRGBA32 ): void {
    sceneryLog && sceneryLog.Encoding && this.rustLock === 0 && ( this.rustEncoding += `encoding${this.id}.encode_color(${rustDrawColor( color )});\n` );

    this.drawTagsBuf.pushU32( DrawTag.COLOR );
    this.drawDataBuf.pushU32( to_premul_u32( color ) );
  }

  // zero: => false, one => color, many => true (icky)
  private add_ramp( color_stops: VelloColorStop[], alpha: number, extend: Extend ): null | true | ColorRGBA32 {
    const offset = this.drawDataBuf.byteLength;
    const stops_start = this.color_stops.length;
    if ( alpha !== 1 ) {
      this.color_stops.push( ...color_stops.map( stop => new VelloColorStop( stop.offset, with_alpha_factor( stop.color, alpha ) ) ) );
    }
    else {
      this.color_stops.push( ...color_stops );
    }
    const stops_end = this.color_stops.length;

    const stopCount = stops_end - stops_start;

    if ( stopCount === 0 ) {
      return null;
    }
    else if ( stopCount === 1 ) {
      assert && assert( this.color_stops.length );

      return this.color_stops.pop()!.color;
    }
    else {
      this.patches.push( new VelloRampPatch( offset, color_stops, extend ) );
      return true;
    }
  }

  // Encodes a linear gradient brush.
  public encode_linear_gradient( x0: number, y0: number, x1: number, y1: number, color_stops: VelloColorStop[], alpha: number, extend: Extend ): void {
    sceneryLog && sceneryLog.Encoding && this.rustLock === 0 && ( this.rustEncoding += `encoding${this.id}.encode_linear_gradient(DrawLinearGradient {index: 0, p0: [${rustF32( x0 )}, ${rustF32( y0 )}], p1: [${rustF32( x1 )}, ${rustF32( y1 )}]}, ${rustColorStops( color_stops )}, ${rustF32( alpha )}, ${extend});\n` );
    sceneryLog && sceneryLog.Encoding && this.rustLock++;

    const result = this.add_ramp( color_stops, alpha, extend );
    if ( result === null ) {
      this.encode_color( 0 );
    }
    else if ( result === true ) {
      this.drawTagsBuf.pushU32( DrawTag.LINEAR_GRADIENT );
      this.drawDataBuf.pushU32( 0 ); // ramp index, will get filled in
      this.drawDataBuf.pushF32( x0 );
      this.drawDataBuf.pushF32( y0 );
      this.drawDataBuf.pushF32( x1 );
      this.drawDataBuf.pushF32( y1 );
    }
    else {
      this.encode_color( result );
    }

    sceneryLog && sceneryLog.Encoding && this.rustLock--;
  }

  // TODO: note the parameter order?
  // Encodes a radial gradient brush.
  public encode_radial_gradient( x0: number, y0: number, r0: number, x1: number, y1: number, r1: number, color_stops: VelloColorStop[], alpha: number, extend: Extend ): void {
    sceneryLog && sceneryLog.Encoding && this.rustLock === 0 && ( this.rustEncoding += `encoding${this.id}.encode_radial_gradient(DrawRadialGradient {index: 0, p0: [${rustF32( x0 )}, ${rustF32( y0 )}], r0: ${rustF32( r0 )}, p1: [${rustF32( x1 )}, ${rustF32( y1 )}], r1: ${rustF32( r1 )}}, ${rustColorStops( color_stops )}, ${rustF32( alpha )}, ${extend});\n` );
    sceneryLog && sceneryLog.Encoding && this.rustLock++;

    // Match Skia's epsilon for radii comparison
    const SKIA_EPSILON = 1 / ( ( 1 << 12 ) >>> 0 );
    if ( x0 === x1 && y0 === y1 && Math.abs( r0 - r1 ) < SKIA_EPSILON ) {
      this.encode_color( 0 );
    }
    else {
      const result = this.add_ramp( color_stops, alpha, extend );
      if ( result === null ) {
        this.encode_color( 0 );
      }
      else if ( result === true ) {
        this.drawTagsBuf.pushU32( DrawTag.RADIAL_GRADIENT );
        this.drawDataBuf.pushU32( 0 ); // ramp index, will get filled in
        this.drawDataBuf.pushF32( x0 );
        this.drawDataBuf.pushF32( y0 );
        this.drawDataBuf.pushF32( x1 );
        this.drawDataBuf.pushF32( y1 );
        this.drawDataBuf.pushF32( r0 );
        this.drawDataBuf.pushF32( r1 );
      }
      else {
        this.encode_color( result );
      }
    }

    sceneryLog && sceneryLog.Encoding && this.rustLock--;
  }

  // Encodes an image brush.
  public encode_image( image: EncodableImage ): void {
    // TODO: sceneryLog.Encoding support!! (easy from BufferImage, hard from SourceImage)
    // TODO: see if it's premultiplied and we don't have to do that!
    this.patches.push( new VelloImagePatch( this.drawDataBuf.byteLength, image ) );
    this.drawTagsBuf.pushU32( DrawTag.IMAGE );

    // packed atlas coordinates (xy) u32
    this.drawDataBuf.pushU32( 0 );

    // Packed image dimensions. (width_height) u32
    this.drawDataBuf.pushU32( ( ( image.width << 16 ) >>> 0 ) | ( image.height & 0xFFFF ) );
  }

  // Encodes a begin clip command.
  public encode_begin_clip( mix: Mix, compose: Compose, alpha: number ): void {
    sceneryLog && sceneryLog.Encoding && this.rustLock === 0 && ( this.rustEncoding += `encoding${this.id}.encode_begin_clip(${mix}, ${compose}, ${rustF32( alpha )});\n` );
    this.drawTagsBuf.pushU32( DrawTag.BEGIN_CLIP );

    // u32 combination of mix and compose
    this.drawDataBuf.pushU32( ( ( mix << 8 ) >>> 0 ) | compose );
    this.drawDataBuf.pushF32( alpha );

    this.n_clips += 1;
    this.n_open_clips += 1;
  }

  // Encodes an end clip command.
  public encode_end_clip(): void {
    sceneryLog && sceneryLog.Encoding && this.rustLock === 0 && ( this.rustEncoding += `encoding${this.id}.encode_end_clip();\n` );
    if ( this.n_open_clips > 0 ) {
      this.drawTagsBuf.pushU32( DrawTag.END_CLIP );
      // This is a dummy path, and will go away with the new clip impl.
      this.pathTagsBuf.pushU8( PathTag.PATH );
      this.n_paths += 1;
      this.n_clips += 1;
      this.n_open_clips -= 1;
    }
  }

  // TODO: make this workaround not needed
  public finalize_scene(): void {
    this.encode_path( true );
    this.move_to( 0, 0 );
    this.line_to( 1, 0 );
    this.close();
    this.finish( true );
  }

  // To encode a kite shape, we'll need to split arcs/elliptical-arcs into bezier curves
  public encode_kite_shape( shape: Shape, isFill: boolean, insertPathMarker: boolean, tolerance: number ): number {
    this.encode_path( isFill );

    shape.subpaths.forEach( subpath => {
      if ( subpath.isDrawable() ) {
        const startPoint = subpath.getFirstSegment().start;
        this.move_to( startPoint.x, startPoint.y );

        subpath.segments.forEach( segment => {
          if ( segment instanceof Line ) {
            this.line_to( segment.end.x, segment.end.y );
          }
          else if ( segment instanceof Quadratic ) {
            this.quad_to( segment.control.x, segment.control.y, segment.end.x, segment.end.y );
          }
          else if ( segment instanceof Cubic ) {
            this.cubic_to( segment.control1.x, segment.control1.y, segment.control2.x, segment.control2.y, segment.end.x, segment.end.y );
          }
          else if ( segment instanceof Arc || segment instanceof EllipticalArc ) {
            // arc or elliptical arc, split with kurbo's setup (not the most optimal).
            // See https://raphlinus.github.io/curves/2021/03/11/bezier-fitting.html for better.

            const maxRadius = segment instanceof Arc ? segment.radius : Math.max( segment.radiusX, segment.radiusY );
            const scaled_err = maxRadius / tolerance;
            const n_err = Math.max( Math.pow( 1.1163 * scaled_err, 1 / 6 ), 3.999999 );

            // TODO: hacked with *4 for now, figure out how to better do this
            const n = Math.ceil( n_err * Math.abs( segment.getAngleDifference() ) * ( 1.0 / ( 2.0 * Math.PI ) ) ) * 4;

            // For now, evenly subdivide
            const segments = n > 1 ? segment.subdivisions( _.range( 1, n ).map( t => t / n ) ) : [ segment ];

            // Create cubics approximations for each segment
            // TODO: performance optimize?
            segments.forEach( subSegment => {
              const start = subSegment.start;
              const middle = subSegment.positionAt( 0.5 );
              const end = subSegment.end;

              // 1/4 start, 1/4 end, 1/2 control, find the control point given the middle (t=0.5) point
              // average + 2 * ( middle - average ) => 2 * middle - average => 2 * middle - ( start + end ) / 2

              // const average = start.average( end );
              // const control = average.plus( middle.minus( average ).timesScalar( 2 ) );

              // mutates middle also
              const control = start.plus( end ).multiplyScalar( -0.5 ).add( middle.multiplyScalar( 2 ) );

              this.quad_to( control.x, control.y, end.x, end.y );
            } );
          }
        } );

        if ( subpath.closed ) {
          this.close();
        }
      }
    } );

    return this.finish( insertPathMarker );
  }

  public print_debug(): void {
    console.log( `path_tags\n${[ ...this.pathTagsBuf.u8Array ].map( x => x.toString() ).join( ', ' )}` );
    console.log( `path_data\n${[ ...this.pathDataBuf.u8Array ].map( x => x.toString() ).join( ', ' )}` );
    console.log( `draw_tags\n${[ ...this.drawTagsBuf.u8Array ].map( x => x.toString() ).join( ', ' )}` );
    console.log( `draw_data\n${[ ...this.drawDataBuf.u8Array ].map( x => x.toString() ).join( ', ' )}` );
    console.log( `transforms\n${this.transforms.map( x => `_ a00:${x.a00} a10:${x.a10} a01:${x.a01} a11:${x.a11} a02:${x.a02} a12:${x.a12}_` ).join( '\n' )}` );
    console.log( `linewidths\n${this.linewidths.map( x => x.toString() ).join( ', ' )}` );
    console.log( `n_paths\n${this.n_paths}` );
    console.log( `n_path_segments\n${this.n_path_segments}` );
    console.log( `n_clips\n${this.n_clips}` );
    console.log( `n_open_clips\n${this.n_open_clips}` );
  }

  // Resolves late bound resources and packs an encoding. Returns the packed
  // layout and computed ramp data.
  public resolve( deviceContext: DeviceContext ): RenderInfo {

    // @ts-expect-error Because it can't detect the filter result type
    deviceContext.ramps.updatePatches( this.patches.filter( patch => patch instanceof VelloRampPatch ) );
    // @ts-expect-error Because it can't detect the filter result type
    deviceContext.atlas.updatePatches( this.patches.filter( patch => patch instanceof VelloImagePatch ) );

    const layout = new Layout();
    layout.n_paths = this.n_paths;
    layout.n_clips = this.n_clips;

    const sceneBufferSizes = new SceneBufferSizes( this );
    const buffer_size = sceneBufferSizes.buffer_size;
    const path_tag_padded = sceneBufferSizes.path_tag_padded;

    const dataBuf = new ByteBuffer( sceneBufferSizes.buffer_size );

    // Path tag stream
    layout.path_tag_base = size_to_words( dataBuf.byteLength );
    dataBuf.pushByteBuffer( this.pathTagsBuf );
    // TODO: what if we... just error if there are open clips? Why are we padding the streams to make this work?
    for ( let i = 0; i < this.n_open_clips; i++ ) {
      dataBuf.pushU8( PathTag.PATH );
    }
    dataBuf.byteLength = path_tag_padded;

    // Path data stream
    layout.path_data_base = size_to_words( dataBuf.byteLength );
    dataBuf.pushByteBuffer( this.pathDataBuf );

    // Draw tag stream
    layout.draw_tag_base = size_to_words( dataBuf.byteLength );
    // Bin data follows draw info
    layout.bin_data_start = _.sum( this.drawTagsBuf.u32Array.map( DrawTag.info_size ) );
    dataBuf.pushByteBuffer( this.drawTagsBuf );
    for ( let i = 0; i < this.n_open_clips; i++ ) {
      dataBuf.pushU32( DrawTag.END_CLIP );
    }

    // Draw data stream
    layout.draw_data_base = size_to_words( dataBuf.byteLength );
    {
      const drawDataOffset = dataBuf.byteLength;
      dataBuf.pushByteBuffer( this.drawDataBuf );

      this.patches.forEach( patch => {
        const byteOffset = drawDataOffset + patch.draw_data_offset;
        let bytes;

        if ( patch instanceof VelloRampPatch ) {
          bytes = u32_to_bytes( ( ( patch.id << 2 ) >>> 0 ) | patch.extend );
        }
        else {
          assert && assert( patch.atlasSubImage );
          bytes = u32_to_bytes( ( patch.atlasSubImage!.x << 16 ) >>> 0 | patch.atlasSubImage!.y );
          // TODO: assume the image fit (if not, we'll need to do something else)
        }

        // Patch data directly into our full output
        dataBuf.fullU8Array.set( bytes, byteOffset );
      } );
    }

    // Transform stream
    // TODO: Float32Array instead of Affine?
    layout.transform_base = size_to_words( dataBuf.byteLength );
    for ( let i = 0; i < this.transforms.length; i++ ) {
      const transform = this.transforms[ i ];
      dataBuf.pushF32( transform.a00 );
      dataBuf.pushF32( transform.a10 );
      dataBuf.pushF32( transform.a01 );
      dataBuf.pushF32( transform.a11 );
      dataBuf.pushF32( transform.a02 );
      dataBuf.pushF32( transform.a12 );
    }

    // Linewidth stream
    layout.linewidth_base = size_to_words( dataBuf.byteLength );
    for ( let i = 0; i < this.linewidths.length; i++ ) {
      dataBuf.pushF32( this.linewidths[ i ] );
    }

    layout.n_draw_objects = layout.n_paths;

    if ( dataBuf.byteLength !== buffer_size ) {
      throw new Error( 'buffer size mismatch' );
    }

    return new RenderInfo( dataBuf.u8Array, layout );
  }
}

scenery.register( 'Encoding', Encoding );
