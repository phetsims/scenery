// Copyright 2023, University of Colorado Boulder

/**
 * An encoding of a Vello shader fragment. Encodings can be combined, and will be resolved to a packed strcuture
 * sent to the GPU.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Vector2 from '../../../../dot/js/Vector2.js';
import Utils from '../../../../dot/js/Utils.js';
import { Affine, AtlasSubImage, BufferImage, ByteBuffer, ColorMatrixFilter, DeviceContext, scenery, SourceImage, WorkgroupSize } from '../../imports.js';
import { Arc, Cubic, EllipticalArc, Line, Quadratic, Shape } from '../../../../kite/js/imports.js';
import Bounds2 from '../../../../dot/js/Bounds2.js';

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

const sizeToWords = ( byteSize: number ): number => byteSize / 4;

const alignUp = ( len: number, alignment: number ): number => {
  return len + ( ( ( ~len ) + 1 ) & ( alignment - 1 ) );
};

const nextMultipleOf = ( val: number, rhs: number ): number => {
  const r = val % rhs;
  return r === 0 ? val : val + ( rhs - r );
};

// Convert u32/f32 to 4 bytes in little endian order
const scratchForBytes = new Uint8Array( 4 );
export const f32ToBytes = ( float: number ): U8[] => {
  const view = new DataView( scratchForBytes.buffer );
  view.setFloat32( 0, float );
  return [ ...scratchForBytes ].reverse();
};
export const u32ToBytes = ( int: number ): U8[] => {
  const view = new DataView( scratchForBytes.buffer );
  view.setUint32( 0, int );
  return [ ...scratchForBytes ].reverse();
};

export const withAlphaFactor = ( color: ColorRGBA32, alpha: number ): ColorRGBA32 => {
  return ( color & 0xffffff00 ) | ( Utils.roundSymmetric( ( color & 0xff ) * alpha ) & 0xff );
};

export const premultiplyRGBA8 = ( rgba8color: ColorRGBA32 ): ColorRGBA32 => {
  const a = ( rgba8color & 0xff ) / 255;
  const r = Utils.roundSymmetric( ( ( rgba8color >>> 24 ) & 0xff ) * a >>> 0 );
  const g = Utils.roundSymmetric( ( ( rgba8color >>> 16 ) & 0xff ) * a >>> 0 );
  const b = Utils.roundSymmetric( ( ( rgba8color >>> 8 ) & 0xff ) * a >>> 0 );
  return ( ( r << 24 ) | ( g << 16 ) | ( b << 8 ) | ( rgba8color & 0xff ) ) >>> 0;
};

const lerpChannel = ( x: number, y: number, a: number ) => {
  return Utils.roundSymmetric( x * ( 1 - a ) + y * a >>> 0 );
};

export const lerpRGBA8 = ( c1: ColorRGBA32, c2: ColorRGBA32, t: number ): ColorRGBA32 => {
  const r = lerpChannel( ( c1 >>> 24 ) & 0xff, ( c2 >>> 24 ) & 0xff, t );
  const g = lerpChannel( ( c1 >>> 16 ) & 0xff, ( c2 >>> 16 ) & 0xff, t );
  const b = lerpChannel( ( c1 >>> 8 ) & 0xff, ( c2 >>> 8 ) & 0xff, t );
  const a = lerpChannel( c1 & 0xff, c2 & 0xff, t );
  return ( ( r << 24 ) | ( g << 16 ) | ( b << 8 ) | a ) >>> 0;
};

let globalEncodingID = 0;
let globalPathID = 0;

export class VelloColorStop {
  public constructor( public readonly offset: number, public readonly color: number ) {}
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

const ExtendMap = {
  [ Extend.Pad ]: 'Extend::Pad',
  [ Extend.Repeat ]: 'Extend::Repeat',
  [ Extend.Reflect ]: 'Extend::Reflect'
};

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

const MixMap = {
  [ Mix.Normal ]: 'Mix::Normal',
  [ Mix.Multiply ]: 'Mix::Multiply',
  [ Mix.Screen ]: 'Mix::Screen',
  [ Mix.Overlay ]: 'Mix::Overlay',
  [ Mix.Darken ]: 'Mix::Darken',
  [ Mix.Lighten ]: 'Mix::Lighten',
  [ Mix.ColorDodge ]: 'Mix::ColorDodge',
  [ Mix.ColorBurn ]: 'Mix::ColorBurn',
  [ Mix.HardLight ]: 'Mix::HardLight',
  [ Mix.SoftLight ]: 'Mix::SoftLight',
  [ Mix.Difference ]: 'Mix::Difference',
  [ Mix.Exclusion ]: 'Mix::Exclusion',
  [ Mix.Hue ]: 'Mix::Hue',
  [ Mix.Saturation ]: 'Mix::Saturation',
  [ Mix.Color ]: 'Mix::Color',
  [ Mix.Luminosity ]: 'Mix::Luminosity',
  [ Mix.Clip ]: 'Mix::Clip'
};

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

const ComposeMap = {
  [ Compose.Clear ]: 'Compose::Clear',
  [ Compose.Copy ]: 'Compose::Copy',
  [ Compose.Dest ]: 'Compose::Dest',
  [ Compose.SrcOver ]: 'Compose::SrcOver',
  [ Compose.DestOver ]: 'Compose::DestOver',
  [ Compose.SrcIn ]: 'Compose::SrcIn',
  [ Compose.DestIn ]: 'Compose::DestIn',
  [ Compose.SrcOut ]: 'Compose::SrcOut',
  [ Compose.DestOut ]: 'Compose::DestOut',
  [ Compose.SrcAtop ]: 'Compose::SrcAtop',
  [ Compose.DestAtop ]: 'Compose::DestAtop',
  [ Compose.Xor ]: 'Compose::Xor',
  [ Compose.Plus ]: 'Compose::Plus',
  [ Compose.PlusLighter ]: 'Compose::PlusLighter'
};

scenery.register( 'Compose', Compose );

export class FilterMatrix {
  public constructor(
    public m00 = 1, public m01 = 0, public m02 = 0, public m03 = 0, public m04 = 0,
    public m10 = 0, public m11 = 1, public m12 = 0, public m13 = 0, public m14 = 0,
    public m20 = 0, public m21 = 0, public m22 = 1, public m23 = 0, public m24 = 0,
    public m30 = 0, public m31 = 0, public m32 = 0, public m33 = 1, public m34 = 0
  ) {

    assert && assert( isFinite( m00 ), 'm00 should be a finite number' );
    assert && assert( isFinite( m01 ), 'm01 should be a finite number' );
    assert && assert( isFinite( m02 ), 'm02 should be a finite number' );
    assert && assert( isFinite( m03 ), 'm03 should be a finite number' );
    assert && assert( isFinite( m04 ), 'm04 should be a finite number' );

    assert && assert( isFinite( m10 ), 'm10 should be a finite number' );
    assert && assert( isFinite( m11 ), 'm11 should be a finite number' );
    assert && assert( isFinite( m12 ), 'm12 should be a finite number' );
    assert && assert( isFinite( m13 ), 'm13 should be a finite number' );
    assert && assert( isFinite( m14 ), 'm14 should be a finite number' );

    assert && assert( isFinite( m20 ), 'm20 should be a finite number' );
    assert && assert( isFinite( m21 ), 'm21 should be a finite number' );
    assert && assert( isFinite( m22 ), 'm22 should be a finite number' );
    assert && assert( isFinite( m23 ), 'm23 should be a finite number' );
    assert && assert( isFinite( m24 ), 'm24 should be a finite number' );

    assert && assert( isFinite( m30 ), 'm30 should be a finite number' );
    assert && assert( isFinite( m31 ), 'm31 should be a finite number' );
    assert && assert( isFinite( m32 ), 'm32 should be a finite number' );
    assert && assert( isFinite( m33 ), 'm33 should be a finite number' );
    assert && assert( isFinite( m34 ), 'm34 should be a finite number' );
  }

  public reset(): this {
    this.m00 = 1;
    this.m01 = 0;
    this.m02 = 0;
    this.m03 = 0;
    this.m04 = 0;
    this.m10 = 0;
    this.m11 = 1;
    this.m12 = 0;
    this.m13 = 0;
    this.m14 = 0;
    this.m20 = 0;
    this.m21 = 0;
    this.m22 = 1;
    this.m23 = 0;
    this.m24 = 0;
    this.m30 = 0;
    this.m31 = 0;
    this.m32 = 0;
    this.m33 = 1;
    this.m34 = 0;

    return this;
  }

  public multiplyAlpha( alpha: number ): this {
    this.m03 *= alpha;
    this.m13 *= alpha;
    this.m23 *= alpha;
    this.m33 *= alpha;

    return this;
  }

  public multiply( other: FilterMatrix | ColorMatrixFilter ): this {

    const m00 = this.m00 * other.m00 + this.m01 * other.m10 + this.m02 * other.m20 + this.m03 * other.m30;
    const m01 = this.m00 * other.m01 + this.m01 * other.m11 + this.m02 * other.m21 + this.m03 * other.m31;
    const m02 = this.m00 * other.m02 + this.m01 * other.m12 + this.m02 * other.m22 + this.m03 * other.m32;
    const m03 = this.m00 * other.m03 + this.m01 * other.m13 + this.m02 * other.m23 + this.m03 * other.m33;
    const m04 = this.m00 * other.m04 + this.m01 * other.m14 + this.m02 * other.m24 + this.m03 * other.m34 + this.m04;

    const m10 = this.m10 * other.m00 + this.m11 * other.m10 + this.m12 * other.m20 + this.m13 * other.m30;
    const m11 = this.m10 * other.m01 + this.m11 * other.m11 + this.m12 * other.m21 + this.m13 * other.m31;
    const m12 = this.m10 * other.m02 + this.m11 * other.m12 + this.m12 * other.m22 + this.m13 * other.m32;
    const m13 = this.m10 * other.m03 + this.m11 * other.m13 + this.m12 * other.m23 + this.m13 * other.m33;
    const m14 = this.m10 * other.m04 + this.m11 * other.m14 + this.m12 * other.m24 + this.m13 * other.m34 + this.m14;

    const m20 = this.m20 * other.m00 + this.m21 * other.m10 + this.m22 * other.m20 + this.m23 * other.m30;
    const m21 = this.m20 * other.m01 + this.m21 * other.m11 + this.m22 * other.m21 + this.m23 * other.m31;
    const m22 = this.m20 * other.m02 + this.m21 * other.m12 + this.m22 * other.m22 + this.m23 * other.m32;
    const m23 = this.m20 * other.m03 + this.m21 * other.m13 + this.m22 * other.m23 + this.m23 * other.m33;
    const m24 = this.m20 * other.m04 + this.m21 * other.m14 + this.m22 * other.m24 + this.m23 * other.m34 + this.m24;

    const m30 = this.m30 * other.m00 + this.m31 * other.m10 + this.m32 * other.m20 + this.m33 * other.m30;
    const m31 = this.m30 * other.m01 + this.m31 * other.m11 + this.m32 * other.m21 + this.m33 * other.m31;
    const m32 = this.m30 * other.m02 + this.m31 * other.m12 + this.m32 * other.m22 + this.m33 * other.m32;
    const m33 = this.m30 * other.m03 + this.m31 * other.m13 + this.m32 * other.m23 + this.m33 * other.m33;
    const m34 = this.m30 * other.m04 + this.m31 * other.m14 + this.m32 * other.m24 + this.m33 * other.m34 + this.m34;

    this.m00 = m00;
    this.m01 = m01;
    this.m02 = m02;
    this.m03 = m03;
    this.m04 = m04;

    this.m10 = m10;
    this.m11 = m11;
    this.m12 = m12;
    this.m13 = m13;
    this.m14 = m14;

    this.m20 = m20;
    this.m21 = m21;
    this.m22 = m22;
    this.m23 = m23;
    this.m24 = m24;

    this.m30 = m30;
    this.m31 = m31;
    this.m32 = m32;
    this.m33 = m33;
    this.m34 = m34;

    return this;
  }
}

scenery.register( 'FilterMatrix', FilterMatrix );

// used to be:
// 0x1 is whether it is a clip (00_0000_0001)
// 0x1c defines the scene_offset (00_0001_1100)   ( x >> 2 ) & 0x7
// 0x3c0 defines the info_offset (11_1100_0000)   ( x >> 6 ) & 0xf
// 0x20 is added on to end-clip (00_0010_0000)

// now:
// 0x1 is whether it is a clip (0000_0000_0001)
// 0x3e is the scene_offset (0000_0011_1110)      ( x >> 1 ) & 0x1f
// 0x3c0 is the info_offset (0011_1100_0000)      ( x >> 6 ) & 0xf
// 0x400 is added on to end-clip (0100_0000_0000)
const createDrawTag = ( isClip: boolean, infoSize: number, sceneSize: number, extra = 0 ): number => {
  return ( isClip ? 1 : 0 ) | ( sceneSize << 1 ) | ( infoSize << 6 ) | extra;
};

// u32
export class DrawTag {
  // No operation.
  public static readonly NOP = 0;

  // Color fill.
  // 0x44 => 0x42
  public static readonly COLOR = createDrawTag( false, 1, 1 );

  // Linear gradient fill.
  // 0x114 => 0x10a
  public static readonly LINEAR_GRADIENT = createDrawTag( false, 4, 5 );

  // Radial gradient fill.
  // 0x29c => 0x28e
  public static readonly RADIAL_GRADIENT = createDrawTag( false, 10, 7 );

  // Image fill.
  // 0x248 => 0x244 => 0x286
  public static readonly IMAGE = createDrawTag( false, 10, 3 );

  // Begin layer/clip.
  // 0x9 => 0x2b
  public static readonly BEGIN_CLIP = createDrawTag( true, 0, 21 );

  // End layer/clip.
  // 0x21 => 0x401
  public static readonly END_CLIP = createDrawTag( true, 0, 0, 0x400 );

  // Returns the size of the info buffer (in u32s) used by this tag.
  public static infoSize( drawTag: U32 ): U32 {
    return ( ( drawTag >>> 6 ) & 0xf ) >>> 0;
  }

  public static sceneSize( drawTag: U32 ): U32 {
    return ( ( drawTag >>> 1 ) & 0x7 ) >>> 0;
  }

  public static isClip( drawTag: U32 ): boolean {
    return ( drawTag & 0x1 ) === 1;
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
  public static isPathSegment( pathTag: U8 ): boolean {
    return PathTag.pathSegmentType( pathTag ) !== 0;
  }

  // Returns true if this is a 32-bit floating point segment.
  public static isF32( pathTag: U8 ): boolean {
    return ( pathTag & PathTag.F32_BIT ) !== 0;
  }

  // Returns true if this segment ends a subpath.
  public static isSubpathEnd( pathTag: U8 ): boolean {
    return ( pathTag & PathTag.SUBPATH_END_BIT ) !== 0;
  }

  // Sets the subpath end bit.
  public static withSubpathEnd( pathTag: U8 ): U8 {
    return pathTag | PathTag.SUBPATH_END_BIT;
  }

  // Returns the segment type.
  public static pathSegmentType( pathTag: U8 ): U8 {
    return pathTag & PathTag.SEGMENT_MASK;
  }
}

export class Layout {

  public numDrawObjects: U32 = 0;
  public numPaths: U32 = 0;
  public numClips: U32 = 0;
  public binningDataStart: U32 = 0;
  public pathTagBase: U32 = 0;
  public pathDataBase: U32 = 0;
  public drawTagBase: U32 = 0;
  public drawDataBase: U32 = 0;
  public transformBase: U32 = 0;
  public linewidthBase: U32 = 0;

  public getPathTagsSize(): number {
    const start = this.pathTagBase * 4;
    const end = this.pathDataBase * 4;
    return end - start;
  }
}

export class SceneBufferSizes {

  // TODO: perhaps pooling objects like these?
  public readonly pathTagPadded: number;
  public readonly bufferSize: number;

  public constructor( encoding: Encoding ) {
    const numPathTags = encoding.pathTagsBuf.byteLength + encoding.numOpenClips;

    // Padded length of the path tag stream in bytes.
    this.pathTagPadded = alignUp( numPathTags, 4 * PATH_REDUCE_WG );

    // Full size of the scene buffer in bytes.
    this.bufferSize = this.pathTagPadded
                      + encoding.pathDataBuf.byteLength // u8
                      + encoding.drawTagsBuf.byteLength + encoding.numOpenClips * 4 // u32 in rust
                      + encoding.drawDataBuf.byteLength // u8
                      + encoding.transforms.length * 6 * 4 // 6xf32
                      + encoding.lineWidths.length * 4; // f32

    // NOTE: because of not using the glyphs feature, our patch_sizes are effectively zero
  }
}

// Uniform render configuration data used by all GPU stages.
///
// This data structure must be kept in sync with the definition in
// shaders/shared/config.wgsl.
export class ConfigUniform {
  public widthInTiles = 0;
  public heightInTiles = 0;
  public targetWidth = 0;
  public targetHeight = 0;
  public baseColor: ColorRGBA32 = 0;
  public layout: Layout;
  public binningSize = 0; // Size of binning buffer allocation (in u32s).
  public tilesSize = 0; // Size of tile buffer allocation (in Tiles).
  public segmentsSize = 0; // Size of segment buffer allocation (in PathSegments).
  public ptclSize = 0; // Size of per-tile command list buffer allocation (in u32s).

  public constructor( layout: Layout ) {
    this.layout = layout;
  }

  public toTypedArray(): Uint8Array {
    const buf = new ByteBuffer( CONFIG_UNIFORM_BYTES );

    buf.pushU32( this.widthInTiles );
    buf.pushU32( this.heightInTiles );
    buf.pushU32( this.targetWidth );
    buf.pushU32( this.targetHeight );
    buf.pushU32( this.baseColor );

    // Layout
    buf.pushU32( this.layout.numDrawObjects );
    buf.pushU32( this.layout.numPaths );
    buf.pushU32( this.layout.numClips );
    buf.pushU32( this.layout.binningDataStart );
    buf.pushU32( this.layout.pathTagBase );
    buf.pushU32( this.layout.pathDataBase );
    buf.pushU32( this.layout.drawTagBase );
    buf.pushU32( this.layout.drawDataBase );
    buf.pushU32( this.layout.transformBase );
    buf.pushU32( this.layout.linewidthBase );

    buf.pushU32( this.binningSize );
    buf.pushU32( this.tilesSize );
    buf.pushU32( this.segmentsSize );
    buf.pushU32( this.ptclSize );

    return buf.u8Array;
  }
}

export class WorkgroupCounts {

  // TODO: pooling
  public useLargePathScan: boolean;
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

  public constructor( layout: Layout, widthInTiles: number, heightInTiles: number, numPathTags: number ) {

    const numPaths = layout.numPaths;
    const numDrawObjects = layout.numDrawObjects;
    const numClips = layout.numClips;
    const pathTagAdded = alignUp( numPathTags, 4 * PATH_REDUCE_WG );
    const pathTagSize = Math.floor( pathTagAdded / ( 4 * PATH_REDUCE_WG ) );
    const useLargePathScan = pathTagSize > PATH_REDUCE_WG;
    const reducedSize = useLargePathScan ? alignUp( pathTagSize, PATH_REDUCE_WG ) : pathTagSize;
    const drawObjectSize = Math.floor( ( numDrawObjects + PATH_BBOX_WG - 1 ) / PATH_BBOX_WG );
    const pathCoarseSize = Math.floor( ( numPathTags + PATH_COARSE_WG - 1 ) / PATH_COARSE_WG );
    const clipReduceSize = Math.floor( Math.max( 0, numClips - 1 ) / CLIP_REDUCE_WG );
    const clipSize = Math.floor( ( numClips + CLIP_REDUCE_WG - 1 ) / CLIP_REDUCE_WG );
    const pathSize = Math.floor( ( numPaths + PATH_BBOX_WG - 1 ) / PATH_BBOX_WG );
    const widthInBins = Math.floor( ( widthInTiles + 15 ) / 16 );
    const heightInBins = Math.floor( ( heightInTiles + 15 ) / 16 );

    this.useLargePathScan = useLargePathScan;
    this.path_reduce = new WorkgroupSize( pathTagSize, 1, 1 );
    this.path_reduce2 = new WorkgroupSize( PATH_REDUCE_WG, 1, 1 );
    this.path_scan1 = new WorkgroupSize( Math.floor( reducedSize / PATH_REDUCE_WG ), 1, 1 );
    this.path_scan = new WorkgroupSize( pathTagSize, 1, 1 );
    this.bbox_clear = new WorkgroupSize( drawObjectSize, 1, 1 );
    this.path_seg = new WorkgroupSize( pathCoarseSize, 1, 1 );
    this.draw_reduce = new WorkgroupSize( drawObjectSize, 1, 1 );
    this.draw_leaf = new WorkgroupSize( drawObjectSize, 1, 1 );
    this.clip_reduce = new WorkgroupSize( clipReduceSize, 1, 1 );
    this.clip_leaf = new WorkgroupSize( clipSize, 1, 1 );
    this.binning = new WorkgroupSize( drawObjectSize, 1, 1 );
    this.tile_alloc = new WorkgroupSize( pathSize, 1, 1 );
    this.path_coarse = new WorkgroupSize( pathCoarseSize, 1, 1 );
    this.backdrop = new WorkgroupSize( pathSize, 1, 1 );
    this.coarse = new WorkgroupSize( widthInBins, heightInBins, 1 );
    this.fine = new WorkgroupSize( widthInTiles, heightInTiles, 1 );
  }
}

export class BufferSize {
  public constructor( public readonly length: number, public readonly bytesPerElement: number ) {}

  // Creates a new buffer size from size in bytes (u32)
  public static from_size_in_bytes( size: number, bytes_per_element: number ): BufferSize {
    return new BufferSize( size / bytes_per_element, bytes_per_element );
  }

  // Returns the number of elements.
  public len(): number {
    return this.length;
  }

  // Returns the size in bytes.
  public getSizeInBytes(): number {
    return this.bytesPerElement * this.length;
  }

  // Returns the size in bytes aligned up to the given value.
  public getAlignedInBytes( alignment: number ): number {
    return alignUp( this.getSizeInBytes(), alignment );
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

  public readonly path_reduced: BufferSize;
  public readonly path_reduced2: BufferSize;
  public readonly path_reduced_scan: BufferSize;
  public readonly path_monoids: BufferSize;
  public readonly path_bboxes: BufferSize;
  public readonly cubics: BufferSize;
  public readonly draw_reduced: BufferSize;
  public readonly draw_monoids: BufferSize;
  public readonly info: BufferSize;
  public readonly clip_inps: BufferSize;
  public readonly clip_els: BufferSize;
  public readonly clip_bics: BufferSize;
  public readonly clip_bboxes: BufferSize;
  public readonly draw_bboxes: BufferSize;
  public readonly bump_alloc: BufferSize;
  public readonly bin_headers: BufferSize;
  public readonly paths: BufferSize;

  // The following buffer sizes have been hand picked to accommodate the vello test scenes as
  // well as paris-30k. These should instead get derived from the scene layout using
  // reasonable heuristics.
  // TODO: derive from scene layout
  public readonly bin_data: BufferSize;
  public readonly tiles: BufferSize;
  public readonly segments: BufferSize;
  public readonly ptcl: BufferSize;

  // layout: &Layout, workgroups: &WorkgroupCounts, n_path_tags: u32
  public constructor( layout: Layout, workgroups: WorkgroupCounts, n_path_tags: number ) {

    const numPaths = layout.numPaths;
    const numDrawObjects = layout.numDrawObjects;
    const numClips = layout.numClips;
    const pathTagSize = workgroups.path_reduce.x;
    const reducedSize = workgroups.useLargePathScan ? alignUp( pathTagSize, PATH_REDUCE_WG ) : pathTagSize;
    this.path_reduced = new BufferSize( reducedSize, PATH_MONOID_BYTES );
    this.path_reduced2 = new BufferSize( PATH_REDUCE_WG, PATH_MONOID_BYTES );
    this.path_reduced_scan = new BufferSize( pathTagSize, PATH_MONOID_BYTES );
    this.path_monoids = new BufferSize( pathTagSize * PATH_REDUCE_WG, PATH_MONOID_BYTES );
    this.path_bboxes = new BufferSize( numPaths, PATH_BBOX_BYTES );
    this.cubics = new BufferSize( n_path_tags, CUBIC_BYTES );
    const drawObjectSize = workgroups.draw_reduce.x;
    this.draw_reduced = new BufferSize( drawObjectSize, DRAW_MONOID_BYTES );
    this.draw_monoids = new BufferSize( numDrawObjects, DRAW_MONOID_BYTES );
    this.info = new BufferSize( layout.binningDataStart, 4 );
    this.clip_inps = new BufferSize( numClips, CLIP_BYTES );
    this.clip_els = new BufferSize( numClips, CLIP_ELEMENT_BYTES );
    this.clip_bics = new BufferSize( Math.floor( numClips / CLIP_REDUCE_WG ), CLIP_BIC_BYTES );
    this.clip_bboxes = new BufferSize( numClips, CLIP_BBOX_BYTES );
    this.draw_bboxes = new BufferSize( numPaths, DRAW_BBOX_BYTES );
    this.bump_alloc = new BufferSize( 1, BUMP_ALLOCATORS_BYTES );
    this.bin_headers = new BufferSize( drawObjectSize * 256, BIN_HEADER_BYTES );
    const numPathsAligned = alignUp( numPaths, 256 );
    this.paths = new BufferSize( numPathsAligned, PATH_BYTES );

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
  public readonly workgroupCounts: WorkgroupCounts;

  // Sizes of all buffer resources.
  public readonly bufferSizes: BufferSizes;

  public readonly configUniform: ConfigUniform;

  public readonly configBytes: Uint8Array;

  public constructor(
    layout: Layout,
    public readonly width: number,
    public readonly height: number,
    public readonly baseColor: ColorRGBA32
  ) {
    const configUniform = new ConfigUniform( layout );

    const newWidth = nextMultipleOf( width, TILE_WIDTH );
    const newHeight = nextMultipleOf( height, TILE_HEIGHT );
    const widthInTiles = newWidth / TILE_WIDTH;
    const heightInTiles = newHeight / TILE_HEIGHT;
    const numPathTags = layout.getPathTagsSize();
    const workgroupCounts = new WorkgroupCounts( layout, widthInTiles, heightInTiles, numPathTags );
    const bufferSizes = new BufferSizes( layout, workgroupCounts, numPathTags );

    this.workgroupCounts = workgroupCounts;
    this.bufferSizes = bufferSizes;

    configUniform.widthInTiles = widthInTiles;
    configUniform.heightInTiles = heightInTiles;
    configUniform.targetWidth = width;
    configUniform.targetHeight = height;
    configUniform.baseColor = premultiplyRGBA8( baseColor );
    configUniform.binningSize = bufferSizes.bin_data.len() - layout.binningDataStart;
    configUniform.tilesSize = bufferSizes.tiles.len();
    configUniform.segmentsSize = bufferSizes.segments.len();
    configUniform.ptclSize = bufferSizes.ptcl.len();

    this.configUniform = configUniform;

    this.configBytes = this.configUniform.toTypedArray();
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

  public constructor(
    public readonly packed: Uint8Array,
    public readonly layout: Layout
  ) {}

  public prepareRender( width: number, height: number, base_color: ColorRGBA32 ): void {
    this.renderConfig = new RenderConfig( this.layout, width, height, base_color );
  }
}

export class VelloPatch {
  public constructor( public readonly draw_data_offset: number ) {}
}

export class VelloImagePatch extends VelloPatch {

  public readonly type = 'image' as const;

  // Filled in by Atlas
  public atlasSubImage: AtlasSubImage | null = null;

  public constructor( drawDataOffset: number, public readonly image: EncodableImage ) {
    super( drawDataOffset );
  }

  public withOffset( drawDataOffset: number ): VelloImagePatch {
    return new VelloImagePatch( drawDataOffset, this.image );
  }
}

export class VelloRampPatch extends VelloPatch {

  public readonly type = 'ramp' as const;

  // Filled in by Ramps
  public id = -1;

  public constructor(
    drawDataOffset: number,
    public readonly stops: VelloColorStop[],
    public extend: Extend
  ) {
    super( drawDataOffset );
  }

  public withOffset( drawDataOffset: number ): VelloRampPatch {
    return new VelloRampPatch( drawDataOffset, this.stops, this.extend );
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
    return `ColorStop {offset: ${rustF32( stop.offset )}, color: peniko::Color {r: 0x${( ( stop.color >>> 24 ) & 0xff ).toString( 16 )}, g: 0x${( ( stop.color >>> 16 ) & 0xff ).toString( 16 )}, b: 0x${( ( stop.color >>> 8 ) & 0xff ).toString( 16 )}, a: 0x${( ( stop.color >>> 0 ) & 0xff ).toString( 16 )}}}`;
  } ).join( ', ' )}].into_iter()`;
};

export default class Encoding {

  public readonly id: number; // For things like output of drawing commands for validation

  public readonly pathTagsBuf = new ByteBuffer(); // path_tags
  public readonly pathDataBuf = new ByteBuffer(); // path_data
  public readonly drawTagsBuf = new ByteBuffer(); // draw_tags // NOTE: was u32 array (effectively) in rust
  public readonly drawDataBuf = new ByteBuffer(); // draw_data
  public readonly transforms: Affine[] = []; // Vec<Transform> in rust, Affine[] in js
  public readonly lineWidths: number[] = []; // Vec<f32> in rust, number[] in js
  public numPaths = 0; // u32
  public numPathSegments = 0; // u32,
  public numClips = 0; // u32
  public numOpenClips = 0; // u32
  public readonly patches: ( VelloImagePatch | VelloRampPatch )[] = []; // Vec<Patch> in rust, Patch[] in js
  public readonly colorStops: VelloColorStop[] = []; // Vec<ColorStop> in rust, VelloColorStop[] in js

  // Embedded PathEncoder
  public readonly firstPoint: Vector2 = new Vector2( 0, 0 ); // mutated
  public state: ( 0x1 | 0x2 | 0x3 ) = Encoding.PATH_START;
  public numEncodedSegments = 0;
  public isFill = true;

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

  public isEmpty(): boolean {
    return this.pathTagsBuf.byteLength === 0;
  }

  // Clears the encoding.
  public reset( isFragment: boolean ): void {
    // Clears the rustEncoding too, reinitalizing it
    // TODO: don't require hardcoding TRUE for isFragment?
    sceneryLog && sceneryLog.Encoding && this.rustLock === 0 && ( this.rustEncoding = `let mut encoding${this.id}: Encoding = Encoding::new();\n` );
    sceneryLog && sceneryLog.Encoding && this.rustLock === 0 && ( this.rustEncoding += `encoding${this.id}.reset(true);\n` );
    this.transforms.length = 0;
    this.pathTagsBuf.clear();
    this.pathDataBuf.clear();
    this.lineWidths.length = 0;
    this.drawDataBuf.clear();
    this.drawTagsBuf.clear();
    this.numPaths = 0;
    this.numPathSegments = 0;
    this.numClips = 0;
    this.numOpenClips = 0;
    this.patches.length = 0;
    this.colorStops.length = 0;
    if ( !isFragment ) {
      this.transforms.push( Affine.IDENTITY );
      this.lineWidths.push( -1.0 );
    }
  }

  // Appends another encoding to this one with an optional transform.
  public append( other: Encoding, transform: Affine | null = null ): void {
    assert && assert( !this.rustLock );

    const initialDrawDataLength = this.drawDataBuf.byteLength;

    this.pathTagsBuf.pushByteBuffer( other.pathTagsBuf );
    this.pathDataBuf.pushByteBuffer( other.pathDataBuf );
    this.drawTagsBuf.pushByteBuffer( other.drawTagsBuf );
    this.drawDataBuf.pushByteBuffer( other.drawDataBuf );
    this.numPaths += other.numPaths;
    this.numPathSegments += other.numPathSegments;
    this.numClips += other.numClips;
    this.numOpenClips += other.numOpenClips;
    if ( transform ) {
      this.transforms.push( ...other.transforms.map( t => transform.times( t ) ) );
    }
    else {
      this.transforms.push( ...other.transforms );
    }
    this.lineWidths.push( ...other.lineWidths );
    this.colorStops.push( ...other.colorStops );
    this.patches.push( ...other.patches.map( patch => patch.withOffset( patch.draw_data_offset + initialDrawDataLength ) ) );

    if ( sceneryLog && sceneryLog.Encoding && this.rustLock === 0 ) {
      if ( !this.rustEncoding?.includes( `let mut encoding${other.id} ` ) ) {
        this.rustEncoding = other.rustEncoding + this.rustEncoding;
      }
      this.rustEncoding += `encoding${this.id}.append(&mut encoding${other.id}, ${transform ? `&Some(${rustTransform( transform )})` : '&None'});\n`;
    }
  }

  public encodeLineWidth( lineWidth: number ): void {
    if ( this.lineWidths[ this.lineWidths.length - 1 ] !== lineWidth ) {
      sceneryLog && sceneryLog.Encoding && this.rustLock === 0 && ( this.rustEncoding += `encoding${this.id}.encode_linewidth(${rustF32( lineWidth )});\n` );
      this.pathTagsBuf.pushU8( PathTag.LINEWIDTH );
      this.lineWidths.push( lineWidth );
    }
  }

  /**
   * @returns Whether a transform was added
   */
  public encodeTransform( transform: Affine ): boolean {
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

  // If `isFill` is true, all subpaths will be automatically closed.
  public encodePath( isFill: boolean ): void {
    this.firstPoint.x = 0;
    this.firstPoint.y = 0;
    this.state = Encoding.PATH_START;
    this.numEncodedSegments = 0;
    this.isFill = isFill;

    if ( sceneryLog && sceneryLog.Encoding && this.rustLock === 0 ) {
      globalPathID++;

      this.rustEncoding += `let mut path_encoder${globalPathID} = encoding${this.id}.encode_path(${isFill});\n`;
    }
  }

  public moveTo( x: number, y: number ): void {
    sceneryLog && sceneryLog.Encoding && this.rustLock === 0 && ( this.rustEncoding += `path_encoder${globalPathID}.move_to(${rustF32( x )}, ${rustF32( y )});\n` );
    sceneryLog && sceneryLog.Encoding && this.rustLock++;

    if ( this.isFill ) {
      this.close();
    }
    this.firstPoint.x = x;
    this.firstPoint.y = y;
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

  public lineTo( x: number, y: number ): void {
    sceneryLog && sceneryLog.Encoding && this.rustLock === 0 && ( this.rustEncoding += `path_encoder${globalPathID}.line_to(${rustF32( x )}, ${rustF32( y )});\n` );
    sceneryLog && sceneryLog.Encoding && this.rustLock++;

    if ( this.state === Encoding.PATH_START ) {
      if ( this.numEncodedSegments === 0 ) {
        // This copies the behavior of kurbo which treats an initial line, quad
        // or curve as a move.
        this.moveTo( x, y );
        sceneryLog && sceneryLog.Encoding && this.rustLock--;
        return;
      }
      this.moveTo( this.firstPoint.x, this.firstPoint.y );
    }
    this.pathDataBuf.pushF32( x );
    this.pathDataBuf.pushF32( y );
    this.pathTagsBuf.pushU8( PathTag.LINE_TO_F32 );
    this.state = Encoding.PATH_NONEMPTY_SUBPATH;
    this.numEncodedSegments += 1;

    sceneryLog && sceneryLog.Encoding && this.rustLock--;
  }

  public quadTo( x1: number, y1: number, x2: number, y2: number ): void {
    sceneryLog && sceneryLog.Encoding && this.rustLock === 0 && ( this.rustEncoding += `path_encoder${globalPathID}.quad_to(${rustF32( x1 )}, ${rustF32( y1 )}, ${rustF32( x2 )}, ${rustF32( y2 )});\n` );
    sceneryLog && sceneryLog.Encoding && this.rustLock++;

    if ( this.state === Encoding.PATH_START ) {
      if ( this.numEncodedSegments === 0 ) {
        this.moveTo( x2, y2 );
        sceneryLog && sceneryLog.Encoding && this.rustLock--;
        return;
      }
      this.moveTo( this.firstPoint.x, this.firstPoint.y );
    }
    this.pathDataBuf.pushF32( x1 );
    this.pathDataBuf.pushF32( y1 );
    this.pathDataBuf.pushF32( x2 );
    this.pathDataBuf.pushF32( y2 );
    this.pathTagsBuf.pushU8( PathTag.QUAD_TO_F32 );
    this.state = Encoding.PATH_NONEMPTY_SUBPATH;
    this.numEncodedSegments += 1;

    sceneryLog && sceneryLog.Encoding && this.rustLock--;
  }

  public cubicTo( x1: number, y1: number, x2: number, y2: number, x3: number, y3: number ): void {
    sceneryLog && sceneryLog.Encoding && this.rustLock === 0 && ( this.rustEncoding += `path_encoder${globalPathID}.cubic_to(${rustF32( x1 )}, ${rustF32( y1 )}, ${rustF32( x2 )}, ${rustF32( y2 )}, ${rustF32( x3 )}, ${rustF32( y3 )});\n` );
    sceneryLog && sceneryLog.Encoding && this.rustLock++;

    if ( this.state === Encoding.PATH_START ) {
      if ( this.numEncodedSegments === 0 ) {
        this.moveTo( x3, y3 );
        sceneryLog && sceneryLog.Encoding && this.rustLock--;
        return;
      }
      this.moveTo( this.firstPoint.x, this.firstPoint.y );
    }
    this.pathDataBuf.pushF32( x1 );
    this.pathDataBuf.pushF32( y1 );
    this.pathDataBuf.pushF32( x2 );
    this.pathDataBuf.pushF32( y2 );
    this.pathDataBuf.pushF32( x3 );
    this.pathDataBuf.pushF32( y3 );
    this.pathTagsBuf.pushU8( PathTag.CUBIC_TO_F32 );
    this.state = Encoding.PATH_NONEMPTY_SUBPATH;
    this.numEncodedSegments += 1;

    sceneryLog && sceneryLog.Encoding && this.rustLock--;
  }

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
    if ( Math.abs( lastX - this.firstPoint.x ) > 1e-8 || Math.abs( lastY - this.firstPoint.y ) > 1e-8 ) {
      this.pathDataBuf.pushF32( this.firstPoint.x );
      this.pathDataBuf.pushF32( this.firstPoint.y );
      this.pathTagsBuf.pushU8( PathTag.withSubpathEnd( PathTag.LINE_TO_F32 ) );
      this.numEncodedSegments += 1;
    }
    else {
      this.setSubpathEndTag();
    }
    this.state = Encoding.PATH_START;
  }

  /**
   * Completes path encoding and returns the actual number of encoded segments.
   *
   * If `insertPathMarker` is true, encodes the [PathTag::PATH] tag to signify
   * the end of a complete path object. Setting this to false allows encoding
   * multiple paths with differing transforms for a single draw object.
   */
  public finish( insertPathMarker: boolean ): number {
    sceneryLog && sceneryLog.Encoding && this.rustLock === 0 && ( this.rustEncoding += `path_encoder${globalPathID}.finish(${insertPathMarker});\n` );
    sceneryLog && sceneryLog.Encoding && this.rustLock++;

    if ( this.isFill ) {
      this.close();
    }
    if ( this.state === Encoding.PATH_MOVE_TO ) {
      this.pathDataBuf.byteLength -= 8;
    }
    if ( this.numEncodedSegments !== 0 ) {
      this.setSubpathEndTag();
      this.numPathSegments += this.numEncodedSegments;
      if ( insertPathMarker ) {
        this.insertPathMarker();
      }
    }

    sceneryLog && sceneryLog.Encoding && this.rustLock--;

    return this.numEncodedSegments;
  }

  public setSubpathEndTag(): void {
    if ( this.pathTagsBuf.byteLength ) {
      // In-place replace, add the "subpath end" flag
      const lastIndex = this.pathTagsBuf.byteLength - 1;

      this.pathTagsBuf.fullU8Array[ lastIndex ] = PathTag.withSubpathEnd( this.pathTagsBuf.fullU8Array[ lastIndex ] );
    }
  }

  // Exposed for glyph handling
  public insertPathMarker(): void {
    sceneryLog && sceneryLog.Encoding && this.rustLock === 0 && ( this.rustEncoding += `encoding${this.id}.path_tags.push(PathTag::PATH);\nencoding${this.id}.n_paths += 1;\n` );
    this.pathTagsBuf.pushU8( PathTag.PATH );
    this.numPaths += 1;
  }

  // Encodes a solid color brush.
  public encodeColor( color: ColorRGBA32 ): void {
    sceneryLog && sceneryLog.Encoding && this.rustLock === 0 && ( this.rustEncoding += `encoding${this.id}.encode_color(${rustDrawColor( color )});\n` );

    this.drawTagsBuf.pushU32( DrawTag.COLOR );
    this.drawDataBuf.pushU32( premultiplyRGBA8( color ) );
  }

  // Swap the last two tags in the path tag stream; used for transformed paints.
  public swapLastPathTags(): void {
    const pathTagsArray = this.pathTagsBuf.fullU8Array;
    const tag = pathTagsArray[ pathTagsArray.length - 2 ];
    pathTagsArray[ pathTagsArray.length - 2 ] = pathTagsArray[ pathTagsArray.length - 1 ];
    pathTagsArray[ pathTagsArray.length - 1 ] = tag;
  }

  // zero: => false, one => color, many => true (icky)
  private addRamp( color_stops: VelloColorStop[], alpha: number, extend: Extend ): null | true | ColorRGBA32 {
    const offset = this.drawDataBuf.byteLength;
    const stopsStart = this.colorStops.length;
    if ( alpha !== 1 ) {
      this.colorStops.push( ...color_stops.map( stop => new VelloColorStop( stop.offset, withAlphaFactor( stop.color, alpha ) ) ) );
    }
    else {
      this.colorStops.push( ...color_stops );
    }
    const stopsEnd = this.colorStops.length;

    const stopCount = stopsEnd - stopsStart;

    if ( stopCount === 0 ) {
      return null;
    }
    else if ( stopCount === 1 ) {
      assert && assert( this.colorStops.length );

      return this.colorStops.pop()!.color;
    }
    else {
      this.patches.push( new VelloRampPatch( offset, color_stops, extend ) );
      return true;
    }
  }

  public encodeLinearGradient( x0: number, y0: number, x1: number, y1: number, colorStops: VelloColorStop[], alpha: number, extend: Extend ): void {
    sceneryLog && sceneryLog.Encoding && this.rustLock === 0 && ( this.rustEncoding += `encoding${this.id}.encode_linear_gradient(DrawLinearGradient {index: 0, p0: [${rustF32( x0 )}, ${rustF32( y0 )}], p1: [${rustF32( x1 )}, ${rustF32( y1 )}]}, ${rustColorStops( colorStops )}, ${rustF32( alpha )}, ${ExtendMap[ extend ]});\n` );
    sceneryLog && sceneryLog.Encoding && this.rustLock++;

    const result = this.addRamp( colorStops, alpha, extend );
    if ( result === null ) {
      this.encodeColor( 0 );
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
      this.encodeColor( result );
    }

    sceneryLog && sceneryLog.Encoding && this.rustLock--;
  }

  public encodeRadialGradient( x0: number, y0: number, r0: number, x1: number, y1: number, r1: number, colorStops: VelloColorStop[], alpha: number, extend: Extend ): void {
    sceneryLog && sceneryLog.Encoding && this.rustLock === 0 && ( this.rustEncoding += `encoding${this.id}.encode_radial_gradient(DrawRadialGradient {index: 0, p0: [${rustF32( x0 )}, ${rustF32( y0 )}], r0: ${rustF32( r0 )}, p1: [${rustF32( x1 )}, ${rustF32( y1 )}], r1: ${rustF32( r1 )}}, ${rustColorStops( colorStops )}, ${rustF32( alpha )}, ${ExtendMap[ extend ]});\n` );
    sceneryLog && sceneryLog.Encoding && this.rustLock++;

    // Match Skia's epsilon for radii comparison
    const SKIA_EPSILON = 1 / ( ( 1 << 12 ) >>> 0 );
    if ( x0 === x1 && y0 === y1 && Math.abs( r0 - r1 ) < SKIA_EPSILON ) {
      this.encodeColor( 0 );
    }
    else {
      const result = this.addRamp( colorStops, alpha, extend );
      if ( result === null ) {
        this.encodeColor( 0 );
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
        this.encodeColor( result );
      }
    }

    sceneryLog && sceneryLog.Encoding && this.rustLock--;
  }

  public encodeImage( image: EncodableImage, extendX: Extend = Extend.Pad, extendY: Extend = Extend.Pad ): void {
    if ( sceneryLog && sceneryLog.Encoding && this.rustLock === 0 ) {
      let u8array;
      if ( image instanceof BufferImage ) {
        u8array = new Uint8Array( image.buffer );
      }
      else {
        const canvas = document.createElement( 'canvas' );
        canvas.width = image.width;
        canvas.height = image.height;
        const context = canvas.getContext( '2d' )!;
        context.drawImage( image.source, 0, 0 );
        u8array = new Uint8Array( context.getImageData( 0, 0, image.width, image.height ).data.buffer );
      }
      const dataString = [ ...u8array ].join( ', ' );
      sceneryLog && sceneryLog.Encoding && this.rustLock === 0 && ( this.rustEncoding += `encoding${this.id}.encode_image(&peniko::Image::new(peniko::Blob::new(std::sync::Arc::new([${dataString}].to_vec())), peniko::Format::Rgba8, ${image.width}, ${image.height}), 1.0);\n` );
    }

    this.patches.push( new VelloImagePatch( this.drawDataBuf.byteLength, image ) );
    this.drawTagsBuf.pushU32( DrawTag.IMAGE );

    // packed atlas coordinates (xy) u32
    this.drawDataBuf.pushU32( 0 );

    // Packed image dimensions. (width_height) u32
    this.drawDataBuf.pushU32( ( ( image.width << 16 ) >>> 0 ) | ( image.height & 0xFFFF ) );

    // Packed extend modes
    this.drawDataBuf.pushU32( ( ( extendX << 2 ) >>> 0 ) | extendY );
  }

  public encodeBeginClip( mix: Mix, compose: Compose, filterMatrix: FilterMatrix ): void {
    sceneryLog && sceneryLog.Encoding && this.rustLock === 0 && ( this.rustEncoding += `encoding${this.id}.encode_begin_clip(BlendMode {mix: ${MixMap[ mix ]}, compose: ${ComposeMap[ compose ]}}, ${rustF32( filterMatrix.m33 )});\n` );
    this.drawTagsBuf.pushU32( DrawTag.BEGIN_CLIP );

    // u32 combination of mix and compose
    this.drawDataBuf.pushU32( ( ( mix << 8 ) >>> 0 ) | compose );

    this.drawDataBuf.pushF32( filterMatrix.m00 );
    this.drawDataBuf.pushF32( filterMatrix.m10 );
    this.drawDataBuf.pushF32( filterMatrix.m20 );
    this.drawDataBuf.pushF32( filterMatrix.m30 );

    this.drawDataBuf.pushF32( filterMatrix.m01 );
    this.drawDataBuf.pushF32( filterMatrix.m11 );
    this.drawDataBuf.pushF32( filterMatrix.m21 );
    this.drawDataBuf.pushF32( filterMatrix.m31 );

    this.drawDataBuf.pushF32( filterMatrix.m02 );
    this.drawDataBuf.pushF32( filterMatrix.m12 );
    this.drawDataBuf.pushF32( filterMatrix.m22 );
    this.drawDataBuf.pushF32( filterMatrix.m32 );

    this.drawDataBuf.pushF32( filterMatrix.m03 );
    this.drawDataBuf.pushF32( filterMatrix.m13 );
    this.drawDataBuf.pushF32( filterMatrix.m23 );
    this.drawDataBuf.pushF32( filterMatrix.m33 );

    this.drawDataBuf.pushF32( filterMatrix.m04 );
    this.drawDataBuf.pushF32( filterMatrix.m14 );
    this.drawDataBuf.pushF32( filterMatrix.m24 );
    this.drawDataBuf.pushF32( filterMatrix.m34 );

    this.numClips += 1;
    this.numOpenClips += 1;
  }

  public encodeEndClip(): void {
    sceneryLog && sceneryLog.Encoding && this.rustLock === 0 && ( this.rustEncoding += `encoding${this.id}.encode_end_clip();\n` );
    if ( this.numOpenClips > 0 ) {
      this.drawTagsBuf.pushU32( DrawTag.END_CLIP );
      // This is a dummy path, and will go away with the new clip impl.
      this.pathTagsBuf.pushU8( PathTag.PATH );
      this.numPaths += 1;
      this.numClips += 1;
      this.numOpenClips -= 1;
    }
  }

  // TODO: make this workaround not needed
  public finalizeScene(): void {
    this.encodePath( true );
    this.moveTo( 0, 0 );
    this.lineTo( 1, 0 );
    this.close();
    this.finish( true );
  }

  public encodeBounds( bounds: Bounds2 ): number {
    return this.encodeRect( bounds.minX, bounds.minY, bounds.maxX, bounds.maxY );
  }

  public encodeRect( x0: number, y0: number, x1: number, y1: number ): number {
    this.encodePath( true );
    this.moveTo( x0, y0 );
    this.lineTo( x1, y0 );
    this.lineTo( x1, y1 );
    this.lineTo( x0, y1 );
    this.close();
    return this.finish( true );
  }

  // To encode a kite shape, we'll need to split arcs/elliptical-arcs into bezier curves
  public encodeShape( shape: Shape, isFill: boolean, insertPathMarker: boolean, tolerance: number ): number {
    this.encodePath( isFill );

    // TODO: better code that isn't tons of forEach's that will kill our GC and add jank
    shape.subpaths.forEach( subpath => {
      if ( subpath.isDrawable() ) {
        const startPoint = subpath.getFirstSegment().start;
        this.moveTo( startPoint.x, startPoint.y );

        subpath.segments.forEach( segment => {
          if ( segment instanceof Line ) {
            this.lineTo( segment.end.x, segment.end.y );
          }
          else if ( segment instanceof Quadratic ) {
            this.quadTo( segment.control.x, segment.control.y, segment.end.x, segment.end.y );
          }
          else if ( segment instanceof Cubic ) {
            this.cubicTo( segment.control1.x, segment.control1.y, segment.control2.x, segment.control2.y, segment.end.x, segment.end.y );
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

              this.quadTo( control.x, control.y, end.x, end.y );
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

  public printDebug(): void {
    console.log( `path_tags\n${[ ...this.pathTagsBuf.u8Array ].map( x => x.toString() ).join( ', ' )}` );
    console.log( `path_data\n${[ ...this.pathDataBuf.u8Array ].map( x => x.toString() ).join( ', ' )}` );
    console.log( `draw_tags\n${[ ...this.drawTagsBuf.u8Array ].map( x => x.toString() ).join( ', ' )}` );
    console.log( `draw_data\n${[ ...this.drawDataBuf.u8Array ].map( x => x.toString() ).join( ', ' )}` );
    console.log( `transforms\n${this.transforms.map( x => `_ a00:${x.a00} a10:${x.a10} a01:${x.a01} a11:${x.a11} a02:${x.a02} a12:${x.a12}_` ).join( '\n' )}` );
    console.log( `linewidths\n${this.lineWidths.map( x => x.toString() ).join( ', ' )}` );
    console.log( `n_paths\n${this.numPaths}` );
    console.log( `n_path_segments\n${this.numPathSegments}` );
    console.log( `n_clips\n${this.numClips}` );
    console.log( `n_open_clips\n${this.numOpenClips}` );
  }

  // Resolves late bound resources and packs an encoding. Returns the packed
  // layout and computed ramp data.
  public resolve( deviceContext: DeviceContext ): RenderInfo {

    // @ts-expect-error Because it can't detect the filter result type
    deviceContext.ramps.updatePatches( this.patches.filter( patch => patch instanceof VelloRampPatch ) );
    // @ts-expect-error Because it can't detect the filter result type
    deviceContext.atlas.updatePatches( this.patches.filter( patch => patch instanceof VelloImagePatch ) );

    const layout = new Layout();
    layout.numPaths = this.numPaths;
    layout.numClips = this.numClips;

    const sceneBufferSizes = new SceneBufferSizes( this );
    const bufferSize = sceneBufferSizes.bufferSize;
    const pathTagPadded = sceneBufferSizes.pathTagPadded;

    const dataBuf = new ByteBuffer( sceneBufferSizes.bufferSize );

    // Path tag stream
    layout.pathTagBase = sizeToWords( dataBuf.byteLength );
    dataBuf.pushByteBuffer( this.pathTagsBuf );
    // TODO: what if we... just error if there are open clips? Why are we padding the streams to make this work?
    for ( let i = 0; i < this.numOpenClips; i++ ) {
      dataBuf.pushU8( PathTag.PATH );
    }
    dataBuf.byteLength = pathTagPadded;

    // Path data stream
    layout.pathDataBase = sizeToWords( dataBuf.byteLength );
    dataBuf.pushByteBuffer( this.pathDataBuf );

    // Draw tag stream
    layout.drawTagBase = sizeToWords( dataBuf.byteLength );
    // Bin data follows draw info
    layout.binningDataStart = _.sum( this.drawTagsBuf.u32Array.map( DrawTag.infoSize ) );
    dataBuf.pushByteBuffer( this.drawTagsBuf );
    for ( let i = 0; i < this.numOpenClips; i++ ) {
      dataBuf.pushU32( DrawTag.END_CLIP );
    }

    // Draw data stream
    layout.drawDataBase = sizeToWords( dataBuf.byteLength );
    {
      const drawDataOffset = dataBuf.byteLength;
      dataBuf.pushByteBuffer( this.drawDataBuf );

      this.patches.forEach( patch => {
        const byteOffset = drawDataOffset + patch.draw_data_offset;
        let bytes;

        if ( patch instanceof VelloRampPatch ) {
          bytes = u32ToBytes( ( ( patch.id << 2 ) >>> 0 ) | patch.extend );
        }
        else {
          assert && assert( patch.atlasSubImage );
          bytes = u32ToBytes( ( patch.atlasSubImage!.x << 16 ) >>> 0 | patch.atlasSubImage!.y );
          // TODO: assume the image fit (if not, we'll need to do something else)
        }

        // Patch data directly into our full output
        dataBuf.fullU8Array.set( bytes, byteOffset );
      } );
    }

    // Transform stream
    // TODO: Float32Array instead of Affine?
    layout.transformBase = sizeToWords( dataBuf.byteLength );
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
    layout.linewidthBase = sizeToWords( dataBuf.byteLength );
    for ( let i = 0; i < this.lineWidths.length; i++ ) {
      dataBuf.pushF32( this.lineWidths[ i ] );
    }

    layout.numDrawObjects = layout.numPaths;

    if ( dataBuf.byteLength !== bufferSize ) {
      throw new Error( 'buffer size mismatch' );
    }

    return new RenderInfo( dataBuf.u8Array, layout );
  }
}

scenery.register( 'Encoding', Encoding );
