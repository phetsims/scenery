// Copyright 2023, University of Colorado Boulder

/**
 * RenderProgram representing a constant color
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { ClippableFace, Color, constantTrue, RenderColorSpace, RenderPath, RenderPathProgram, RenderProgram, scenery, SerializedRenderPath } from '../../../imports.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Matrix3 from '../../../../../dot/js/Matrix3.js';
import Vector4 from '../../../../../dot/js/Vector4.js';

// TODO: consider transforms as a node itself? Meh probably excessive?

export default class RenderColor extends RenderPathProgram {
  public constructor( path: RenderPath | null, public color: Vector4 ) {
    super( path );
  }

  public override transformed( transform: Matrix3 ): RenderProgram {
    return new RenderColor( this.getTransformedPath( transform ), this.color );
  }

  public override equals( other: RenderProgram ): boolean {
    if ( this === other ) { return true; }
    return super.equals( other ) &&
           other instanceof RenderColor &&
           this.color.equals( other.color );
  }

  public override isFullyTransparent(): boolean {
    return this.color.w === 0;
  }

  public override isFullyOpaque(): boolean {
    return this.path === null && this.color.w === 1;
  }

  public override replace( callback: ( program: RenderProgram ) => RenderProgram | null ): RenderProgram {
    const replaced = callback( this );
    if ( replaced ) {
      return replaced;
    }
    else {
      return new RenderColor( this.path, this.color );
    }
  }

  public override simplify( pathTest: ( renderPath: RenderPath ) => boolean = constantTrue ): RenderProgram {
    if ( this.isInPath( pathTest ) ) {
      return new RenderColor( null, this.color );
    }
    else {
      return RenderColor.TRANSPARENT;
    }
  }

  public override evaluate(
    face: ClippableFace | null,
    area: number,
    centroid: Vector2,
    minX: number,
    minY: number,
    maxX: number,
    maxY: number,
    pathTest: ( renderPath: RenderPath ) => boolean = constantTrue
  ): Vector4 {
    if ( this.isInPath( pathTest ) ) {
      return this.color;
    }
    else {
      return Vector4.ZERO;
    }
  }

  public override toRecursiveString( indent: string ): string {
    return `${indent}RenderColor (${this.path ? this.path.id : 'null'}, color:${this.color.toString()})`;
  }

  public static fromColor( path: RenderPath | null, color: Color ): RenderColor {
    return new RenderColor( path, RenderColor.colorToPremultipliedLinear( color ) );
  }

  public static colorToPremultipliedLinear( color: Color ): Vector4 {
    // https://entropymine.com/imageworsener/srgbformula/
    // sRGB to Linear
    // 0 ≤ S ≤ 0.0404482362771082 : L = S/12.92
    // 0.0404482362771082 < S ≤ 1 : L = ((S+0.055)/1.055)^2.4

    // Linear to sRGB
    // 0 ≤ L ≤ 0.00313066844250063 : S = L×12.92
    // 0.00313066844250063 < L ≤ 1 : S = 1.055×L^1/2.4 − 0.055

    const sRGB = new Vector4(
      color.red / 255,
      color.green / 255,
      color.blue / 255,
      color.alpha
    );

    return RenderColor.premultiply( RenderColor.sRGBToLinear( sRGB ) );
  }

  public static premultipliedLinearToColor( premultiplied: Vector4 ): Color {
    const sRGB = RenderColor.linearToSRGB( RenderColor.unpremultiply( premultiplied ) );

    return new Color(
      sRGB.x * 255,
      sRGB.y * 255,
      sRGB.z * 255,
      sRGB.w
    );
  }

  public static sRGBToLinear( sRGB: Vector4 ): Vector4 {
    return new Vector4(
      sRGB.x <= 0.0404482362771082 ? sRGB.x / 12.92 : Math.pow( ( sRGB.x + 0.055 ) / 1.055, 2.4 ),
      sRGB.y <= 0.0404482362771082 ? sRGB.y / 12.92 : Math.pow( ( sRGB.y + 0.055 ) / 1.055, 2.4 ),
      sRGB.z <= 0.0404482362771082 ? sRGB.z / 12.92 : Math.pow( ( sRGB.z + 0.055 ) / 1.055, 2.4 ),
      sRGB.w
    );
  }

  public static linearToSRGB( linear: Vector4 ): Vector4 {
    return new Vector4(
      linear.x <= 0.00313066844250063 ? linear.x * 12.92 : 1.055 * Math.pow( linear.x, 1 / 2.4 ) - 0.055,
      linear.y <= 0.00313066844250063 ? linear.y * 12.92 : 1.055 * Math.pow( linear.y, 1 / 2.4 ) - 0.055,
      linear.z <= 0.00313066844250063 ? linear.z * 12.92 : 1.055 * Math.pow( linear.z, 1 / 2.4 ) - 0.055,
      linear.w
    );
  }

  // Oklab is a perceptually uniform color space, which is useful for color blending.
  // https://bottosson.github.io/posts/oklab/
  // returned as (L,a,b,alpha)
  public static linearToOklab( linear: Vector4 ): Vector4 {
    // TODO: isolate matrices out
    const l = 0.4122214708 * linear.x + 0.5363325363 * linear.y + 0.0514459929 * linear.z;
    const m = 0.2119034982 * linear.x + 0.6806995451 * linear.y + 0.1073969566 * linear.z;
    const s = 0.0883024619 * linear.x + 0.2817188376 * linear.y + 0.6299787005 * linear.z;

    const l_ = Math.cbrt( l );
    const m_ = Math.cbrt( m );
    const s_ = Math.cbrt( s );

    return new Vector4(
      0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
      1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
      0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
      linear.w
    );
  }

  public static oklabToLinear( oklab: Vector4 ): Vector4 {
    const l_ = oklab.x + 0.3963377774 * oklab.y + 0.2158037573 * oklab.z;
    const m_ = oklab.x - 0.1055613458 * oklab.y - 0.0638541728 * oklab.z;
    const s_ = oklab.x - 0.0894841775 * oklab.y - 1.2914855480 * oklab.z;

    const l = l_ * l_ * l_;
    const m = m_ * m_ * m_;
    const s = s_ * s_ * s_;

    return new Vector4(
    4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
      oklab.w
    );
  }

  public static premultiply( color: Vector4 ): Vector4 {
    return new Vector4(
      color.x * color.w,
      color.y * color.w,
      color.z * color.w,
      color.w
    );
  }

  public static unpremultiply( color: Vector4 ): Vector4 {
    return color.w === 0 ? Vector4.ZERO : new Vector4(
      color.x / color.w,
      color.y / color.w,
      color.z / color.w,
      color.w
    );
  }

  public static ratioBlend(
    zeroColor: Vector4,
    oneColor: Vector4,
    ratio: number,
    colorSpace: RenderColorSpace = RenderColorSpace.LinearUnpremultipliedSRGB
  ): Vector4 {
    const minusRatio = 1 - ratio;

    switch( colorSpace ) {
      case RenderColorSpace.LinearUnpremultipliedSRGB:
        return new Vector4(
          zeroColor.x * minusRatio + oneColor.x * ratio,
          zeroColor.y * minusRatio + oneColor.y * ratio,
          zeroColor.z * minusRatio + oneColor.z * ratio,
          zeroColor.w * minusRatio + oneColor.w * ratio
        );
      case RenderColorSpace.SRGB:
      {
        // TODO: This is.... crazy to have so many premultiply/unpremultiply calls, just to work around the sRGB
        // TODO: nonlinear transfer function?

        // TODO: reduce allocations?
        const zeroSRGB = RenderColor.premultiply( RenderColor.linearToSRGB( RenderColor.unpremultiply( zeroColor ) ) );
        const oneSRGB = RenderColor.premultiply( RenderColor.linearToSRGB( RenderColor.unpremultiply( oneColor ) ) );
        return RenderColor.premultiply( RenderColor.sRGBToLinear( RenderColor.unpremultiply( new Vector4(
          zeroSRGB.x * minusRatio + oneSRGB.x * ratio,
          zeroSRGB.y * minusRatio + oneSRGB.y * ratio,
          zeroSRGB.z * minusRatio + oneSRGB.z * ratio,
          zeroSRGB.w * minusRatio + oneSRGB.w * ratio
        ) ) ) );
      }
      case RenderColorSpace.Oklab:
      {
        // TODO: DO we really have a need to blend in this space?
        // TODO: reduce allocations?
        const zeroOklab = RenderColor.premultiply( RenderColor.linearToOklab( RenderColor.unpremultiply( zeroColor ) ) );
        const oneOklab = RenderColor.premultiply( RenderColor.linearToOklab( RenderColor.unpremultiply( oneColor ) ) );
        return RenderColor.premultiply( RenderColor.oklabToLinear( RenderColor.unpremultiply( new Vector4(
          zeroOklab.x * minusRatio + oneOklab.x * ratio,
          zeroOklab.y * minusRatio + oneOklab.y * ratio,
          zeroOklab.z * minusRatio + oneOklab.z * ratio,
          zeroOklab.w * minusRatio + oneOklab.w * ratio
        ) ) ) );
      }
      default:
        throw new Error( `Invalid color space: ${colorSpace}` );
    }
  }

  public static readonly TRANSPARENT = new RenderColor( null, Vector4.ZERO );

  public override serialize(): SerializedRenderColor {
    return {
      type: 'RenderColor',
      path: this.path ? this.path.serialize() : null,
      color: { r: this.color.x, g: this.color.y, b: this.color.z, a: this.color.w }
    };
  }

  public static override deserialize( obj: SerializedRenderColor ): RenderColor {
    return new RenderColor( obj.path ? RenderPath.deserialize( obj.path ) : null, new Vector4( obj.color.r, obj.color.g, obj.color.b, obj.color.a ) );
  }
}

scenery.register( 'RenderColor', RenderColor );

export type SerializedRenderColor = {
  type: 'RenderColor';
  path: SerializedRenderPath | null;
  color: { r: number; g: number; b: number; a: number };
};
