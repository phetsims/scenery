// Copyright 2023, University of Colorado Boulder

/**
 * RenderProgram representing a constant color
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { ClippableFace, Color, constantTrue, RenderPath, RenderProgram, scenery } from '../../../imports.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Matrix3 from '../../../../../dot/js/Matrix3.js';
import Vector4 from '../../../../../dot/js/Vector4.js';

// TODO: consider transforms as a node itself? Meh probably excessive?

const scratchCombinedVector = new Vector4( 0, 0, 0, 0 );

export default class RenderColor extends RenderProgram {
  public constructor(
    public color: Vector4
  ) {
    super();
  }

  public override transformed( transform: Matrix3 ): RenderProgram {
    return new RenderColor( this.color );
  }

  public override equals( other: RenderProgram ): boolean {
    if ( this === other ) { return true; }
    return other instanceof RenderColor &&
           this.color.equals( other.color );
  }

  public override isFullyTransparent(): boolean {
    return this.color.w === 0;
  }

  public override isFullyOpaque(): boolean {
    return this.color.w === 1;
  }

  public override needsFace(): boolean {
    return false;
  }

  public override needsArea(): boolean {
    return false;
  }

  public override needsCentroid(): boolean {
    return false;
  }

  public override replace( callback: ( program: RenderProgram ) => RenderProgram | null ): RenderProgram {
    const replaced = callback( this );
    if ( replaced ) {
      return replaced;
    }
    else {
      return new RenderColor( this.color );
    }
  }

  public override simplify( pathTest: ( renderPath: RenderPath ) => boolean = constantTrue ): RenderProgram {
    return this;
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
    return this.color;
  }

  public override toRecursiveString( indent: string ): string {
    return `${indent}RenderColor color:${this.color.toString()})`;
  }

  public static premultipliedSRGBToLinearPremultipliedSRGB( color: Vector4 ): Vector4 {
    // TODO: performance improvements? We'll be killing the GC with all these Vector4s
    return RenderColor.premultiply( RenderColor.sRGBToLinear( RenderColor.unpremultiply( color ) ) );
  }

  // TODO: can we combine these methods of sRGB conversion without losing performance?
  // TODO: TODO: TODO: NOTE IN THIS FUNCTION it goes to sRGB255
  public static linearPremultipliedSRGBToSRGB255( color: Vector4 ): Vector4 {
    const accumulation = color;
    const a = accumulation.w;

    if ( a > 0 ) {
      // unpremultiply
      const x = accumulation.x / a;
      const y = accumulation.y / a;
      const z = accumulation.z / a;

      // linear to sRGB
      const r = x <= 0.00313066844250063 ? x * 12.92 : 1.055 * Math.pow( x, 1 / 2.4 ) - 0.055;
      const g = y <= 0.00313066844250063 ? y * 12.92 : 1.055 * Math.pow( y, 1 / 2.4 ) - 0.055;
      const b = z <= 0.00313066844250063 ? z * 12.92 : 1.055 * Math.pow( z, 1 / 2.4 ) - 0.055;

      return scratchCombinedVector.setXYZW(
        r * 255,
        g * 255,
        b * 255,
        a * 255
      );
    }
    else {
      return scratchCombinedVector.setXYZW( 0, 0, 0, 0 );
    }
  }

  public static premultipliedSRGBFromColor( color: Color ): RenderColor {
    return new RenderColor( RenderColor.premultiply( RenderColor.colorToSRGB( color ) ) );
  }

  public static premultipliedOklabFromColor( color: Color ): RenderColor {
    return new RenderColor( RenderColor.premultiply( RenderColor.linearToOklab( RenderColor.sRGBToLinear( RenderColor.colorToSRGB( color ) ) ) ) );
  }

  public static colorToSRGB( color: Color ): Vector4 {
    return new Vector4(
      color.red / 255,
      color.green / 255,
      color.blue / 255,
      color.alpha
    );
  }

  public static premultipliedSRGBToColor( premultiplied: Vector4 ): Color {
    const sRGB = RenderColor.unpremultiply( premultiplied );

    return new Color(
      sRGB.x * 255,
      sRGB.y * 255,
      sRGB.z * 255,
      sRGB.w
    );
  }

  public static sRGBToLinear( sRGB: Vector4 ): Vector4 {
    // https://entropymine.com/imageworsener/srgbformula/ (a more precise formula for sRGB)
    // sRGB to Linear
    // 0 ≤ S ≤ 0.0404482362771082 : L = S/12.92
    // 0.0404482362771082 < S ≤ 1 : L = ((S+0.055)/1.055)^2.4
    return new Vector4(
      sRGB.x <= 0.0404482362771082 ? sRGB.x / 12.92 : Math.pow( ( sRGB.x + 0.055 ) / 1.055, 2.4 ),
      sRGB.y <= 0.0404482362771082 ? sRGB.y / 12.92 : Math.pow( ( sRGB.y + 0.055 ) / 1.055, 2.4 ),
      sRGB.z <= 0.0404482362771082 ? sRGB.z / 12.92 : Math.pow( ( sRGB.z + 0.055 ) / 1.055, 2.4 ),
      sRGB.w
    );
  }

  public static linearToSRGB( linear: Vector4 ): Vector4 {
    // https://entropymine.com/imageworsener/srgbformula/ (a more precise formula for sRGB)
    // Linear to sRGB
    // 0 ≤ L ≤ 0.00313066844250063 : S = L×12.92
    // 0.00313066844250063 < L ≤ 1 : S = 1.055×L^1/2.4 − 0.055
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
    ratio: number
  ): Vector4 {
    const minusRatio = 1 - ratio;

    return new Vector4(
      zeroColor.x * minusRatio + oneColor.x * ratio,
      zeroColor.y * minusRatio + oneColor.y * ratio,
      zeroColor.z * minusRatio + oneColor.z * ratio,
      zeroColor.w * minusRatio + oneColor.w * ratio
    );
  }

  public static readonly TRANSPARENT = new RenderColor( Vector4.ZERO );

  public override serialize(): SerializedRenderColor {
    return {
      type: 'RenderColor',
      color: { r: this.color.x, g: this.color.y, b: this.color.z, a: this.color.w }
    };
  }

  public static override deserialize( obj: SerializedRenderColor ): RenderColor {
    return new RenderColor( new Vector4( obj.color.r, obj.color.g, obj.color.b, obj.color.a ) );
  }
}

scenery.register( 'RenderColor', RenderColor );

export type SerializedRenderColor = {
  type: 'RenderColor';
  color: { r: number; g: number; b: number; a: number };
};
