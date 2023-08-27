// Copyright 2023, University of Colorado Boulder

/**
 * RenderProgram representing a constant color
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { ClippableFace, Color, RenderColorSpace, RenderProgram, scenery } from '../../../imports.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Vector4 from '../../../../../dot/js/Vector4.js';
import Vector3 from '../../../../../dot/js/Vector3.js';
import Matrix3 from '../../../../../dot/js/Matrix3.js';
import ConstructorOf from '../../../../../phet-core/js/types/ConstructorOf.js';

// TODO: consider transforms as a node itself? Meh probably excessive?

const scratchCombinedVector = new Vector4( 0, 0, 0, 0 );

const sRGBRedChromaticity = new Vector2( 0.64, 0.33 );
const sRGBGreenChromaticity = new Vector2( 0.3, 0.6 );
const sRGBBlueChromaticity = new Vector2( 0.15, 0.06 );
const sRGBWhiteChromaticity = new Vector2( 0.312713, 0.329016 );

const dciP3RedChromaticity = new Vector2( 0.68, 0.32 );
const dciP3GreenChromaticity = new Vector2( 0.265, 0.69 );
const dciP3BlueChromaticity = new Vector2( 0.15, 0.06 );

export default class RenderColor extends RenderProgram {
  public constructor(
    public color: Vector4
  ) {
    super();
  }

  public static from( ...args: ConstructorParameters<ConstructorOf<Color>> ): RenderColor {
    // @ts-expect-error We're passing Color's constructor arguments in
    const color = new Color( ...args );
    return new RenderColor( new Vector4(
      color.red / 255,
      color.green / 255,
      color.blue / 255,
      color.alpha
    ) );
  }

  public override getName(): string {
    return 'RenderColor';
  }

  public override getChildren(): RenderProgram[] {
    return [];
  }

  public override withChildren( children: RenderProgram[] ): RenderColor {
    assert && assert( children.length === 0 );
    return this;
  }

  protected override equalsTyped( other: this ): boolean {
    return this.color.equals( other.color );
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

  public override simplified(): RenderProgram {
    return this;
  }

  public override evaluate(
    face: ClippableFace | null,
    area: number,
    centroid: Vector2,
    minX: number,
    minY: number,
    maxX: number,
    maxY: number
  ): Vector4 {
    return this.color;
  }

  protected override getExtraDebugString(): string {
    return `${this.color.x}, ${this.color.y}, ${this.color.z}, ${this.color.w}`;
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

  public static premultipliedDisplayP3FromColor( color: Color ): RenderColor {
    return new RenderColor( RenderColor.premultiply( RenderColor.linearDisplayP3ToDisplayP3( RenderColor.linearToLinearDisplayP3( RenderColor.displayP3ToLinearDisplayP3( RenderColor.colorToSRGB( color ) ) ) ) ) );
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

  public static multiplyMatrixTimesColor( matrix: Matrix3, color: Vector4 ): Vector4 {
    return new Vector4(
      matrix.m00() * color.x + matrix.m01() * color.y + matrix.m02() * color.z,
      matrix.m10() * color.x + matrix.m11() * color.y + matrix.m12() * color.z,
      matrix.m20() * color.x + matrix.m21() * color.y + matrix.m22() * color.z,
      color.w
    );
  }

  public static linearToLinearDisplayP3( color: Vector4 ): Vector4 {
    return RenderColor.multiplyMatrixTimesColor( RenderColor.sRGBToDisplayP3Matrix, color );
  }

  public static linearDisplayP3ToLinear( color: Vector4 ): Vector4 {
    return RenderColor.multiplyMatrixTimesColor( RenderColor.displayP3TosRGBMatrix, color );
  }

  public static linearDisplayP3ToDisplayP3( linear: Vector4 ): Vector4 {
    return RenderColor.linearToSRGB( linear ); // same transfer curve
  }

  public static displayP3ToLinearDisplayP3( displayP3: Vector4 ): Vector4 {
    return RenderColor.sRGBToLinear( displayP3 ); // same transfer curve
  }

  // A radian-based oklch?
  public static oklabToOklch( oklab: Vector4 ): Vector4 {
    const c = Math.sqrt( oklab.y * oklab.y + oklab.z * oklab.z );
    const h = Math.atan2( oklab.z, oklab.y );
    return new Vector4( oklab.x, c, h, oklab.w );
  }

  // A radian-based oklch
  public static oklchToOklab( oklch: Vector4 ): Vector4 {
    return new Vector4( oklch.x, oklch.y * Math.cos( oklch.z ), oklch.y * Math.sin( oklch.z ), oklch.w );
  }

  public static linearToOklch( linear: Vector4 ): Vector4 {
    return RenderColor.oklabToOklch( RenderColor.linearToOklab( linear ) );
  }

  // TODO: consistent "linear" naming? (means linear SRGB here)
  public static oklchToLinear( oklch: Vector4 ): Vector4 {
    return RenderColor.oklabToLinear( RenderColor.oklchToOklab( oklch ) );
  }

  public static linearDisplayP3ToOklch( linear: Vector4 ): Vector4 {
    return RenderColor.linearToOklch( RenderColor.linearDisplayP3ToLinear( linear ) );
  }

  public static oklchToLinearDisplayP3( oklch: Vector4 ): Vector4 {
    return RenderColor.linearToLinearDisplayP3( RenderColor.oklchToLinear( oklch ) );
  }

  public static convert( color: Vector4, fromSpace: RenderColorSpace, toSpace: RenderColorSpace ): Vector4 {
    if ( fromSpace === toSpace ) {
      return color;
    }

    if ( assert ) {
      // If we add more, add in the conversions here
      const spaces = [
        RenderColorSpace.XYZ,
        RenderColorSpace.xyY,
        RenderColorSpace.sRGB,
        RenderColorSpace.premultipliedSRGB,
        RenderColorSpace.linearSRGB,
        RenderColorSpace.premultipliedLinearSRGB,
        RenderColorSpace.displayP3,
        RenderColorSpace.premultipliedDisplayP3,
        RenderColorSpace.linearDisplayP3,
        RenderColorSpace.premultipliedLinearDisplayP3,
        RenderColorSpace.oklab,
        RenderColorSpace.premultipliedOklab
      ];

      assert( spaces.includes( fromSpace ) );
      assert( spaces.includes( toSpace ) );
    }

    if ( fromSpace.name === toSpace.name ) {
      if ( fromSpace.isLinear === toSpace.isLinear ) {
        // Just a premultiply change!
        return fromSpace.isPremultiplied ? RenderColor.unpremultiply( color ) : RenderColor.premultiply( color );
      }
      else {
        // We're different in linearity!
        if ( fromSpace.isPremultiplied ) {
          color = RenderColor.unpremultiply( color );
        }
        if ( fromSpace.name === 'srgb' || fromSpace.name === 'display-p3' ) {
          // sRGB transfer function
          color = fromSpace.isLinear ? RenderColor.linearToSRGB( color ) : RenderColor.sRGBToLinear( color );
        }
        if ( toSpace.isPremultiplied ) {
          color = RenderColor.premultiply( color );
        }
        return color;
      }
    }
    else {
      // essentially, we'll convert to linear sRGB and back

      if ( fromSpace.isPremultiplied ) {
        color = RenderColor.unpremultiply( color );
      }

      if ( fromSpace === RenderColorSpace.xyY ) {
        color = color.y === 0 ? new Vector4( 0, 0, 0, color.w ) : new Vector4(
          // TODO: separate out into a function
          color.x * color.z / color.y,
          color.z,
          ( 1 - color.x - color.y ) * color.z / color.y,
          color.w
        );
      }
      if ( fromSpace === RenderColorSpace.xyY || fromSpace === RenderColorSpace.XYZ ) {
        color = RenderColor.multiplyMatrixTimesColor( RenderColor.XYZTosRGBMatrix, color );
      }
      if (
        fromSpace === RenderColorSpace.sRGB ||
        fromSpace === RenderColorSpace.premultipliedSRGB ||
        fromSpace === RenderColorSpace.displayP3 ||
        fromSpace === RenderColorSpace.premultipliedDisplayP3
      ) {
        color = RenderColor.sRGBToLinear( color );
      }
      if ( fromSpace === RenderColorSpace.displayP3 || fromSpace === RenderColorSpace.premultipliedDisplayP3 ) {
        color = RenderColor.linearDisplayP3ToLinear( color );
      }
      if ( fromSpace === RenderColorSpace.oklab || fromSpace === RenderColorSpace.premultipliedOklab ) {
        color = RenderColor.oklabToLinear( color );
      }

      // Now reverse the process, but for the other color space
      if ( toSpace === RenderColorSpace.oklab || toSpace === RenderColorSpace.premultipliedOklab ) {
        color = RenderColor.linearToOklab( color );
      }
      if ( toSpace === RenderColorSpace.displayP3 || toSpace === RenderColorSpace.premultipliedDisplayP3 ) {
        color = RenderColor.linearToLinearDisplayP3( color );
      }
      if (
        toSpace === RenderColorSpace.sRGB ||
        toSpace === RenderColorSpace.premultipliedSRGB ||
        toSpace === RenderColorSpace.displayP3 ||
        toSpace === RenderColorSpace.premultipliedDisplayP3
      ) {
        color = RenderColor.linearToSRGB( color );
      }
      if ( toSpace === RenderColorSpace.xyY || toSpace === RenderColorSpace.XYZ ) {
        color = RenderColor.multiplyMatrixTimesColor( RenderColor.sRGBToXYZMatrix, color );
      }
      if ( toSpace === RenderColorSpace.xyY ) {
        color = ( color.x + color.y + color.z === 0 ) ? new Vector4(
          // TODO: white point change to the other functions, I think we have some of this duplicated.
          // TODO: separate out into a function
          // using white point for D65
          sRGBWhiteChromaticity.x,
          sRGBWhiteChromaticity.y,
          0,
          color.w
        ) : new Vector4(
          color.x / ( color.x + color.y + color.z ),
          color.y / ( color.x + color.y + color.z ),
          color.y,
          color.w
        );
      }

      if ( toSpace.isPremultiplied ) {
        color = RenderColor.premultiply( color );
      }

      return color;
    }
  }

  // ONLY remaps the r,g,b parts, not alpha
  public static isColorInRange( color: Vector4 ): boolean {
    return color.x >= 0 && color.x <= 1 &&
           color.y >= 0 && color.y <= 1 &&
           color.z >= 0 && color.z <= 1;
  }

  /**
   * Relative colorimetric mapping. We could add more of a perceptual intent, but this is a good start.
   *
   * Modeled after https://drafts.csswg.org/css-color-4/#binsearch
   */
  public static gamutMapColor( color: Vector4, toOklab: ( c: Vector4 ) => Vector4, fromOklab: ( c: Vector4 ) => Vector4 ): Vector4 {
    if ( RenderColor.isColorInRange( color ) ) {
      return color;
    }

    const oklab = toOklab( color ).copy(); // we'll mutate it
    if ( oklab.x <= 0 ) {
      return new Vector4( 0, 0, 0, color.w );
    }
    else if ( oklab.x >= 1 ) {
      return new Vector4( 1, 1, 1, color.w );
    }

    const chroma = new Vector2( oklab.y, oklab.z );

    // Bisection of chroma
    let lowChroma = 0;
    let highChroma = 1;
    let clipped: Vector4 | null = null;

    while ( highChroma - lowChroma > 1e-4 ) {
      const testChroma = ( lowChroma + highChroma ) * 0.5;
      oklab.y = chroma.x * testChroma;
      oklab.z = chroma.y * testChroma;

      const mapped = fromOklab( oklab );
      const isInColorRange = RenderColor.isColorInRange( mapped );
      clipped = isInColorRange ? mapped : new Vector4(
        Math.max( 0, Math.min( 1, mapped.x ) ),
        Math.max( 0, Math.min( 1, mapped.y ) ),
        Math.max( 0, Math.min( 1, mapped.z ) ),
        mapped.w
      );

      // JND (just noticeable difference) of 0.02, per the spec at https://drafts.csswg.org/css-color/#css-gamut-mapping
      if ( isInColorRange || ( toOklab( clipped ).distance( oklab ) <= 0.02 ) ) {
        lowChroma = testChroma;
      }
      else {
        highChroma = testChroma;
      }
    }

    const potentialResult = fromOklab( oklab );
    if ( RenderColor.isColorInRange( potentialResult ) ) {
      return potentialResult;
    }
    else {
      assert && assert( clipped );
      return clipped!;
    }
  }

  public static gamutMapLinearSRGB( color: Vector4 ): Vector4 {
    return RenderColor.gamutMapColor( color, RenderColor.linearToOklab, RenderColor.oklabToLinear );
  }

  public static gamutMapLinearDisplayP3( color: Vector4 ): Vector4 {
    return RenderColor.gamutMapColor( color, RenderColor.linearDisplayP3ToOklab, RenderColor.oklabToLinearDisplayP3 );
  }

  public static gamutMapSRGB( color: Vector4 ): Vector4 {
    if ( RenderColor.isColorInRange( color ) ) {
      return color;
    }
    else {
      return RenderColor.linearToSRGB( RenderColor.gamutMapLinearSRGB( RenderColor.sRGBToLinear( color ) ) );
    }
  }

  public static gamutMapDisplayP3( color: Vector4 ): Vector4 {
    if ( RenderColor.isColorInRange( color ) ) {
      return color;
    }
    else {
      return RenderColor.linearDisplayP3ToDisplayP3( RenderColor.gamutMapLinearDisplayP3( RenderColor.displayP3ToLinearDisplayP3( color ) ) );
    }
  }

  /**
   * OUTPUTS unpremultiplied sRGB, with a valid alpha value
   */
  public static gamutMapPremultipliedSRGB( color: Vector4 ): Vector4 {
    if ( color.w <= 1e-8 ) {
      return Vector4.ZERO;
    }

    const mapped = RenderColor.gamutMapSRGB( RenderColor.unpremultiply( color ) );

    if ( color.w > 1 ) {
      return new Vector4( mapped.x, mapped.y, mapped.z, 1 );
    }
    else {
      return mapped;
    }
  }

  /**
   * OUTPUTS unpremultiplied Display P3, with a valid alpha value
   */
  public static gamutMapPremultipliedDisplayP3( color: Vector4 ): Vector4 {
    if ( color.w <= 1e-8 ) {
      return Vector4.ZERO;
    }

    const mapped = RenderColor.gamutMapDisplayP3( RenderColor.unpremultiply( color ) );

    if ( color.w > 1 ) {
      return new Vector4( mapped.x, mapped.y, mapped.z, 1 );
    }
    else {
      return mapped;
    }
  }

  public static oklabToLinearDisplayP3( oklab: Vector4 ): Vector4 {
    return RenderColor.linearToLinearDisplayP3( RenderColor.oklabToLinear( oklab ) );
  }

  public static linearDisplayP3ToOklab( linearP3: Vector4 ): Vector4 {
    return RenderColor.linearToOklab( RenderColor.linearDisplayP3ToLinear( linearP3 ) );
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

  public static xyYToXYZ( xyY: Vector3 ): Vector3 {
    /**
     * ( color.x + color.y + color.z === 0 ) ? new Vector4(
          // TODO: white point change to the other functions, I think we have some of this duplicated.
          // TODO: separate out into a function
          // using white point for D65
          sRGBWhiteChromaticity.x,
          sRGBWhiteChromaticity.y,
          0,
          color.w
        )
     */
    return new Vector3(
      xyY.x * xyY.z / xyY.y,
      xyY.z,
      ( 1 - xyY.x - xyY.y ) * xyY.z / xyY.y
    );
  }

  public static xyToXYZ( xy: Vector2 ): Vector3 {
    return RenderColor.xyYToXYZ( new Vector3( xy.x, xy.y, 1 ) );
  }

  public static xyzToLinear( xyz: Vector4 ): Vector4 {
    return RenderColor.multiplyMatrixTimesColor( RenderColor.XYZTosRGBMatrix, xyz );
  }

  public static linearToXYZ( linear: Vector4 ): Vector4 {
    return RenderColor.multiplyMatrixTimesColor( RenderColor.sRGBToXYZMatrix, linear );
  }

  public static getMatrixRGBToXYZ(
    redChromaticity: Vector2,
    greenChromaticity: Vector2,
    blueChromaticity: Vector2,
    whiteChromaticity: Vector2
  ): Matrix3 {

    // Based on https://mina86.com/2019/srgb-xyz-matrix/, could not get Bruce Lindbloom's formulas to work.

    const whiteXYZ = RenderColor.xyToXYZ( whiteChromaticity );
    const redPrimeXYZ = RenderColor.xyToXYZ( redChromaticity );
    const greenPrimeXYZ = RenderColor.xyToXYZ( greenChromaticity );
    const bluePrimeXYZ = RenderColor.xyToXYZ( blueChromaticity );
    const matrixPrime = Matrix3.rowMajor(
      redPrimeXYZ.x, greenPrimeXYZ.x, bluePrimeXYZ.x,
      redPrimeXYZ.y, greenPrimeXYZ.y, bluePrimeXYZ.y,
      redPrimeXYZ.z, greenPrimeXYZ.z, bluePrimeXYZ.z
    );
    const rgbY = matrixPrime.inverted().timesVector3( whiteXYZ );
    return matrixPrime.timesMatrix( Matrix3.rowMajor(
      rgbY.x, 0, 0,
      0, rgbY.y, 0,
      0, 0, rgbY.z
    ) );
  }

  public static sRGBToXYZMatrix = RenderColor.getMatrixRGBToXYZ(
    sRGBRedChromaticity,
    sRGBGreenChromaticity,
    sRGBBlueChromaticity,
    sRGBWhiteChromaticity
  );

  public static XYZTosRGBMatrix = RenderColor.sRGBToXYZMatrix.inverted();

  public static displayP3ToXYZMatrix = RenderColor.getMatrixRGBToXYZ(
    dciP3RedChromaticity,
    dciP3GreenChromaticity,
    dciP3BlueChromaticity,
    sRGBWhiteChromaticity
  );

  public static XYZToDisplayP3Matrix = RenderColor.displayP3ToXYZMatrix.inverted();

  public static sRGBToDisplayP3Matrix = RenderColor.XYZToDisplayP3Matrix.timesMatrix( RenderColor.sRGBToXYZMatrix );

  public static displayP3TosRGBMatrix = RenderColor.sRGBToDisplayP3Matrix.inverted();

  public static canvasSupportsDisplayP3(): boolean {
    const canvas = document.createElement( 'canvas' );
    try {
      // Errors might be thrown if the option is supported but system requirements are not met
      const context = canvas.getContext( '2d', { colorSpace: 'display-p3' } );
      return context!.getContextAttributes().colorSpace === 'display-p3';
    }
    catch{
      return false;
    }
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
