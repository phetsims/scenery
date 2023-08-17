// Copyright 2023, University of Colorado Boulder

/**
 * RenderProgram for an image
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { constantTrue, RenderColor, RenderExtend, RenderImageable, RenderPath, RenderPathProgram, RenderProgram, scenery } from '../../../imports.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Matrix3 from '../../../../../dot/js/Matrix3.js';
import Vector4 from '../../../../../dot/js/Vector4.js';
import Utils from '../../../../../dot/js/Utils.js';
import RenderColorSpace from './RenderColorSpace.js';

export default class RenderImage extends RenderPathProgram {
  public constructor(
    path: RenderPath | null,
    public readonly transform: Matrix3,
    public readonly image: RenderImageable,
    public readonly extendX: RenderExtend,
    public readonly extendY: RenderExtend
  ) {
    super( path );
  }

  public override transformed( transform: Matrix3 ): RenderProgram {
    return new RenderImage( this.getTransformedPath( transform ), transform.timesMatrix( this.transform ), this.image, this.extendX, this.extendY );
  }

  public override equals( other: RenderProgram ): boolean {
    if ( this === other ) { return true; }
    return super.equals( other ) &&
      other instanceof RenderImage &&
      this.transform.equals( other.transform ) &&
      this.image === other.image &&
      this.extendX === other.extendX &&
      this.extendY === other.extendY;
  }

  public override isFullyTransparent(): boolean {
    return false;
  }

  public override isFullyOpaque(): boolean {
    return false;
  }

  public override replace( callback: ( program: RenderProgram ) => RenderProgram | null ): RenderProgram {
    const replaced = callback( this );
    if ( replaced ) {
      return replaced;
    }
    else {
      return new RenderImage( this.path, this.transform, this.image, this.extendX, this.extendY );
    }
  }

  public override simplify( pathTest: ( renderPath: RenderPath ) => boolean = constantTrue ): RenderProgram {
    if ( this.isInPath( pathTest ) ) {
      return new RenderImage( null, this.transform, this.image, this.extendX, this.extendY );
    }
    else {
      return RenderColor.TRANSPARENT;
    }
  }

  public override evaluate( point: Vector2, pathTest: ( renderPath: RenderPath ) => boolean = constantTrue ): Vector4 {
    if ( !this.isInPath( pathTest ) ) {
      return Vector4.ZERO;
    }

    const localPoint = this.transform.inverted().timesVector2( point );
    const tx = localPoint.x / this.image.width;
    const ty = localPoint.y / this.image.height;
    const mappedX = RenderImage.extend( this.extendX, tx );
    const mappedY = RenderImage.extend( this.extendY, ty );

    // TODO: better sampling
    const color = this.image.evaluate( Math.floor( mappedX * this.image.width ), Math.floor( mappedY * this.image.height ) );

    switch( this.image.colorSpace ) {
      case RenderColorSpace.LinearUnpremultipliedSRGB:
        return color;
      case RenderColorSpace.SRGB:
        return RenderColor.premultiply( RenderColor.sRGBToLinear( color ) );
      case RenderColorSpace.Oklab:
        return RenderColor.premultiply( RenderColor.oklabToLinear( color ) );
      default:
        throw new Error( 'unknown color space: ' + this.image.colorSpace );
    }
  }

  public override toRecursiveString( indent: string ): string {
    return `${indent}RenderImage (${this.path ? this.path.id : 'null'})`;
  }

  public static extend( extend: RenderExtend, t: number ): number {
    switch( extend ) {
      case RenderExtend.Pad:
        return Utils.clamp( t, 0, 1 );
      case RenderExtend.Repeat:
        return t - Math.floor( t );
      case RenderExtend.Reflect:
        return Math.abs( t - 2.0 * Utils.roundSymmetric( 0.5 * t ) );
        // return ( Math.floor( t ) % 2 === 0 ? t : 1 - t ) - Math.floor( t );
      default:
        throw new Error( 'Unknown RenderExtend' );
    }
  }

  // Integer version of extend_mode.
  // Given size=4, provide the following patterns:
  //
  // input:  -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
  //
  // pad:     0,  0,  0,  0,  0,  0, 0, 1, 2, 3, 3, 3, 3, 3, 3, 3
  // repeat:  2,  3,  0,  1,  2,  3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1
  // reflect: 2,  3,  3,  2,  1,  0, 0, 1, 2, 3, 3, 2, 1, 0, 0, 1
  public static extendInteger( i: number, size: number, extend: RenderExtend ): number {
    switch( extend ) {
      case RenderExtend.Pad: {
        return Utils.clamp( i, 0, size - 1 );
      }
      case RenderExtend.Repeat: {
        if ( i >= 0 ) {
          return i % size;
        }
        else {
          return size - ( ( -i - 1 ) % size ) - 1;
        }
      }
      case RenderExtend.Reflect: {
        // easier to convert both to positive (with a repeat offset)
        const positiveI = i < 0 ? -i - 1 : i;

        const section = positiveI % ( size * 2 );
        if ( section < size ) {
          return section;
        }
        else {
          return 2 * size - section - 1;
        }
      }
      default: {
        throw new Error( 'Unknown RenderExtend' );
      }
    }
  }
}

scenery.register( 'RenderImage', RenderImage );
