// Copyright 2023, University of Colorado Boulder

/**
 * RenderProgram to convert sRGB => linear sRGB
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { ClippableFace, RenderColor, RenderColorSpace, RenderLinearDisplayP3ToLinearSRGB, RenderLinearSRGBToLinearDisplayP3, RenderLinearSRGBToOklab, RenderLinearSRGBToSRGB, RenderOklabToLinearSRGB, RenderPremultiply, RenderProgram, RenderSRGBToLinearSRGB, RenderUnary, RenderUnpremultiply, scenery, SerializedRenderProgram } from '../../../imports.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Vector4 from '../../../../../dot/js/Vector4.js';

export default abstract class RenderColorSpaceConversion extends RenderUnary {
  protected constructor(
    program: RenderProgram,
    public readonly convert: ( color: Vector4 ) => Vector4
  ) {
    super( program );
  }

  public static convert( renderProgram: RenderProgram, fromSpace: RenderColorSpace, toSpace: RenderColorSpace ): RenderProgram {
    if ( fromSpace === toSpace ) {
      return renderProgram;
    }

    if ( assert ) {
      // If we add more, add in the conversions here
      // NOTE: not really worrying about XYZ or xyY
      const spaces = [
        RenderColorSpace.sRGB,
        RenderColorSpace.premultipliedSRGB,
        RenderColorSpace.linearSRGB,
        RenderColorSpace.premultipliedLinearSRGB,
        RenderColorSpace.displayP3,
        RenderColorSpace.premultipliedDisplayP3,
        RenderColorSpace.linearDisplayP3,
        RenderColorSpace.premultipliedLinearDisplayP3,
        RenderColorSpace.oklab,
        RenderColorSpace.premultipliedOklab,
        RenderColorSpace.linearOklab,
        RenderColorSpace.premultipliedLinearOklab
      ];

      assert( spaces.includes( fromSpace ) );
      assert( spaces.includes( toSpace ) );
    }

    if ( fromSpace.name === toSpace.name ) {
      if ( fromSpace.isLinear === toSpace.isLinear ) {
        // Just a premultiply change!
        return fromSpace.isPremultiplied ? new RenderUnpremultiply( renderProgram ) : new RenderPremultiply( renderProgram );
      }
      else {
        // We're different in linearity!
        if ( fromSpace.isPremultiplied ) {
          renderProgram = new RenderUnpremultiply( renderProgram );
        }
        if ( fromSpace.name === 'srgb' || fromSpace.name === 'display-p3' ) {
          // sRGB transfer function
          renderProgram = fromSpace.isLinear ? new RenderLinearSRGBToSRGB( renderProgram ) : new RenderSRGBToLinearSRGB( renderProgram );
        }
        if ( toSpace.isPremultiplied ) {
          renderProgram = new RenderPremultiply( renderProgram );
        }
        return renderProgram;
      }
    }
    else {
      // essentially, we'll convert to linear sRGB and back

      if ( fromSpace.isPremultiplied ) {
        renderProgram = new RenderUnpremultiply( renderProgram );
      }

      if (
        fromSpace === RenderColorSpace.sRGB ||
        fromSpace === RenderColorSpace.premultipliedSRGB ||
        fromSpace === RenderColorSpace.displayP3 ||
        fromSpace === RenderColorSpace.premultipliedDisplayP3
      ) {
        renderProgram = new RenderSRGBToLinearSRGB( renderProgram );
      }
      if ( fromSpace === RenderColorSpace.displayP3 || fromSpace === RenderColorSpace.premultipliedDisplayP3 ) {
        renderProgram = new RenderLinearDisplayP3ToLinearSRGB( renderProgram );
      }
      if ( fromSpace === RenderColorSpace.oklab || fromSpace === RenderColorSpace.premultipliedOklab ) {
        renderProgram = new RenderOklabToLinearSRGB( renderProgram );
      }

      // Now reverse the process, but for the other color space
      if ( toSpace === RenderColorSpace.oklab || toSpace === RenderColorSpace.premultipliedOklab ) {
        renderProgram = new RenderLinearSRGBToOklab( renderProgram );
      }
      if ( toSpace === RenderColorSpace.displayP3 || toSpace === RenderColorSpace.premultipliedDisplayP3 ) {
        renderProgram = new RenderLinearSRGBToLinearDisplayP3( renderProgram );
      }
      if (
        toSpace === RenderColorSpace.sRGB ||
        toSpace === RenderColorSpace.premultipliedSRGB ||
        toSpace === RenderColorSpace.displayP3 ||
        toSpace === RenderColorSpace.premultipliedDisplayP3
      ) {
        renderProgram = new RenderLinearSRGBToSRGB( renderProgram );
      }

      if ( toSpace.isPremultiplied ) {
        renderProgram = new RenderPremultiply( renderProgram );
      }

      return renderProgram;
    }
  }

  public override simplified(): RenderProgram {
    const program = this.program.simplified();

    if ( program.isFullyTransparent() ) {
      return RenderColor.TRANSPARENT;
    }

    if ( program instanceof RenderColor ) {
      return new RenderColor( this.convert( program.color ) );
    }
    else {
      return this.withChildren( [ program ] );
    }
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
    const source = this.program.evaluate( face, area, centroid, minX, minY, maxX, maxY );

    return this.convert( source );
  }

  public static displayP3ToSRGB( renderProgram: RenderProgram ): RenderProgram {
    return new RenderLinearSRGBToSRGB( new RenderLinearDisplayP3ToLinearSRGB( new RenderSRGBToLinearSRGB( renderProgram ) ) );
  }

  public static sRGBToDisplayP3( renderProgram: RenderProgram ): RenderProgram {
    return new RenderLinearSRGBToSRGB( new RenderLinearSRGBToLinearDisplayP3( new RenderSRGBToLinearSRGB( renderProgram ) ) );
  }

  public static displayP3ToLinearSRGB( renderProgram: RenderProgram ): RenderProgram {
    return new RenderLinearDisplayP3ToLinearSRGB( new RenderSRGBToLinearSRGB( renderProgram ) );
  }

  public static linearSRGBToDisplayP3( renderProgram: RenderProgram ): RenderProgram {
    return new RenderLinearSRGBToSRGB( new RenderLinearSRGBToLinearDisplayP3( renderProgram ) );
  }

  public static oklabToSRGB( renderProgram: RenderProgram ): RenderProgram {
    return new RenderLinearSRGBToSRGB( new RenderOklabToLinearSRGB( renderProgram ) );
  }

  public static sRGBToOklab( renderProgram: RenderProgram ): RenderProgram {
    return new RenderLinearSRGBToOklab( new RenderSRGBToLinearSRGB( renderProgram ) );
  }

  public static oklabToDisplayP3( renderProgram: RenderProgram ): RenderProgram {
    return new RenderLinearSRGBToSRGB( new RenderLinearSRGBToLinearDisplayP3( new RenderOklabToLinearSRGB( renderProgram ) ) );
  }

  public static displayP3ToOklab( renderProgram: RenderProgram ): RenderProgram {
    return new RenderLinearSRGBToOklab( new RenderLinearDisplayP3ToLinearSRGB( new RenderSRGBToLinearSRGB( renderProgram ) ) );
  }

  public static premulSRGBToPremulLinearSRGB( renderProgram: RenderProgram ): RenderProgram {
    return new RenderPremultiply( new RenderSRGBToLinearSRGB( new RenderUnpremultiply( renderProgram ) ) );
  }

  public static premulLinearSRGBToPremulSRGB( renderProgram: RenderProgram ): RenderProgram {
    return new RenderPremultiply( new RenderLinearSRGBToSRGB( new RenderUnpremultiply( renderProgram ) ) );
  }

  public static premulLinearSRGBToPremulDisplayP3( renderProgram: RenderProgram ): RenderProgram {
    return new RenderPremultiply( RenderColorSpaceConversion.linearSRGBToDisplayP3( new RenderUnpremultiply( renderProgram ) ) );
  }

  public static premulDisplayP3ToPremulLinearSRGB( renderProgram: RenderProgram ): RenderProgram {
    return new RenderPremultiply( RenderColorSpaceConversion.displayP3ToLinearSRGB( new RenderUnpremultiply( renderProgram ) ) );
  }

  public static premulDisplayP3ToPremulSRGB( renderProgram: RenderProgram ): RenderProgram {
    return new RenderPremultiply( RenderColorSpaceConversion.displayP3ToSRGB( new RenderUnpremultiply( renderProgram ) ) );
  }

  public static premulSRGBToPremulDisplayP3( renderProgram: RenderProgram ): RenderProgram {
    return new RenderPremultiply( RenderColorSpaceConversion.sRGBToDisplayP3( new RenderUnpremultiply( renderProgram ) ) );
  }

  public static premulOklabToPremulSRGB( renderProgram: RenderProgram ): RenderProgram {
    return new RenderPremultiply( RenderColorSpaceConversion.oklabToSRGB( new RenderUnpremultiply( renderProgram ) ) );
  }

  public static premulSRGBToPremulOklab( renderProgram: RenderProgram ): RenderProgram {
    return new RenderPremultiply( RenderColorSpaceConversion.sRGBToOklab( new RenderUnpremultiply( renderProgram ) ) );
  }

  public static premulOklabToPremulDisplayP3( renderProgram: RenderProgram ): RenderProgram {
    return new RenderPremultiply( RenderColorSpaceConversion.oklabToDisplayP3( new RenderUnpremultiply( renderProgram ) ) );
  }

  public static premulDisplayP3ToPremulOklab( renderProgram: RenderProgram ): RenderProgram {
    return new RenderPremultiply( RenderColorSpaceConversion.displayP3ToOklab( new RenderUnpremultiply( renderProgram ) ) );
  }

  public override serialize(): SerializedRenderColorSpaceConversion {
    return {
      type: 'RenderColorSpaceConversion',
      subtype: this.getName(),
      program: this.program.serialize()
    };
  }

  public static override deserialize( obj: SerializedRenderColorSpaceConversion ): RenderColorSpaceConversion {
    const program = RenderProgram.deserialize( obj.program );

    if ( obj.subtype === 'RenderPremultiply' ) {
      return new RenderPremultiply( program );
    }
    else if ( obj.subtype === 'RenderUnpremultiply' ) {
      return new RenderUnpremultiply( program );
    }
    else if ( obj.subtype === 'RenderLinearSRGBToOklab' ) {
      return new RenderLinearSRGBToOklab( program );
    }
    else if ( obj.subtype === 'RenderLinearSRGBToSRGB' ) {
      return new RenderLinearSRGBToSRGB( program );
    }
    else if ( obj.subtype === 'RenderOklabToLinearSRGB' ) {
      return new RenderOklabToLinearSRGB( program );
    }
    else if ( obj.subtype === 'RenderSRGBToLinearSRGB' ) {
      return new RenderSRGBToLinearSRGB( program );
    }
    else if ( obj.subtype === 'RenderLinearDisplayP3ToLinearSRGB' ) {
      return new RenderLinearDisplayP3ToLinearSRGB( program );
    }
    else if ( obj.subtype === 'RenderLinearSRGBToLinearDisplayP3' ) {
      return new RenderLinearSRGBToLinearDisplayP3( program );
    }
    else {
      throw new Error( `Unrecognized subtype: ${obj.subtype}` );
    }
  }
}

scenery.register( 'RenderColorSpaceConversion', RenderColorSpaceConversion );

export type SerializedRenderColorSpaceConversion = {
  type: 'RenderColorSpaceConversion';
  subtype: string;
  program: SerializedRenderProgram;
};
