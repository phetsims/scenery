// Copyright 2023, University of Colorado Boulder

/**
 * RenderProgram to convert sRGB => linear sRGB
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { ClippableFace, RenderColor, RenderLinearDisplayP3ToLinearSRGB, RenderLinearSRGBToLinearDisplayP3, RenderLinearSRGBToOklab, RenderLinearSRGBToSRGB, RenderOklabToLinearSRGB, RenderPremultiply, RenderProgram, RenderSRGBToLinearSRGB, RenderUnary, RenderUnpremultiply, scenery, SerializedRenderProgram } from '../../../imports.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Vector4 from '../../../../../dot/js/Vector4.js';

export default abstract class RenderColorSpaceConversion extends RenderUnary {
  protected constructor(
    program: RenderProgram,
    public readonly convert: ( color: Vector4 ) => Vector4
  ) {
    super( program );
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

  public static oklabToSRGB( renderProgram: RenderProgram ): RenderProgram {
    return new RenderLinearSRGBToSRGB( new RenderOklabToLinearSRGB( renderProgram ) );
  }

  public static sRGBToOklab( renderProgram: RenderProgram ): RenderProgram {
    return new RenderLinearSRGBToOklab( new RenderSRGBToLinearSRGB( renderProgram ) );
  }

  public static premulSRGBToPremulLinearSRGB( renderProgram: RenderProgram ): RenderProgram {
    return new RenderPremultiply( new RenderSRGBToLinearSRGB( new RenderUnpremultiply( renderProgram ) ) );
  }

  public static premulLinearSRGBToPremulSRGB( renderProgram: RenderProgram ): RenderProgram {
    return new RenderPremultiply( new RenderLinearSRGBToSRGB( new RenderUnpremultiply( renderProgram ) ) );
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
