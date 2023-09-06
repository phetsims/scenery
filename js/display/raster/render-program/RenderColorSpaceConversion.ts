// Copyright 2023, University of Colorado Boulder

/**
 * RenderProgram to convert between color spaces. Should not change whether something is transparent or opaque
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { ClippableFace, RenderColor, RenderColorSpace, RenderLinearDisplayP3ToLinearSRGB, RenderLinearSRGBToLinearDisplayP3, RenderLinearSRGBToOklab, RenderLinearSRGBToSRGB, RenderOklabToLinearSRGB, RenderPremultiply, RenderProgram, RenderSRGBToLinearSRGB, RenderUnpremultiply, scenery, SerializedRenderProgram } from '../../../imports.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Vector4 from '../../../../../dot/js/Vector4.js';
import Constructor from '../../../../../phet-core/js/types/Constructor.js';

export default abstract class RenderColorSpaceConversion extends RenderProgram {

  public inverse?: Constructor<RenderColorSpaceConversion>;

  protected constructor(
    public readonly program: RenderProgram,
    public readonly convert: ( color: Vector4 ) => Vector4
  ) {
    super(
      [ program ],
      program.isFullyTransparent,
      program.isFullyOpaque
    );
  }

  // TODO: add a helper on RenderProgram
  public static convert( renderProgram: RenderProgram, fromSpace: RenderColorSpace, toSpace: RenderColorSpace ): RenderProgram {
    if ( fromSpace === toSpace ) {
      return renderProgram;
    }

    if ( fromSpace.isPremultiplied ) {
      renderProgram = new RenderUnpremultiply( renderProgram );
    }
    if ( !fromSpace.isLinear ) {
      renderProgram = fromSpace.toLinearRenderProgram!( renderProgram );
    }
    renderProgram = fromSpace.linearToLinearSRGBRenderProgram!( renderProgram );
    renderProgram = toSpace.linearSRGBToLinearRenderProgram!( renderProgram );
    if ( !toSpace.isLinear ) {
      renderProgram = toSpace.fromLinearRenderProgram!( renderProgram );
    }
    if ( toSpace.isPremultiplied ) {
      renderProgram = new RenderPremultiply( renderProgram );
    }
    return renderProgram.simplified();
  }

  public override getSimplified( children: RenderProgram[] ): RenderProgram | null {
    const program = children[ 0 ];

    if ( program.isFullyTransparent ) {
      return RenderColor.TRANSPARENT;
    }
    else if ( program instanceof RenderColor ) {
      return new RenderColor( this.convert( program.color ) );
    }
    else if ( this.inverse && program instanceof this.inverse ) {
      return program.program;
    }
    else {
      return null;
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
