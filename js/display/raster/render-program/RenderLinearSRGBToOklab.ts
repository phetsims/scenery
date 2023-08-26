// Copyright 2023, University of Colorado Boulder

/**
 * RenderProgram to convert linear sRGB => Oklab
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { ClippableFace, constantTrue, RenderColor, RenderPath, RenderProgram, scenery, SerializedRenderProgram } from '../../../imports.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Vector4 from '../../../../../dot/js/Vector4.js';

export default class RenderLinearSRGBToOklab extends RenderProgram {
  public constructor(
    public readonly program: RenderProgram
  ) {
    super();
  }

  public override getName(): string {
    return 'RenderLinearSRGBToOklab';
  }

  public override getChildren(): RenderProgram[] {
    return [ this.program ];
  }

  public override withChildren( children: RenderProgram[] ): RenderLinearSRGBToOklab {
    assert && assert( children.length === 1 );
    return new RenderLinearSRGBToOklab( children[ 0 ] );
  }

  public override simplify( pathTest: ( renderPath: RenderPath ) => boolean = constantTrue ): RenderProgram {
    const program = this.program.simplify( pathTest );

    if ( program.isFullyTransparent() ) {
      return RenderColor.TRANSPARENT;
    }

    // Now we're "inside" our path
    if ( program instanceof RenderColor ) {
      return new RenderColor( RenderColor.linearToOklab( program.color ) );
    }
    else {
      return new RenderLinearSRGBToOklab( program );
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
    const source = this.program.evaluate( face, area, centroid, minX, minY, maxX, maxY, pathTest );

    return RenderColor.linearToOklab( source );
  }

  public override serialize(): SerializedRenderLinearSRGBToOklab {
    return {
      type: 'RenderLinearSRGBToOklab',
      program: this.program.serialize()
    };
  }

  public static override deserialize( obj: SerializedRenderLinearSRGBToOklab ): RenderLinearSRGBToOklab {
    return new RenderLinearSRGBToOklab( RenderProgram.deserialize( obj.program ) );
  }
}

scenery.register( 'RenderLinearSRGBToOklab', RenderLinearSRGBToOklab );

export type SerializedRenderLinearSRGBToOklab = {
  type: 'RenderLinearSRGBToOklab';
  program: SerializedRenderProgram;
};
