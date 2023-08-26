// Copyright 2023, University of Colorado Boulder

/**
 * RenderProgram to convert linear sRGB => sRGB
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { ClippableFace, constantTrue, RenderColor, RenderPath, RenderProgram, scenery, SerializedRenderProgram } from '../../../imports.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Vector4 from '../../../../../dot/js/Vector4.js';

export default class RenderLinearSRGBToSRGB extends RenderProgram {
  public constructor(
    public readonly program: RenderProgram
  ) {
    super();
  }

  public override getName(): string {
    return 'RenderLinearSRGBToSRGB';
  }

  public override getChildren(): RenderProgram[] {
    return [ this.program ];
  }

  public override withChildren( children: RenderProgram[] ): RenderLinearSRGBToSRGB {
    assert && assert( children.length === 1 );
    return new RenderLinearSRGBToSRGB( children[ 0 ] );
  }

  public override simplify( pathTest: ( renderPath: RenderPath ) => boolean = constantTrue ): RenderProgram {
    const program = this.program.simplify( pathTest );

    if ( program.isFullyTransparent() ) {
      return RenderColor.TRANSPARENT;
    }

    // Now we're "inside" our path
    if ( program instanceof RenderColor ) {
      return new RenderColor( RenderColor.linearToSRGB( program.color ) );
    }
    else {
      return new RenderLinearSRGBToSRGB( program );
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

    return RenderColor.linearToSRGB( source );
  }

  public override serialize(): SerializedRenderLinearSRGBToSRGB {
    return {
      type: 'RenderLinearSRGBToSRGB',
      program: this.program.serialize()
    };
  }

  public static override deserialize( obj: SerializedRenderLinearSRGBToSRGB ): RenderLinearSRGBToSRGB {
    return new RenderLinearSRGBToSRGB( RenderProgram.deserialize( obj.program ) );
  }
}

scenery.register( 'RenderLinearSRGBToSRGB', RenderLinearSRGBToSRGB );

export type SerializedRenderLinearSRGBToSRGB = {
  type: 'RenderLinearSRGBToSRGB';
  program: SerializedRenderProgram;
};
