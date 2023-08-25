// Copyright 2023, University of Colorado Boulder

/**
 * A gradient stop for linear/radial gradients
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { ClippableFace, constantTrue, RenderColor, RenderPath, RenderProgram, scenery, SerializedRenderProgram } from '../../../imports.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Vector4 from '../../../../../dot/js/Vector4.js';

export default class RenderGradientStop {
  public constructor( public readonly ratio: number, public readonly program: RenderProgram ) {
    assert && assert( ratio >= 0 && ratio <= 1 );
  }

  public static evaluate(
    face: ClippableFace | null, // if null, it is fully covered
    area: number,
    centroid: Vector2,
    minX: number,
    minY: number,
    maxX: number,
    maxY: number,
    stops: RenderGradientStop[],
    t: number,
    pathTest: ( renderPath: RenderPath ) => boolean = constantTrue
  ): Vector4 {
    let i = -1;
    while ( i < stops.length - 1 && stops[ i + 1 ].ratio < t ) {
      i++;
    }
    if ( i === -1 ) {
      return stops[ 0 ].program.evaluate( face, area, centroid, minX, minY, maxX, maxY, pathTest );
    }
    else if ( i === stops.length - 1 ) {
      return stops[ i ].program.evaluate( face, area, centroid, minX, minY, maxX, maxY, pathTest );
    }
    else {
      const before = stops[ i ];
      const after = stops[ i + 1 ];
      const ratio = ( t - before.ratio ) / ( after.ratio - before.ratio );

      const beforeColor = before.program.evaluate( face, area, centroid, minX, minY, maxX, maxY, pathTest );
      const afterColor = after.program.evaluate( face, area, centroid, minX, minY, maxX, maxY, pathTest );

      return RenderColor.ratioBlend( beforeColor, afterColor, ratio );
    }
  }

  public serialize(): SerializedRenderGradientStop {
    return {
      ratio: this.ratio,
      program: this.program.serialize()
    };
  }

  public static deserialize( obj: SerializedRenderGradientStop ): RenderGradientStop {
    return new RenderGradientStop( obj.ratio, RenderProgram.deserialize( obj.program ) );
  }
}

scenery.register( 'RenderGradientStop', RenderGradientStop );

export type SerializedRenderGradientStop = {
  ratio: number;
  program: SerializedRenderProgram;
};
