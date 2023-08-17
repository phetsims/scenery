// Copyright 2023, University of Colorado Boulder

/**
 * A gradient stop for linear/radial gradients
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { constantTrue, RenderColor, RenderColorSpace, RenderPath, RenderProgram, scenery } from '../../../imports.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Vector4 from '../../../../../dot/js/Vector4.js';

export default class RenderGradientStop {
  public constructor( public readonly ratio: number, public readonly program: RenderProgram ) {
    assert && assert( ratio >= 0 && ratio <= 1 );
  }

  public static evaluate(
    point: Vector2,
    stops: RenderGradientStop[],
    t: number,
    colorSpace: RenderColorSpace,
    pathTest: ( renderPath: RenderPath ) => boolean = constantTrue
  ): Vector4 {
    let i = -1;
    while ( i < stops.length - 1 && stops[ i + 1 ].ratio < t ) {
      i++;
    }
    if ( i === -1 ) {
      return stops[ 0 ].program.evaluate( point, pathTest );
    }
    else if ( i === stops.length - 1 ) {
      return stops[ i ].program.evaluate( point, pathTest );
    }
    else {
      const before = stops[ i ];
      const after = stops[ i + 1 ];
      const ratio = ( t - before.ratio ) / ( after.ratio - before.ratio );

      const beforeColor = before.program.evaluate( point, pathTest );
      const afterColor = after.program.evaluate( point, pathTest );

      return RenderColor.ratioBlend( beforeColor, afterColor, ratio, colorSpace );
    }
  }
}

scenery.register( 'RenderGradientStop', RenderGradientStop );
