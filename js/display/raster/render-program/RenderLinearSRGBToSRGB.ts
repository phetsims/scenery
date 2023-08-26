// Copyright 2023, University of Colorado Boulder

/**
 * RenderProgram to convert linear sRGB => sRGB
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { RenderColor, RenderColorSpaceConversion, RenderProgram, scenery } from '../../../imports.js';

export default class RenderLinearSRGBToSRGB extends RenderColorSpaceConversion {
  public constructor(
    program: RenderProgram
  ) {
    super( program, RenderColor.linearToSRGB );
  }

  public override getName(): string {
    return 'RenderLinearSRGBToSRGB';
  }

  public override withChildren( children: RenderProgram[] ): RenderLinearSRGBToSRGB {
    assert && assert( children.length === 1 );
    return new RenderLinearSRGBToSRGB( children[ 0 ] );
  }
}

scenery.register( 'RenderLinearSRGBToSRGB', RenderLinearSRGBToSRGB );
