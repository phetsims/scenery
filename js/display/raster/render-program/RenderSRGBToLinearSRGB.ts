// Copyright 2023, University of Colorado Boulder

/**
 * RenderProgram to convert sRGB => linear sRGB
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { RenderColor, RenderColorSpaceConversion, RenderProgram, scenery } from '../../../imports.js';

export default class RenderSRGBToLinearSRGB extends RenderColorSpaceConversion {
  public constructor(
    program: RenderProgram
  ) {
    super( program, RenderColor.sRGBToLinear );
  }

  public override getName(): string {
    return 'RenderSRGBToLinearSRGB';
  }

  public override withChildren( children: RenderProgram[] ): RenderSRGBToLinearSRGB {
    assert && assert( children.length === 1 );
    return new RenderSRGBToLinearSRGB( children[ 0 ] );
  }
}

scenery.register( 'RenderSRGBToLinearSRGB', RenderSRGBToLinearSRGB );
