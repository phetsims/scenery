// Copyright 2023, University of Colorado Boulder

/**
 * RenderProgram to convert linear Display P3 => linear sRGB
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { RenderColor, RenderColorSpaceConversion, RenderProgram, scenery } from '../../../imports.js';

export default class RenderLinearDisplayP3ToLinearSRGB extends RenderColorSpaceConversion {
  public constructor(
    program: RenderProgram
  ) {
    super( program, RenderColor.linearDisplayP3ToLinear );
  }

  public override getName(): string {
    return 'RenderLinearDisplayP3ToLinearSRGB';
  }

  public override withChildren( children: RenderProgram[] ): RenderLinearDisplayP3ToLinearSRGB {
    assert && assert( children.length === 1 );
    return new RenderLinearDisplayP3ToLinearSRGB( children[ 0 ] );
  }
}

scenery.register( 'RenderLinearDisplayP3ToLinearSRGB', RenderLinearDisplayP3ToLinearSRGB );
