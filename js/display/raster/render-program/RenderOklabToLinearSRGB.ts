// Copyright 2023, University of Colorado Boulder

/**
 * RenderProgram to convert Oklab => linear sRGB
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { RenderColor, RenderColorSpaceConversion, RenderProgram, scenery } from '../../../imports.js';

export default class RenderOklabToLinearSRGB extends RenderColorSpaceConversion {
  public constructor(
    program: RenderProgram
  ) {
    super( program, RenderColor.oklabToLinear );
  }

  public override getName(): string {
    return 'RenderOklabToLinearSRGB';
  }

  public override withChildren( children: RenderProgram[] ): RenderOklabToLinearSRGB {
    assert && assert( children.length === 1 );
    return new RenderOklabToLinearSRGB( children[ 0 ] );
  }
}

scenery.register( 'RenderOklabToLinearSRGB', RenderOklabToLinearSRGB );
