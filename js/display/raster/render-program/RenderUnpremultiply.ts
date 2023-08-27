// Copyright 2023, University of Colorado Boulder

/**
 * RenderProgram to unpremultiply the input color
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { RenderColor, RenderColorSpaceConversion, RenderPremultiply, RenderProgram, scenery } from '../../../imports.js';

export default class RenderUnpremultiply extends RenderColorSpaceConversion {
  public constructor(
    program: RenderProgram
  ) {
    super( program, RenderColor.unpremultiply );
  }

  public override getName(): string {
    return 'RenderUnpremultiply';
  }

  public override withChildren( children: RenderProgram[] ): RenderUnpremultiply {
    assert && assert( children.length === 1 );
    return new RenderUnpremultiply( children[ 0 ] );
  }
}

RenderPremultiply.prototype.inverse = RenderUnpremultiply;
RenderUnpremultiply.prototype.inverse = RenderPremultiply;

scenery.register( 'RenderUnpremultiply', RenderUnpremultiply );
