// Copyright 2023, University of Colorado Boulder

/**
 * RenderProgram to premultiply the input color
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { RenderColor, RenderColorSpaceConversion, RenderProgram, scenery } from '../../../imports.js';

export default class RenderPremultiply extends RenderColorSpaceConversion {
  public constructor(
    program: RenderProgram
  ) {
    super( program, RenderColor.premultiply );
  }

  public override getName(): string {
    return 'RenderPremultiply';
  }

  public override withChildren( children: RenderProgram[] ): RenderPremultiply {
    assert && assert( children.length === 1 );
    return new RenderPremultiply( children[ 0 ] );
  }
}

scenery.register( 'RenderPremultiply', RenderPremultiply );
