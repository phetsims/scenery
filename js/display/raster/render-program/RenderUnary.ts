// Copyright 2023, University of Colorado Boulder

/**
 * Shared code for unary RenderPrograms
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { RenderProgram, scenery } from '../../../imports.js';

export default abstract class RenderUnary extends RenderProgram {
  protected constructor(
    public readonly program: RenderProgram
  ) {
    super();
  }

  public override getChildren(): RenderProgram[] {
    return [ this.program ];
  }
}

scenery.register( 'RenderUnary', RenderUnary );
