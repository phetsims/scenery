// Copyright 2023, University of Colorado Boulder

/**
 * RenderProgram to convert linear sRGB => linear Display P3
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { RenderColor, RenderColorSpaceConversion, RenderLinearDisplayP3ToLinearSRGB, RenderProgram, scenery } from '../../../imports.js';

export default class RenderLinearSRGBToLinearDisplayP3 extends RenderColorSpaceConversion {
  public constructor(
    program: RenderProgram
  ) {
    super( program, RenderColor.linearToLinearDisplayP3 );
  }

  public override getName(): string {
    return 'RenderLinearSRGBToLinearDisplayP3';
  }

  public override withChildren( children: RenderProgram[] ): RenderLinearSRGBToLinearDisplayP3 {
    assert && assert( children.length === 1 );
    return new RenderLinearSRGBToLinearDisplayP3( children[ 0 ] );
  }
}

RenderLinearSRGBToLinearDisplayP3.prototype.inverse = RenderLinearDisplayP3ToLinearSRGB;
RenderLinearDisplayP3ToLinearSRGB.prototype.inverse = RenderLinearSRGBToLinearDisplayP3;

scenery.register( 'RenderLinearSRGBToLinearDisplayP3', RenderLinearSRGBToLinearDisplayP3 );
