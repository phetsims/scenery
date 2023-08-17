// Copyright 2023, University of Colorado Boulder

/**
 * Abstract RenderProgram that has a path associated with it.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { RenderPath, RenderProgram, scenery } from '../../../imports.js';
import Matrix3 from '../../../../../dot/js/Matrix3.js';

export default abstract class RenderPathProgram extends RenderProgram {
  protected constructor( public readonly path: RenderPath | null ) {
    super();
  }

  public override equals( other: RenderProgram ): boolean {
    if ( this === other ) { return true; }
    return other instanceof RenderPathProgram &&
           this.path === other.path;
  }

  public isInPath( pathTest: ( renderPath: RenderPath ) => boolean ): boolean {
    return !this.path || pathTest( this.path );
  }

  public getTransformedPath( transform: Matrix3 ): RenderPath | null {
    return this.path ? this.path.transformed( transform ) : null;
  }
}

scenery.register( 'RenderPathProgram', RenderPathProgram );
