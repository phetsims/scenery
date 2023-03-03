// Copyright 2023, University of Colorado Boulder

/**
 * For completely blank lines in RichText
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */
import Bounds2 from '../../../../dot/js/Bounds2.js';
import Pool, { TPoolable } from '../../../../phet-core/js/Pool.js';
import { Node, RichTextCleanable, scenery } from '../../imports.js';

class RichTextVerticalSpacer extends RichTextCleanable( Node ) implements TPoolable {
  public constructor( height: number ) {
    super();

    this.initialize( height );
  }

  public initialize( height: number ): this {

    this.localBounds = new Bounds2( 0, 0, 0, height );

    return this;
  }

  public freeToPool(): void {
    RichTextVerticalSpacer.pool.freeToPool( this );
  }

  public static readonly pool = new Pool( RichTextVerticalSpacer );
}

scenery.register( 'RichTextVerticalSpacer', RichTextVerticalSpacer );

export default RichTextVerticalSpacer;
