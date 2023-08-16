// Copyright 2023, University of Colorado Boulder

/**
 * Like Vector2, but with BigRationals
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { BigRational, scenery } from '../../../imports.js';

export default class BigRationalVector2 {
  public constructor( public x: BigRational, public y: BigRational ) {}

  public equals( vector: BigRationalVector2 ): boolean {
    return this.x.equals( vector.x ) && this.y.equals( vector.y );
  }

  public toString(): string {
    return `(${this.x}, ${this.y})`;
  }

  // TODO
}

scenery.register( 'BigRationalVector2', BigRationalVector2 );
