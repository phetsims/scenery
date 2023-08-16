// Copyright 2023, University of Colorado Boulder

/**
 * Like Vector2, but with BigInts
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { scenery } from '../../../imports.js';

export default class BigIntVector2 {
  public constructor( public x: bigint, public y: bigint ) {}

  public equals( vector: BigIntVector2 ): boolean {
    return this.x === vector.x && this.y === vector.y;
  }

  public toString(): string {
    return `(${this.x}, ${this.y})`;
  }

  // TODO
}

scenery.register( 'BigIntVector2', BigIntVector2 );
