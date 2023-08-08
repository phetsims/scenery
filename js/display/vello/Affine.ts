// Copyright 2023, University of Colorado Boulder

/**
 * An affine matrix - TODO: just replace this with typed arrays
 *
 * TODO: pooling?
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { scenery } from '../../imports.js';

export default class Affine {
  public constructor(
    public a00: number, public a10: number, public a01: number, public a11: number,
    public a02: number, public a12: number
  ) {}

  public times( affine: Affine ): Affine {
    // TODO: Affine (and this method) are a hot spot IF we are doing client-side matrix stuff
    const a00 = this.a00 * affine.a00 + this.a01 * affine.a10;
    const a01 = this.a00 * affine.a01 + this.a01 * affine.a11;
    const a02 = this.a00 * affine.a02 + this.a01 * affine.a12 + this.a02;
    const a10 = this.a10 * affine.a00 + this.a11 * affine.a10;
    const a11 = this.a10 * affine.a01 + this.a11 * affine.a11;
    const a12 = this.a10 * affine.a02 + this.a11 * affine.a12 + this.a12;
    return new Affine( a00, a10, a01, a11, a02, a12 );
  }

  public equals( affine: Affine ): boolean {
    return this.a00 === affine.a00 &&
           this.a10 === affine.a10 &&
           this.a01 === affine.a01 &&
           this.a11 === affine.a11 &&
           this.a02 === affine.a02 &&
           this.a12 === affine.a12;
  }

  public static readonly IDENTITY = new Affine( 1, 0, 0, 1, 0, 0 );
}

scenery.register( 'Affine', Affine );
