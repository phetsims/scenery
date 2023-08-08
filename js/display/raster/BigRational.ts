// Copyright 2023, University of Colorado Boulder

/**
 * Contains an import-style snippet of shader code, with dependencies on other snippets.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { scenery } from '../../imports.js';

export default class BigRational {

  public numerator: bigint;
  public denominator: bigint;

  public constructor( numerator: bigint | number, denominator: bigint | number ) {
    this.numerator = BigInt( numerator );
    this.denominator = BigInt( denominator );

    if ( denominator === 0n ) {
      throw new Error( 'Division by zero' );
    }

    if ( denominator < 0n ) {
      this.numerator = -this.numerator;
      this.denominator = -this.denominator;
    }
  }

  public copy(): BigRational {
    return new BigRational( this.numerator, this.denominator );
  }

  // lazy implementation NOT meant to be in JS due to excess reduction
  public plus( rational: BigRational ): BigRational {
    return new BigRational(
      this.numerator * rational.denominator + this.denominator * rational.numerator,
      this.denominator * rational.denominator
    ).reduced();
  }

  // lazy implementation NOT meant to be in JS due to excess reduction
  public minus( rational: BigRational ): BigRational {
    return new BigRational(
      this.numerator * rational.denominator - this.denominator * rational.numerator,
      this.denominator * rational.denominator
    ).reduced();
  }

  // lazy implementation NOT meant to be in JS due to excess reduction
  public times( rational: BigRational ): BigRational {
    return new BigRational(
      this.numerator * rational.numerator,
      this.denominator * rational.denominator
    ).reduced();
  }

  public reduce(): void {
    if ( this.numerator === 0n ) {
      this.denominator = 1n;
      return;
    }
    else if ( this.denominator === 1n ) {
      return;
    }

    const absNumerator = this.numerator < 0n ? -this.numerator : this.numerator;
    const gcd = BigRational.gcdBigInt( absNumerator, this.denominator );

    if ( gcd !== 1n ) {
      this.numerator /= gcd;
      this.denominator /= gcd;
    }
  }

  public reduced(): BigRational {
    const result = this.copy();
    result.reduce();
    return result;
  }

  public isZero(): boolean {
    return this.numerator === 0n;
  }

  public ratioTest(): number {
    if ( this.numerator === 0n || this.numerator === this.denominator ) {
      return 1;
    }
    else if ( this.numerator > 0n && this.numerator < this.denominator ) {
      return 2;
    }
    else {
      return 0;
    }
  }

  public equalsCrossMul( other: BigRational ): boolean {
    return this.numerator * other.denominator === this.denominator * other.numerator;
  }

  public compareCrossMul( other: BigRational ): number {
    const thisCross = this.numerator * other.denominator;
    const otherCross = this.denominator * other.numerator;
    return thisCross < otherCross ? -1 : thisCross > otherCross ? 1 : 0;
  }

  // NOT for WGSL, slow
  public equals( other: BigRational ): boolean {
    const thisReduced = this.reduced();
    const otherReduced = other.reduced();
    return thisReduced.numerator === otherReduced.numerator && thisReduced.denominator === otherReduced.denominator;
  }

  public toString(): string {
    return this.denominator === 1n ? `${this.numerator}` : `${this.numerator}/${this.denominator}`;
  }

  public static readonly ZERO = new BigRational( 0n, 1n );
  public static readonly ONE = new BigRational( 1n, 1n );

  public static whole( numerator: number | bigint ): BigRational {
    return new BigRational( numerator, 1n );
  }
  public static inverse( numerator: number | bigint ): BigRational {
    return new BigRational( 1n, numerator );
  }

  public static gcdBigInt( a: bigint, b: bigint ): bigint {
    while ( b !== 0n ) {
      const t = b;
      b = a % b;
      a = t;
    }
    return a;
  }
}

scenery.register( 'BigRational', BigRational );
