// Copyright 2023, University of Colorado Boulder

/**
 * Utilities for Hilbert space-filling curve mapping
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { IntegerEdge, scenery } from '../../../imports.js';

type P2 = 'x' | '-x' | 'y' | '-y';
type P3 = 'x' | '-x' | 'y' | '-y' | 'z' | '-z';
type P4 = 'x' | '-x' | 'y' | '-y' | 'z' | '-z' | 'w' | '-w';
type P5 = 'x' | '-x' | 'y' | '-y' | 'z' | '-z' | 'w' | '-w' | 'v' | '-v';
type P6 = 'x' | '-x' | 'y' | '-y' | 'z' | '-z' | 'w' | '-w' | 'v' | '-v' | 'u' | '-u';
type Hilbert = Hilbert2 | Hilbert3 | Hilbert4 | Hilbert5 | Hilbert6;

export default class HilbertMapping {

  public static sortCenterSize( integerEdges: IntegerEdge[], scale: number ): void {
    integerEdges.sort( ( a, b ) => {
      return HilbertMapping.getHilbert4Compare(
        0.5 * ( a.x0 + a.x1 ) * scale,
        0.5 * ( a.y0 + a.y1 ) * scale,
        0.2 + 0.01 * ( a.x1 - a.x0 ) * scale,
        0.2 + 0.01 * ( a.y1 - a.y0 ) * scale,
        0.5 * ( b.x0 + b.x1 ) * scale,
        0.5 * ( b.y0 + b.y1 ) * scale,
        0.2 + 0.01 * ( b.x1 - b.x0 ) * scale,
        0.2 + 0.01 * ( b.y1 - b.y0 ) * scale
      );
    } );
  }

  public static sortMinMax( integerEdges: IntegerEdge[], scale: number ): void {
    integerEdges.sort( ( a, b ) => {
      return HilbertMapping.getHilbert4Compare(
        a.bounds.minX * scale,
        a.bounds.minY * scale,
        a.bounds.maxX * scale,
        a.bounds.maxY * scale,
        b.bounds.minX * scale,
        b.bounds.minY * scale,
        b.bounds.maxX * scale,
        b.bounds.maxY * scale
      );
    } );
  }

  public static sortMinMaxSize( integerEdges: IntegerEdge[], scale: number ): void {
    integerEdges.sort( ( a, b ) => {
      return HilbertMapping.getHilbert6Compare(
        a.bounds.minX * scale,
        a.bounds.minY * scale,
        a.bounds.maxX * scale,
        a.bounds.maxY * scale,
        0.2 + 0.01 * ( a.x1 - a.x0 ) * scale,
        0.2 + 0.01 * ( a.y1 - a.y0 ) * scale,
        b.bounds.minX * scale,
        b.bounds.minY * scale,
        b.bounds.maxX * scale,
        b.bounds.maxY * scale,
        0.2 + 0.01 * ( b.x1 - b.x0 ) * scale,
        0.2 + 0.01 * ( b.y1 - b.y0 ) * scale
      );
    } );
  }

  public static sortCenterMinMax( integerEdges: IntegerEdge[], scale: number ): void {
    integerEdges.sort( ( a, b ) => {
      return HilbertMapping.getHilbert6Compare(
        0.5 * ( a.x0 + a.x1 ) * scale,
        0.5 * ( a.y0 + a.y1 ) * scale,
        a.bounds.minX * scale,
        a.bounds.minY * scale,
        a.bounds.maxX * scale,
        a.bounds.maxY * scale,
        0.5 * ( b.x0 + b.x1 ) * scale,
        0.5 * ( b.y0 + b.y1 ) * scale,
        b.bounds.minX * scale,
        b.bounds.minY * scale,
        b.bounds.maxX * scale,
        b.bounds.maxY * scale
      );
    } );
  }

  public static binaryToGray( n: number ): number {
    return n ^ ( n >> 1 );
  }

  public static grayToBinary( n: number ): number {
    let mask = n;
    while ( mask ) {
      mask >>= 1;
      n ^= mask;
    }
    return n;
  }

  public static grayToBinary32( n: number ): number {
    n ^= n >> 16;
    n ^= n >> 8;
    n ^= n >> 4;
    n ^= n >> 2;
    n ^= n >> 1;
    return n;
  }

  public static grayToBinary16( n: number ): number {
    n ^= n >> 8;
    n ^= n >> 4;
    n ^= n >> 2;
    n ^= n >> 1;
    return n;
  }

  public static grayToBinary8( n: number ): number {
    n ^= n >> 4;
    n ^= n >> 2;
    n ^= n >> 1;
    return n;
  }

  public static grayToBinary4( n: number ): number {
    n ^= n >> 2;
    n ^= n >> 1;
    return n;
  }

  public static grayToBinary2( n: number ): number {
    n ^= n >> 1;
    return n;
  }

  public static getHilbertBits( hilbert: Hilbert, dimension: number, bits = 64 ): number {
    let result = 0;
    let iteration = 1;
    while ( bits > 0 ) {
      hilbert.apply();
      result += hilbert.i * Math.pow( 2, -dimension * iteration );
      bits -= dimension;
      iteration++;
    }
    return result;
  }

  public static getHilbertCompare( a: Hilbert, b: Hilbert, iterLimit = 20 ): number {
    let iterations = 1;
    do {
      a.apply();
      b.apply();
    }
    while ( a.i === b.i && iterations++ <= iterLimit );

    return a.i === b.i ? 0 : ( a.i < b.i ? -1 : 1 );
  }

  public static getHilbert2Bits( x: number, y: number, bits = 64 ): number {
    return HilbertMapping.getHilbertBits( scratch2a.set( x, y ), 2, bits );
  }

  public static getHilbert3Bits( x: number, y: number, z: number, bits = 64 ): number {
    return HilbertMapping.getHilbertBits( scratch3a.set( x, y, z ), 3, bits );
  }

  public static getHilbert4Bits( x: number, y: number, z: number, w: number, bits = 64 ): number {
    return HilbertMapping.getHilbertBits( scratch4a.set( x, y, z, w ), 4, bits );
  }

  public static getHilbert5Bits( x: number, y: number, z: number, w: number, v: number, bits = 64 ): number {
    return HilbertMapping.getHilbertBits( scratch5a.set( x, y, z, w, v ), 5, bits );
  }

  public static getHilbert6Bits( x: number, y: number, z: number, w: number, v: number, u: number, bits = 64 ): number {
    return HilbertMapping.getHilbertBits( scratch6a.set( x, y, z, w, v, u ), 6, bits );
  }

  public static getHilbert2Compare( x0: number, y0: number, x1: number, y1: number, iterLimit = 32 ): number {
    return HilbertMapping.getHilbertCompare( scratch2a.set( x0, y0 ), scratch2b.set( x1, y1 ), iterLimit );
  }

  public static getHilbert3Compare( x0: number, y0: number, z0: number, x1: number, y1: number, z1: number, iterLimit = 22 ): number {
    return HilbertMapping.getHilbertCompare( scratch3a.set( x0, y0, z0 ), scratch3b.set( x1, y1, z1 ), iterLimit );
  }

  public static getHilbert4Compare( x0: number, y0: number, z0: number, w0: number, x1: number, y1: number, z1: number, w1: number, iterLimit = 16 ): number {
    return HilbertMapping.getHilbertCompare( scratch4a.set( x0, y0, z0, w0 ), scratch4b.set( x1, y1, z1, w1 ), iterLimit );
  }

  public static getHilbert5Compare( x0: number, y0: number, z0: number, w0: number, v0: number, x1: number, y1: number, z1: number, w1: number, v1: number, iterLimit = 13 ): number {
    return HilbertMapping.getHilbertCompare( scratch5a.set( x0, y0, z0, w0, v0 ), scratch5b.set( x1, y1, z1, w1, v1 ), iterLimit );
  }

  public static getHilbert6Compare( x0: number, y0: number, z0: number, w0: number, v0: number, u0: number, x1: number, y1: number, z1: number, w1: number, v1: number, u1: number, iterLimit = 11 ): number {
    return HilbertMapping.getHilbertCompare( scratch6a.set( x0, y0, z0, w0, v0, u0 ), scratch6b.set( x1, y1, z1, w1, v1, u1 ), iterLimit );
  }
}

scenery.register( 'HilbertMapping', HilbertMapping );

export class Hilbert2 {
  public n = 0;
  public i = 0;

  public constructor(
    public x: number,
    public y: number
  ) {
    assert && assert( Math.abs( x ) <= 1 );
    assert && assert( Math.abs( y ) <= 1 );
  }

  public set( x: number, y: number ): this {
    this.x = x;
    this.y = y;
    return this;
  }

  public apply(): void {
    const xPlus = this.x >= 0;
    const yPlus = this.y >= 0;
    this.n = ( xPlus ? 0x1 : 0 ) | ( yPlus ? 0x2 : 0 );
    this.i = HilbertMapping.grayToBinary2( this.n );
    this.x = 2 * ( this.x + ( xPlus ? -0.5 : 0.5 ) );
    this.y = 2 * ( this.y + ( yPlus ? -0.5 : 0.5 ) );
    this.permute();
  }

  private static permutations = [
    [ 'y', 'x' ],
    [ 'x', 'y' ],
    [ 'x', 'y' ],
    [ '-y', '-x' ]
  ] as const;

  private getValue( p: P2 ): number {
    switch( p ) {
      case 'x':
        return this.x;
      case '-x':
        return -this.x;
      case 'y':
        return this.y;
      case '-y':
        return -this.y;
      default:
        throw new Error( 'unreachable' );
    }
  }

  private permute(): void {
    const permutation = Hilbert2.permutations[ this.i ];
    const x = this.getValue( permutation[ 0 ] );
    const y = this.getValue( permutation[ 1 ] );
    this.x = x;
    this.y = y;
  }
}

scenery.register( 'Hilbert2', Hilbert2 );

export class Hilbert3 {
  public n = 0;
  public i = 0;

  public constructor(
    public x: number,
    public y: number,
    public z: number
  ) {
    assert && assert( Math.abs( x ) <= 1 );
    assert && assert( Math.abs( y ) <= 1 );
    assert && assert( Math.abs( z ) <= 1 );
  }

  public set( x: number, y: number, z: number ): this {
    this.x = x;
    this.y = y;
    this.z = z;
    return this;
  }

  public apply(): void {
    const xPlus = this.x >= 0;
    const yPlus = this.y >= 0;
    const zPlus = this.z >= 0;
    this.n = ( xPlus ? 0x1 : 0 ) | ( yPlus ? 0x2 : 0 ) | ( zPlus ? 0x4 : 0 );
    this.i = HilbertMapping.grayToBinary4( this.n );
    this.x = 2 * ( this.x + ( xPlus ? -0.5 : 0.5 ) );
    this.y = 2 * ( this.y + ( yPlus ? -0.5 : 0.5 ) );
    this.z = 2 * ( this.z + ( zPlus ? -0.5 : 0.5 ) );
    this.permute();
  }

  // Map[(Differences[#][[{1, 2, 4}]]) &, Partition[HilbertCurve[2, 3][[1]], 8], {1}]
  private static permutations = [
    [ 'y', 'z', 'x' ],
    [ 'z', 'x', 'y' ],
    [ 'z', 'x', 'y' ],
    [ '-x', '-y', 'z' ],
    [ '-x', '-y', 'z' ],
    [ '-z', 'x', '-y' ],
    [ '-z', 'x', '-y' ],
    [ 'y', '-z', '-x' ]
  ] as const;

  private getValue( p: P3 ): number {
    switch( p ) {
      case 'x':
        return this.x;
      case '-x':
        return -this.x;
      case 'y':
        return this.y;
      case '-y':
        return -this.y;
      case 'z':
        return this.z;
      case '-z':
        return -this.z;
      default:
        throw new Error( 'unreachable' );
    }
  }

  private permute(): void {
    const permutation = Hilbert3.permutations[ this.i ];
    const x = this.getValue( permutation[ 0 ] );
    const y = this.getValue( permutation[ 1 ] );
    const z = this.getValue( permutation[ 2 ] );
    this.x = x;
    this.y = y;
    this.z = z;
  }
}

scenery.register( 'Hilbert3', Hilbert3 );

export class Hilbert4 {
  public n = 0;
  public i = 0;

  public constructor(
    public x: number,
    public y: number,
    public z: number,
    public w: number
  ) {
    assert && assert( Math.abs( x ) <= 1 );
    assert && assert( Math.abs( y ) <= 1 );
    assert && assert( Math.abs( z ) <= 1 );
    assert && assert( Math.abs( w ) <= 1 );
  }

  public set( x: number, y: number, z: number, w: number ): this {
    this.x = x;
    this.y = y;
    this.z = z;
    this.w = w;
    return this;
  }

  public apply(): void {
    const xPlus = this.x >= 0;
    const yPlus = this.y >= 0;
    const zPlus = this.z >= 0;
    const wPlus = this.w >= 0;
    this.n = ( xPlus ? 0x1 : 0 ) | ( yPlus ? 0x2 : 0 ) | ( zPlus ? 0x4 : 0 ) | ( wPlus ? 0x8 : 0 );
    this.i = HilbertMapping.grayToBinary4( this.n );
    this.x = 2 * ( this.x + ( xPlus ? -0.5 : 0.5 ) );
    this.y = 2 * ( this.y + ( yPlus ? -0.5 : 0.5 ) );
    this.z = 2 * ( this.z + ( zPlus ? -0.5 : 0.5 ) );
    this.w = 2 * ( this.w + ( wPlus ? -0.5 : 0.5 ) );
    this.permute();
  }

  // Map[(Differences[#][[{1, 2, 4, 8}]]) &, Partition[HilbertCurve[2, 4][[1]], 16], {1}]
  private static permutations = [
    [ 'y', 'z', 'w', 'x' ],
    [ 'z', 'w', 'x', 'y' ],
    [ 'z', 'w', 'x', 'y' ],
    [ 'w', '-x', '-y', 'z' ],
    [ 'w', '-x', '-y', 'z' ],
    [ '-z', 'w', 'x', '-y' ],
    [ '-z', 'w', 'x', '-y' ],
    [ '-x', 'y', '-z', 'w' ],
    [ '-x', 'y', '-z', 'w' ],
    [ '-z', '-w', 'x', 'y' ],
    [ '-z', '-w', 'x', 'y' ],
    [ '-w', '-x', '-y', '-z' ],
    [ '-w', '-x', '-y', '-z' ],
    [ 'z', '-w', 'x', '-y' ],
    [ 'z', '-w', 'x', '-y' ],
    [ 'y', 'z', '-w', '-x' ]
  ] as const;

  private getValue( p: P4 ): number {
    switch( p ) {
      case 'x':
        return this.x;
      case '-x':
        return -this.x;
      case 'y':
        return this.y;
      case '-y':
        return -this.y;
      case 'z':
        return this.z;
      case '-z':
        return -this.z;
      case 'w':
        return this.w;
      case '-w':
        return -this.w;
      default:
        throw new Error( 'unreachable' );
    }
  }

  private permute(): void {
    const permutation = Hilbert4.permutations[ this.i ];
    const x = this.getValue( permutation[ 0 ] );
    const y = this.getValue( permutation[ 1 ] );
    const z = this.getValue( permutation[ 2 ] );
    const w = this.getValue( permutation[ 3 ] );
    this.x = x;
    this.y = y;
    this.z = z;
    this.w = w;
  }
}

scenery.register( 'Hilbert4', Hilbert4 );

export class Hilbert5 {
  public n = 0;
  public i = 0;

  public constructor(
    public x: number,
    public y: number,
    public z: number,
    public w: number,
    public v: number
  ) {
    assert && assert( Math.abs( x ) <= 1 );
    assert && assert( Math.abs( y ) <= 1 );
    assert && assert( Math.abs( z ) <= 1 );
    assert && assert( Math.abs( w ) <= 1 );
    assert && assert( Math.abs( v ) <= 1 );
  }

  public set( x: number, y: number, z: number, w: number, v: number ): this {
    this.x = x;
    this.y = y;
    this.z = z;
    this.w = w;
    this.v = v;
    return this;
  }

  public apply(): void {
    const xPlus = this.x >= 0;
    const yPlus = this.y >= 0;
    const zPlus = this.z >= 0;
    const wPlus = this.w >= 0;
    const vPlus = this.v >= 0;
    this.n = ( xPlus ? 0x1 : 0 ) | ( yPlus ? 0x2 : 0 ) | ( zPlus ? 0x4 : 0 ) | ( wPlus ? 0x8 : 0 ) | ( vPlus ? 0x10 : 0 );
    this.i = HilbertMapping.grayToBinary8( this.n );
    this.x = 2 * ( this.x + ( xPlus ? -0.5 : 0.5 ) );
    this.y = 2 * ( this.y + ( yPlus ? -0.5 : 0.5 ) );
    this.z = 2 * ( this.z + ( zPlus ? -0.5 : 0.5 ) );
    this.w = 2 * ( this.w + ( wPlus ? -0.5 : 0.5 ) );
    this.v = 2 * ( this.v + ( vPlus ? -0.5 : 0.5 ) );
    this.permute();
  }

  // Map[(Differences[#][[{1, 2, 4, 8, 16}]]) &, Partition[HilbertCurve[2, 5][[1]], 32], {1}]
  private static permutations = [
    [ 'y', 'z', 'w', 'v', 'x' ],
    [ 'z', 'w', 'v', 'x', 'y' ],
    [ 'z', 'w', 'v', 'x', 'y' ],
    [ 'w', 'v', '-x', '-y', 'z' ],
    [ 'w', 'v', '-x', '-y', 'z' ],
    [ '-z', 'w', 'v', 'x', '-y' ],
    [ '-z', 'w', 'v', 'x', '-y' ],
    [ 'v', '-x', 'y', '-z', 'w' ],
    [ 'v', '-x', 'y', '-z', 'w' ],
    [ '-z', '-w', 'v', 'x', 'y' ],
    [ '-z', '-w', 'v', 'x', 'y' ],
    [ '-w', 'v', '-x', '-y', '-z' ],
    [ '-w', 'v', '-x', '-y', '-z' ],
    [ 'z', '-w', 'v', 'x', '-y' ],
    [ 'z', '-w', 'v', 'x', '-y' ],
    [ '-x', 'y', 'z', '-w', 'v' ],
    [ '-x', 'y', 'z', '-w', 'v' ],
    [ 'z', '-w', '-v', 'x', 'y' ],
    [ 'z', '-w', '-v', 'x', 'y' ],
    [ '-w', '-v', '-x', '-y', 'z' ],
    [ '-w', '-v', '-x', '-y', 'z' ],
    [ '-z', '-w', '-v', 'x', '-y' ],
    [ '-z', '-w', '-v', 'x', '-y' ],
    [ '-v', '-x', 'y', '-z', '-w' ],
    [ '-v', '-x', 'y', '-z', '-w' ],
    [ '-z', 'w', '-v', 'x', 'y' ],
    [ '-z', 'w', '-v', 'x', 'y' ],
    [ 'w', '-v', '-x', '-y', '-z' ],
    [ 'w', '-v', '-x', '-y', '-z' ],
    [ 'z', 'w', '-v', 'x', '-y' ],
    [ 'z', 'w', '-v', 'x', '-y' ],
    [ 'y', 'z', 'w', '-v', '-x' ]
  ] as const;

  private getValue( p: P5 ): number {
    switch( p ) {
      case 'x':
        return this.x;
      case '-x':
        return -this.x;
      case 'y':
        return this.y;
      case '-y':
        return -this.y;
      case 'z':
        return this.z;
      case '-z':
        return -this.z;
      case 'w':
        return this.w;
      case '-w':
        return -this.w;
      case 'v':
        return this.v;
      case '-v':
        return -this.v;
      default:
        throw new Error( 'unreachable' );
    }
  }

  private permute(): void {
    const permutation = Hilbert5.permutations[ this.i ];
    const x = this.getValue( permutation[ 0 ] );
    const y = this.getValue( permutation[ 1 ] );
    const z = this.getValue( permutation[ 2 ] );
    const w = this.getValue( permutation[ 3 ] );
    const v = this.getValue( permutation[ 4 ] );
    this.x = x;
    this.y = y;
    this.z = z;
    this.w = w;
    this.v = v;
  }
}

scenery.register( 'Hilbert5', Hilbert5 );

export class Hilbert6 {
  public n = 0;
  public i = 0;

  public constructor(
    public x: number,
    public y: number,
    public z: number,
    public w: number,
    public v: number,
    public u: number
  ) {
    assert && assert( Math.abs( x ) <= 1 );
    assert && assert( Math.abs( y ) <= 1 );
    assert && assert( Math.abs( z ) <= 1 );
    assert && assert( Math.abs( w ) <= 1 );
    assert && assert( Math.abs( v ) <= 1 );
    assert && assert( Math.abs( u ) <= 1 );
  }

  public set( x: number, y: number, z: number, w: number, v: number, u: number ): this {
    this.x = x;
    this.y = y;
    this.z = z;
    this.w = w;
    this.v = v;
    this.u = u;
    return this;
  }

  public apply(): void {
    const xPlus = this.x >= 0;
    const yPlus = this.y >= 0;
    const zPlus = this.z >= 0;
    const wPlus = this.w >= 0;
    const vPlus = this.v >= 0;
    const uPlus = this.u >= 0;
    this.n = ( xPlus ? 0x1 : 0 ) | ( yPlus ? 0x2 : 0 ) | ( zPlus ? 0x4 : 0 ) | ( wPlus ? 0x8 : 0 ) | ( vPlus ? 0x10 : 0 ) | ( uPlus ? 0x20 : 0 );
    this.i = HilbertMapping.grayToBinary8( this.n );
    this.x = 2 * ( this.x + ( xPlus ? -0.5 : 0.5 ) );
    this.y = 2 * ( this.y + ( yPlus ? -0.5 : 0.5 ) );
    this.z = 2 * ( this.z + ( zPlus ? -0.5 : 0.5 ) );
    this.w = 2 * ( this.w + ( wPlus ? -0.5 : 0.5 ) );
    this.v = 2 * ( this.v + ( vPlus ? -0.5 : 0.5 ) );
    this.u = 2 * ( this.u + ( uPlus ? -0.5 : 0.5 ) );
    this.permute();
  }

  // Map[(Differences[#][[{1, 2, 4, 8, 16, 32}]]) &, Partition[HilbertCurve[2, 6][[1]], 64], {1}]
  private static permutations = [
    [ 'y', 'z', 'w', 'v', 'u', 'x' ],
    [ 'z', 'w', 'v', 'u', 'x', 'y' ],
    [ 'z', 'w', 'v', 'u', 'x', 'y' ],
    [ 'w', 'v', 'u', '-x', '-y', 'z' ],
    [ 'w', 'v', 'u', '-x', '-y', 'z' ],
    [ '-z', 'w', 'v', 'u', 'x', '-y' ],
    [ '-z', 'w', 'v', 'u', 'x', '-y' ],
    [ 'v', 'u', '-x', 'y', '-z', 'w' ],
    [ 'v', 'u', '-x', 'y', '-z', 'w' ],
    [ '-z', '-w', 'v', 'u', 'x', 'y' ],
    [ '-z', '-w', 'v', 'u', 'x', 'y' ],
    [ '-w', 'v', 'u', '-x', '-y', '-z' ],
    [ '-w', 'v', 'u', '-x', '-y', '-z' ],
    [ 'z', '-w', 'v', 'u', 'x', '-y' ],
    [ 'z', '-w', 'v', 'u', 'x', '-y' ],
    [ 'u', '-x', 'y', 'z', '-w', 'v' ],
    [ 'u', '-x', 'y', 'z', '-w', 'v' ],
    [ 'z', '-w', '-v', 'u', 'x', 'y' ],
    [ 'z', '-w', '-v', 'u', 'x', 'y' ],
    [ '-w', '-v', 'u', '-x', '-y', 'z' ],
    [ '-w', '-v', 'u', '-x', '-y', 'z' ],
    [ '-z', '-w', '-v', 'u', 'x', '-y' ],
    [ '-z', '-w', '-v', 'u', 'x', '-y' ],
    [ '-v', 'u', '-x', 'y', '-z', '-w' ],
    [ '-v', 'u', '-x', 'y', '-z', '-w' ],
    [ '-z', 'w', '-v', 'u', 'x', 'y' ],
    [ '-z', 'w', '-v', 'u', 'x', 'y' ],
    [ 'w', '-v', 'u', '-x', '-y', '-z' ],
    [ 'w', '-v', 'u', '-x', '-y', '-z' ],
    [ 'z', 'w', '-v', 'u', 'x', '-y' ],
    [ 'z', 'w', '-v', 'u', 'x', '-y' ],
    [ '-x', 'y', 'z', 'w', '-v', 'u' ],
    [ '-x', 'y', 'z', 'w', '-v', 'u' ],
    [ 'z', 'w', '-v', '-u', 'x', 'y' ],
    [ 'z', 'w', '-v', '-u', 'x', 'y' ],
    [ 'w', '-v', '-u', '-x', '-y', 'z' ],
    [ 'w', '-v', '-u', '-x', '-y', 'z' ],
    [ '-z', 'w', '-v', '-u', 'x', '-y' ],
    [ '-z', 'w', '-v', '-u', 'x', '-y' ],
    [ '-v', '-u', '-x', 'y', '-z', 'w' ],
    [ '-v', '-u', '-x', 'y', '-z', 'w' ],
    [ '-z', '-w', '-v', '-u', 'x', 'y' ],
    [ '-z', '-w', '-v', '-u', 'x', 'y' ],
    [ '-w', '-v', '-u', '-x', '-y', '-z' ],
    [ '-w', '-v', '-u', '-x', '-y', '-z' ],
    [ 'z', '-w', '-v', '-u', 'x', '-y' ],
    [ 'z', '-w', '-v', '-u', 'x', '-y' ],
    [ '-u', '-x', 'y', 'z', '-w', '-v' ],
    [ '-u', '-x', 'y', 'z', '-w', '-v' ],
    [ 'z', '-w', 'v', '-u', 'x', 'y' ],
    [ 'z', '-w', 'v', '-u', 'x', 'y' ],
    [ '-w', 'v', '-u', '-x', '-y', 'z' ],
    [ '-w', 'v', '-u', '-x', '-y', 'z' ],
    [ '-z', '-w', 'v', '-u', 'x', '-y' ],
    [ '-z', '-w', 'v', '-u', 'x', '-y' ],
    [ 'v', '-u', '-x', 'y', '-z', '-w' ],
    [ 'v', '-u', '-x', 'y', '-z', '-w' ],
    [ '-z', 'w', 'v', '-u', 'x', 'y' ],
    [ '-z', 'w', 'v', '-u', 'x', 'y' ],
    [ 'w', 'v', '-u', '-x', '-y', '-z' ],
    [ 'w', 'v', '-u', '-x', '-y', '-z' ],
    [ 'z', 'w', 'v', '-u', 'x', '-y' ],
    [ 'z', 'w', 'v', '-u', 'x', '-y' ],
    [ 'y', 'z', 'w', 'v', '-u', '-x' ]
  ] as const;

  private getValue( p: P6 ): number {
    switch( p ) {
      case 'x':
        return this.x;
      case '-x':
        return -this.x;
      case 'y':
        return this.y;
      case '-y':
        return -this.y;
      case 'z':
        return this.z;
      case '-z':
        return -this.z;
      case 'w':
        return this.w;
      case '-w':
        return -this.w;
      case 'v':
        return this.v;
      case '-v':
        return -this.v;
      case 'u':
        return this.u;
      case '-u':
        return -this.u;
      default:
        throw new Error( 'unreachable' );
    }
  }

  private permute(): void {
    const permutation = Hilbert6.permutations[ this.i ];
    const x = this.getValue( permutation[ 0 ] );
    const y = this.getValue( permutation[ 1 ] );
    const z = this.getValue( permutation[ 2 ] );
    const w = this.getValue( permutation[ 3 ] );
    const v = this.getValue( permutation[ 4 ] );
    const u = this.getValue( permutation[ 5 ] );
    this.x = x;
    this.y = y;
    this.z = z;
    this.w = w;
    this.v = v;
    this.u = u;
  }
}

scenery.register( 'Hilbert6', Hilbert6 );

const scratch2a = new Hilbert2( 0, 0 );
const scratch3a = new Hilbert3( 0, 0, 0 );
const scratch4a = new Hilbert4( 0, 0, 0, 0 );
const scratch5a = new Hilbert5( 0, 0, 0, 0, 0 );
const scratch6a = new Hilbert6( 0, 0, 0, 0, 0, 0 );
const scratch2b = new Hilbert2( 0, 0 );
const scratch3b = new Hilbert3( 0, 0, 0 );
const scratch4b = new Hilbert4( 0, 0, 0, 0 );
const scratch5b = new Hilbert5( 0, 0, 0, 0, 0 );
const scratch6b = new Hilbert6( 0, 0, 0, 0, 0, 0 );
