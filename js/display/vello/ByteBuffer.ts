// Copyright 2023, University of Colorado Boulder

/**
 * An appendable/settable buffer of bytes
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { F32, f32_to_bytes, scenery, U32, u32_to_bytes, U8 } from '../../imports.js';

export default class ByteBuffer {

  private _byteLength: number;
  private _arrayBuffer: ArrayBuffer;
  private _f32Array: Float32Array;
  private _u32Array: Uint32Array;
  private _u8Array: Uint8Array;

  public constructor( initialSize = 512 ) {
    this._byteLength = 0;

    // TODO: resizable buffers once supported by Firefox, use maxByteLength (no copying!!!)
    this._arrayBuffer = new ArrayBuffer( initialSize );
    this._f32Array = new Float32Array( this._arrayBuffer );
    this._u32Array = new Uint32Array( this._arrayBuffer );
    this._u8Array = new Uint8Array( this._arrayBuffer );
  }

  // Direct access, for when performance is helpful
  public get fullU8Array(): Uint8Array {
    return this._u8Array;
  }
  public get fullU32Array(): Uint32Array {
    return this._u32Array;
  }
  public get fullF32Array(): Float32Array {
    return this._f32Array;
  }

  public get u8Array(): Uint8Array {
    return new Uint8Array( this._arrayBuffer, 0, this._byteLength );
  }
  public get u32Array(): Uint32Array {
    return new Uint32Array( this._arrayBuffer, 0, this._byteLength / 4 );
  }
  public get f32Array(): Float32Array {
    return new Float32Array( this._arrayBuffer, 0, this._byteLength / 4 );
  }

  public clear(): void {
    this._byteLength = 0;
    this._u8Array.fill( 0 );
  }

  public pushByteBuffer( byteBuffer: ByteBuffer ): void {
    // TODO: this is a hot spot, optimize
    this.ensureSpaceFor( byteBuffer._byteLength );

    this._u8Array.set( byteBuffer._u8Array.slice( 0, byteBuffer._byteLength ), this._byteLength );
    this._byteLength += byteBuffer._byteLength;
  }

  public pushF32( f32: F32 ): void {
    this.ensureSpaceFor( 4 );
    // If aligned, use the faster _f32Array
    if ( this._byteLength % 4 === 0 ) {
      this._f32Array[ this._byteLength / 4 ] = f32;
    }
    else {
      const bytes = f32_to_bytes( f32 );
      this._u8Array.set( bytes, this._byteLength );
    }
    this._byteLength += 4;
  }

  public pushU32( u32: U32 ): void {
    this.ensureSpaceFor( 4 );
    // If aligned, use the faster _u32Array
    if ( this._byteLength % 4 === 0 ) {
      this._u32Array[ this._byteLength / 4 ] = u32;
    }
    else {
      const bytes = u32_to_bytes( u32 );
      this._u8Array.set( bytes, this._byteLength );
    }
    this._byteLength += 4;
  }

  public pushReversedU32( u32: U32 ): void {
    this.ensureSpaceFor( 4 );

    const bytes = u32_to_bytes( u32 ).reverse();
    this._u8Array.set( bytes, this._byteLength );

    this._byteLength += 4;
  }

  public pushU8( u8: U8 ): void {
    this.ensureSpaceFor( 1 );
    this._u8Array[ this._byteLength ] = u8;
    this._byteLength += 1;
  }

  public get byteLength(): number {
    return this._byteLength;
  }

  public set byteLength( byteLength: number ) {
    // Don't actually expand below
    if ( byteLength > this._arrayBuffer.byteLength ) {
      this.resize( byteLength );
    }
    this._byteLength = byteLength;
  }

  private ensureSpaceFor( byteLength: number ): void {
    const requiredByteLength = this._byteLength + byteLength;
    if ( this._byteLength + byteLength > this._arrayBuffer.byteLength ) {
      this.resize( Math.max( this._arrayBuffer.byteLength * 2, requiredByteLength ) );
    }
  }

  // NOTE: this MAY truncate
  public resize( byteLength = 0 ): void {
    // TODO: This is a hot-spot!
    byteLength = byteLength || this._arrayBuffer.byteLength * 2;
    byteLength = Math.ceil( byteLength / 4 ) * 4; // Round up to nearest 4 (for alignment)
    // Double the size of the _arrayBuffer by default, copying memory
    const newArrayBuffer = new ArrayBuffer( byteLength );
    const newU8Array = new Uint8Array( newArrayBuffer );
    newU8Array.set( this._u8Array.slice( 0, Math.min( this._byteLength, byteLength ) ) );
    this._arrayBuffer = newArrayBuffer;
    this._f32Array = new Float32Array( this._arrayBuffer );
    this._u32Array = new Uint32Array( this._arrayBuffer );
    this._u8Array = new Uint8Array( this._arrayBuffer );
  }
}

scenery.register( 'ByteBuffer', ByteBuffer );
