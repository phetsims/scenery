// Copyright 2023, University of Colorado Boulder

/**
 * TypeScript wrapper and utilities around Guillotiere atlas allocation
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import guillotiere, { GuillotiereAtlas } from './guillotiere.js';
import guillotiere_wasm from './guillotiere_wasm.js';
import { base64ToU8, scenery } from '../../imports.js';
import IntentionalAny from '../../../../phet-core/js/types/IntentionalAny.js';
import asyncLoader from '../../../../phet-core/js/asyncLoader.js';
import Bounds2 from '../../../../dot/js/Bounds2.js';

const loadPromise = guillotiere( base64ToU8( guillotiere_wasm ) );

// Record whether it's loaded, so we can skip the asyncLoader lock if so
let loaded = false;
const lock = asyncLoader.createLock( 'guillotiere wasm' );
loadPromise.then( () => {
  // To debug memory leaks, give this function a `wasm` parameter, and access `wasm.memory.buffer.byteLength`.
  loaded = true;
  lock();
} ).catch( ( err: IntentionalAny ) => {
  lock();
  throw err;
} );

type ChangeList = {
  changes: { old_id: number; new_id: number; new_x: number; new_y: number }[];
  failures: { old_id: number }[];
};

export default class AtlasAllocator {

  private readonly atlas: GuillotiereAtlas;

  // We'll need to look things up by index, and then remap them on "rearrange" resizes
  private readonly binMap = new Map<number, AtlasBin>();

  public constructor( width: number, height: number ) {
    assert && assert( loaded );

    this.atlas = new GuillotiereAtlas( width, height );
  }

  public allocate( width: number, height: number ): AtlasBin | null {
    const internalBin = this.atlas.allocate( width, height );
    if ( internalBin.is_valid() ) {
      const bin = new AtlasBin( internalBin.id, internalBin.x, internalBin.y, width, height );
      this.binMap.set( bin.id, bin );
      internalBin.free(); // it got copied, right?
      return bin;
    }
    else {
      internalBin.free();
      return null;
    }
  }

  public deallocate( bin: AtlasBin ): void {
    this.atlas.deallocate( bin.id );
    this.binMap.delete( bin.id );
  }

  public isEmpty(): boolean {
    return this.atlas.is_empty();
  }

  public clear(): void {
    this.atlas.clear();
  }

  // Tries to make things more compact, without changing the size
  public rearrange(): void {
    const changeList = JSON.parse( this.atlas.rearrange() ) as ChangeList;
    this.applyChangeList( changeList );
  }

  // Tries to make things more compact AND changes the size
  public resizeAndRearrange( width: number, height: number ): void {
    const json = this.atlas.resize_and_rearrange( width, height );
    const changeList = JSON.parse( json ) as ChangeList;
    this.applyChangeList( changeList );
  }

  // Changes the size (increase only), leaving locations as-is
  public grow( width: number, height: number ): void {
    this.atlas.grow( width, height );
  }

  // Guillotiere hands us a change list, which we need to apply to our bins
  private applyChangeList( changeList: ChangeList ): boolean {
    const binsToRemap = [];

    // Extract out bins, and update their info
    for ( let i = 0; i < changeList.changes.length; i++ ) {
      const change = changeList.changes[ i ];
      const bin = this.binMap.get( change.old_id )!;
      this.binMap.delete( bin.id );
      binsToRemap.push( bin );
      bin.id = change.new_id;
      bin.x = change.new_x;
      bin.y = change.new_y;
    }

    // Re-insert bins
    for ( let i = 0; i < binsToRemap.length; i++ ) {
      const bin = binsToRemap[ i ];
      this.binMap.set( bin.id, bin );
    }

    // Somewhat handle failures
    for ( let i = 0; i < changeList.failures.length; i++ ) {
      const failure = changeList.failures[ i ];
      const bin = this.binMap.get( failure.old_id )!;
      this.binMap.delete( bin.id );
    }

    return changeList.failures.length === 0;
  }

  public dispose(): void {
    this.atlas.free(); // We don't keep references to the temporary allocations
  }
}

export class AtlasBin {
  public constructor(
    public id: number,
    public x: number,
    public y: number,
    public readonly width: number,
    public readonly height: number
  ) {}

  public get bounds(): Bounds2 {
    return new Bounds2( this.x, this.y, this.x + this.width, this.y + this.height );
  }
}

scenery.register( 'AtlasAllocator', AtlasAllocator );
