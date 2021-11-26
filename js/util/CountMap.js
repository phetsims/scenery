// Copyright 2020-2021, University of Colorado Boulder

/**
 * Data structure that handles creating/destroying related objects that need to exist when something's count is >=1
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Poolable from '../../../phet-core/js/Poolable.js';
import { scenery } from '../imports.js';

class CountMap {
  /**
   * @param {function(*):*} create
   * @param {function(*,*)} destroy
   */
  constructor( create, destroy ) {

    // @private
    this.create = create;
    this.destroy = destroy;

    // @private {Map.<*,Entry>}
    this.map = new Map();
  }

  /**
   * @public
   *
   * @param {*} key
   * @param {number} [quantity]
   */
  increment( key, quantity = 1 ) {
    assert && assert( typeof quantity === 'number' );
    assert && assert( quantity >= 1 );

    if ( this.map.has( key ) ) {
      this.map.get( key ).count += quantity;
    }
    else {
      const value = this.create( key );
      const entry = CountMapEntry.createFromPool( quantity, key, value );
      this.map.set( key, entry );
    }
  }

  /**
   * @public
   *
   * @param {*} key
   * @param {number} [quantity]
   */
  decrement( key, quantity = 1 ) {
    assert && assert( typeof quantity === 'number' );
    assert && assert( quantity >= 1 );

    const entry = this.map.get( key );

    // Was an old comment of
    // > since the block may have been disposed (yikes!), we have a defensive set-up here
    // So we're playing it extra defensive here for now
    if ( entry ) {
      entry.count -= quantity;
      if ( entry.count < 1 ) {
        this.destroy( key, entry.value );
        this.map.delete( key );
        entry.dispose();
      }
    }
  }

  /**
   * @public
   *
   * @param {*} key
   * @returns {*}
   */
  get( key ) {
    return this.map.get( key ).value;
  }

  /**
   * @public
   *
   * NOTE: We COULD try to collect all of the CountMapEntries... but that seems like an awful lot of CPU.
   * If GC is an issue from this, we can add more logic
   */
  clear() {
    this.map.clear();
  }
}

class CountMapEntry {
  /**
   * @param {number} count
   * @param {*} key
   * @param {*} value
   */
  constructor( count, key, value ) {
    this.initialize( count, key, value );
  }

  /**
   * @public
   *
   * @param {number} count
   * @param {*} key
   * @param {*} value
   */
  initialize( count, key, value ) {

    // @public {number}
    this.count = count;

    // @public {*}
    this.key = key;
    this.value = value;
  }

  /**
   * Releases references
   * @public
   */
  dispose() {
    // Null things out, to prevent leaks (in case)
    this.key = null;
    this.value = null;

    this.freeToPool();
  }
}

Poolable.mixInto( CountMapEntry, {
  initialize: CountMapEntry.prototype.initialize
} );

scenery.register( 'CountMap', CountMap );
export default CountMap;