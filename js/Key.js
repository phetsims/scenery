// Copyright 2002-2012, University of Colorado

/**
 * Tracks a single key-press
 *
 * TODO: general key-press implementation
 * TODO: consider separate handling for keys in general.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  
  var Finger = require( 'SCENERY/Finger' );
  
  var Key = function( key, event ) {
    Finger.call( this );
    
    this.key = key;
    this.isKey = true;
    this.trail = null;
  };
  
  Key.prototype = _.extend( {}, Finger.prototype, {
    constructor: Key
  } );
  
  return Key;
} );
