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
  
  var scenery = require( 'SCENERY/scenery' );
  
  var Pointer = require( 'SCENERY/input/Pointer' ); // Inherits from Pointer
  
  scenery.Key = function( key, event ) {
    Pointer.call( this );
    
    this.key = key;
    this.isKey = true;
    this.trail = null;
    this.type = 'key';
  };
  var Key = scenery.Key;
  
  Key.prototype = _.extend( {}, Pointer.prototype, {
    constructor: Key
  } );
  
  return Key;
} );
