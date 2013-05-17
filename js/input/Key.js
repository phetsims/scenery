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
  'use strict';
  
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  
  var Pointer = require( 'SCENERY/input/Pointer' ); // Inherits from Pointer
  
  scenery.Key = function Key( event ) {
    Pointer.call( this );
    
    this.event = event; // event.keyCode event.charCode
    this.isKey = true; // compared to isMouse/isPen/isTouch
    this.trail = null;
    this.type = 'key';
  };
  var Key = scenery.Key;
  
  inherit( Key, Pointer, {
    
  } );
  
  return Key;
} );
