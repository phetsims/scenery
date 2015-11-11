// Copyright 2014-2015, University of Colorado Boulder

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

  function Key( event ) {
    Pointer.call( this );

    this.event = event; // event.keyCode event.charCode
    this.isKey = true; // compared to isMouse/isPen/isTouch
    this.trail = null;
    this.type = 'key';
  }

  scenery.register( 'Key', Key );

  return inherit( Pointer, Key, {
    firesGenericEvent: false // don't fire 'down', 'up' and the other generic events
  } );
} );