// Copyright 2013-2015, University of Colorado Boulder


/**
 * Tracks a stylus ('pen') or something with tilt and pressure information
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );

  var Pointer = require( 'SCENERY/input/Pointer' ); // extends Pointer

  /**
   * @extends Pointer
   * @param {number} id
   * @param {Vector2} point
   * @param {DOMEvent} event
   * @constructor
   */
  function Pen( id, point, event ) {
    Pointer.call( this, point, true, 'pen' ); // true: pen pointers always start in the down state

    this.id = id;

    sceneryLog && sceneryLog.Pointer && sceneryLog.Pointer( 'Created ' + this.toString() );
  }

  scenery.register( 'Pen', Pen );

  inherit( Pointer, Pen, {

    move: function( point, event ) {
      var pointChanged = this.hasPointChanged( point );
      // if ( this.point ) { this.point.freeToPool(); }
      this.point = point;
      return pointChanged;
    },

    end: function( point, event ) {
      var pointChanged = this.hasPointChanged( point );
      // if ( this.point ) { this.point.freeToPool(); }
      this.point = point;
      this.isDown = false;
      return pointChanged;
    },

    cancel: function( point, event ) {
      var pointChanged = this.hasPointChanged( point );
      // if ( this.point ) { this.point.freeToPool(); }
      this.point = point;
      this.isDown = false;
      return pointChanged;
    },

    toString: function() {
      return 'Pen#' + this.id;
    }
  } );

  return Pen;
} );
