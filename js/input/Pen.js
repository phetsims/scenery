// Copyright 2002-2014, University of Colorado

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

  scenery.Pen = function Pen( id, point, event ) {
    Pointer.call( this );

    this.id = id;
    this.point = point;
    this.isPen = true;
    this.trail = null;

    this.isDown = true; // pens always start down? TODO: is this true with pointer events?

    this.type = 'pen';
  };
  var Pen = scenery.Pen;

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
