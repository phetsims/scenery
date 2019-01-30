// Copyright 2013-2015, University of Colorado Boulder

/**
 * Tracks a single touch point
 *
 * IE guidelines for Touch-friendly sites: http://blogs.msdn.com/b/ie/archive/2012/04/20/guidelines-for-building-touch-friendly-sites.aspx
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
  function Touch( id, point, event ) {
    Pointer.call( this, point, true, 'touch' ); // true: touches always start in the down state

    this.id = id;

    sceneryLog && sceneryLog.Pointer && sceneryLog.Pointer( 'Created ' + this.toString() );
  }

  scenery.register( 'Touch', Touch );

  inherit( Pointer, Touch, {

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
      return 'Touch#' + this.id;
    }
  } );

  return Touch;
} );
