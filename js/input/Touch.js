// Copyright 2013-2019, University of Colorado Boulder

/**
 * Tracks a single touch point
 *
 * IE guidelines for Touch-friendly sites: http://blogs.msdn.com/b/ie/archive/2012/04/20/guidelines-for-building-touch-friendly-sites.aspx
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( require => {
  'use strict';

  const inherit = require( 'PHET_CORE/inherit' );
  const scenery = require( 'SCENERY/scenery' );

  const Pointer = require( 'SCENERY/input/Pointer' ); // extends Pointer

  /**
   * @extends Pointer
   *
   * @param {number} id
   * @param {Vector2} point
   * @param {Event} event
   * @constructor
   */
  function Touch( id, point, event ) {
    Pointer.call( this, point, true, 'touch' ); // true: touches always start in the down state

    // @public {number} - For tracking which touch is which
    this.id = id;

    sceneryLog && sceneryLog.Pointer && sceneryLog.Pointer( 'Created ' + this.toString() );
  }

  scenery.register( 'Touch', Touch );

  inherit( Pointer, Touch, {

    /**
     * Sets information in this Touch for a given touch move.
     * @public (scenery-internal)
     *
     * @param {Vector2} point
     * @param {Event} event
     * @returns {boolean} - Whether the point changed
     */
    move: function( point, event ) {
      const pointChanged = this.hasPointChanged( point );

      this.point = point;
      return pointChanged;
    },

    /**
     * Sets information in this Touch for a given touch end.
     * @public (scenery-internal)
     *
     * @param {Vector2} point
     * @param {Event} event
     * @returns {boolean} - Whether the point changed
     */
    end: function( point, event ) {
      const pointChanged = this.hasPointChanged( point );

      this.point = point;
      this.isDown = false;
      return pointChanged;
    },

    /**
     * Sets information in this Touch for a given touch cancel.
     * @public (scenery-internal)
     *
     * @param {Vector2} point
     * @param {Event} event
     * @returns {boolean} - Whether the point changed
     */
    cancel: function( point, event ) {
      const pointChanged = this.hasPointChanged( point );

      this.point = point;
      this.isDown = false;
      return pointChanged;
    },

    /**
     * Returns an improved string representation of this object.
     * @public
     * @override
     *
     * @returns {string}
     */
    toString: function() {
      return 'Touch#' + this.id;
    }
  } );

  return Touch;
} );
