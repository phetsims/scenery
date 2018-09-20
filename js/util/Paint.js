// Copyright 2014-2017, University of Colorado Boulder

/**
 * Base type for gradients and patterns (and NOT the only type for fills/strokes)
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );

  var globalId = 1;

  /**
   * @constructor
   * @extends {Object}
   */
  function Paint() {
    // @public (scenery-internal) {string}
    this.id = 'paint' + globalId++;

    // @protected {Matrix3|null}
    this.transformMatrix = null;
  }

  scenery.register( 'Paint', Paint );

  inherit( Object, Paint, {
    // @public {boolean}
    isPaint: true,

    /**
     * Returns an object that can be passed to a Canvas context's fillStyle or strokeStyle.
     * @public
     *
     * @returns {*}
     */
    getCanvasStyle: function() {
      throw new Error( 'abstract method' );
    },

    /**
     * Sets how this paint (pattern/gradient) is transformed, compared with the local coordinate frame of where it is
     * used.
     * @public
     *
     * NOTE: This should only be used before the pattern/gradient is ever displayed.
     * TODO: Catch if this is violated?
     *
     * @param {Matrix3} transformMatrix
     * @returns {Paint} - for chaining
     */
    setTransformMatrix: function( transformMatrix ) {
      if ( this.transformMatrix !== transformMatrix ) {
        this.transformMatrix = transformMatrix;
      }
      return this;
    }
  } );

  return Paint;
} );
