// Copyright 2014-2017, University of Colorado Boulder

/**
 * Base type for gradients and patterns (and NOT the only type for fills/strokes)
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var Color = require( 'SCENERY/util/Color' );
  var inherit = require( 'PHET_CORE/inherit' );
  var Property = require( 'AXON/Property' );
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
  }, {
    /**
     * Returns whether the given object is a general "paint" (something that can be provided to a fill/stroke).
     * @public
     *
     * @param {*}
     * @returns {boolean}
     */
    isPaint: function( paint ) {
      return paint === null ||
             typeof paint === 'string' ||
             paint instanceof Color ||
             paint instanceof Paint ||
             ( paint instanceof Property && ( typeof paint.value === 'string' || paint.value instanceof Color ) );

    },

    /**
     * Takes a snapshot of the given paint, returning the current color where possible.
     * @public
     *
     * @param {null|string|Color|Property.<string|Color>|LinearGradient|RadialGradient|Pattern} paint
     * @returns {Color}
     */
    toColor: function( paint ) {
      if ( typeof paint === 'string' ) {
        return new Color( paint );
      }
      if ( paint instanceof Color ) {
        return paint.copy();
      }
      if ( paint instanceof Property ) {
        return Paint.toColor( paint.value );
      }
      if ( paint instanceof scenery.Gradient ) {
        // Average the stops
        var color = Color.TRANSPARENT;
        var quantity = 0;
        paint.stops.forEach( function( stop ) {
          color = color.blend( Paint.toColor( stop.color ), 1 / ( quantity + 1 ) );
        } );
        return color;
      }

      // Fall-through value (null, Pattern, etc.)
      return Color.TRANSPARENT;
    }
  } );

  return Paint;
} );
