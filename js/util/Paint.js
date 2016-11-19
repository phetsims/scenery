// Copyright 2014-2016, University of Colorado Boulder

/**
 * Base type for gradients and patterns (and NOT the only type for fills/strokes)
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  require( 'SCENERY/util/Color' );
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );

  var globalId = 1;

  function Paint() {
    this.id = 'paint' + globalId++;

    this.transformMatrix = null;
  }

  scenery.register( 'Paint', Paint );

  inherit( Object, Paint, {
    // abstract getCanvasStyle: function()

    isPaint: true,

    setTransformMatrix: function( transformMatrix ) {
      // TODO: invalidate?
      if ( this.transformMatrix !== transformMatrix ) {
        this.transformMatrix = transformMatrix;
      }
      return this;
    }
  } );

  return Paint;
} );
