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

  function Paint() {
    // @public (scenery-internal) {string}
    this.id = 'paint' + globalId++;

    // @protected {Matrix3|null}
    this.transformMatrix = null;
  }

  scenery.register( 'Paint', Paint );

  inherit( Object, Paint, {
    isPaint: true,

    // TODO: setting this after use of the paint is not currently supported
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
