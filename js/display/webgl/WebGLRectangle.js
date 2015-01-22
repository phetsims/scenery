//  Copyright 2002-2014, University of Colorado Boulder

/**
 * A scenery Rectangle can actually take many structurally different forms: stroked/unstroked,
 * rounded/squared and these would change how it is handled by the WebGL shaders & buffer data.
 * This class watches a scenery Rectangle and decides which shader implementation to use.
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var inherit = require( 'PHET_CORE/inherit' );
  var SquareUnstrokedRectangle = require( 'SCENERY/display/webgl/SquareUnstrokedRectangle' );

  /**
   *
   * @constructor
   */
  function WebGLRectangle( webglRenderer, rectangle ) {
    this.webglRenderer = webglRenderer;
    this.rectangle = rectangle;
    this.update( rectangle );
  }

  return inherit( Object, WebGLRectangle, {
    create: function() {
      this.rectangleHandle = (!this.rectangle.isRounded() && this.rectangle.stroke === null) ?
                             new SquareUnstrokedRectangle() :
                             (this.rectangle.isRounded()) ?
                             null :
                             null;
//                             new RoundedRectangle() :
//                             new RoundedStrokedRectangle();
    },
    update: function() {

      if ( this.rectangleHandle === null ) {
        this.create();
      }
      else if ( this.rectangleHandle.supports( this.rectangle ) ) {
        this.rectangleHandle.update( this.rectangle );
      }
      else {
        this.rectangleHandle.dispose();
        this.create();
      }
    }
  } );
} );