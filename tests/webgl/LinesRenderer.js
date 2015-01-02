//  Copyright 2002-2014, University of Colorado Boulder

/**
 * Demonstration of a custom WebGL renderer (like CanvasNode), in this case which draws lines.
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var inherit = require( 'PHET_CORE/inherit' );

  // shaders
  var lineVertexShader = require( 'text!SCENERY/display/webgl/lines.vert' );
  var lineFragmentShader = require( 'text!SCENERY/display/webgl/lines.frag' );

  /**
   *
   * @constructor
   */
  function LinesRenderer() {
  }

  return inherit( Object, LinesRenderer, {
    draw: function() {}
  } );
} );