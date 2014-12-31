//  Copyright 2002-2014, University of Colorado Boulder

/**
 * Simple test harness for experimenting with WebGL features.
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var inherit = require( 'PHET_CORE/inherit' );
  var Rectangle = require( 'SCENERY/nodes/Rectangle' );
  var Path = require( 'SCENERY/nodes/Path' );
  var Shape = require( 'KITE/Shape' );
  var Color = require( 'SCENERY/util/Color' );
  var WebGLRenderer = require( 'SCENERY/display/webgl/WebGLRenderer' );

  // images
  var mountains = require( 'image!ENERGY_SKATE_PARK_BASICS/mountains.png' );

  return {
    start: function() {

      var webGLRenderer = new WebGLRenderer();

      // TODO: Add a uniform matrix4 for transforming vertices to the -1,-1,1,1 rectangle
      var colorTriangleBufferData = webGLRenderer.colorTriangleRenderer.colorTriangleBufferData;
      colorTriangleBufferData.createFromRectangle( new Rectangle( 150, 200, 1024 / 2, 100, {fill: 'red'} ), 0.5 );
      colorTriangleBufferData.createFromTriangle( 100, 200, 300, 300, 550, 300, 'blue', 0.4 );

      webGLRenderer.colorTriangleRenderer.bindVertexBuffer();
      webGLRenderer.colorTriangleRenderer.bindColorBuffer();

      webGLRenderer.start();

      webGLRenderer.events.on( 'step', function() {

      } );

      console.log( 'total triangles', colorTriangleBufferData.vertexArray.length / 3 );
    }};
} );