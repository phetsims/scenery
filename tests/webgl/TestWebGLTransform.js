// Copyright 2002-2014, University of Colorado Boulder

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
      var sceneryRectangle = new Rectangle( 150, 200, 1024 / 2, 100, { fill: 'red', rotation: Math.PI / 16 } );
      var myRectangle = colorTriangleBufferData.createFromRectangle( sceneryRectangle, 0.5 );

      webGLRenderer.colorTriangleRenderer.bindVertexBuffer();
      webGLRenderer.start();

      webGLRenderer.events.on( 'step', function() {

        sceneryRectangle.resetTransform();

        var angle = Date.now() / 1000 * 2 * Math.PI / 10;
        sceneryRectangle.rotateAround( { x: 600, y: 600 }, angle );
        myRectangle.setTransform( sceneryRectangle.getLocalToGlobalMatrix().toMatrix4() );
        webGLRenderer.colorTriangleRenderer.updateTriangleBuffer( myRectangle );
      } );

      console.log( 'total triangles', colorTriangleBufferData.vertexArray.length / 3 );
    }
  };
} );