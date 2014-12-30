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

  return {
    start: function() {

      var webGLRenderer = new WebGLRenderer();

      // TODO: Add a uniform matrix4 for transforming vertices to the -1,-1,1,1 rectangle
      webGLRenderer.colorTriangleRenderer.colorTriangleBufferData.createFromRectangle( new Rectangle( 0, 0, 1024 / 2, 100, {fill: 'red'} ) );
      webGLRenderer.colorTriangleRenderer.colorTriangleBufferData.createFromRectangle( new Rectangle( 100, 0, 100, 100, {fill: 'green'} ) );
      var rectangle = new Rectangle( 200, 300, 100, 100, {fill: 'blue'} );
      var rectangleGeometry = webGLRenderer.colorTriangleRenderer.colorTriangleBufferData.createFromRectangle( rectangle );
      webGLRenderer.colorTriangleRenderer.colorTriangleBufferData.createFromPath( new Rectangle( 100, 100, 100, 100, 20, 20, {fill: 'blue'} ) );
      webGLRenderer.colorTriangleRenderer.colorTriangleBufferData.createFromPath( new Path( Shape.circle( 300, 300, 50 ), {fill: 'blue'} ) );
      webGLRenderer.colorTriangleRenderer.colorTriangleBufferData.createFromPath( new Path( Shape.circle( 600, 600, 200 ), {fill: 'red'} ) );

      for ( var i = 0; i < 50; i++ ) {
        var circle = Shape.circle( 600 * Math.random(), 600 * Math.random(), 50 * Math.random() );
        var path = new Path( circle, {
          fill: new Color( Math.random() * 255, Math.random() * 255, Math.random() * 255, 1 )
        } );
        webGLRenderer.colorTriangleRenderer.colorTriangleBufferData.createFromPath( path );
      }

      var t1 = webGLRenderer.colorTriangleRenderer.colorTriangleBufferData.createFromTriangle( 100, 100, 200, 100, 150, 200 );
      var t2 = webGLRenderer.colorTriangleRenderer.colorTriangleBufferData.createFromTriangle( 100, 100, 200, 100, 150, 200 );
      var t3 = webGLRenderer.colorTriangleRenderer.colorTriangleBufferData.createFromTriangle( 100, 200, 200, 200, 150, 300 );

      //Show something from another module
      webGLRenderer.textureRenderer.colorTriangleBufferData.createFromRectangle( new Rectangle( 0, 0, 1024 / 2, 1024 / 2, {fill: 'black'} ) );
      webGLRenderer.textureRenderer.bindVertexBuffer();
      webGLRenderer.textureRenderer.bindColorBuffer();

      webGLRenderer.colorTriangleRenderer.bindVertexBuffer();
      webGLRenderer.colorTriangleRenderer.bindColorBuffer();

      webGLRenderer.start();

      webGLRenderer.events.on( 'step', function() {
        var rectX = Math.cos( Date.now() / 1000.0 * 2 * Math.PI / 2 ) * 100 + 300;
//      var rectX = 50;
//
        rectangleGeometry.setXWidth( rectX, 100 );
//
        t2.setTriangle( 100 + rectX, 100, 200 + rectX, 100, 150 + rectX, 200 );

        webGLRenderer.colorTriangleRenderer.updateTriangleBuffer( t2 );
        webGLRenderer.colorTriangleRenderer.updateTriangleBuffer( rectangleGeometry );
      } );

      console.log( 'total triangles', webGLRenderer.colorTriangleRenderer.colorTriangleBufferData.vertexArray.length / 3 );
    }};
} );