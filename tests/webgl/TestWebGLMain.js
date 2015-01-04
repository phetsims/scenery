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
  var LinesRenderer = require( 'SCENERY/../tests/webgl/LinesRenderer' );
  var Matrix4 = require( 'DOT/Matrix4' );

  // images
  var mountains = require( 'image!ENERGY_SKATE_PARK_BASICS/mountains.png' );

  return {
    start: function() {

      var webGLRenderer = new WebGLRenderer();

      // TODO: Add a uniform matrix4 for transforming vertices to the -1,-1,1,1 rectangle
      var colorTriangleBufferData = webGLRenderer.colorTriangleRenderer.colorTriangleBufferData;
      colorTriangleBufferData.createFromRectangle( new Rectangle( 0, 0, 1024 / 2, 100, {fill: 'red'} ), 0.5 );
      colorTriangleBufferData.createFromRectangle( new Rectangle( 100, 0, 100, 100, {fill: 'green'} ), 0.5 );
      var rectangle = new Rectangle( 200, 300, 100, 100, {fill: 'blue'}, 0.5 );
      var rectangleGeometry = colorTriangleBufferData.createFromRectangle( rectangle, 0.5 );
      colorTriangleBufferData.createFromPath( new Rectangle( 100, 100, 100, 100, 20, 20, {fill: 'blue'} ), 0.5 );
      colorTriangleBufferData.createFromPath( new Path( Shape.circle( 300, 300, 50 ), {fill: 'blue'} ), 0.5 );
      colorTriangleBufferData.createFromPath( new Path( Shape.circle( 600, 600, 200 ), {fill: 'red'} ), 0.5 );

      // Sample shape that will rotate
      var sceneryRectangle = new Rectangle( 150, 200, 1024 / 2, 100, {fill: 'red', rotation: Math.PI / 16} );
      var myRectangle = colorTriangleBufferData.createFromRectangle( sceneryRectangle, 0.5 );

      for ( var i = 0; i < 50; i++ ) {
        var circle = Shape.circle( 600 * Math.random(), 600 * Math.random(), 50 * Math.random() );
        var path = new Path( circle, {
          fill: new Color( Math.random() * 255, Math.random() * 255, Math.random() * 255, 1 )
        } );
        colorTriangleBufferData.createFromPath( path, 0.5 );
      }

      var t1 = colorTriangleBufferData.createFromTriangle( 100, 100, 200, 100, 150, 200, 'black', 0.5 );
      var t2 = colorTriangleBufferData.createFromTriangle( 100, 100, 200, 100, 150, 200, 'red', 0.5 );
      var t3 = colorTriangleBufferData.createFromTriangle( 100, 200, 200, 200, 150, 300, 'blue', 0.5 );

      //Show something from another module
      var images = [];
      for ( var i = 0; i < 100; i++ ) {
        var matrix4 = Matrix4.identity();
        var image = webGLRenderer.textureRenderer.textureBufferData.createFromImage( i * 2, 0, 256, 256, mountains, matrix4 );
        images.push( image );
      }
      webGLRenderer.textureRenderer.bindVertexBuffer();

      webGLRenderer.colorTriangleRenderer.bindVertexBuffer();

      webGLRenderer.addCustomWebGLRenderer( new LinesRenderer( webGLRenderer.gl, webGLRenderer.backingScale, webGLRenderer.canvas ) );

      webGLRenderer.start();

      webGLRenderer.events.on( 'step', function() {
        var rectX = Math.cos( Date.now() / 1000.0 * 2 * Math.PI / 2 ) * 100 + 300;
//      var rectX = 50;
//
        rectangleGeometry.setXWidth( rectX, 100 );
//
        t2.setTriangle( 100 + rectX, 100, 200 + rectX, 100, 150 + rectX, 200 );

        // Transform the shape
        sceneryRectangle.resetTransform();
        var angle = Date.now() / 1000 * 2 * Math.PI / 10;
        sceneryRectangle.rotateAround( {x: 600, y: 600}, angle );
        myRectangle.setTransform( sceneryRectangle.getLocalToGlobalMatrix().toMatrix4() );

        webGLRenderer.colorTriangleRenderer.updateTriangleBuffer( myRectangle );
        webGLRenderer.colorTriangleRenderer.updateTriangleBuffer( t2 );
        webGLRenderer.colorTriangleRenderer.updateTriangleBuffer( rectangleGeometry );

        for ( var i = 0; i < images.length; i++ ) {
          var image = images[i];
          var y = rectX + i * 10;
          image.setTransform( Matrix4.translation( i * 2, y / i, 0 ) );
          webGLRenderer.textureRenderer.updateTriangleBuffer( image );
        }

        // Experimental alternative to bufferSubData
//        webGLRenderer.colorTriangleRenderer.reBufferData();
      } );

      console.log( 'total triangles', colorTriangleBufferData.vertexArray.length / 13 / 3 );
    }};
} );