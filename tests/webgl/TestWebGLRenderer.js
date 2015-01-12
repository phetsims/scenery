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
  var Image = require( 'SCENERY/nodes/Image' );
  var Path = require( 'SCENERY/nodes/Path' );
  var Shape = require( 'KITE/Shape' );
  var Color = require( 'SCENERY/util/Color' );
  var WebGLRenderer = require( 'SCENERY/display/webgl/WebGLRenderer' );
  var LinesRenderer = require( 'SCENERY/../tests/webgl/LinesRenderer' );
  var Matrix4 = require( 'DOT/Matrix4' );
  var SquareUnstrokedRectangle = require( 'SCENERY/display/webgl/SquareUnstrokedRectangle' );

  // images
  var mountains = require( 'image!ENERGY_SKATE_PARK_BASICS/mountains.png' );

  return {
    start: function() {

      var webGLRenderer = new WebGLRenderer();

      // TODO: Add a uniform matrix4 for transforming vertices to the -1,-1,1,1 rectangle
      var colorTriangleBufferData = webGLRenderer.colorTriangleRenderer.colorTriangleBufferData;
      var rect1 = new SquareUnstrokedRectangle( webGLRenderer.colorTriangleRenderer, new Rectangle( 0, 0, 1024 / 2, 100, {fill: 'red'} ), 0.5 );
      var rect2 = new SquareUnstrokedRectangle( webGLRenderer.colorTriangleRenderer, new Rectangle( 100, 0, 100, 100, {fill: 'green'} ), 0.5 );
      var blueSquare = new SquareUnstrokedRectangle( webGLRenderer.colorTriangleRenderer, new Rectangle( 200, 300, 100, 100, {fill: 'blue'} ), 0.5 );
      colorTriangleBufferData.createFromPath( new Rectangle( 100, 100, 100, 100, 20, 20, {fill: 'blue'} ), 0.5 );
      colorTriangleBufferData.createFromPath( new Path( Shape.circle( 300, 300, 50 ), {fill: 'blue'} ), 0.5 );
      colorTriangleBufferData.createFromPath( new Path( Shape.circle( 600, 600, 200 ), {fill: 'red'} ), 0.5 );

      // Sample shape that will rotate
      var largeMovingRectangle = new SquareUnstrokedRectangle( webGLRenderer.colorTriangleRenderer, new Rectangle( 150, 200, 1024 / 2, 100, {fill: 'red', rotation: Math.PI / 16} ), 0.5 );

      for ( var i = 0; i < 50; i++ ) {
        var circle = Shape.circle( 600 * Math.random(), 600 * Math.random(), 50 * Math.random() );
        var path = new Path( circle, {
          fill: new Color( Math.random() * 255, Math.random() * 255, Math.random() * 255, 1 )
        } );
        colorTriangleBufferData.createFromPath( path, 0.5 );
      }

      colorTriangleBufferData.createFromTriangle( 100, 100, 200, 100, 150, 200, 'black', 0.5 );
      var redTriangle = colorTriangleBufferData.createFromTriangle( 100, 100, 200, 100, 150, 200, 'red', 0.5 );
      colorTriangleBufferData.createFromTriangle( 100, 200, 200, 200, 150, 300, 'blue', 0.5 );

      //Show something from another module
      var images = [];
      for ( var i = 0; i < 100; i++ ) {
        var imageNode = new Image( mountains, {x: i * 2, y: 0} );
        var image = webGLRenderer.textureRenderer.textureBufferData.createFromImageNode( imageNode, Math.random() );
        images.push( image );
      }
      webGLRenderer.textureRenderer.bindVertexBuffer();
      webGLRenderer.textureRenderer.bindDirtyTextures();

      webGLRenderer.colorTriangleRenderer.bindVertexBuffer();

      webGLRenderer.addCustomWebGLRenderer( new LinesRenderer( webGLRenderer.gl, webGLRenderer.backingScale, webGLRenderer.canvas ) );

      webGLRenderer.start();

      webGLRenderer.events.on( 'step', function() {
        var rectX = Math.cos( Date.now() / 1000.0 * 2 * Math.PI / 2 ) * 100 + 300;

        blueSquare.rectangle.rectX = rectX;
        blueSquare.update();

        redTriangle.setTriangle( 100 + rectX, 100, 200 + rectX, 100, 150 + rectX, 200 );

        // Transform the shape
        largeMovingRectangle.rectangle.resetTransform();
        var angle = Date.now() / 1000 * 2 * Math.PI / 10;
        largeMovingRectangle.rectangle.rotateAround( {x: 600, y: 600}, angle );
        largeMovingRectangle.update();

        webGLRenderer.colorTriangleRenderer.updateTriangleBuffer( redTriangle );

        for ( var i = 0; i < images.length; i++ ) {
          var image = images[i];
          var y = rectX + i * 10;
          var translateX = i * 2;
          var translateY = y / (i + 1);
          debugger;
          image.setTransform( Matrix4.translation( translateX, translateY, 0 ) );
          webGLRenderer.textureRenderer.updateTriangleBuffer( image );
        }

        // Experimental alternative to bufferSubData
//        webGLRenderer.colorTriangleRenderer.reBufferData();
      } );

      console.log( 'total triangles', colorTriangleBufferData.vertexArray.length / 13 / 3 );
    }};
} );