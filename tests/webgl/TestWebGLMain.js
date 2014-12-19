//  Copyright 2002-2014, University of Colorado Boulder

/**
 *
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
  var TestWebGL = require( 'SCENERY/display/webgl/TestWebGL' );

  /**
   *
   * @constructor
   */
  function TestWebGLMain() {
  }

  return {start: function() {

    var testWebGL = new TestWebGL();

    // TODO: Add a uniform matrix4 for transforming vertices to the -1,-1,1,1 rectangle
    testWebGL.trianglesGeometry.createFromRectangle( new Rectangle( 0, 0, 1024 / 2, 100, {fill: 'red'} ) );
    testWebGL.trianglesGeometry.createFromRectangle( new Rectangle( 100, 0, 100, 100, {fill: 'green'} ) );
    testWebGL.trianglesGeometry.createFromRectangle( new Rectangle( 200, 0, 100, 100, {fill: 'blue'} ) );
    testWebGL.trianglesGeometry.createFromPath( new Rectangle( 100, 100, 100, 100, 20, 20, {fill: 'blue'} ) );
    testWebGL.trianglesGeometry.createFromPath( new Path( Shape.circle( 300, 300, 50 ), {fill: 'blue'} ) );
    testWebGL.trianglesGeometry.createFromPath( new Path( Shape.circle( 600, 600, 200 ), {fill: 'red'} ) );

    for ( var i = 0; i < 50; i++ ) {
      var circle = Shape.circle( 600 * Math.random(), 600 * Math.random(), 50 * Math.random() );
      var path = new Path( circle, {
        fill: new Color( Math.random() * 255, Math.random() * 255, Math.random() * 255, 1 )
      } );
      testWebGL.trianglesGeometry.createFromPath( path );
    }
    testWebGL.bindVertexBuffer();
    testWebGL.bindColorBuffer();

//      testWebGL.draw();

    testWebGL.start();
  }};
} );