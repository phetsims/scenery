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

  /**
   *
   * @constructor
   */
  function SquareUnstrokedRectangle( colorTriangleRenderer, rectangle, z ) {
    assert && assert( z !== undefined );
    this.z = z;
    this.rectangle = rectangle;
    this.colorTriangleBufferData = colorTriangleRenderer.colorTriangleBufferData;
    this.colorTriangleRenderer = colorTriangleRenderer;

    // Reserve two triangles
    this.indexObject = this.colorTriangleBufferData.reserveVertices( 6 );
    this.startIndex = this.indexObject.startIndex;
    this.endIndex = this.indexObject.endIndex;//exclusive (not included)

    this.update();
  }

  return inherit( Object, SquareUnstrokedRectangle, {
    update: function() {

      var x = this.rectangle._rectX;
      var y = this.rectangle._rectY;
      var width = this.rectangle._rectWidth;
      var height = this.rectangle._rectHeight;
      var z = this.z;

      //TODO: Use this
      var color = this.rectangle.fillColor;//read only reference

      // TODO: maybe better to update in fragment shader?  It depends how often we update()
      var r = color.red / 255;
      var g = color.green / 255;
      var b = color.blue / 255;
      var a = color.alpha;

      var matrix4 = this.rectangle.getLocalToGlobalMatrix().toMatrix4();

      // TODO: Maybe create arrays here every time is creating too much garbage.
      // TODO: Could just unroll this later once things settle down.
      var newElements = [
        // Top left
        //TODO: Maybe should be m03 for last element, see Matrix3.toAffineMatrix4
        x, y, z, r, g, b, a, /*               */matrix4.m00(), matrix4.m01(), matrix4.m03(), matrix4.m10(), matrix4.m11(), matrix4.m13(),
        (x + width), y, z, r, g, b, a, /*     */matrix4.m00(), matrix4.m01(), matrix4.m03(), matrix4.m10(), matrix4.m11(), matrix4.m13(),
        x, y + height, z, r, g, b, a, /*      */matrix4.m00(), matrix4.m01(), matrix4.m03(), matrix4.m10(), matrix4.m11(), matrix4.m13(),

        // Bottom right
        (x + width), y + height, z, r, g, b, a, matrix4.m00(), matrix4.m01(), matrix4.m03(), matrix4.m10(), matrix4.m11(), matrix4.m13(),
        (x + width), y, z, r, g, b, a, /*     */matrix4.m00(), matrix4.m01(), matrix4.m03(), matrix4.m10(), matrix4.m11(), matrix4.m13(),
        x, y + height, z, r, g, b, a, /*      */matrix4.m00(), matrix4.m01(), matrix4.m03(), matrix4.m10(), matrix4.m11(), matrix4.m13()
      ];
      for ( var i = 0; i < newElements.length; i++ ) {
        this.colorTriangleBufferData.vertexArray[ this.startIndex + i ] = newElements[ i ];
      }
      this.colorTriangleRenderer.updateTriangleBuffer( this );
    }
  } );
} );