//  Copyright 2002-2014, University of Colorado Boulder

/**
 * This WebGL renderer is used to draw images as textures on rectangles.
 * TODO: Can this same pattern be used for interleaved texture coordinates? (Or other interleaved data?)
 * TODO: Work in progress, much to be done here!
 * TODO: Add this file to the list of scenery files (for jshint, etc.)
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
  function TextureBufferData() {

    //TODO: Use Float32Array -- though we will have to account for the fact that they have a fixed size
    this.vertexArray = [];
    this.textureCoordinates = [];
  }

  return inherit( Object, TextureBufferData, {

    createFromImage: function( x, y, width, height, image ) {
      var textureBufferData = this;
      var index = this.vertexArray.length;

      var x1 = x;
      var x2 = x + width;
      var y1 = y;
      var y2 = y + height;

      this.vertexArray.push(
        x1, y1,
        x2, y1,
        x1, y2,
        x1, y2,
        x2, y1,
        x2, y2
      );

      // Add the same color for all vertices (solid fill rectangle).
      // TODO: some way to reduce this amount of elements!
      this.textureCoordinates.push(
        0.0, 0.0,
        1.0, 0.0,
        0.0, 1.0,

        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0
      );

      //Track the index so it can delete itself, update itself, etc.
      //TODO: Move to a separate class.
      return {
        initialState: {x: x, y: y, width: width, height: height},
        index: index,
        endIndex: textureBufferData.vertexArray.length,
        setXWidth: function( x, width ) {
          textureBufferData.vertexArray[index] = x;
          textureBufferData.vertexArray[index + 2] = x + width;
          textureBufferData.vertexArray[index + 4] = x;
          textureBufferData.vertexArray[index + 6] = x + width;
          textureBufferData.vertexArray[index + 8] = x + width;
          textureBufferData.vertexArray[index + 10] = x;
        },
        setRect: function( x, y, width, height ) {

          textureBufferData.vertexArray[index] = x;
          textureBufferData.vertexArray[index + 1] = y;

          textureBufferData.vertexArray[index + 2] = x + width;
          textureBufferData.vertexArray[index + 3] = y;

          textureBufferData.vertexArray[index + 4] = x;
          textureBufferData.vertexArray[index + 5] = y + height;

          textureBufferData.vertexArray[index + 6] = x + width;
          textureBufferData.vertexArray[index + 7] = y + height;

          textureBufferData.vertexArray[index + 8] = x + width;
          textureBufferData.vertexArray[index + 9] = y;

          textureBufferData.vertexArray[index + 10] = x;
          textureBufferData.vertexArray[index + 11] = y + height;
        }
      };
    }
  } );
} );