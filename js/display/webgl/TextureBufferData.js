// Copyright 2002-2014, University of Colorado Boulder

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
    this.vertexArray = [];
    this.elementsPerVertex = 11;
  }

  return inherit( Object, TextureBufferData, {
    reserveVertices: function( numVertices ) {
      var startIndex = this.vertexArray.length;
      for ( var i = 0; i < numVertices; i++ ) {
        for ( var k = 0; k < this.elementsPerVertex; k++ ) {
          this.vertexArray.push( 0 );
        }
      }
      var endIndex = this.vertexArray.length;
      return { startIndex: startIndex, endIndex: endIndex };
    }
    //createFromImageNode: function( imageNode, z, frameRange ) {
    //  return this.createFromImage( imageNode.x, imageNode.y, z,
    //    imageNode._image.width, imageNode._image.height, imageNode.image, imageNode.getLocalToGlobalMatrix().toMatrix4(), frameRange );
    //}
  } );
} );