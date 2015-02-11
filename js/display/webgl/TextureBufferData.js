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
    this.vertexArray = [];
  }

  return inherit( Object, TextureBufferData, {
    //createFromImageNode: function( imageNode, z, frameRange ) {
    //  return this.createFromImage( imageNode.x, imageNode.y, z,
    //    imageNode._image.width, imageNode._image.height, imageNode.image, imageNode.getLocalToGlobalMatrix().toMatrix4(), frameRange );
    //}
  } );
} );