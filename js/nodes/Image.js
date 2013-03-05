// Copyright 2002-2012, University of Colorado

/**
 * Images
 *
 * TODO: setImage / getImage and the whole toolchain that uses that
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  "use strict";
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var Bounds2 = require( 'DOT/Bounds2' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  var Node = require( 'SCENERY/nodes/Node' ); // Image inherits from Node
  var Renderer = require( 'SCENERY/layers/Renderer' ); // we need to specify the Renderer in the prototype
  var objectCreate = require( 'SCENERY/util/Util' ).objectCreate;
  
  scenery.Image = function( image, options ) {
    Node.call( this, options );
    
    this._image = image;
    
    this.invalidateImage();
  };
  var Image = scenery.Image;
  
  Image.prototype = objectCreate( Node.prototype );
  Image.prototype.constructor = Image;
  
  Image.prototype.invalidateImage = function() {
    this.invalidateSelf( new Bounds2( 0, 0, this._image.width, this._image.height ) );
  };
  
  Image.prototype.getImage = function() {
    return this._image;
  };
  
  Image.prototype.setImage = function( image ) {
    if ( this._image !== image ) {
      this._image = image;
      this.invalidateImage();
    }
    return this;
  };

  // TODO: add SVG / DOM support
  Image.prototype.paintCanvas = function( state ) {
    var layer = state.layer;
    var context = layer.context;
    context.drawImage( this._image, 0, 0 );
  };
  
  Image.prototype.paintWebGL = function( state ) {
    throw new Error( 'Image.prototype.paintWebGL unimplemented' );
  };
  
  Image.prototype.hasSelf = function() {
    return true;
  };
  
  Image.prototype._mutatorKeys = [ 'image' ].concat( Node.prototype._mutatorKeys );
  
  Image.prototype._supportedRenderers = [ Renderer.Canvas ];
  
  Object.defineProperty( Image.prototype, 'image', { set: Image.prototype.setImage, get: Image.prototype.getImage } );
  
  return Image;
} );


