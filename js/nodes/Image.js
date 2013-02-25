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
  
  var Node = require( 'SCENERY/Node' ); // Image inherits from Node
  var Backend = require( 'SCENERY/layers/Backend' ); // we need to specify the Backend in the prototype
  var objectCreate = require( 'SCENERY/util/Util' ).objectCreate;
  
  scenery.Image = function( image, options ) {
    Node.call( this, options );
    
    this.image = image;
    
    this.invalidateSelf( new Bounds2( 0, 0, image.width, image.height ) );
  };
  var Image = scenery.Image;
  
  Image.prototype = objectCreate( Node.prototype );
  Image.prototype.constructor = Image;

  // TODO: add SVG / DOM support
  Image.prototype.paintCanvas = function( state ) {
    var layer = state.layer;
    var context = layer.context;
    context.drawImage( this.image, 0, 0 );
  };
  
  Image.prototype.paintWebGL = function( state ) {
    throw new Error( 'Image.prototype.paintWebGL unimplemented' );
  };
  
  Image.prototype.hasSelf = function() {
    return true;
  };
  
  Image.prototype._supportedBackends = [ Backend.Canvas ];
  
  return Image;
} );


