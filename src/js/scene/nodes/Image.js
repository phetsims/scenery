// Copyright 2002-2012, University of Colorado

/**
 * Images
 *
 * TODO: setImage / getImage and the whole toolchain that uses that
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

var scenery = scenery || {};

(function(){
  "use strict";
  
  scenery.Image = function( image, params ) {
    scenery.Node.call( this, params );
    
    this.image = image;
    
    this.invalidateSelf( new phet.math.Bounds2( 0, 0, image.width, image.height ) );
  };
  var Image = scenery.Image;
  
  Image.prototype = phet.Object.create( scenery.Node.prototype );
  Image.prototype.constructor = Image;
  
  Image.prototype.renderSelf = function( state ) {
    // TODO: add SVG / DOM support
    if ( state.isCanvasState() ) {
      var layer = state.layer;
      var context = layer.context;
      context.drawImage( this.image, 0, 0 );
    }
  };
  
  Image.prototype.hasSelf = function() {
    return true;
  };
  
  Image.prototype._supportedLayerTypes = [ scenery.LayerType.Canvas ];
})();


