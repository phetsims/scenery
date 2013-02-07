// Copyright 2002-2012, University of Colorado

/**
 * Images
 *
 * @author Jonathan Olson
 */

var phet = phet || {};
phet.scene = phet.scene || {};

(function(){
    "use strict";
    
    phet.scene.Image = function( image, params ) {
        phet.scene.Node.call( this, params );
        
        this.image = image;
        
        this.invalidateSelf( new phet.math.Bounds2( 0, 0, image.width, image.height ) );
    };
    var Image = phet.scene.Image;
    
    Image.prototype = phet.Object.create( phet.scene.Node.prototype );
    Image.prototype.constructor = Image;
    
    Image.prototype.renderSelf = function( state ) {
        // TODO: add SVG / DOM support
        if ( state.isCanvasState() ) {
            var layer = state.layer;
            var context = layer.context;
            context.drawImage( this.image, 0, 0 );
        }
    };
})();


