// Copyright 2002-2012, University of Colorado

/**
 * A rectangle shape, with an optional stroke or fill
 *
 * TODO: consider for deprecation / removal if the regular rectangle performance can be satisfactory
 *
 * @author Jonathan Olson
 */

var phet = phet || {};
phet.scene = phet.scene || {};

(function(){
    "use strict";
    
    phet.scene.Rectangle = function ( args ) {
        phet.scene.Node.call( this );
        
        this.args = {};
        
        this.update( args );
    };
    
    var Rectangle = phet.scene.Rectangle;
    
    Rectangle.parameters = [ 'x', 'y', 'width', 'height' ];
    
    Rectangle.prototype = Object.create( phet.scene.Node.prototype );
    Rectangle.prototype.constructor = Rectangle;
    
    Rectangle.prototype.update = function( args ) {
        // copy values into args
        for( var key in args ) {
            this.args[key] = args[key];
        }
        
        // TODO: bounds handling for stroke widths / caps, etc.
        this.invalidateSelf( new phet.math.Bounds2( this.args.x, this.args.y, this.args.x + this.args.width, this.args.y + this.args.height ) );
    };
    
    // whether this node's rendering contains a specific point in local coordinates
    Rectangle.prototype.containsPointSelf = function( point ) {
        // TODO: consider stroke width!
        return point.x >= this.args.x && point.y >= this.args.y && point.x <= this.args.x + this.args.width && point.y <= this.args.y + this.args.height;
    };
    
    // whether the bounding box intersects this node
    Rectangle.prototype.intersectsBoundsSelf = function( bounds ) {
        return !this.getSelfBounds().intersection( bounds ).isEmpty();
    };
    
    Rectangle.prototype.renderSelf = function ( state ) {
        if( state.isCanvasState() ) {
            var layer = state.layer;
            var context = layer.context;
            if( this.hasFill() ) {
                layer.setFillStyle( this.getFill() );
                context.fillRect( this.args.x, this.args.y, this.args.width, this.args.height );
            }
            
            // stroke deprecated, since the bounds are unused. use Shape instead for rectangles
        }
    };
})();


