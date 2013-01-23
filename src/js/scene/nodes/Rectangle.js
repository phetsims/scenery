// Copyright 2002-2012, University of Colorado

/**
 * A rectangle shape, with an optional stroke or fill
 *
 * @author Jonathan Olson
 */

var phet = phet || {};
phet.scene = phet.scene || {};

(function(){
    phet.scene.Rectangle = function ( args ) {
        phet.scene.Node.call( this );
        
        this.args = {};
        
        this.update( args );
    };
    
    var Rectangle = phet.scene.Rectangle;
    
    Rectangle.parameters = [ 'x', 'y', 'width', 'height', 'stroke', 'fill' ];
    
    Rectangle.prototype = Object.create( phet.scene.Node.prototype );
    Rectangle.prototype.constructor = Rectangle;
    
    Rectangle.prototype.update = function( args ) {
        // copy values into args
        for( var key in args ) {
            this.args[key] = args[key];
        }
        
        this.x = args.x === undefined ? this.x : args.x;
        this.y = args.y === undefined ? this.y : args.y;
        this.width = args.width === undefined ? this.width : args.width;
        this.height = args.height === undefined ? this.height : args.height;
        this.stroke = args.stroke === undefined ? this.stroke : args.stroke;
        this.fill = args.fill === undefined ? this.fill : args.fill;
        
        // TODO: bounds handling for stroke widths / caps, etc.
        this.invalidateSelf( new phet.math.Bounds2( this.x, this.y, this.x + this.width, this.y + this.height ) );
    };
    
    // whether this node's rendering contains a specific point in local coordinates
    Rectangle.prototype.containsPointSelf = function( point ) {
        // TODO: consider stroke width!
        return point.x >= this.x && point.y >= this.y && point.x <= this.x + this.width && point.y <= this.y + this.height;
    };
    
    // whether the bounding box intersects this node
    Rectangle.prototype.intersectsBoundsSelf = function( bounds ) {
        return !this.getSelfBounds().intersection( bounds ).isEmpty();
    };
    
    Rectangle.prototype.renderSelf = function ( state ) {
        if( state.isCanvasState() ) {
            var layer = state.layer;
            var context = layer.context;
            if( this.fill ) {
                layer.setFillStyle( this.fill );
                context.fillRect( this.x, this.y, this.width, this.height );
            }
            if( this.stroke ) {
                layer.setStrokeStyle( this.stroke );
                context.strokeRect( this.x, this.y, this.width, this.height );
            }
        }
    };
})();


