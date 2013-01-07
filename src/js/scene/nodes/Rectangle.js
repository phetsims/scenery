// Copyright 2002-2012, University of Colorado

var phet = phet || {};
phet.scene = phet.scene || {};
phet.scene.nodes = phet.scene.nodes || {};

(function(){
    phet.scene.nodes.Rectangle = function ( args ) {
        phet.scene.Node.call( this );
        
        this.args = {};
        
        this.update( args );
    };

    var Rectangle = phet.scene.nodes.Rectangle;
    
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


