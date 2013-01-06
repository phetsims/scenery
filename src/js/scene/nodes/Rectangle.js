// Copyright 2002-2012, University of Colorado

var phet = phet || {};
phet.scene = phet.scene || {};
phet.scene.nodes = phet.scene.nodes || {};

(function(){
    phet.scene.nodes.Rectangle = function ( args ) {
        phet.scene.Node.call( this );
        
        this.args = args;
        this.x = args.x;
        this.y = args.y;
        this.width = args.width;
        this.height = args.height;
        
        // TODO: handle strokes and fills in a better way
        this.stroke = args.stroke;
        this.fill = args.fill;
    };

    var Rectangle = phet.scene.nodes.Rectangle;

    Rectangle.prototype = Object.create( phet.scene.Node.prototype );
    Rectangle.prototype.constructor = Rectangle;
    
    Rectangle.prototype.renderSelf = function ( state ) {
        if( state.isCanvasState ) {
            if( this.fill ) {
                state.setFillStyle( this.fill );
                state.context.fillRect( this.x, this.y, this.width, this.height );
            }
            if( this.stroke ) {
                state.setStrokeStyle( this.stroke );
                state.context.strokeRect( this.x, this.y, this.width, this.height );
            }
        }
    };
})();


