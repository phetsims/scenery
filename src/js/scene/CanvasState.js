// Copyright 2002-2012, University of Colorado

var phet = phet || {};
phet.scene = phet.scene || {};

(function(){
    phet.scene.CanvasState = function( name ) {
        this.transform = new phet.math.Transform3();
        this.context = null;
        this.fillStyle = null;
        this.strokeStyle = null;
        this.isCanvasState = true;
    }

    var CanvasState = phet.scene.CanvasState;

    CanvasState.prototype = {
        constructor: CanvasState,
        
        setFillStyle: function( style ) {
            if( this.fillStyle != style ) {
                this.fillStyle = style;
                this.context.fillStyle = style;
            }
        },
        
        setStrokeStyle: function( style ) {
            if( this.strokeStyle != style ) {
                this.strokeStyle = style;
                this.context.strokeStyle = style;
            }
        }
    };
})();