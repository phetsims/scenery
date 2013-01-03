// Copyright 2002-2012, University of Colorado

var phet = phet || {};
phet.scene = phet.scene || {};

(function(){
    phet.scene.RenderState = function( name ) {
        this.transform = new phet.math.Transform3();
        this.context = null;
        this.fillStyle = null;
        this.strokeStyle = null;
    }

    var RenderState = phet.scene.RenderState;

    RenderState.prototype = {
        constructor: RenderState,
        
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