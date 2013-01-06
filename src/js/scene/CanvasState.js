// Copyright 2002-2012, University of Colorado

var phet = phet || {};
phet.scene = phet.scene || {};

(function(){
    phet.scene.CanvasState = function() {
        this.transform = new phet.math.Transform3();
        this.context = null;
        
        // styles initially set to null so we will always reset them
        this.fillStyle = null;
        this.strokeStyle = null;
        
        this.isCanvasState = true;
    }

    var CanvasState = phet.scene.CanvasState;

    CanvasState.prototype = {
        constructor: CanvasState,
        
        // TODO: consider a stack-based model for transforms?
        applyTransformationMatrix: function( matrix ) {
            this.transform.append( matrix );
            this.context.transform( 
                // inlined array entries
                matrix.entries[0],
                matrix.entries[1],
                matrix.entries[3],
                matrix.entries[4],
                matrix.entries[6],
                matrix.entries[7]
            );
        },
        
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