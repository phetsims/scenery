// Copyright 2002-2012, University of Colorado

var phet = phet || {};
phet.scene = phet.scene || {};
phet.scene.layers = phet.scene.layers || {};

(function(){
    var Bounds2 = phet.math.Bounds2;
    
    // assumes main is wrapped with JQuery
    phet.scene.layers.CanvasLayer = function ( main ) {
        this.canvas = document.createElement( 'canvas' );
        this.canvas.width = main.width();
        this.canvas.height = main.height();
        main.append( this.canvas );
        
        this.context = phet.canvas.initCanvas( this.canvas );
        
        this.isCanvasLayer = true;
        
        this.dirtyBounds = Bounds2.NOTHING;
    };

    var CanvasLayer = phet.scene.layers.CanvasLayer;
    
    CanvasLayer.prototype = {
        constructor: CanvasLayer,
        
        initialize: function( matrix ) {
            // set the context's transform to the current transformation matrix
            this.context.setTransform(
                // inlined array entries
                matrix.entries[0],
                matrix.entries[1],
                matrix.entries[3],
                matrix.entries[4],
                matrix.entries[6],
                matrix.entries[7]
            );
        },
        
        // TODO: consider a stack-based model for transforms?
        applyTransformationMatrix: function( matrix ) {
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
        
        markDirtyRegion: function( bounds ) {
            // TODO: for performance, consider more than just a single dirty bounding box
            this.dirtyBounds = this.dirtyBounds.union( bounds );
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


