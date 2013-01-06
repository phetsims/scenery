// Copyright 2002-2012, University of Colorado

var phet = phet || {};
phet.scene = phet.scene || {};
phet.scene.layers = phet.scene.layers || {};

(function(){
    // assumes main is wrapped with JQuery
    phet.scene.layers.CanvasLayer = function ( main ) {
        this.canvas = document.createElement( 'canvas' );
        this.canvas.width = main.width();
        this.canvas.height = main.height();
        main.append( this.canvas );
        
        this.context = phet.canvas.initCanvas( this.canvas );
    };

    var CanvasLayer = phet.scene.layers.CanvasLayer;
    
    CanvasLayer.prototype = {
        constructor: CanvasLayer,
        
        createState: function( otherState ) {
            var state = new phet.scene.CanvasState();
            
            // rely on all states having an instance of Transform3
            state.transform = otherState.transform;
            
            // use this layer's context
            state.context = this.context;
            
            return state;
        }
    };
})();


