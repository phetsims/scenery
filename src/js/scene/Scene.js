// Copyright 2002-2012, University of Colorado

/**
 * Main scene
 *
 * @author Jonathan Olson
 */

var phet = phet || {};
phet.scene = phet.scene || {};

(function(){
    phet.scene.Scene = function() {
        this.root = new phet.scene.Node();
        
        // default to a canvas layer type, but this can be changed
        this.root.layerType = phet.scene.layers.CanvasLayer;
    }

    var Scene = phet.scene.Scene;

    Scene.prototype = {
        constructor: Scene,
        
        renderScene: function() {
            phet.assert( this.root.parent == null );
            phet.assert( this.root.isLayerRoot() );
            
            // validating bounds, similar to Piccolo2d
            this.root.validateBounds();
            
            // TODO: render only dirty regions
            this.root.render( new phet.scene.RenderState() );
        }
    };
})();