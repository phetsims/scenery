// Copyright 2002-2012, University of Colorado

/**
 * Mutable state passed through the scene graph rendering process that stores
 * the current transformation and layer.
 *
 * @author Jonathan Olson
 */

var phet = phet || {};
phet.scene = phet.scene || {};

(function(){
    phet.scene.RenderState = function() {
        this.transform = new phet.math.Transform3();
        
        this.layer = null;
    }

    var RenderState = phet.scene.RenderState;

    RenderState.prototype = {
        constructor: RenderState,
        
        switchToLayer: function( layer ) {
            this.layer = layer;
            
            // give the layer the current transformation matrix
            layer.initialize( this.transform.matrix );
        },
        
        isCanvasState: function() {
            return this.layer.isCanvasLayer;
        },
        
        // TODO: consider a stack-based model for transforms?
        applyTransformationMatrix: function( matrix ) {
            this.transform.append( matrix );
            this.layer.applyTransformationMatrix( matrix );
        }
    };
})();