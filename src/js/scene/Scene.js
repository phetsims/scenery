// Copyright 2002-2012, University of Colorado

/**
 * Main scene
 *
 * @author Jonathan Olson
 */

var phet = phet || {};
phet.scene = phet.scene || {};

(function(){
    phet.scene.Scene = function( main ) {
        this.root = new phet.scene.Node();
        
        // default to a canvas layer type, but this can be changed
        this.root.layerType = phet.scene.layers.CanvasLayer;
        
        this.layers = [];
        
        this.main = main;
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
            var state = new phet.scene.RenderState();
            this.root.render( state );
            state.finish(); // handle cleanup for the last layer
        },
        
        clearLayers: function() {
            this.main.empty();
            this.layers = [];
        },
        
        addLayer: function( layer ) {
            this.main.append( layer );
            this.layers.push( layer );
        },
        
        // called on the root node when any layer-relevant changes are made
        // TODO: add flags for this to happen, and call during renderFull. set flags on necessary functions
        rebuildLayers: function() {
            // verify that this node is the effective root
            phet.assert( this.root.parent == null );
            
            // root needs to contain a layer type reference
            phet.assert( this.root.layerType != null );
            
            // a few variables for the closurelayer
            var main = this.main;
            var layer = this;
            
            // remove everything from our container, so we can fill it in with fresh layers
            this.clearLayers();
            
            // for handling layers in depth-first fashion
            function recursiveRebuild( node, baseLayerType ) {
                var hasLayer = node.layerType != null;
                if( !hasLayer ) {
                    // sanity checks, in case a layerType was removed
                    node._layerBeforeRender = null;
                    node._layerAfterRender = null;
                } else {
                    // layers created in order, later
                    
                    // change the base layer type for the layer children
                    baseLayerType = node.layerType;
                }
                
                // for stacking, add the "before" layer before recursion
                if( hasLayer ) {
                    node._layerBeforeRender = new node.layerType( main );
                    layer.addLayer( node._layerBeforeRender );
                }
                
                // handle layers for children
                _.each( node.children, function( child ) {
                    recursiveRebuild( child, baseLayerType );
                } );
                
                // and the "after" layer after recursion, on top of any child layers
                if( hasLayer ) {
                    node._layerAfterRender = new baseLayerType( main );
                    layer.addLayer( node._layerAfterRender );
                }
            }
            
            // get the layer constructor
            var rootLayerType = this.root.layerType;
            
            // create the first layer (will be the only layer if no other nodes have a layerType)
            var startingLayer = new rootLayerType( main );
            this.root._layerBeforeRender = startingLayer;
            layer.addLayer( startingLayer );
            // no "after" layer needed for the root, since nothing is rendered after it
            this.root._layerAfterRender = null;
            
            _.each( this.root.children, function( child ) {
                recursiveRebuild( child, rootLayerType );
            } );
        }
    };
})();