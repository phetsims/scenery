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
            
            _.each( this.layers, function( layer ) {
                layer.clearDirtyRegions();
            } );
        },
        
        clearLayers: function() {
            this.main.empty();
            this.layers = [];
        },
        
        // handles creation and adds it to our internal list
        createLayer: function( constructor, args ) {
            var layer = new constructor( args );
            this.layers.push( layer );
            return layer;
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
            var scene = this;
            
            // used so we can give each layer a "start" and "end" node
            
            var lastLayer = null;
            
            // mutable layer arguments (since z-indexing needs to be increased, etc.)
            var layerArgs = {
                main: main,
                zIndex: 1 // needs to be incremented by the canvas
            };
            
            // remove everything from our container, so we can fill it in with fresh layers
            this.clearLayers();
            
            function layerChange( node, isBeforeRender ) {
                // the previous layer ends at this node, so mark that down
                lastLayer.endNode = node;
                
                // create the new layer, and give it the proper node reference
                var newLayer = scene.createLayer( node.layerType, layerArgs );
                if( isBeforeRender ) {
                    node._layerBeforeRender = newLayer;
                } else {
                    node._layerAfterRender = newLayer;
                }
                
                // mark this node as the beginning of the layer
                newLayer.startNode = node;
                
                // and prepare for the next call
                lastLayer = newLayer;
            }
            
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
                    layerChange( node, true );
                }
                
                // handle layers for children
                _.each( node.children, function( child ) {
                    recursiveRebuild( child, baseLayerType );
                } );
                
                // and the "after" layer after recursion, on top of any child layers
                if( hasLayer ) {
                    layerChange( node, false );
                }
            }
            
            // get the layer constructor
            var rootLayerType = this.root.layerType;
            
            // create the first layer (will be the only layer if no other nodes have a layerType)
            var startingLayer = scene.createLayer( rootLayerType, layerArgs );
            
            // no "after" layer needed for the root, since nothing is rendered after it
            this.root._layerBeforeRender = startingLayer;
            this.root._layerAfterRender = null;
            
            startingLayer.startNode = this.root;
            lastLayer = startingLayer;
            
            // step through the recursion
            _.each( this.root.children, function( child ) {
                recursiveRebuild( child, rootLayerType );
            } );
            
            lastLayer.endNode = this.root;
        }
    };
})();