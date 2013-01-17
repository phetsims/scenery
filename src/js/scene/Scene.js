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
        
        this.root._isRoot = true;
        this.root.scene = this;
        
        this.layers = [];
        
        // the greatest common "path" of nodes, i.e. ancestors arrays. after modifications, this will tell us the smallest subtree needed to re-layer
        this.dirtyLayerPath = null;
        
        this.main = main;
        
        this.sceneBounds = new phet.math.Bounds2( 0, 0, main.width(), main.height() );
        
        // default to a canvas layer type, but this can be changed
        // called here AFTER the root is initialized, so that we set the correct dirtyLayerPath and get a layer rebuild / refresh as necessary
        this.root.setLayerType( phet.scene.layers.CanvasLayer ); 
    }

    var Scene = phet.scene.Scene;
    
    function fullRender( node, state ) {
        node.enterState( state );
        
        if( node.visible ) {
            node.renderSelf( state );
            
            var children = node.children;
            
            // check if we need to filter the children we render, and ignore nodes with few children (but allow 2, since that may prevent branches)
            if( state.childRestrictedBounds && children.length > 1 ) {
                var localRestrictedBounds = node.globalToLocalBounds( state.childRestrictedBounds );
                
                // don't filter if every child is inside the bounds
                if( !localRestrictedBounds.containsBounds( node.parentToLocalBounds( node._bounds ) ) ) {
                    children = node.getChildrenWithinBounds( localRestrictedBounds );
                }
            }
            
            _.each( children, function( child ) {
                fullRender( child, state );
            } );
        }
        
        node.exitState( state );
    }

    Scene.prototype = {
        constructor: Scene,
        
        renderScene: function() {
            phet.assert( this.root.parent == null );
            phet.assert( this.root.isLayerRoot() );
            
            // validating bounds, similar to Piccolo2d
            this.root.validateBounds();
            
            // TODO: render only dirty regions
            var state = new phet.scene.RenderState();
            fullRender( this.root, state );
            state.finish(); // handle cleanup for the last layer
            
            _.each( this.layers, function( layer ) {
                layer.resetDirtyRegions();
            } );
        },
        
        updateScene: function( args ) {
            phet.assert( this.root.parent == null );
            phet.assert( this.root.isLayerRoot() );
            
            // validating bounds, similar to Piccolo2d
            this.root.validateBounds();
            this.root.validatePaint();
            
            var scene = this;
            
            _.each( this.layers, function( layer ) {
                // don't repaint clean layers
                if( layer.isDirty() ) {
                    scene.updateLayer( layer, args );
                }
            } );
        },
        
        updateLayer: function( layer, args ) {
            // TODO: only render in dirty rectangles (modify state and checks?)
            var state = new phet.scene.RenderState();
            
            // switches to (and initializes) the layer
            var dirtyBounds = layer.getDirtyBounds();
            var visibleDirtyBounds = layer.getDirtyBounds().intersection( this.sceneBounds );
            layer.prepareDirtyRegions();
            state.pushClipShape( phet.scene.Shape.bounds( visibleDirtyBounds ) );
            state.childRestrictedBounds = visibleDirtyBounds;
            state.switchToLayer( layer );
            state.multiLayerRender = false; // don't allow it to switch layers at the start / end nodes
            
            if( layer.startNode == layer.endNode ) {
                // the node and all of its descendants can just be rendered
                var node = layer.startNode;
                
                // walk up from the root to the node, applying transforms, clips, etc.
                _.each( node.getAncestors(), function( ancestor ) {
                    ancestor.enterState( state );
                    
                    if( !ancestor.visible ) {
                        return; // completely bail
                    }
                } );
                
                // then render the node and its children
                fullRender( node, state );
                
                // cooldown on the layer. we don't have to walk the state back down to the root
                state.finish();
            } else {
                var startPath = layer.startNode.getPathToRoot();
                var endPath = layer.endNode.getPathToRoot();
                
                var minLength = Math.min( startPath.length, endPath.length );
                
                var depth;
                
                // run through parts that are the same start and end, since everything we want is under here
                for( depth = 0; depth < minLength; depth++ ) {
                    // bail on the first difference
                    if( startPath[depth] != endPath[depth] ) {
                        break;
                    }
                    
                    if( !startPath[depth].visible ) {
                        return; // none of our layer visible, bail
                    }
                    
                    // apply transforms, clips, etc. that wouldn't be applied later
                    // -2, since we will start rendering at [depth-1], and this will include everything
                    if( depth > 2 ) {
                        startPath[depth-2].enterState( state );
                    }
                }
                
                // now, our depth is the index of the first difference between startPath and endPath.
                
                /* Rendering partial contents under this node, with depth indexing into the startPath/endPath.
                 * 
                 * Since we want to render just a single layer (and only nodes inside that layer), we need to handle children in one of three cases:
                 * a) outside of bounds (none of child or its descendants in our layer):
                 *          don't render it at all
                 * b) at a boundary (some of the child or its descendants are in our layer):
                 *          continue our recursive partial render through this child, but ONLY if the boundary is compatible (low bound <==> beforeRender, high bound <==> afterRender)
                 * c) inside of bounds (all of the child and descendants are in our layer):
                 *          render it like normally for greater efficiency
                 * 
                 * Our startPath and endPath are the paths in the scene-graph from the root to both the nodes that mark the start and end of our layer.
                 * Any node that is rendered must have a path that is essentially "between" these two.
                 *
                 * Depth is masked as a parameter here from the outside scope.
                 */ 
                function recursivePartialRender( node, depth, hasLowBound, hasHighBound ) {
                    if( !hasLowBound && !hasHighBound ) {
                        // if there are no bounds restrictions on children, just do a straight rendering
                        fullRender( node, state );
                        return;
                    }
                    
                    // we are now assured that startPath[depth] != endPath[depth], so each child is either high-bounded or low-bounded
                    
                    node.enterState( state );
            
                    if( node.visible ) {
                        if( !hasLowBound ) {
                            node.renderSelf( state );
                        }
                        
                        var passedLowBound = !hasLowBound;
                        
                        // check to see if we need to filter what children are rendered based on restricted bounds
                        var localRestrictedBounds;
                        var filterChildren = false; 
                        if( state.childRestrictedBounds && node.children.length > 1 ) {
                            localRestrictedBounds = node.globalToLocalBounds( state.childRestrictedBounds );
                            
                            // don't filter if all children will be inside the bounds
                            filterChildren = !localRestrictedBounds.containsBounds( node.parentToLocalBounds( node._bounds ) )
                        }
                        
                        // uses a classic for loop so we can bail out early
                        for( var i = 0; i < node.children.length; i++ ) {
                            var child = node.children[i];
                            
                            if( filterChildren && !localRestrictedBounds.intersectsBounds( child._bounds ) ) {
                                continue;
                            }
                            
                            // due to the calling conditions, we should be assured that startPath[depth] != endPath[depth]
                            
                            if( !passedLowBound ) {
                                // if we haven't passed the low bound, it MUST exist, and we are either (a) not rendered, or (b) low-bounded
                                
                                if( startPath[depth] == child ) {
                                    // only recursively render if the switch is "before" the node
                                    if( startPath[depth]._layerBeforeRender == layer ) {
                                        recursivePartialRender( child, depth + 1, startPath.length != depth, false ); // if it has a low-bound, it can't have a high bound
                                    }
                                
                                    // for later children, we have passed the low bound.
                                    passedLowBound = true;
                                }
                                
                                // either way, go on to the next nodes
                                continue;
                            }
                            
                            if( hasHighBound && endPath[depth] == child ) {
                                // only recursively render if the switch is "after" the node
                                if( endPath[depth]._layerAfterRender == layer ) {
                                    // high-bounded here
                                    recursivePartialRender( child, depth + 1, false, endPath.length != depth ); // if it has a high-bound, it can't have a low bound
                                }
                                
                                // don't render any more children, since we passed the high bound
                                break;
                            }
                            
                            // we should now be in-between both the high and low bounds with no further restrictions, so just carry out the rendering
                            fullRender( child, state );
                        }
                    }
                    
                    node.exitState( state );
                }
                
                recursivePartialRender( startPath[depth-1], depth, startPath.length != depth, endPath.length != depth );
                
                // for layer cooldown
                state.finish();
            }
            
            // we are done rendering, so reset the layer's dirty regions
            layer.resetDirtyRegions();
        },
        
        // attempt to render everything currently visible in the scene to an external canvas. allows copying from canvas layers straight to the other canvas
        renderToCanvas: function( canvas, context ) {
            context.clearRect( 0, 0, canvas.width, canvas.height );
            _.each( this.layers, function( layer ) {
                layer.renderToCanvas( canvas, context );
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
            phet.assert( this.root.isLayerRoot() );
            
            // a few variables for the closurelayer
            var main = this.main;
            var scene = this;
            
            // used so we can give each layer a "start" and "end" node
            var lastLayer = null;
            
            // mutable layer arguments (since z-indexing needs to be increased, etc.)
            var layerArgs = {
                main: main,
                scene: this
            };
            
            // remove everything from our container, so we can fill it in with fresh layers
            this.clearLayers();
            
            function layerChange( node, isBeforeRender ) {
                // the previous layer ends at this node, so mark that down
                lastLayer.endNode = node;
                
                // create the new layer, and give it the proper node reference
                var newLayer = scene.createLayer( node._layerType, layerArgs );
                if( isBeforeRender ) {
                    node._layerBeforeRender = newLayer;
                } else {
                    node._layerAfterRender = newLayer;
                }
                
                // mark this node as the beginning of the layer
                newLayer.startNode = node;
                
                // hook the layers together so they know about each other
                lastLayer.nextLayer = newLayer;
                newLayer.previousLayer = lastLayer;
                
                // and prepare for the next call
                lastLayer = newLayer;
            }
            
            // for handling layers in depth-first fashion
            function recursiveRebuild( node, baseLayerType ) {
                var hasLayer = node._layerType != null;
                if( !hasLayer ) {
                    // sanity checks, in case a layerType was removed
                    node._layerBeforeRender = null;
                    node._layerAfterRender = null;
                } else {
                    // layers created in order, later
                    
                    // change the base layer type for the layer children
                    baseLayerType = node._layerType;
                }
                
                // for stacking, add the "before" layer before recursion
                if( hasLayer ) {
                    layerChange( node, true );
                }
                
                // let the node know what layer it's self will render inside of
                node._layerReference = lastLayer;
                
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
            var rootLayerType = this.root._layerType;
            
            // create the first layer (will be the only layer if no other nodes have a layerType)
            var startingLayer = scene.createLayer( rootLayerType, layerArgs );
            
            // no "after" layer needed for the root, since nothing is rendered after it
            this.root._layerBeforeRender = startingLayer;
            this.root._layerAfterRender = null;
            
            startingLayer.startNode = this.root;
            lastLayer = startingLayer;
            this.root._layerReference = startingLayer;
            
            // step through the recursion
            _.each( this.root.children, function( child ) {
                recursiveRebuild( child, rootLayerType );
            } );
            
            lastLayer.endNode = this.root;
            
            this.reindexLayers();
        },
        
        // after layer changes, the layers should have their zIndex updated
        reindexLayers: function() {
            var index = 1;
            _.each( this.layers, function( layer ) {
                // layers increment indices as needed
                index = layer.reindex( index );
            } );
        },
        
        clearAllLayers: function() {
            _.each( this.layers, function( layer ) {
                layer.prepareBounds( phet.math.Bounds2.EVERYTHING );
            } );
        },
        
        layersDirtyUnder: function( node ) {
            if( !this.dirtyLayerPath ) {
                this.dirtyLayerPath = node.getPathToRoot();
            } else {
                var nodePath = node.getPathToRoot();
                var maxIndex = Math.min( nodePath.length, this.dirtyLayerPath.length );
                for( var i = 0; i < maxIndex; i++ ) {
                    // cut the dirty layer path off before the first discrepancy
                    if( nodePath[i] != this.dirtyLayerPath[i] ) {
                        this.dirtyLayerPath = _.first( this.dirtyLayerPath, i );
                        break;
                    }
                }
            }
        },
        
        refreshLayers: function() {
            // TODO: THIS FUNCTION: refreshLayers
            
            /*
                PSEUDOCODE:
                
                identify pre,post layers
                remove [pre+1,post] from scene
                if content:
                    relayer starting with pre
                reassign "post" nodes to last content layer (could be pre, if so check if post==pre => reassignment not necessary)
                reattach pre <==> pre+1, last <==> post.next, and set up before/after handling
                reindex
            */
            
            // TODO: ensure that "valley" layers are created, even if currently unnecessary. this allows more efficient insertChild handling
            this.rebuildLayers();
            
            // reset the dirty layer path
            this.dirtyLayerPath = null;
        }
    };
})();