// Copyright 2002-2012, University of Colorado

var phet = phet || {};
phet.scene = phet.scene || {};

(function(){
    var Bounds2 = phet.math.Bounds2;
    
	phet.scene.Node = function( name ) {
        // user-editable properties
        this.visible = true;
        this.layerType = null; // null indicates there is no layer root here. otherwise should be a layer constructor function
        
        this.children = [];
        this.transform = new phet.math.Transform3();
        this.parent = null;
        
        // layer-specific data, currently updated in the rebuildLayers step
        this._layerBeforeRender = null; // for layer changes
        this._layerAfterRender = null;
        
        // bounds handling
        this._bounds = Bounds2.NOTHING; // in "parent" coordinates
        this._selfBounds = Bounds2.NOTHING; // in "local" coordinates
        this._childBounds = Bounds2.NOTHING; // in "local" coordinates
        this._boundsDirty = true;
        this._selfBoundsDirty = true;
        this._childBoundsDirty = true;
    }

	var Node = phet.scene.Node;
    var Matrix3 = phet.math.Matrix3;
    
	Node.prototype = {
		constructor: Node,
        
        // main render function for the root
        renderFull: function() {
            phet.assert( this.parent == null );
            phet.assert( this.isLayerRoot() );
            
            this.validateBounds();
            
            // TODO: render only dirty regions
            this.render( new phet.scene.RenderState() );
        },
        
        render: function( state ) {
            if( this._layerBeforeRender ) {
                state.switchToLayer( this._layerBeforeRender );
            }
            
            if ( !this.transform.isIdentity() ) {
                // TODO: consider a stack-based model for transforms?
                state.applyTransformationMatrix( this.transform.matrix );
            }

            this.preRender( state );

            // TODO: consider allowing render passes here?
            this.renderSelf( state );
            this.renderChildren( state );

            this.postRender( state );

            if ( !this.transform.isIdentity() ) {
                state.applyTransformationMatrix( this.transform.inverse );
            }
            
            if( this._layerAfterRender ) {
                state.switchToLayer( this._layerAfterRender );
            }
        },
        
        renderSelf: function ( state ) {

        },

        renderChildren: function ( state ) {
            for ( var i = 0; i < this.children.length; i++ ) {
                this.children[i].render( state );
            }
        },

        preRender: function ( state ) {
            
        },

        postRender: function ( state ) {
            
        },

        addChild: function ( node ) {
            phet.assert( node !== null && node !== undefined );
            if ( this.isChild( node ) ) {
                return;
            }
            if ( node.parent !== null ) {
                node.parent.removeChild( node );
            }
            node.parent = this;
            this.children.push( node );
            
            this.invalidateBounds();
            this._childBoundsDirty = true;
        },

        removeChild: function ( node ) {
            phet.assert( this.isChild( node ) );

            node.parent = null;
            this.children.splice( this.children.indexOf( node ), 1 );
            
            this.invalidateBounds();
        },

        hasParent: function () {
            return this.parent !== null && this.parent !== undefined;
        },

        detach: function () {
            if ( this.hasParent() ) {
                this.parent.removeChild( this );
            }
        },

        isChild: function ( potentialChild ) {
            phet.assert( (potentialChild.parent === this ) === (this.children.indexOf( potentialChild ) != -1) );
            return potentialChild.parent === this;
        },
        
        isLayerRoot: function() {
            return this.layerType != null;
        },
        
        getLayer: function() {
            phet.assert( this.isLayerRoot() );
            return this._layerBeforeRender;
        },

        translate: function ( x, y ) {
            this.transform.append( Matrix3.translation( x, y ) );
            this.invalidateBounds();
        },

        // scale( s ) is also supported
        scale: function ( x, y ) {
            this.transform.append( Matrix3.scaling( x, y ) );
            this.invalidateBounds();
        },

        rotate: function ( angle ) {
            this.transform.append( Matrix3.rotation2( angle ) );
            this.invalidateBounds();
        },
        
        rebuildLayers: function( main ) {
            // verify that this node is the effective root
            phet.assert( this.parent == null );
            
            // root needs to contain a layer type reference
            phet.assert( this.layerType != null );
            
            main.empty();
            
            function recursiveRebuild( node, baseLayerType ) {
                var hasLayer = node.layerType != null;
                if( !hasLayer ) {
                    // sanity checks, in case a layerType was removed
                    node._layerBeforeRender = null;
                    node._layerAfterRender = null;
                } else {
                    // create the layers for before/after
                    node._layerBeforeRender = new node.layerType( main );
                    node._layerAfterRender = new baseLayerType( main );
                    
                    // change the base layer type for the layer children
                    baseLayerType = node.layerType;
                }
                
                // for stacking, add the "before" layer before recursion
                if( hasLayer ) {
                    main.append( node._layerBeforeRender );
                }
                
                // handle layers for children
                _.each( node.children, function( child ) {
                    recursiveRebuild( child, baseLayerType );
                } );
                
                // and the "after" layer after recursion, on top of any child layers
                if( hasLayer ) {
                    main.append( node._layerAfterRender );
                }
            }
            
            // get the layer constructor
            var rootLayerType = this.layerType;
            
            // create the first layer (will be the only layer if no other nodes have a layerType)
            var startingLayer = new rootLayerType( main );
            main.append( startingLayer );
            this._layerBeforeRender = startingLayer;
            // no "after" layer needed for the root, since nothing is rendered after it
            this._layerAfterRender = null;
            
            _.each( this.children, function( child ) {
                recursiveRebuild( child, rootLayerType );
            } );
        },
        
        // bounds assumed to be in the local coordinate frame, below this node's transform
        markDirtyRegion: function( bounds ) {
            var layer = this.findLayer();
            
            // if there is no layer, ignore the markDirtyRegion call
            if( layer != null ) {
                layer.markDirtyRegion( this.localToGlobalBounds( bounds ) );
            }
        },
        
        getSelfBounds: function() {
            return this._selfBounds;
        },
        
        getBounds: function() {
            this.validateBounds();
            return this._bounds;
        },
        
        validateBounds: function() {
            // TODO: why is _selfBoundsDirty even needed?
            if( this._selfBoundsDirty ) {
                this._selfBoundsDirty = false;
            }
            
            // validate bounds of children if necessary
            if( this._childBoundsDirty ) {
                // have each child validate their own bounds
                _.each( this.children, function( child ) {
                    child.validateBounds();
                } );
                
                // and recompute our _childBounds
                this._childBounds = Bounds2.NOTHING;
                var that = this;
                _.each( this.children, function( child ) {
                    that._childBounds = that._childBounds.union( child._bounds );
                } );
                
                this._childBoundsDirty = false;
            }
            
            // TODO: layout here?
            
            if( this._boundsDirty ) {
                var oldBounds = this._bounds;
                
                var that = this;
                
                var newBounds = this.localToParentBounds( this._selfBounds ).union( that.localToParentBounds( this._childBounds ) );
                
                if( !newBounds.equals( oldBounds ) ) {
                    this._bounds = newBounds;
                    
                    if( this.parent != null ) {
                        this.parent.invalidateBounds();
                    }
                    this.markDirtyRegion( this.parentToLocalBounds( oldBounds ) );
                    
                    // TODO: fire off event listeners?
                }
                
                this._boundsDirty = false;
            }
        },
        
        findLayerRoot: function() {
            var node = this;
            while( node != null ) {
                if( node.isLayerRoot() ) {
                    return node;
                }
                node = node.parent;
            }
            
            // no layer root found
            return null;
        },
        
        findLayer: function() {
            var root = this.findLayerRoot();
            return root != null ? root.getLayer() : null;
        },
        
        invalidateBounds: function() {
            this._boundsDirty = true;
            
            // and set flags for all ancestors
            var node = this.parent;
            while( node != null ) {
                // TODO: for performance, consider cutting this once we detect a node with this as true
                node._childBoundsDirty = true;
                node = node.parent;
            }
        },
        
        setSelfBounds: function( newBounds ) {
            // if these bounds are different than current self bounds
            if( !this._selfBounds.equals( newBounds ) ) {
                // mark the old region to be repainted
                this.markDirtyRegion( this._selfBounds );
                
                // set repaint flags
                this._selfBoundsDirty = true;
                this.invalidateBounds();
                
                // record the new bounds
                this._selfBounds = newBounds;
            }
        },
        
        // apply this node's transform to the point
        localToParentPoint: function( point ) {
            return this.transform.transformPosition2( point );
        },
        
        localToParentBounds: function( bounds ) {
            return this.transform.transformBounds2( bounds );
        },
        
        // apply the inverse of this node's transform to the point
        parentToLocalPoint: function( point ) {
            return this.transform.inversePosition2( point );
        },
        
        parentToLocalBounds: function( bounds ) {
            return this.transform.inverseBounds2( bounds );
        },
        
        // apply this node's transform (and then all of its parents' transforms) to the point
        localToGlobalPoint: function( point ) {
            var node = this;
            while( node != null ) {
                point = node.transform.transformPosition2( point );
                node = node.parent;
            }
            return point;
        },
        
        localToGlobalBounds: function( bounds ) {
            var node = this;
            while( node != null ) {
                bounds = node.transform.transformBounds2( bounds );
                node = node.parent;
            }
            return bounds;
        },
        
        globalToLocalPoint: function( point ) {
            var node = this;
            
            // we need to apply the transformations in the reverse order, so we temporarily store them
            var transforms = [];
            while( node != null ) {
                transforms.push( node.transform );
                node = node.parent;
            }
            
            // iterate from the back forwards (from the root node to here)
            for( var i = transforms.length - 1; i >=0; i-- ) {
                point = transforms[i].inversePosition2( point );
            }
            return point;
        },
        
        globalToLocalBounds: function( bounds ) {
            var node = this;
            
            // we need to apply the transformations in the reverse order, so we temporarily store them
            var transforms = [];
            while( node != null ) {
                transforms.push( node.transform );
                node = node.parent;
            }
            
            // iterate from the back forwards (from the root node to here)
            for( var i = transforms.length - 1; i >=0; i-- ) {
                bounds = transforms[i].inverseBounds2( bounds );
            }
            return bounds;
        }
	};
})();