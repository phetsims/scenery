// Copyright 2002-2012, University of Colorado

/**
 * A node for the phet-scene scene graph. Supports only tree-style graphs at the moment.
 * Handles multiple layers with assorted types (canvas, svg, DOM, etc.), and bounds
 * computation
 *
 * TODO: investigate handling DAGs (directed acyclic graphs)
 *
 * @author Jonathan Olson
 */

var phet = phet || {};
phet.scene = phet.scene || {};

(function(){
    var Bounds2 = phet.math.Bounds2;
    var Shape = phet.scene.Shape;
    
    // TODO: consider an args-style constructor here!
    phet.scene.Node = function() {
        this.visible = true;
        
        // type of layer to be created for content under this node.
        // if non-null, this node is a layer root, and layerType should be a layer constructor function
        this.layerType = null;
        
        // This node and all children will be clipped by this shape (in addition to any other clipping shapes).
        // The shape should be in the local coordinate frame
        this.clipShape = null;
        
        this.children = [];
        this.transform = new phet.math.Transform3();
        this.parent = null;
        
        // layer-specific data, currently updated in the rebuildLayers step
        this._layerBeforeRender = null; // layer to swap to before rendering this node
        this._layerAfterRender = null; // layer to swap to after rendering this node
        this._layerReference = null; // stores a reference to this node's corresponding layer (the one used to render it's self content)
        
        // bounds handling
        this._bounds = Bounds2.NOTHING; // for this node and its children, in "parent" coordinates
        this._selfBounds = Bounds2.NOTHING; // just for this node, in "local" coordinates
        this._childBounds = Bounds2.NOTHING; // just for children, in "local" coordinates
        this._boundsDirty = true;
        this._selfBoundsDirty = true;
        this._childBoundsDirty = true;
        
        // dirty region handling
        this._paintDirty = false;
        this._childPaintDirty = false;
        this._oldPaintMarked = false; // flag indicates the last rendered bounds of this node and all descendants are marked for a repaint already
        
        // shape used for rendering
        this._shape = null;
        // fill/stroke for shapes
        this._stroke = null;
        this._fill = null;
        
        this._lineDrawingStyles = new Shape.LineStyles();
    }
    
    var Node = phet.scene.Node;
    var Matrix3 = phet.math.Matrix3;
    
    Node.prototype = {
        constructor: Node,
        
        enterState: function( state ) {
            // switch layers if needed
            if( this._layerBeforeRender ) {
                state.switchToLayer( this._layerBeforeRender );
            }
            
            // apply this node's transform
            if ( !this.transform.isIdentity() ) {
                // TODO: consider a stack-based model for transforms?
                state.applyTransformationMatrix( this.transform.getMatrix() );
            }
            
            if( this.clipShape ) {
                state.pushClipShape( this.clipShape );
            }
        },
        
        exitState: function( state ) {
            if( this.clipShape ) {
                state.popClipShape();
            }
            
            // apply the inverse of this node's transform
            if ( !this.transform.isIdentity() ) {
                state.applyTransformationMatrix( this.transform.getInverse() );
            }
            
            // switch layers if needed
            if( this._layerAfterRender ) {
                state.switchToLayer( this._layerAfterRender );
            }
        },
        
        // override to render typical leaf behavior (although possible to use for non-leaf nodes also)
        renderSelf: function ( state ) {
            
            // by default, render a shape if it exists
            if( this.hasShape() ) {
                if( state.isCanvasState() ) {
                    var layer = state.layer;
                    var context = layer.context;
                    
                    // TODO: fill/stroke delay optimizations?
                    context.beginPath();
                    this._shape.writeToContext( context );
                    
                    if( this._fill ) {
                        layer.setFillStyle( this._fill );
                        context.fill();
                    }
                    if( this._stroke ) {
                        layer.setStrokeStyle( this._stroke );
                        layer.setLineWidth( this.getLineWidth() );
                        context.stroke();
                    }
                } else {
                    throw new Error( 'layer type shape rendering not implemented' );
                }
            }
        },
        
        // override to run before rendering of this node is done
        preRender: function ( state ) {
            
        },
        
        // override to run just after this node and its children are rendered
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
            
            node.invalidateBounds();
            node.invalidatePaint();
            
            // synchronize layer references for adding subtrees without layer types.
            if( node._layerReference != this._layerReference ) {
                // TODO: THIS CODE IS BROKEN - adding in a "valley" layer will cause layering issues and breaks, along with other undefined behavior
                var layerReference = this._layerReference;
                node.walkDepthFirst( function( child ) {
                    child._layerReference = layerReference;
                } );
            }
        },
        
        removeChild: function ( node ) {
            phet.assert( this.isChild( node ) );
            
            node.markOldPaint();
            
            node.parent = null;
            this.children.splice( this.children.indexOf( node ), 1 );
            
            this.invalidateBounds();
        },
        
        hasParent: function () {
            return this.parent !== null && this.parent !== undefined;
        },
        
        // remove this node from its parent
        detach: function () {
            if ( this.hasParent() ) {
                this.parent.removeChild( this );
            }
        },
        
        isChild: function ( potentialChild ) {
            phet.assert( (potentialChild.parent === this ) === (this.children.indexOf( potentialChild ) != -1) );
            return potentialChild.parent === this;
        },
        
        // does this node have an associated layerType (are the contents of this node rendered separately from its ancestors)
        isLayerRoot: function() {
            return this.layerType != null;
        },
        
        // the first layer associated with this node (can be multiple layers if children of this node are also layer roots)
        getLayer: function() {
            phet.assert( this.isLayerRoot() );
            return this._layerBeforeRender;
        },
        
        // TODO: how to handle point vs x,y
        translate: function( x, y ) {
            // mark old bounds as needing a repaint
            this.appendMatrix( Matrix3.translation( x, y ) );
        },
        
        // scale( s ) is also supported
        scale: function( x, y ) {
            this.appendMatrix( Matrix3.scaling( x, y ) );
        },
        
        rotate: function( angle ) {
            this.appendMatrix( Matrix3.rotation2( angle ) );
        },
        
        // TODO: how to handle x,y?
        setTranslation: function( x, y ) {
            var translation = this.getTranslation();
            this.translate( x - translation.x, y - translation.y );
        },
        
        // append a transformation matrix to our local transform
        appendMatrix: function( matrix ) {
            // invalidate paint TODO improve methods for this
            this.markOldPaint();
            
            this.transform.append( matrix );
            
            this.invalidateBounds();
            this.invalidatePaint();
        },
        
        getLineWidth: function() {
            return this._lineDrawingStyles.lineWidth;
        },
        
        setLineWidth: function( lineWidth ) {
            if( this.getLineWidth() != lineWidth ) {
                this.markOldSelfPaint(); // since the previous line width may have been wider
                
                this._lineDrawingStyles.lineWidth = lineWidth;
                
                this.invalidateBounds();
                this.invalidatePaint();
            }
        },
        
        getFill: function() {
            return this._fill;
        },
        
        setFill: function( fill ) {
            if( this.getFill() != fill ) {
                this._fill = fill;
                this.invalidatePaint();
            }
        },
        
        getStroke: function() {
            return this._stroke;
        },
        
        setStroke: function( stroke ) {
            if( this.getStroke() != stroke ) {
                // since this can actually change the bounds, we need to handle a few things differently than the fill
                this.markOldSelfPaint();
                
                this._stroke = stroke;
                this.invalidatePaint();
                this.invalidateBounds();
            }
        },
        
        // bounds assumed to be in the local coordinate frame, below this node's transform
        markDirtyRegion: function( bounds ) {
            var globalBounds = this.localToGlobalBounds( bounds );
            _.each( this.findDescendantLayers(), function( layer ) {
                layer.markDirtyRegion( globalBounds );
            } );
        },
        
        markOldSelfPaint: function() {
            this.markOldPaint( true );
        },
        
        // marks the last-rendered bounds of this node and optionally all of its descendants as needing a repaint
        markOldPaint: function( justSelf ) {
            var node = this;
            var alreadyMarked = this._oldPaintMarked;
            while( !alreadyMarked && node.parent != null ) {
                node = node.parent;
                alreadyMarked = node._oldPaintMarked; 
            }
            
            // we want to not do this marking if possible multiple times for the same sub-tree, so we check flags first
            if( !alreadyMarked ) {
                if( justSelf ) {
                    this.markDirtyRegion( this._selfBounds );
                } else {
                    this.markDirtyRegion( this.parentToLocalBounds( this._bounds ) );
                    this._oldPaintMarked = true; // don't mark this in self calls, because we don't use the full bounds
                }
            }
        },
        
        // the bounds for content in renderSelf(), in "local" coordinates
        getSelfBounds: function() {
            return this._selfBounds;
        },
        
        // the bounds for content in render(), in "parent" coordinates
        getBounds: function() {
            this.validateBounds();
            return this._bounds;
        },
        
        // ensure that cached bounds stored on this node (and all children) are accurate
        validateBounds: function() {
            if( this._selfBoundsDirty ) {
                this._selfBoundsDirty = false;
                
                // if our self bounds changed, make sure to paint the area where our new bounds are
                this.markDirtyRegion( this._selfBounds );
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
                    // this.markDirtyRegion( this.parentToLocalBounds( oldBounds ) );
                    
                    // TODO: fire off event listeners?
                }
                
                this._boundsDirty = false;
            }
        },
        
        validatePaint: function() {
            // if dirty, mark the region
            if( this._paintDirty ) {
                this.markDirtyRegion( this.parentToLocalBounds( this._bounds ) );
                this._paintDirty = false;
            }
            
            // clear flags and recurse
            if( this._childPaintDirty || this._oldPaintMarked ) {
                this._childPaintDirty = false;
                this._oldPaintMarked = false;
                
                _.each( this.children, function( child ) {
                    child.validatePaint();
                } );
            }
        },
        
        // find the layer to which this node will be rendered
        findLayer: function() {
            return this._layerReference;
        },
        
        // find all layers for which this node (and all its descendants) will be rendered
        findDescendantLayers: function() {
            var firstLayer = this.findLayer();
            
            if( firstLayer == null ) {
                return [];
            }
            
            // run a node out to the last child until we reach a leaf, to find the last layer we will render to in this subtree
            var node = this;
            while( node.children.length != 0 ) {
                node = _.last( node.children );
            }
            var lastLayer = node.findLayer();
            
            // collect all layers between the first and last layers
            var layers = [firstLayer];
            var layer = firstLayer;
            while( layer != lastLayer && layer.nextLayer != null ) {
                layer = layer.nextLayer;
                layers.push( layer );
            }
            return layers;
        },
        
        
        
        // mark the bounds of this node as invalid, so it is recomputed before it is accessed again
        invalidateBounds: function() {
            this._boundsDirty = true;
            
            // and set flags for all ancestors
            var node = this.parent;
            while( node != null && !node._childBoundsDirty ) {
                node._childBoundsDirty = true;
                node = node.parent;
            }
        },
        
        // mark the paint of this node as invalid, so its new region will be painted
        invalidatePaint: function() {
            this._paintDirty = true;
            
            // and set flags for all ancestors (but bail if already marked, since this guarantees lower nodes will be marked)
            var node = this.parent;
            while( node != null && !node._childPaintDirty ) {
                node._childPaintDirty = true;
                node = node.parent;
            }
        },
        
        // called to notify that renderSelf will display different paint, with possibly different bounds
        invalidateSelf: function( newBounds ) {
            // mark the old region to be repainted, regardless of whether the actual bounds change
            this.markOldSelfPaint();
            
            // if these bounds are different than current self bounds
            if( !this._selfBounds.equals( newBounds ) ) {
                // set repaint flags
                this._selfBoundsDirty = true;
                this.invalidateBounds();
                
                // record the new bounds
                this._selfBounds = newBounds;
            }
            
            this.invalidatePaint();
        },
        
        // checking for whether a point (in parent coordinates) is contained in this sub-tree
        containsPoint: function( point ) {
            // update bounds for pruning
            this.validateBounds();
            
            // bail quickly if this doesn't hit our computed bounds
            if( !this._bounds.containsPoint( point ) ) { return false; }
            
            // point in the local coordinate frame. computed after the main bounds check, so we can bail out there efficiently
            var localPoint = this.transform.inversePosition2( point );
            
            // check children first, since they are rendered later
            if( this.children.length > 0 && this._childBounds.containsPoint( localPoint ) ) {
                
                // manual iteration here so we can return directly, and so we can iterate backwards (last node is in front)
                for( var i = this.children.length - 1; i >= 0; i-- ) {
                    var child = this.children[i];
                    
                    // the child will have the point in its parent's coordinate frame (i.e. this node's frame)
                    if( child.containsPoint( localPoint ) ) {
                        return true;
                    }
                }
            }
            
            // didn't hit our children, so check ourself as a last resort
            if( this._selfBounds.containsPoint( point ) ) {
                return this.containsPointSelf( point );
            }
        },
        
        childrenWithinBounds: function( bounds ) {
            return _.filter( this.children, function( child ) { return !child._bounds.intersection( bounds ).isEmpty(); } );
        },
        
        // override for computation of whether a point is inside the content rendered in renderSelf
        containsPointSelf: function( point ) {
            return false;
        },
        
        hasShape: function() {
            return this._shape != null;
        },
        
        // sets the shape drawn, or null to remove the shape
        setShape: function( shape ) {
            this._shape = shape;
            
            this.invalidateSelf( shape.computeBounds( ) );
        },
        
        getShape: function() {
            return this._shape;
        },
        
        walkDepthFirst: function( callback ) {
            callback( this );
            _.each( this.children, function( child ) {
                child.walkDepthFirst( callback );
            } );
        },
        
        
        
        // returns a list of ancestors of this node, with the root first
        ancestors: function() {
            return this.parent ? this.parent.pathToRoot() : [];
        },
        
        // like ancestors(), but includes the current node as well
        pathToRoot: function() {
            var result = [];
            var node = this;
            
            while( node != null ) {
                result.unshift( node );
                node = node.parent;
            }
            
            return result;
        },
        
        getTranslation: function() {
            return this.transform.getMatrix().translation();
        },
        
        getScaling: function() {
            return this.transform.getMatrix().scaling();
        },
        
        getRotation: function() {
            return this.transform.getMatrix().rotation();
        },
        
        getX: function() {
            return getTranslation().x;
        },
        
        getY: function() {
            return getTranslation().y;
        },
        
        /*---------------------------------------------------------------------------*
        * Coordinate transform methods
        *----------------------------------------------------------------------------*/        
        
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