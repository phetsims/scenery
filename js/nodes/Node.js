// Copyright 2002-2012, University of Colorado

/**
 * A node for the Scenery scene graph. Supports general directed acyclic graphics (DAGs).
 * Handles multiple layers with assorted types (Canvas 2D, SVG, DOM, WebGL, etc.).
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  "use strict";
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var Bounds2 = require( 'DOT/Bounds2' );
  var Transform3 = require( 'DOT/Transform3' );
  var Matrix3 = require( 'DOT/Matrix3' );
  var clamp = require( 'DOT/Util' ).clamp;
  
  var scenery = require( 'SCENERY/scenery' );
  var LayerStrategy = require( 'SCENERY/layers/LayerStrategy' ); // used to set the default layer strategy on the prototype
  // require( 'SCENERY/layers/Renderer' ); // commented out so Require.js doesn't balk at the circular dependency
  
  // TODO: FIXME: Why do I have to comment out this dependency?
  // require( 'SCENERY/util/Trail' );
  // require( 'SCENERY/util/TrailPointer' );
  
  var globalIdCounter = 1;
  
  /*
   * Available keys for use in the options parameter object for a vanilla Node (not inherited), in the order they are executed in:
   *
   * children:         A list of children to add (in order)
   * cursor:           Will display the specified CSS cursor when the mouse is over this Node or one of its descendents. The Scene needs to have input listeners attached with an initialize method first.
   * visible:          If false, this node (and its children) will not be displayed (or get input events)
   * pickable:         If false, this node (and its children) will not get input events
   * translation:      Sets the translation of the node to either the specified dot.Vector2 value, or the x,y values from an object (e.g. translation: { x: 1, y: 2 } )
   * x:                Sets the x-translation of the node
   * y:                Sets the y-translation of the node
   * rotation:         Sets the rotation of the node in radians
   * scale:            Sets the scale of the node. Supports either a number (same x-y scale), or a dot.Vector2 / object with ob.x and ob.y to set the scale for each axis independently
   * left:             Sets the x-translation so that the left (min X) of the bounding box (in the parent coordinate frame) is at the specified value
   * right:            Sets the x-translation so that the right (max X) of the bounding box (in the parent coordinate frame) is at the specified value
   * top:              Sets the y-translation so that the top (min Y) of the bounding box (in the parent coordinate frame) is at the specified value
   * bottom:           Sets the y-translation so that the bottom (min Y) of the bounding box (in the parent coordinate frame) is at the specified value
   * centerX:          Sets the x-translation so that the horizontal center of the bounding box (in the parent coordinate frame) is at the specified value
   * centerY:          Sets the y-translation so that the vertical center of the bounding box (in the parent coordinate frame) is at the specified value
   * renderer:         Forces Scenery to use the specific renderer (canvas/svg) to display this node (and if possible, children). Accepts both strings (e.g. 'canvas', 'svg', etc.) or actual Renderer objects (e.g. Renderer.Canvas, Renderer.SVG, etc.)
   * rendererOptions:  Parameter object that is passed to the created layer, and can affect how the layering process works.
   * layerSplit:       Forces a split between layers before and after this node (and its children) have been rendered. Useful for performance with Canvas-based renderers.
   * layerSplitBefore: Forces a split between layers before this node (and its children) have been rendered. Useful for performance with Canvas-based renderers.
   * layerSplitAfter:  Forces a split between layers after this node (and its children) have been rendered. Useful for performance with Canvas-based renderers.
   */
  scenery.Node = function Node( options ) {
    var self = this;
    
    // assign a unique ID to this node (allows trails to get a unique list of IDs)
    this._id = globalIdCounter++;
    
    // Whether this node (and its children) will be visible when the scene is updated. Visible nodes by default will not be pickable either
    this._visible = true;
    
    // Opacity from 0 to 1
    this._opacity = 1;
    
    // Whether hit testing will check for this node (and its children).
    this._pickable = true;
    
    // This node and all children will be clipped by this shape (in addition to any other clipping shapes).
    // The shape should be in the local coordinate frame
    this._clipShape = null;
    
    // the CSS cursor to be displayed over this node. null should be the default (inherit) value
    this._cursor = null;
    
    this._children = []; // ordered
    this._parents = []; // unordered
    
    /*
     * Set up the transform reference. we add a listener so that the transform itself can be modified directly
     * by reference, or node.transform = <transform> / node.setTransform() can be used to change the transform reference.
     * Both should trigger the necessary event notifications for Scenery to keep track internally.
     */
    this._transform = new Transform3();
    this._transformListener = {
      before: function() { self.beforeTransformChange(); },
      after: function() { self.afterTransformChange(); }
    };
    this._transform.addTransformListener( this._transformListener );
    
    this._inputListeners = []; // for user input handling (mouse/touch)
    this._eventListeners = []; // for internal events like paint invalidation, layer invalidation, etc.
    
    // TODO: add getter/setters that will be able to invalidate whether this node is under any pointers, etc.
    this._includeStrokeInHitRegion = false;
    
    // bounds handling
    this._bounds = Bounds2.NOTHING; // for this node and its children, in "parent" coordinates
    this._selfBounds = Bounds2.NOTHING; // just for this node, in "local" coordinates
    this._childBounds = Bounds2.NOTHING; // just for children, in "local" coordinates
    this._boundsDirty = true;
    this._selfBoundsDirty = this.isPainted();
    this._childBoundsDirty = true;
    
    // dirty region handling
    this._paintDirty = false;
    this._childPaintDirty = false;
    this._oldPaintMarked = false; // flag indicates the last rendered bounds of this node and all descendants are marked for a repaint already
    
    // what type of renderer should be forced for this node.
    this._renderer = null;
    this._rendererOptions = null; // options that will determine the layer type
    this._rendererLayerType = null; // cached layer type that is used by the LayerStrategy
    
    // whether layers should be split before and/or after this node. setting both will put this node and its children into a separate layer
    this._layerSplitBefore = false;
    this._layerSplitAfter = false;
    
    if ( options ) {
      this.mutate( options );
    }
  };
  var Node = scenery.Node;
  
  Node.prototype = {
    constructor: Node,
    
    insertChild: function( index, node ) {
      assert && assert( node !== null && node !== undefined, 'insertChild cannot insert a null/undefined child' );
      assert && assert( !_.contains( this._children, node ), 'Parent already contains child' );
      assert && assert( node !== this, 'Cannot add self as a child' );

      //Add peer for accessibility to the other scene
      if (window.accessibilityScene && node.peer){
        window.accessibilityScene.addChild(node.peer);
      }
      
      node._parents.push( this );
      this._children.splice( index, 0, node );
      
      node.invalidateBounds();
      node.invalidatePaint();
      
      this.dispatchEvent( 'markForInsertion', {
        parent: this,
        child: node,
        index: index
      } );
      
      this.dispatchEvent( 'stitch', { match: false } );
    },
    
    addChild: function( node ) {
      this.insertChild( this._children.length, node );
    },
    
    removeChild: function( node ) {
      assert && assert( this.isChild( node ) );
      
      node.markOldPaint();
      
      var indexOfParent = _.indexOf( node._parents, this );
      var indexOfChild = _.indexOf( this._children, node );
      
      this.dispatchEvent( 'markForRemoval', {
        parent: this,
        child: node,
        index: indexOfChild
      } );
      
      node._parents.splice( indexOfParent, 1 );
      this._children.splice( indexOfChild, 1 );
      
      this.invalidateBounds();
      
      this.dispatchEvent( 'stitch', { match: false } );

      //Remove peer for accessibility to the other scene
      if (window.accessibilityScene && node.peer){
        window.accessibilityScene.removeChild(node.peer);
      }
    },
    
    // TODO: efficiency by batching calls?
    setChildren: function( children ) {
      var node = this;
      if ( this._children !== children ) {
        _.each( this._children.slice( 0 ), function( child ) {
          node.removeChild( child );
        } );
        _.each( children, function( child ) {
          node.addChild( child );
        } );
      }
    },
    
    getChildren: function() {
      return this._children.slice( 0 ); // create a defensive copy
    },
    
    getParents: function() {
      return this._parents.slice( 0 ); // create a defensive copy
    },
    
    indexOfParent: function( parent ) {
      return _.indexOf( this._parents, parent );
    },
    
    indexOfChild: function( child ) {
      return _.indexOf( this._children, child );
    },
    
    moveToFront: function() {
      var self = this;
      _.each( this._parents.slice( 0 ), function( parent ) {
        parent.moveChildToFront( self );
      } );
    },
    
    moveChildToFront: function( child ) {
      if ( this.indexOfChild( child ) !== this._children.length - 1 ) {
        this.removeChild( child );
        this.addChild( child );
      }
    },
    
    moveToBack: function() {
      var self = this;
      _.each( this._parents.slice( 0 ), function( parent ) {
        parent.moveChildToBack( self );
      } );
    },
    
    moveChildToBack: function( child ) {
      if ( this.indexOfChild( child ) !== 0 ) {
        this.removeChild( child );
        this.insertChild( 0, child );
      }
    },
    
    // remove this node from its parents
    detach: function() {
      var that = this;
      _.each( this._parents.slice( 0 ), function( parent ) {
        parent.removeChild( that );
      } );
    },
    
    // ensure that cached bounds stored on this node (and all children) are accurate
    validateBounds: function() {
      var that = this;
      
      if ( this._selfBoundsDirty ) {
        // note: this should only be triggered if the bounds were actually changed, since we have a guard in place at invalidateSelf()
        this._selfBoundsDirty = false;
        
        // if our self bounds changed, make sure to paint the area where our new bounds are
        this.markDirtyRegion( this._selfBounds );
        
        // TODO: consider changing to parameter object (that may be a problem for the GC overhead)
        this.fireEvent( 'selfBounds', this._selfBounds );
      }
      
      // validate bounds of children if necessary
      if ( this._childBoundsDirty ) {
        // have each child validate their own bounds
        _.each( this._children, function( child ) {
          child.validateBounds();
        } );
        
        var oldChildBounds = this._childBounds;
        
        // and recompute our _childBounds
        this._childBounds = Bounds2.NOTHING;
        
        _.each( this._children, function( child ) {
          that._childBounds = that._childBounds.union( child._bounds );
        } );
        
        this._childBoundsDirty = false;
        
        if ( !this._childBounds.equals( oldChildBounds ) ) {
          // TODO: consider changing to parameter object (that may be a problem for the GC overhead)
          this.fireEvent( 'childBounds', this._childBounds );
        }
      }
      
      // TODO: layout here?
      
      if ( this._boundsDirty ) {
        var oldBounds = this._bounds;
        
        var newBounds = this.localToParentBounds( this._selfBounds ).union( that.localToParentBounds( this._childBounds ) );
        var changed = !newBounds.equals( oldBounds );
        
        if ( changed ) {
          this._bounds = newBounds;
          
          _.each( this._parents, function( parent ) {
            parent.invalidateBounds();
          } );
          
          // TODO: consider changing to parameter object (that may be a problem for the GC overhead)
          this.fireEvent( 'bounds', this._bounds );
        }
        
        this._boundsDirty = false;
      }
    },
    
    validatePaint: function() {
      // if dirty, mark the region
      if ( this._paintDirty ) {
        this.markDirtyRegion( this.parentToLocalBounds( this._bounds ) );
        this._paintDirty = false;
      }
      
      // clear flags and recurse
      if ( this._childPaintDirty || this._oldPaintMarked ) {
        this._childPaintDirty = false;
        this._oldPaintMarked = false;
        
        _.each( this._children, function( child ) {
          child.validatePaint();
        } );
      }
    },
    
    // mark the bounds of this node as invalid, so it is recomputed before it is accessed again
    invalidateBounds: function() {
      this._boundsDirty = true;
      
      // and set flags for all ancestors
      _.each( this._parents, function( parent ) {
        parent.invalidateChildBounds();
      } );
    },
    
    // recursively tag all ancestors with _childBoundsDirty
    invalidateChildBounds: function() {
      // don't bother updating if we've already been tagged
      if ( !this._childBoundsDirty ) {
        this._childBoundsDirty = true;
        _.each( this._parents, function( parent ) {
          parent.invalidateChildBounds();
        } );
      }
    },
    
    // mark the paint of this node as invalid, so its new region will be painted
    invalidatePaint: function() {
      this._paintDirty = true;
      
      // and set flags for all ancestors
      _.each( this._parents, function( parent ) {
        parent.invalidateChildPaint();
      } );
    },
    
    // recursively tag all ancestors with _childPaintDirty
    invalidateChildPaint: function() {
      // don't bother updating if we've already been tagged
      if ( !this._childPaintDirty ) {
        this._childPaintDirty = true;
        _.each( this._parents, function( parent ) {
          parent.invalidateChildPaint();
        } );
      }
    },
    
    // called to notify that self rendering will display different paint, with possibly different bounds
    invalidateSelf: function( newBounds ) {
      assert && assert( newBounds.isEmpty() || newBounds.isFinite() , "Bounds must be empty or finite in invalidateSelf");
      
      // mark the old region to be repainted, regardless of whether the actual bounds change
      this.markOldSelfPaint();
      
      // if these bounds are different than current self bounds
      if ( !this._selfBounds.equals( newBounds ) ) {
        // set repaint flags
        this._selfBoundsDirty = true;
        this.invalidateBounds();
        
        // record the new bounds
        this._selfBounds = newBounds;
      }
      
      this.invalidatePaint();
    },
    
    // bounds assumed to be in the local coordinate frame, below this node's transform
    markDirtyRegion: function( bounds ) {
      this.dispatchEventWithTransform( 'dirtyBounds', {
        node: this,
        bounds: bounds
      } );
    },
    
    markOldSelfPaint: function() {
      this.markOldPaint( true );
    },
    
    // should be called whenever something triggers changes for how this node is layered
    markLayerRefreshNeeded: function() {
      this.dispatchEvent( 'markForLayerRefresh', {} );
      
      this.dispatchEvent( 'stitch', { match: true } );
    },
    
    // marks the last-rendered bounds of this node and optionally all of its descendants as needing a repaint
    markOldPaint: function( justSelf ) {
      function ancestorHasOldPaint( node ) {
        if( node._oldPaintMarked ) {
          return true;
        }
        return _.some( node._parents, function( parent ) {
          return ancestorHasOldPaint( parent );
        } );
      }
      
      var alreadyMarked = ancestorHasOldPaint( this );
      
      // we want to not do this marking if possible multiple times for the same sub-tree, so we check flags first
      if ( !alreadyMarked ) {
        if ( justSelf ) {
          this.markDirtyRegion( this._selfBounds );
        } else {
          this.markDirtyRegion( this.parentToLocalBounds( this._bounds ) );
          this._oldPaintMarked = true; // don't mark this in self calls, because we don't use the full bounds
        }
      }
    },
    
    isChild: function( potentialChild ) {
      var ourChild = _.contains( this._children, potentialChild );
      var itsParent = _.contains( potentialChild._parents, this );
      assert && assert( ourChild === itsParent );
      return ourChild;
    },
    
    // the bounds for self content in "local" coordinates
    getSelfBounds: function() {
      return this._selfBounds;
    },
    
    getChildBounds: function() {
      this.validateBounds();
      return this._childBounds;
    },
    
    // the bounds for content in render(), in "parent" coordinates
    getBounds: function() {
      this.validateBounds();
      return this._bounds;
    },
    
    /*
     * Return a trail to the top node (if any, otherwise null) whose self-rendered area contains the
     * point (in parent coordinates).
     *
     * If options.pruneInvisible is false, invisible nodes will be allowed in the trail.
     * If options.pruneUnpickable is false, unpickable nodes will be allowed in the trail.
     */
    trailUnderPoint: function( point, options ) {
      assert && assert( point, 'trailUnderPointer requires a point' );
      
      var pruneInvisible = ( !options || options.pruneInvisible === undefined ) ? true : options.pruneInvisible;
      var pruneUnpickable = ( !options || options.pruneUnpickable === undefined ) ? true : options.pruneUnpickable;
      
      if ( pruneInvisible && !this.isVisible() ) {
        return null;
      }
      if ( pruneUnpickable && !this.isPickable() ) {
        return null;
      }
      
      // update bounds for pruning
      this.validateBounds();
      
      // bail quickly if this doesn't hit our computed bounds
      if ( !this._bounds.containsPoint( point ) ) { return null; }
      
      // point in the local coordinate frame. computed after the main bounds check, so we can bail out there efficiently
      var localPoint = this._transform.inversePosition2( point );
      
      // check children first, since they are rendered later
      if ( this._children.length > 0 && this._childBounds.containsPoint( localPoint ) ) {
        
        // manual iteration here so we can return directly, and so we can iterate backwards (last node is in front)
        for ( var i = this._children.length - 1; i >= 0; i-- ) {
          var child = this._children[i];
          
          var childHit = child.trailUnderPoint( localPoint, options );
          
          // the child will have the point in its parent's coordinate frame (i.e. this node's frame)
          if ( childHit ) {
            childHit.addAncestor( this, i );
            return childHit;
          }
        }
      }
      
      // didn't hit our children, so check ourself as a last resort
      if ( this._selfBounds.containsPoint( localPoint ) && this.containsPointSelf( localPoint ) ) {
        return new scenery.Trail( this );
      }
      
      // signal no hit
      return null;
    },
    
    // checking for whether a point (in parent coordinates) is contained in this sub-tree
    containsPoint: function( point ) {
      return this.trailUnderPoint( point ) !== null;
    },
    
    // override for computation of whether a point is inside the self content
    // point is considered to be in the local coordinate frame
    containsPointSelf: function( point ) {
      // if self bounds are not null default to checking self bounds
      return this._selfBounds.containsPoint( point );
    },
    
    // whether this node's self intersects the specified bounds, in the local coordinate frame
    intersectsBoundsSelf: function( bounds ) {
      // if self bounds are not null, child should override this
      return this._selfBounds.intersectsBounds( bounds );
    },
    
    isPainted: function() {
      return false;
    },
    
    hasParent: function() {
      return this._parents.length !== 0;
    },
    
    hasChildren: function() {
      return this._children.length > 0;
    },
    
    walkDepthFirst: function( callback ) {
      callback( this );
      _.each( this._children, function( child ) {
        child.walkDepthFirst( callback );
      } );
    },
    
    getChildrenWithinBounds: function( bounds ) {
      return _.filter( this._children, function( child ) { return !child._bounds.intersection( bounds ).isEmpty(); } );
    },
    
    // TODO: set this up with a mix-in for a generic notifier?
    addInputListener: function( listener ) {
      // don't allow listeners to be added multiple times
      if ( _.indexOf( this._inputListeners, listener ) === -1 ) {
        this._inputListeners.push( listener );
      }
    },
    
    removeInputListener: function( listener ) {
      // ensure the listener is in our list
      assert && assert( _.indexOf( this._inputListeners, listener ) !== -1 );
      
      this._inputListeners.splice( _.indexOf( this._inputListeners, listener ), 1 );
    },
    
    getInputListeners: function() {
      return this._inputListeners.slice( 0 ); // defensive copy
    },
    
    // TODO: set this up with a mix-in for a generic notifier?
    addEventListener: function( listener ) {
      // don't allow listeners to be added multiple times
      if ( _.indexOf( this._eventListeners, listener ) === -1 ) {
        this._eventListeners.push( listener );
      }
    },
    
    removeEventListener: function( listener ) {
      // ensure the listener is in our list
      assert && assert( _.indexOf( this._eventListeners, listener ) !== -1 );
      
      this._eventListeners.splice( _.indexOf( this._eventListeners, listener ), 1 );
    },
    
    getEventListeners: function() {
      return this._eventListeners.slice( 0 ); // defensive copy
    },
    
    /*
     * Fires an event to all event listeners attached to this node. It does not bubble down to
     * all ancestors with trails, like dispatchEvent does. Use fireEvent when you only want an event
     * that is relevant for a specific node, and ancestors don't need to be notified.
     */
    fireEvent: function( type, args ) {
      _.each( this.getEventListeners(), function( eventListener ) {
        if ( eventListener[type] ) {
          eventListener[type]( args );
        }
      } );
    },
    
    /*
     * Dispatches an event across all possible Trails ending in this node.
     *
     * For example, if the scene has two children A and B, and both of those nodes have X as a child,
     * dispatching an event on X will fire the event with the following trails:
     * on X     with trail [ X ]
     * on A     with trail [ A, X ]
     * on scene with trail [ scene, A, X ]
     * on B     with trail [ B, X ]
     * on scene with trail [ scene, B, X ]
     *
     * This allows you to add a listener on any node to get notifications for all of the trails that the
     * event is relevant for (e.g. marks dirty paint region for both places X was on the scene).
     */
    dispatchEvent: function( type, args ) {
      var trail = new scenery.Trail();
      
      function recursiveEventDispatch( node ) {
        trail.addAncestor( node );
        
        args.trail = trail;
        
        node.fireEvent( type, args );
        
        _.each( node._parents, function( parent ) {
          recursiveEventDispatch( parent );
        } );
        
        trail.removeAncestor();
      }
      
      recursiveEventDispatch( this );
    },
    
    // dispatches events with the transform computed from parent of the "root" to the local frame
    dispatchEventWithTransform: function( type, args ) {
      var trail = new scenery.Trail();
      var transformStack = [ new Transform3() ];
      
      function recursiveEventDispatch( node ) {
        trail.addAncestor( node );
        
        transformStack.push( new Transform3( node.getMatrix().timesMatrix( transformStack[transformStack.length-1].getMatrix() ) ) );
        args.transform = transformStack[transformStack.length-1];
        args.trail = trail;
        
        node.fireEvent( type, args );
        
        _.each( node._parents, function( parent ) {
          recursiveEventDispatch( parent );
        } );
        
        transformStack.pop();
        
        trail.removeAncestor();
      }
      
      recursiveEventDispatch( this );
    },
    
    // TODO: consider renaming to translateBy to match scaleBy
    translate: function( x, y, prependInstead ) {
      if ( typeof x === 'number' ) {
        // translate( x, y, prependInstead )
        if ( prependInstead ) {
          this.prependMatrix( Matrix3.translation( x, y ) );
        } else {
          this.appendMatrix( Matrix3.translation( x, y ) );
        }
      } else {
        // translate( vector, prependInstead )
        var vector = x;
        this.translate( vector.x, vector.y, y ); // forward to full version
      }
    },
    
    // scale( s ) is also supported, which will scale both dimensions by the same amount. renamed from 'scale' to satisfy the setter/getter
    scale: function( x, y, prependInstead ) {
      if ( typeof x === 'number' ) {
        if ( y === undefined ) {
          // scale( scale )
          this.appendMatrix( Matrix3.scaling( x, x ) );
        } else {
          // scale( x, y, prependInstead )
          if ( prependInstead ) {
            this.prependMatrix( Matrix3.scaling( x, y ) );
          } else {
            this.appendMatrix( Matrix3.scaling( x, y ) );
          }
        }
      } else {
        // scale( vector, prependInstead ) or scale( { x: x, y: y }, prependInstead )
        var vector = x;
        this.scale( vector.x, vector.y, y ); // forward to full version
      }
    },
    
    // TODO: consider naming to rotateBy to match scaleBy (due to scale property / method name conflict)
    rotate: function( angle, prependInstead ) {
      if ( prependInstead ) {
        this.prependMatrix( Matrix3.rotation2( angle ) );
      } else {
        this.appendMatrix( Matrix3.rotation2( angle ) );
      }
    },
    
    // point should be in the parent coordinate frame
    // TODO: determine whether this should use the appendMatrix method
    rotateAround: function( point, angle ) {
      var matrix = Matrix3.translation( -point.x, -point.y );
      matrix = Matrix3.rotation2( angle ).timesMatrix( matrix );
      matrix = Matrix3.translation( point.x, point.y ).timesMatrix( matrix );
      this.prependMatrix( matrix );
    },
    
    getX: function() {
      return this.getTranslation().x;
    },
    
    setX: function( x ) {
      this.setTranslation( x, this.getY() );
      return this;
    },
    
    getY: function() {
      return this.getTranslation().y;
    },
    
    setY: function( y ) {
      this.setTranslation( this.getX(), y );
      return this;
    },
    
    // returns a vector with an entry for each axis, e.g. (5,2) for an Affine-style matrix with rows ((5,0,0),(0,2,0),(0,0,1))
    getScaleVector: function() {
      return this._transform.getMatrix().getScaleVector();
    },
    
    // supports setScaleMagnitude( 5 ) for both dimensions, setScaleMagnitude( 5, 3 ) for each dimension separately, or setScaleMagnitude( new Vector2( x, y ) )
    setScaleMagnitude: function( a, b ) {
      var currentScale = this.getScaleVector();
      
      if ( typeof a === 'number' ) {
        if ( b === undefined ) {
          // to map setScaleMagnitude( scale ) => setScaleMagnitude( scale, scale )
          b = a;
        }
        // setScaleMagnitude( x, y )
        this.appendMatrix( Matrix3.scaling( a / currentScale.x, b / currentScale.y ) );
      } else {
        // setScaleMagnitude( vector ), where we set the x-scale to vector.x and y-scale to vector.y
        this.appendMatrix( Matrix3.scaling( a.x / currentScale.x, a.y / currentScale.y ) );
      }
      return this;
    },
    
    getRotation: function() {
      return this._transform.getMatrix().getRotation();
    },
    
    setRotation: function( rotation ) {
      this.appendMatrix( Matrix3.rotation2( rotation - this.getRotation() ) );
      return this;
    },
    
    // supports setTranslation( x, y ) or setTranslation( new Vector2( x, y ) ) .. or technically setTranslation( { x: x, y: y } )
    setTranslation: function( a, b ) {
      var translation = this.getTranslation();
      
      if ( typeof a === 'number' ) {
        this.translate( a - translation.x, b - translation.y, true );
      } else {
        this.translate( a.x - translation.x, a.y - translation.y, true );
      }
      return this;
    },
    
    getTranslation: function() {
      return this._transform.getMatrix().getTranslation();
    },
    
    // append a transformation matrix to our local transform
    appendMatrix: function( matrix ) {
      this._transform.append( matrix );
    },
    
    // prepend a transformation matrix to our local transform
    prependMatrix: function( matrix ) {
      this._transform.prepend( matrix );
    },
    
    setMatrix: function( matrix ) {
      this._transform.set( matrix );
    },
    
    getMatrix: function() {
      return this._transform.getMatrix();
    },
    
    // change the actual transform reference (not just the actual transform)
    setTransform: function( transform ) {
      if ( this._transform !== transform ) {
        // since our referenced transform doesn't change, we need to trigger the before/after ourselves
        this.beforeTransformChange();
        
        // swap the transform and move the listener to the new one
        this._transform.removeTransformListener( this._transformListener ); // don't leak memory!
        this._transform = transform;
        this._transform.addTransformListener( this._transformListener );
        
        this.afterTransformChange();
      }
    },
    
    getTransform: function() {
      // for now, return an actual copy. we can consider listening to changes in the future
      return this._transform;
    },
    
    resetTransform: function() {
      this.setMatrix( Matrix3.IDENTITY );
    },
    
    // called before our transform is changed
    beforeTransformChange: function() {
      // mark our old bounds as dirty, so that any dirty region repainting will include not just our new position, but also our old position
      this.markOldPaint();
    },
    
    // called after our transform is changed
    afterTransformChange: function() {
      this.dispatchEventWithTransform( 'transform', {
        node: this,
        type: 'transform',
        matrix: this._transform.getMatrix()
      } );
      this.invalidateBounds();
      this.invalidatePaint();
    },
    
    // the left bound of this node, in the parent coordinate frame
    getLeft: function() {
      return this.getBounds().minX;
    },
    
    // shifts this node horizontally so that its left bound (in the parent coordinate frame) is 'left'
    setLeft: function( left ) {
      this.translate( left - this.getLeft(), 0, true );
      return this; // allow chaining
    },
    
    // the right bound of this node, in the parent coordinate frame
    getRight: function() {
      return this.getBounds().maxX;
    },
    
    // shifts this node horizontally so that its right bound (in the parent coordinate frame) is 'right'
    setRight: function( right ) {
      this.translate( right - this.getRight(), 0, true );
      return this; // allow chaining
    },
    
    getCenter: function() {
      return this.getBounds().getCenter();
    },
    
    setCenter: function( center ) {
      this.translate( center.minus( this.getCenter() ) );
      return this;
    },
    
    getCenterX: function() {
      return this.getBounds().getCenterX();
    },
    
    setCenterX: function( x ) {
      this.translate( x - this.getCenterX(), 0, true );
      return this; // allow chaining
    },
    
    getCenterY: function() {
      return this.getBounds().getCenterY();
    },
    
    setCenterY: function( y ) {
      this.translate( 0, y - this.getCenterY(), true );
      return this; // allow chaining
    },
    
    // the top bound of this node, in the parent coordinate frame
    getTop: function() {
      return this.getBounds().minY;
    },
    
    // shifts this node vertically so that its top bound (in the parent coordinate frame) is 'top'
    setTop: function( top ) {
      this.translate( 0, top - this.getTop(), true );
      return this; // allow chaining
    },
    
    // the bottom bound of this node, in the parent coordinate frame
    getBottom: function() {
      return this.getBounds().maxY;
    },
    
    // shifts this node vertically so that its bottom bound (in the parent coordinate frame) is 'bottom'
    setBottom: function( bottom ) {
      this.translate( 0, bottom - this.getBottom(), true );
      return this; // allow chaining
    },
    
    getWidth: function() {
      return this.getBounds().getWidth();
    },
    
    getHeight: function() {
      return this.getBounds().getHeight();
    },
    
    getId: function() {
      return this._id;
    },
    
    isVisible: function() {
      return this._visible;
    },
    
    setVisible: function( visible ) {
      if ( visible !== this._visible ) {
        if ( this._visible ) {
          this.markOldSelfPaint();
        }
        
        this._visible = visible;
        
        this.invalidatePaint();
      }
      return this;
    },
    
    getOpacity: function() {
      return this._opacity;
    },
    
    setOpacity: function( opacity ) {
      var clampedOpacity = clamp( opacity, 0, 1 );
      if ( clampedOpacity !== this._opacity ) {
        this.markOldPaint();
        
        this._opacity = clampedOpacity;
        
        this.invalidatePaint();
      }
    },
    
    isPickable: function() {
      return this._pickable;
    },
    
    setPickable: function( pickable ) {
      if ( this._pickable !== pickable ) {
        // no paint or invalidation changes for now, since this is only handled for the mouse
        this._pickable = pickable;
        
        // TODO: invalidate the cursor somehow?
      }
    },
    
    setCursor: function( cursor ) {
      // TODO: consider a mapping of types to set reasonable defaults
      /*
      auto default none inherit help pointer progress wait crosshair text vertical-text alias copy move no-drop not-allowed
      e-resize n-resize w-resize s-resize nw-resize ne-resize se-resize sw-resize ew-resize ns-resize nesw-resize nwse-resize
      context-menu cell col-resize row-resize all-scroll url( ... ) --> does it support data URLs?
       */
      
      // allow the 'auto' cursor type to let the ancestors or scene pick the cursor type
      this._cursor = cursor === "auto" ? null : cursor;
    },
    
    getCursor: function() {
      return this._cursor;
    },
    
    updateLayerType: function() {
      if ( this._renderer && this._rendererOptions ) {
        // TODO: factor this check out! Make RendererOptions its own class?
        // TODO: FIXME: support undoing this!
        // ensure that if we are passing a CSS transform, we pass this node as the baseNode
        if ( this._rendererOptions.cssTransform || this._rendererOptions.cssTranslation || this._rendererOptions.cssRotation || this._rendererOptions.cssScale ) {
          this._rendererOptions.baseNode = this;
        } else if ( this._rendererOptions.hasOwnProperty( 'baseNode' ) ) {
          delete this._rendererOptions.baseNode; // don't override, let the scene pass in the scene
        }
        // if we set renderer and rendererOptions, only then do we want to trigger a specific layer type
        this._rendererLayerType = this._renderer.createLayerType( this._rendererOptions );
      } else {
        this._rendererLayerType = null; // nothing signaled, since we want to support multiple layer types (including if we specify a renderer)
      }
    },
    
    getRendererLayerType: function() {
      return this._rendererLayerType;
    },
    
    hasRendererLayerType: function() {
      return !!this._rendererLayerType;
    },
    
    setRenderer: function( renderer ) {
      var newRenderer;
      if ( typeof renderer === 'string' ) {
        assert && assert( scenery.Renderer[renderer], 'unknown renderer in setRenderer: ' + renderer );
        newRenderer = scenery.Renderer[renderer];
      } else if ( renderer instanceof scenery.Renderer ) {
        newRenderer = renderer;
      } else if ( !renderer ) {
        newRenderer = null;
      } else {
        throw new Error( 'unrecognized type of renderer: ' + renderer );
      }
      if ( newRenderer !== this._renderer ) {
        assert && assert( !this.isPainted() || !newRenderer || _.contains( this._supportedRenderers, newRenderer ), 'renderer ' + newRenderer + ' not supported by ' + this.constructor.name );
        this._renderer = newRenderer;
        
        this.updateLayerType();
        this.markLayerRefreshNeeded();
      }
    },
    
    getRenderer: function() {
      return this._renderer;
    },
    
    hasRenderer: function() {
      return !!this._renderer;
    },
    
    setRendererOptions: function( options ) {
      // TODO: consider checking options based on the specified 'renderer'?
      this._rendererOptions = options;
      
      this.updateLayerType();
      this.markLayerRefreshNeeded();
    },
    
    getRendererOptions: function() {
      return this._rendererOptions;
    },
    
    hasRendererOptions: function() {
      return !!this._rendererOptions;
    },
    
    setLayerSplitBefore: function( split ) {
      if ( this._layerSplitBefore !== split ) {
        this._layerSplitBefore = split;
        this.markLayerRefreshNeeded();
      }
    },
    
    isLayerSplitBefore: function() {
      return this._layerSplitBefore;
    },
    
    setLayerSplitAfter: function( split ) {
      if ( this._layerSplitAfter !== split ) {
        this._layerSplitAfter = split;
        this.markLayerRefreshNeeded();
      }
    },
    
    isLayerSplitAfter: function() {
      return this._layerSplitAfter;
    },
    
    setLayerSplit: function( split ) {
      if ( split !== this._layerSplitBefore || split !== this._layerSplitAfter ) {
        this._layerSplitBefore = split;
        this._layerSplitAfter = split;
        this.markLayerRefreshNeeded();
      }
    },
    
    // returns a unique trail (if it exists) where each node in the ancestor chain has 0 or 1 parents
    getUniqueTrail: function() {
      var trail = new scenery.Trail();
      var node = this;
      
      while ( node ) {
        trail.addAncestor( node );
        assert && assert( node._parents.length <= 1 );
        node = node._parents[0]; // should be undefined if there aren't any parents
      }
      
      return trail;
    },
    
    debugText: function() {
      var startPointer = new scenery.TrailPointer( new scenery.Trail( this ), true );
      var endPointer = new scenery.TrailPointer( new scenery.Trail( this ), false );
      
      var depth = 0;
      
      startPointer.depthFirstUntil( endPointer, function( pointer ) {
        if ( pointer.isBefore ) {
          // hackish way of multiplying a string
          var padding = new Array( depth * 2 ).join( ' ' );
          console.log( padding + pointer.trail.lastNode().getId() + ' ' + pointer.trail.toString() );
        }
        depth += pointer.isBefore ? 1 : -1;
      }, false );
    },
    
    /*
     * Renders this node to a canvas. If toCanvas( callback ) is used, the canvas will contain the node's
     * entire bounds.
     *
     * callback( canvas, x, y ) is called, where x and y offsets are computed if not specified.
     */
    toCanvas: function( callback, x, y, width, height ) {
      var self = this;
      
      var padding = 2; // padding used if x and y are not set
      
      var bounds = this.getBounds();
      x = x !== undefined ? x : Math.ceil( padding - bounds.minX );
      y = y !== undefined ? y : Math.ceil( padding - bounds.minY );
      width = width !== undefined ? width : Math.ceil( x + bounds.getWidth() + padding );
      height = height !== undefined ? height : Math.ceil( y + bounds.getHeight() + padding );
      
      var canvas = document.createElement( 'canvas' );
      canvas.width = width;
      canvas.height = height;
      var context = canvas.getContext( '2d' );
      
      var $div = $( document.createElement( 'div' ) );
      $div.width( width ).height( height );
      var scene = new scenery.Scene( $div );
      
      scene.addChild( self );
      scene.x = x;
      scene.y = y;
      scene.updateScene();
      
      scene.renderToCanvas( canvas, context, function() {
        callback( canvas, x, y );
        
        // let us be garbage collected
        scene.removeChild( self );
      } );
    },
    
    // gives a data URI, with the same parameter handling as Node.toCanvas()
    toDataURL: function( callback, x, y, width, height ) {
      this.toCanvas( function( canvas, x, y ) {
        // this x and y shadow the outside parameters, and will be different if the outside parameters are undefined
        callback( canvas.toDataURL(), x, y );
      }, x, y, width, height );
    },
    
    // gives an HTMLImageElement with the same parameter handling as Node.toCanvas()
    toImage: function( callback, x, y, width, height ) {
      this.toDataURL( function( url, x, y ) {
        // this x and y shadow the outside parameters, and will be different if the outside parameters are undefined
        var img = document.createElement( 'img' );
        img.onload = function() {
          callback( img, x, y );
          delete img.onload;
        };
        img.src = url;
      }, x, y, width, height );
    },
    
    /*---------------------------------------------------------------------------*
    * Coordinate transform methods
    *----------------------------------------------------------------------------*/
    
    // apply this node's transform to the point
    localToParentPoint: function( point ) {
      return this._transform.transformPosition2( point );
    },
    
    localToParentBounds: function( bounds ) {
      return this._transform.transformBounds2( bounds );
    },
    
    // apply the inverse of this node's transform to the point
    parentToLocalPoint: function( point ) {
      return this._transform.inversePosition2( point );
    },
    
    parentToLocalBounds: function( bounds ) {
      return this._transform.inverseBounds2( bounds );
    },
    
    // apply this node's transform (and then all of its parents' transforms) to the point
    localToGlobalPoint: function( point ) {
      var node = this;
      while ( node !== null ) {
        point = node._transform.transformPosition2( point );
        assert && assert( node._parents[1] === undefined, 'localToGlobalPoint unable to work for DAG' );
        node = node._parents[0];
      }
      return point;
    },
    
    localToGlobalBounds: function( bounds ) {
      var node = this;
      while ( node !== null ) {
        bounds = node._transform.transformBounds2( bounds );
        assert && assert( node._parents[1] === undefined, 'localToGlobalBounds unable to work for DAG' );
        node = node._parents[0];
      }
      return bounds;
    },
    
    globalToLocalPoint: function( point ) {
      var node = this;
      
      // we need to apply the transformations in the reverse order, so we temporarily store them
      var transforms = [];
      while ( node !== null ) {
        transforms.push( node._transform );
        assert && assert( node._parents[1] === undefined, 'globalToLocalPoint unable to work for DAG' );
        node = node._parents[0];
      }
      
      // iterate from the back forwards (from the root node to here)
      for ( var i = transforms.length - 1; i >=0; i-- ) {
        point = transforms[i].inversePosition2( point );
      }
      return point;
    },
    
    globalToLocalBounds: function( bounds ) {
      var node = this;
      
      // we need to apply the transformations in the reverse order, so we temporarily store them
      var transforms = [];
      while ( node !== null ) {
        transforms.push( node._transform );
        assert && assert( node._parents[1] === undefined, 'globalToLocalBounds unable to work for DAG' );
        node = node._parents[0];
      }
      
      // iterate from the back forwards (from the root node to here)
      for ( var i = transforms.length - 1; i >=0; i-- ) {
        bounds = transforms[i].inverseBounds2( bounds );
      }
      return bounds;
    },
    
    /*---------------------------------------------------------------------------*
    * ES5 get/set
    *----------------------------------------------------------------------------*/
    
    set layerSplit( value ) { this.setLayerSplit( value ); },
    get layerSplit() { throw new Error( 'You can\'t get a layerSplit property, since it modifies two separate properties' ); },
    
    set layerSplitBefore( value ) { this.setLayerSplitBefore( value ); },
    get layerSplitBefore() { return this.isLayerSplitBefore(); },
    
    set layerSplitAfter( value ) { this.setLayerSplitAfter( value ); },
    get layerSplitAfter() { return this.isLayerSplitAfter(); },
    
    set renderer( value ) { this.setRenderer( value ); },
    get renderer() { return this.getRenderer(); },
    
    set rendererOptions( value ) { this.setRendererOptions( value ); },
    get rendererOptions() { return this.getRendererOptions(); },
    
    set cursor( value ) { this.setCursor( value ); },
    get cursor() { return this.getCursor(); },
    
    set visible( value ) { this.setVisible( value ); },
    get visible() { return this.isVisible(); },
    
    set opacity( value ) { this.setOpacity( value ); },
    get opacity() { return this.getOpacity(); },
    
    set pickable( value ) { this.setPickable( value ); },
    get pickable() { return this.isPickable(); },
    
    set transform( value ) { this.setTransform( value ); },
    get transform() { return this.getTransform(); },
    
    set matrix( value ) { this.setMatrix( value ); },
    get matrix() { return this.getMatrix(); },
    
    set translation( value ) { this.setTranslation( value ); },
    get translation() { return this.getTranslation(); },
    
    set rotation( value ) { this.setRotation( value ); },
    get rotation() { return this.getRotation(); },
    
    set x( value ) { this.setX( value ); },
    get x() { return this.getX(); },
    
    set y( value ) { this.setY( value ); },
    get y() { return this.getY(); },
    
    set left( value ) { this.setLeft( value ); },
    get left() { return this.getLeft(); },
    
    set right( value ) { this.setRight( value ); },
    get right() { return this.getRight(); },
    
    set top( value ) { this.setTop( value ); },
    get top() { return this.getTop(); },
    
    set bottom( value ) { this.setBottom( value ); },
    get bottom() { return this.getBottom(); },
    
    set center( value ) { this.setCenter( value ); },
    get center() { return this.getCenter(); },
    
    set centerX( value ) { this.setCenterX( value ); },
    get centerX() { return this.getCenterX(); },
    
    set centerY( value ) { this.setCenterY( value ); },
    get centerY() { return this.getCenterY(); },
    
    set children( value ) { this.setChildren( value ); },
    get children() { return this.getChildren(); },
    
    get parents() { return this.getParents(); },
    
    get width() { return this.getWidth(); },
    get height() { return this.getHeight(); },
    get bounds() { return this.getBounds(); },
    get selfBounds() { return this.getSelfBounds(); },
    get childBounds() { return this.getChildBounds(); },
    get id() { return this.getId(); },
    
    mutate: function( options ) {
      var node = this;
      
      _.each( this._mutatorKeys, function( key ) {
        if ( options[key] !== undefined ) {
          var descriptor = Object.getOwnPropertyDescriptor( Node.prototype, key );
          
          // if the key refers to a function that is not ES5 writable, it will execute that function with the single argument
          if ( descriptor && typeof descriptor.value === 'function' ) {
            node[key]( options[key] );
          } else {
            node[key] = options[key];
          }
        }
      } );
      
      return this; // allow chaining
    },
    
    toString: function( spaces ) {
      spaces = spaces || '';
      var props = this.getPropString( spaces + '  ' );
      return spaces + this.getBasicConstructor( props ? ( '\n' + props + '\n' + spaces ) : '' );
    },
    
    getBasicConstructor: function( propLines ) {
      return 'new scenery.Node( {' + propLines + '} )';
    },
    
    getPropString: function( spaces ) {
      var self = this;
      
      var result = '';
      function addProp( key, value, nowrap ) {
        if ( result ) {
          result += ',\n';
        }
        if ( !nowrap && typeof value === 'string' ) {
          result += spaces + key + ': \'' + value + '\'';
        } else {
          result += spaces + key + ': ' + value;
        }
      }
      
      if ( this._children.length ) {
        var childString = '';
        _.each( this._children, function( child ) {
          if ( childString ) {
            childString += ',\n';
          }
          childString += child.toString( spaces + '  ' );
        } );
        addProp( 'children', '[\n' + childString + '\n' + spaces + ']', true );
      }
      
      // direct copy props
      if ( this.cursor ) { addProp( 'cursor', this.cursor ); }
      if ( !this.visible ) { addProp( 'visible', this.visible ); }
      if ( !this.pickable ) { addProp( 'pickable', this.pickable ); }
      if ( this.opacity !== 1 ) { addProp( 'opacity', this.opacity ); }
      
      if ( !this.transform.isIdentity() ) {
        var m = this.transform.getMatrix();
        addProp( 'matrix', 'new dot.Matrix3( ' + m.m00() + ', ' + m.m01() + ', ' + m.m02() + ', ' +
                                                 m.m10() + ', ' + m.m11() + ', ' + m.m12() + ', ' +
                                                 m.m20() + ', ' + m.m21() + ', ' + m.m22() + ' )', true );
      }
      
      if ( this.renderer ) {
        addProp( 'renderer', this.renderer.name );
        if ( this.rendererOptions ) {
          addProp( 'rendererOptions', JSON.stringify( this.rendererOptions ), true );
        }
      }
      
      if ( this._layerSplitBefore ) {
        addProp( 'layerSplitBefore', true );
      }
      
      if ( this._layerSplitAfter ) {
        addProp( 'layerSplitAfter', true );
      }
      
      return result;
    }
  };
  
  /*
   * This is an array of property (setter) names for Node.mutate(), which are also used when creating nodes with parameter objects.
   *
   * E.g. new scenery.Node( { x: 5, rotation: 20 } ) will create a Path, and apply setters in the order below (node.x = 5; node.rotation = 20)
   *
   * The order below is important! Don't change this without knowing the implications.
   * NOTE: translation-based mutators come before rotation/scale, since typically we think of their operations occuring "after" the rotation / scaling
   * NOTE: left/right/top/bottom/centerX/centerY are at the end, since they rely potentially on rotation / scaling changes of bounds that may happen beforehand
   * TODO: using more than one of {translation,x,left,right,centerX} or {translation,y,top,bottom,centerY} should be considered an error
   * TODO: move fill / stroke setting to mixins
   */
  Node.prototype._mutatorKeys = [ 'children', 'cursor', 'visible', 'pickable', 'opacity', 'matrix', 'translation', 'x', 'y', 'rotation', 'scale',
                                  'left', 'right', 'top', 'bottom', 'center', 'centerX', 'centerY', 'renderer', 'rendererOptions',
                                  'layerSplit', 'layerSplitBefore', 'layerSplitAfter' ];
  
  Node.prototype._supportedRenderers = [];
  
  Node.prototype.layerStrategy = LayerStrategy;
  
  return Node;
} );
