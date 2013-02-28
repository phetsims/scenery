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
  
  var scenery = require( 'SCENERY/scenery' );
  var LayerStrategy = require( 'SCENERY/layers/LayerStrategy' ); // used to set the default layer strategy on the prototype
  // require( 'SCENERY/layers/Renderer' ); // commented out so Require.js doesn't balk at the circular dependency
  
  // TODO: FIXME: Why do I have to comment out this dependency?
  // require( 'SCENERY/Trail' );
  
  var globalIdCounter = 1;
  
  // TODO: consider an args-style constructor here!
  scenery.Node = function( options ) {
    // assign a unique ID to this node (allows trails to )
    this._id = globalIdCounter++;
    
    // TODO: hide as _visible, add setter/getter
    this._visible = true;
    
    // This node and all children will be clipped by this shape (in addition to any other clipping shapes).
    // The shape should be in the local coordinate frame
    this._clipShape = null;
    
    // the CSS cursor to be displayed over this node. null should be the default (inherit) value
    this._cursor = null;
    
    // TODO: consider defensive copy getters?
    this.children = []; // ordered
    this.parents = []; // unordered
    
    this.transform = new Transform3();
    
    this._inputListeners = []; // for user input handling (mouse/touch)
    this._eventListeners = []; // for internal events like paint invalidation, layer invalidation, etc.
    
    // TODO: add getter/setters that will be able to invalidate whether this node is under any fingers, etc.
    this._includeStrokeInHitRegion = false;
    
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
    
    enterState: function( state, trail ) {
      // apply this node's transform
      if ( !this.transform.isIdentity() ) {
        // TODO: consider a stack-based model for transforms?
        state.applyTransformationMatrix( this.transform.getMatrix() );
      }
      
      if ( this._clipShape ) {
        // push the clipping shape in the global coordinate frame (relative to the trail)
        state.pushClipShape( trail.getTransform().transformShape( this._clipShape ) );
      }
    },
    
    exitState: function( state, trail ) {
      if ( this._clipShape ) {
        state.popClipShape();
      }
      
      // apply the inverse of this node's transform
      if ( !this.transform.isIdentity() ) {
        state.applyTransformationMatrix( this.transform.getInverse() );
      }
    },
    
    insertChild: function( node, index ) {
      assert && assert( node !== null && node !== undefined && !_.contains( this.children, node ) );
      
      node.parents.push( this );
      this.children.splice( index, 0, node );
      
      node.invalidateBounds();
      node.invalidatePaint();
      
      this.dispatchEvent( 'insertChild', {
        parent: this,
        child: node,
        index: index
      } );
    },
    
    addChild: function( node ) {
      this.insertChild( node, this.children.length );
    },
    
    removeChild: function ( node ) {
      assert && assert( this.isChild( node ) );
      
      node.markOldPaint();
      
      var indexOfParent = _.indexOf( node.parents, this );
      var indexOfChild = _.indexOf( this.children, node );
      
      node.parents.splice( indexOfParent, 1 );
      this.children.splice( indexOfChild, 1 );
      
      this.invalidateBounds();
      
      this.dispatchEvent( 'removeChild', {
        parent: this,
        child: node,
        index: indexOfChild
      } );
    },
    
    // TODO: efficiency by batching calls?
    setChildren: function( children ) {
      var node = this;
      if ( this.children !== children ) {
        _.each( this.children.slice( 0 ), function( child ) {
          node.removeChild( node );
        } );
        _.each( children, function( child ) {
          node.addChild( node );
        } );
      }
    },
    
    // remove this node from its parents
    detach: function () {
      var that = this;
      _.each( this.parents.slice( 0 ), function( parent ) {
        parent.removeChild( that );
      } );
    },
    
    // ensure that cached bounds stored on this node (and all children) are accurate
    validateBounds: function() {
      var that = this;
      
      if ( this._selfBoundsDirty ) {
        this._selfBoundsDirty = false;
        
        // if our self bounds changed, make sure to paint the area where our new bounds are
        this.markDirtyRegion( this._selfBounds );
      }
      
      // validate bounds of children if necessary
      if ( this._childBoundsDirty ) {
        // have each child validate their own bounds
        _.each( this.children, function( child ) {
          child.validateBounds();
        } );
        
        // and recompute our _childBounds
        this._childBounds = Bounds2.NOTHING;
        
        _.each( this.children, function( child ) {
          that._childBounds = that._childBounds.union( child._bounds );
        } );
        
        this._childBoundsDirty = false;
      }
      
      // TODO: layout here?
      
      if ( this._boundsDirty ) {
        var oldBounds = this._bounds;
        
        var newBounds = this.localToParentBounds( this._selfBounds ).union( that.localToParentBounds( this._childBounds ) );
        
        if ( !newBounds.equals( oldBounds ) ) {
          this._bounds = newBounds;
          
          _.each( this.parents, function( parent ) {
            parent.invalidateBounds();
          } );
          // TODO: fire off event listeners?
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
        
        _.each( this.children, function( child ) {
          child.validatePaint();
        } );
      }
    },
    
    // mark the bounds of this node as invalid, so it is recomputed before it is accessed again
    invalidateBounds: function() {
      this._boundsDirty = true;
      
      // and set flags for all ancestors
      _.each( this.parents, function( parent ) {
        parent.invalidateChildBounds();
      } );
    },
    
    // recursively tag all ancestors with _childBoundsDirty
    invalidateChildBounds: function() {
      // don't bother updating if we've already been tagged
      if ( !this._childBoundsDirty ) {
        this._childBoundsDirty = true;
        _.each( this.parents, function( parent ) {
          parent.invalidateChildBounds();
        } );
      }
    },
    
    // mark the paint of this node as invalid, so its new region will be painted
    invalidatePaint: function() {
      this._paintDirty = true;
      
      // and set flags for all ancestors
      _.each( this.parents, function( parent ) {
        parent.invalidateChildPaint();
      } );
    },
    
    // recursively tag all ancestors with _childPaintDirty
    invalidateChildPaint: function() {
      // don't bother updating if we've already been tagged
      if ( !this._childPaintDirty ) {
        this._childPaintDirty = true;
        _.each( this.parents, function( parent ) {
          parent.invalidateChildPaint();
        } );
      }
    },
    
    // called to notify that self rendering will display different paint, with possibly different bounds
    invalidateSelf: function( newBounds ) {
      assert && assert( !isNaN( newBounds.x() ) );
      
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
      this.dispatchEvent( 'layerRefresh', {
        node: this
      } );
    },
    
    // marks the last-rendered bounds of this node and optionally all of its descendants as needing a repaint
    markOldPaint: function( justSelf ) {
      function ancestorHasOldPaint( node ) {
        if( node._oldPaintMarked ) {
          return true;
        }
        return _.some( node.parents, function( parent ) {
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
    
    isChild: function ( potentialChild ) {
      var ourChild = _.contains( this.children, potentialChild );
      var itsParent = _.contains( potentialChild.parents, this );
      assert && assert( ourChild === itsParent );
      return ourChild;
    },
    
    // the bounds for self content in "local" coordinates
    getSelfBounds: function() {
      return this._selfBounds;
    },
    
    // the bounds for content in render(), in "parent" coordinates
    getBounds: function() {
      this.validateBounds();
      return this._bounds;
    },
    
    // return the top node (if any, otherwise null) whose self-rendered area contains the point (in parent coordinates).
    trailUnderPoint: function( point ) {
      assert && assert( point, 'trailUnderPointer requires a point' );
      // update bounds for pruning
      this.validateBounds();
      
      // bail quickly if this doesn't hit our computed bounds
      if ( !this._bounds.containsPoint( point ) ) { return null; }
      
      // point in the local coordinate frame. computed after the main bounds check, so we can bail out there efficiently
      var localPoint = this.transform.inversePosition2( point );
      
      // check children first, since they are rendered later
      if ( this.children.length > 0 && this._childBounds.containsPoint( localPoint ) ) {
        
        // manual iteration here so we can return directly, and so we can iterate backwards (last node is in front)
        for ( var i = this.children.length - 1; i >= 0; i-- ) {
          var child = this.children[i];
          
          var childHit = child.trailUnderPoint( localPoint );
          
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
    
    hasSelf: function() {
      return false;
    },
    
    hasParent: function() {
      return this.parents.length !== 0;
    },
    
    hasChildren: function() {
      return this.children.length > 0;
    },
    
    walkDepthFirst: function( callback ) {
      callback( this );
      _.each( this.children, function( child ) {
        child.walkDepthFirst( callback );
      } );
    },
    
    getChildrenWithinBounds: function( bounds ) {
      return _.filter( this.children, function( child ) { return !child._bounds.intersection( bounds ).isEmpty(); } );
    },
    
    getChildren: function() {
      return this.children.slice( 0 ); // create a defensive copy
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
    
    // dispatches an event across all possible Trails ending in this node
    dispatchEvent: function( type, args ) {
      var trail = new scenery.Trail();
      
      function recursiveEventDispatch( node ) {
        trail.addAncestor( node );
        
        args.trail = trail;
        
        _.each( node.getEventListeners(), function( eventListener ) {
          if ( eventListener[type] ) {
            eventListener[type]( args );
          }
        } );
        
        _.each( node.parents, function( parent ) {
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
        
        _.each( node.getEventListeners(), function( eventListener ) {
          if ( eventListener[type] ) {
            eventListener[type]( args );
          }
        } );
        
        _.each( node.parents, function( parent ) {
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
    
    // scaleBy( s ) is also supported, which will scale both dimensions by the same amount. renamed from 'scale' to satisfy the setter/getter
    scaleBy: function( x, y, prependInstead ) {
      if ( typeof x === 'number' ) {
        if ( y === undefined ) {
          // scaleBy( scale )
          this.appendMatrix( Matrix3.scaling( x, x ) );
        } else {
          // scaleBy( x, y, prependInstead )
          if ( prependInstead ) {
            this.prependMatrix( Matrix3.scaling( x, y ) );
          } else {
            this.appendMatrix( Matrix3.scaling( x, y ) );
          }
        }
      } else {
        // scaleBy( vector, prependInstead )
        var vector = x;
        this.scaleBy( vector.x, vector.y, y ); // forward to full version
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
    getScale: function() {
      return this.transform.getMatrix().scaling();
    },
    
    // supports setScale( 5 ) for both dimensions, setScale( 5, 3 ) for each dimension separately, or setScale( new Vector2( x, y ) )
    setScale: function( a, b ) {
      var currentScale = this.getScale();
      
      if ( typeof a === 'number' ) {
        if ( b === undefined ) {
          // to map setScale( scale ) => setScale( scale, scale )
          b = a;
        }
        // setScale( x, y )
        this.appendMatrix( Matrix3.scaling( a / currentScale.x, b / currentScale.y ) );
      } else {
        // setScale( vector ), where we set the x-scale to vector.x and y-scale to vector.y
        this.appendMatrix( Matrix3.scaling( a.x / currentScale.x, a.y / currentScale.y ) );
      }
      return this;
    },
    
    getRotation: function() {
      return this.transform.getMatrix().rotation();
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
      return this.transform.getMatrix().translation();
    },
    
    notifyTransformChange: function( matrix, type ) {
      this.dispatchEventWithTransform( 'transform', {
        node: this,
        type: type,
        matrix: matrix
      } );
    },
    
    // append a transformation matrix to our local transform
    appendMatrix: function( matrix ) {
      // invalidate paint TODO improve methods for this
      this.markOldPaint();
      
      this.transform.append( matrix );
      
      this.notifyTransformChange( matrix, 'append' );
      this.invalidateBounds();
      this.invalidatePaint();
    },
    
    // prepend a transformation matrix to our local transform
    prependMatrix: function( matrix ) {
      // invalidate paint TODO improve methods for this
      this.markOldPaint();
      
      this.transform.prepend( matrix );
      
      this.notifyTransformChange( matrix, 'prepend' );
      this.invalidateBounds();
      this.invalidatePaint();
    },
    
    setMatrix: function( matrix ) {
      this.markOldPaint();
      
      this.transform.set( matrix );
      
      this.notifyTransformChange( matrix, 'set' );
      this.invalidateBounds();
      this.invalidatePaint();
    },
    
    getMatrix: function() {
      return this.transform.getMatrix();
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
    
    getCenterX: function() {
      return this.getBounds().centerX();
    },
    
    setCenterX: function( x ) {
      this.translate( x - this.getCenterX(), 0, true );
      return this; // allow chaining
    },
    
    getCenterY: function() {
      return this.getBounds().centerY();
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
      return this.getBounds().width();
    },
    
    getHeight: function() {
      return this.getBounds().height();
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
        
        if ( visible ) {
          this.invalidatePaint();
        }
      }
      return this;
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
      } else {
        throw new Error( 'unrecognized type of renderer: ' + renderer );
      }
      if ( newRenderer !== this._renderer ) {
        assert && assert( !this.hasSelf() || _.contains( this._supportedRenderers, newRenderer ), 'renderer ' + newRenderer + ' not supported by ' + this );
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
    
    // returns a unique trail (if it exists) where each node in the ancestor chain has 0 or 1 parents
    getUniqueTrail: function() {
      var trail = new scenery.Trail();
      var node = this;
      
      while ( node ) {
        trail.addAncestor( node );
        assert && assert( node.parents.length <= 1 );
        node = node.parents[0]; // should be undefined if there aren't any parents
      }
      
      return trail;
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
    
    /*---------------------------------------------------------------------------*
    * ES5 get/set
    *----------------------------------------------------------------------------*/
    
    set layerSplitBefore( value ) { this.setLayerSplitBefore( value ); },
    get layerSplitBefore() { return this.isLayerSplitBefore(); },
    
    set layerSplitAfter( value ) { this.setLayerSplitAfter( value ); },
    get layerSplitAfter() { return this.isLayerSplitAfter(); },
    
    set renderer( value ) { this.setRenderer( value ); },
    get renderer() { return this.getRenderer(); },
    
    set rendererOptions( value ) { this.setRendererOptions( value ); },
    get rendererOptions() { return this.getRendererOptions(); },
    
    set cursor( value ) { this.setCursor( value ); },
    get cursor() { return this.isCursor(); },
    
    set visible( value ) { this.setVisible( value ); },
    get visible() { return this.isVisible(); },
    
    set matrix( value ) { this.setMatrix( value ); },
    get matrix() { return this.getMatrix(); },
    
    set translation( value ) { this.setTranslation( value ); },
    get translation() { return this.getTranslation(); },
    
    set rotation( value ) { this.setRotation( value ); },
    get rotation() { return this.getRotation(); },
    
    set scale( value ) { this.setScale( value ); },
    get scale() { return this.getScale(); },
    
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
    
    set centerX( value ) { this.setCenterX( value ); },
    get centerX() { return this.getCenterX(); },
    
    set centerY( value ) { this.setCenterY( value ); },
    get centerY() { return this.getCenterY(); },
    
    get width() { return this.getWidth(); },
    get height() { return this.getHeight(); },
    get id() { return this.getId(); },
    
    mutate: function( options ) {
      
      var node = this;
      
      _.each( this._mutatorKeys, function( key ) {
        if ( options[key] !== undefined ) {
          node[key] = options[key];
        }
      } );
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
  Node.prototype._mutatorKeys = [ 'cursor', 'visible', 'translation', 'x', 'y', 'rotation', 'scale',
                                  'left', 'right', 'top', 'bottom', 'centerX', 'centerY', 'renderer', 'rendererOptions',
                                  'layerSplitBefore', 'layerSplitAfter' ];
  
  Node.prototype._supportedRenderers = [];
  
  Node.prototype.layerStrategy = LayerStrategy.DefaultLayerStrategy;
  
  return Node;
} );
