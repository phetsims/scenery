// Copyright 2002-2014, University of Colorado Boulder

/**
 * A node for the Scenery scene graph. Supports general directed acyclic graphics (DAGs).
 * Handles multiple layers with assorted types (Canvas 2D, SVG, DOM, WebGL, etc.).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var extend = require( 'PHET_CORE/extend' );

  var Events = require( 'AXON/Events' );

  var Bounds2 = require( 'DOT/Bounds2' );
  var Transform3 = require( 'DOT/Transform3' );
  var Matrix3 = require( 'DOT/Matrix3' );
  var Vector2 = require( 'DOT/Vector2' );
  var clamp = require( 'DOT/Util' ).clamp;

  var Shape = require( 'KITE/Shape' );

  var scenery = require( 'SCENERY/scenery' );
  require( 'SCENERY/util/RendererSummary' );
  require( 'SCENERY/util/CanvasContextWrapper' );
  // require( 'SCENERY/display/Renderer' ); // commented out so Require.js doesn't balk at the circular dependency

  // TODO: FIXME: Why do I have to comment out this dependency?
  // require( 'SCENERY/util/Trail' );
  // require( 'SCENERY/util/TrailPointer' );

  var globalIdCounter = 1;

  var eventsRequiringBoundsValidation = {
    'childBounds': true,
    'localBounds': true,
    'bounds': true
  };

  var trailUnderPointerOptions = {};

  function defaultTrailPredicate( node ) {
    return node._parents.length === 0;
  }

  /*
   * Available keys for use in the options parameter object for a vanilla Node (not inherited), in the order they are executed in:
   *
   * children:         A list of children to add (in order)
   * cursor:           Will display the specified CSS cursor when the mouse is over this Node or one of its descendents. The Scene needs to have input listeners attached with an initialize method first.
   * visible:          If false, this node (and its children) will not be displayed (or get input events)
   * pickable:         If false, this node (and its children) will not get input events
   * opacity:          Sets the opacity in the range [0,1]
   * matrix:           Sets the {Matrix3} transformation matrix (sets translation, rotation and scaling)
   * translation:      Sets the translation of the node to either the specified dot.Vector2 value, or the x,y values from an object (e.g. translation: { x: 1, y: 2 } )
   * x:                Sets the x-translation of the node
   * y:                Sets the y-translation of the node
   * rotation:         Sets the rotation of the node in radians
   * scale:            Sets the scale of the node. Supports either a number (same x-y scale), or a dot.Vector2 / object with ob.x and ob.y to set the scale for each axis independently
   * leftTop:          Sets the translation so that the left-top corner of the bounding box (in the parent coordinate frame) is at the specified point
   * centerTop:        Sets the translation so that the center of the top edge of the bounding box (in the parent coordinate frame) is at the specified point
   * rightTop:         Sets the translation so that the right-top corner of the bounding box (in the parent coordinate frame) is at the specified point
   * leftCenter:       Sets the translation so that the center of the left edge of the bounding box (in the parent coordinate frame) is at the specified point
   * center:           Sets the translation so that the center the bounding box (in the parent coordinate frame) is at the specified point
   * rightCenter:      Sets the translation so that the center of the right edge of the bounding box (in the parent coordinate frame) is at the specified point
   * leftBottom:       Sets the translation so that the left-bottom corner of the bounding box (in the parent coordinate frame) is at the specified point
   * centerBottom:     Sets the translation so that the center of the bottom edge of the bounding box (in the parent coordinate frame) is at the specified point
   * rightBottom:      Sets the translation so that the right-bottom corner of the bounding box (in the parent coordinate frame) is at the specified point
   * left:             Sets the x-translation so that the left (min X) of the bounding box (in the parent coordinate frame) is at the specified value
   * right:            Sets the x-translation so that the right (max X) of the bounding box (in the parent coordinate frame) is at the specified value
   * top:              Sets the y-translation so that the top (min Y) of the bounding box (in the parent coordinate frame) is at the specified value
   * bottom:           Sets the y-translation so that the bottom (min Y) of the bounding box (in the parent coordinate frame) is at the specified value
   * centerX:          Sets the x-translation so that the horizontal center of the bounding box (in the parent coordinate frame) is at the specified value
   * centerY:          Sets the y-translation so that the vertical center of the bounding box (in the parent coordinate frame) is at the specified value
   * renderer:         Forces Scenery to use the specific renderer (canvas/svg) to display this node (and if possible, children). Accepts both strings (e.g. 'canvas', 'svg', etc.) or actual Renderer objects (e.g. Renderer.Canvas, Renderer.SVG, etc.)
   * rendererOptions:  Parameter object that is passed to the created layer, and can affect how the layering process works.
   * layerSplit:       Forces a split between layers before and after this node (and its children) have been rendered. Useful for performance with Canvas-based renderers.
   * mouseArea:        Shape (in local coordinate frame) that overrides the 'hit area' for mouse input.
   * touchArea:        Shape (in local coordinate frame) that overrides the 'hit area' for touch input.
   * clipArea:         Shape (in local coordinate frame) that causes any graphics outside of the shape to be invisible (for the node and any children).
   * transformBounds:  Whether to compute tighter parent bounding boxes for rotated bounding boxes, or to just use the bounding box of the rotated bounding box.
   * focusable:        True if the node should be able to receive keyboard focus.
   */
  scenery.Node = function Node( options ) {
    var self = this;

    // supertype call to axon.Events (should just initialize a few properties here, notably _eventListeners and _staticEventListeners)
    Events.call( this );

    // assign a unique ID to this node (allows trails to get a unique list of IDs)
    this._id = globalIdCounter++;

    // all of the Instances tracking this Node
    this._instances = [];

    // drawable states that need to be updated on mutations. generally added by SVG and DOM elements that need to closely track state (possibly by Canvas to maintain dirty state)
    this._drawables = [];

    // Whether this node (and its children) will be visible when the scene is updated. Visible nodes by default will not be pickable either
    this._visible = true;

    // Opacity from 0 to 1
    this._opacity = 1;

    // Whether this node (and its subtree) will allow hit-testing (and thus user interaction). Notably:
    // pickable: null  - default. Node is only pickable if it (or an ancestor/descendant) has either an input listener or pickable: true set
    // pickable: false - Node (and subtree) is pickable, just like if there is an input listener
    // pickable: true  - Node is unpickable (only has an effect when underneath a node with an input listener / pickable: true set)
    this._pickable = null;

    // This node and all children will be clipped by this shape (in addition to any other clipping shapes).
    // {Shape} The shape should be in the local coordinate frame
    this._clipArea = null;

    // areas for hit intersection. if set on a Node, no descendants can handle events
    this._mouseArea = null; // {Shape|Bounds2} for mouse position          in the local coordinate frame
    this._touchArea = null; // {Shape|Bounds2} for touch and pen position  in the local coordinate frame

    // the CSS cursor to be displayed over this node. null should be the default (inherit) value
    this._cursor = null;

    this._children = []; // ordered
    this._parents = []; // unordered

    this._peers = []; // array of peer factories: { element: ..., options: ... }, where element can be an element or a string
    this._liveRegions = []; // array of live region instances

    // whether we will do more accurate (and tight) bounds computations for rotations and shears
    this._transformBounds = false;

    /*
     * Set up the transform reference. we add a listener so that the transform itself can be modified directly
     * by reference, or node.transform = <transform> / node.setTransform() can be used to change the transform reference.
     * Both should trigger the necessary event notifications for Scenery to keep track internally.
     */
    this._transform = new Transform3();
    this._transformListener = {
      // TODO: performance handling so we don't need to do two recursions!
      before: function() { self.beforeTransformChange(); },
      after: function() { self.afterTransformChange(); }
    };
    this._transform.addTransformListener( this._transformListener );

    this._inputListeners = []; // for user input handling (mouse/touch)

    // bounds handling
    this._bounds = Bounds2.NOTHING;      // for this node and its children, in "parent" coordinates
    this._localBounds = Bounds2.NOTHING; // for this node and its children, in "local" coordinates
    this._selfBounds = Bounds2.NOTHING;  // just for this node, in "local" coordinates
    this._childBounds = Bounds2.NOTHING; // just for children, in "local" coordinates
    this._localBoundsOverridden = false; // whether our localBounds have been set (with the ES5 setter / setLocalBounds()) to a custom value
    this._boundsDirty = true;
    this._localBoundsDirty = true;
    this._childBoundsDirty = true;

    // Similar to bounds, but includes any mouse/touch areas respectively, and excludes areas that would be pruned in hit-testing.
    // They are validated separately (independent from normal bounds validation), but should now always be non-null (since we now properly handle pruning)
    this._mouseBounds = Bounds2.NOTHING.copy(); // NOTE: MUTABLE! mouse/touch bounds are purely internal
    this._touchBounds = Bounds2.NOTHING.copy(); // NOTE: MUTABLE! mouse/touch bounds are purely internal
    this._mouseBoundsDirty = true; // whether the bounds are marked as dirty
    this._touchBoundsDirty = true; // whether the bounds are marked as dirty
    this._mouseBoundsHadListener = false; // since we only walk the dirty flags up ancestors, we need a way to re-evaluate descendants when the existence of effective listeners changes
    this._touchBoundsHadListener = false; // since we only walk the dirty flags up ancestors, we need a way to re-evaluate descendants when the existence of effective listeners changes

    // where rendering-specific settings are stored
    this._hints = {
      renderer: 0,          // what type of renderer should be forced for this node.
      usesOpacity: false,   // whether it is ancitipated that opacity will be switched on
      layerSplit: false,    // whether layers should be split before and after this node
      cssTransform: false,  // whether this node and its subtree should handle transforms by using a CSS transform of a div
      fullResolution: false // when rendered as Canvas, whether we should use full (device) resolution on retina-like devices
    };

    // the subtree pickable count is #pickable:true + #inputListeners, since we can prune subtrees with a pickable count of 0
    this._subtreePickableCount = 0;

    // a bitmask which specifies which renderers this node (and only this node, not its subtree) supports.
    this._rendererBitmask = scenery.bitmaskNodeDefault;

    // a bitmask-like summary of what renderers and options are supported by this node and all of its descendants
    this._rendererSummary = new scenery.RendererSummary( this );

    // So we can traverse only the subtrees that require bounds validation for events firing.
    // This is a sum of the number of events requiring bounds validation on this Node, plus the number of children whose count is non-zero.
    // NOTE: this means that if A has a child B, and B has a boundsEventCount of 5, it only contributes 1 to A's count. This allows us to
    // have changes localized (increasing B's count won't change A or any of A's ancestors), and guarantees that we will know whether a subtree
    // has bounds listeners. Also important: decreasing B's boundsEventCount down to 0 will allow A to decrease its count by 1, without having
    // to check its other children (if we were just using a boolean value, this operation would require A to check if any OTHER children besides
    // B had bounds listeners)
    this._boundsEventCount = 0;
    this._boundsEventSelfCount = 0; // this signals that we can validateBounds() on this subtree and we don't have to traverse further

    if ( options ) {
      this.mutate( options );
    }

    phetAllocation && phetAllocation( 'Node' );
  };
  var Node = scenery.Node;

  inherit( Object, Node, extend( {
    insertChild: function( index, node ) {
      assert && assert( node !== null && node !== undefined, 'insertChild cannot insert a null/undefined child' );
      assert && assert( !_.contains( this._children, node ), 'Parent already contains child' );
      assert && assert( node !== this, 'Cannot add self as a child' );

      // needs to be early to prevent re-entrant children modifications
      this.changePickableCount( node._subtreePickableCount );
      this.changeBoundsEventCount( node._boundsEventCount > 0 ? 1 : 0 );
      this._rendererSummary.bitmaskChange( scenery.bitmaskAll, node._rendererSummary.bitmask );

      node._parents.push( this );
      this._children.splice( index, 0, node );

      node.invalidateBounds();
      this._boundsDirty = true; // like calling this.invalidateBounds(), but we already marked all ancestors with dirty child bounds

      this.trigger2( 'childInserted', node, index );
    },

    addChild: function( node ) {
      this.insertChild( this._children.length, node );
    },

    removeChild: function( node ) {
      assert && assert( node );
      assert && assert( this.isChild( node ) );

      var indexOfChild = _.indexOf( this._children, node );

      this.removeChildWithIndex( node, indexOfChild );
    },

    removeChildAt: function( index ) {
      assert && assert( index >= 0 );
      assert && assert( index < this._children.length );

      var node = this._children[ index ];

      this.removeChildWithIndex( node, index );
    },

    // meant for internal use
    removeChildWithIndex: function( node, indexOfChild ) {
      assert && assert( node );
      assert && assert( this.isChild( node ) );
      assert && assert( this._children[ indexOfChild ] === node );

      // needs to be early to prevent re-entrant children modifications
      this.changePickableCount( -node._subtreePickableCount );
      this.changeBoundsEventCount( node._boundsEventCount > 0 ? -1 : 0 );
      this._rendererSummary.bitmaskChange( node._rendererSummary.bitmask, scenery.bitmaskAll );

      var indexOfParent = _.indexOf( node._parents, this );

      node._parents.splice( indexOfParent, 1 );
      this._children.splice( indexOfChild, 1 );

      this.invalidateBounds();
      this._childBoundsDirty = true; // force recomputation of child bounds after removing a child

      this.trigger2( 'childRemoved', node, indexOfChild );
    },

    removeAllChildren: function() {
      this.setChildren( [] );
    },

    // TODO: efficiency by batching calls?
    setChildren: function( children ) {
      if ( this._children !== children ) {
        // remove all children in a way where we don't have to copy the child array for safety
        while ( this._children.length ) {
          this.removeChild( this._children[ this._children.length - 1 ] );
        }

        var len = children.length;
        for ( var i = 0; i < len; i++ ) {
          this.addChild( children[ i ] );
        }
      }
    },

    getChildren: function() {
      // TODO: ensure we are not triggering this in Scenery code when not necessary!
      return this._children.slice( 0 ); // create a defensive copy
    },

    getChildrenCount: function() {
      return this._children.length;
    },

    getParents: function() {
      return this._parents.slice( 0 ); // create a defensive copy
    },

    // returns a single parent if it exists, otherwise null (no parents), or an assertion failure (multiple parents)
    getParent: function() {
      assert && assert( this._parents.length <= 1, 'Cannot call getParent on a node with multiple parents' );
      return this._parents.length ? this._parents[ 0 ] : null;
    },

    getChildAt: function( index ) {
      return this._children[ index ];
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

    // propagate the pickable count change down to our ancestors
    changePickableCount: function( n ) {
      this._subtreePickableCount += n;
      assert && assert( this._subtreePickableCount >= 0, 'subtree pickable count should be guaranteed to be >= 0' );
      var len = this._parents.length;
      for ( var i = 0; i < len; i++ ) {
        this._parents[ i ].changePickableCount( n );
      }

      // changing pickability can affect the mouseBounds/touchBounds used for hit testing
      this.invalidateMouseTouchBounds();
    },

    // update our event count, usually by 1 or -1. see docs on _boundsEventCount in constructor
    changeBoundsEventCount: function( n ) {
      if ( n !== 0 ) {
        var zeroBefore = this._boundsEventCount === 0;

        this._boundsEventCount += n;
        assert && assert( this._boundsEventCount >= 0, 'subtree bounds event count should be guaranteed to be >= 0' );

        var zeroAfter = this._boundsEventCount === 0;

        if ( zeroBefore !== zeroAfter ) {
          // parents will only have their count
          var parentDelta = zeroBefore ? 1 : -1;

          var len = this._parents.length;
          for ( var i = 0; i < len; i++ ) {
            this._parents[ i ].changeBoundsEventCount( parentDelta );
          }
        }
      }
    },

    // currently, there is no way to remove peers. if a string is passed as the element pattern, it will be turned into an element
    addPeer: function( element, options ) {
      assert && assert( !this.instances.length, 'Cannot call addPeer after a node has instances (yet)' );

      this._peers.push( { element: element, options: options } );
    },

    /**
     * @param property any object that has es5 getter for 'value' es5 setter for value, and
     */
    addLiveRegion: function( property, options ) {
      this._liveRegions.push( { property: property, options: options } );
    },

    // Ensure that cached bounds stored on this node (and all children) are accurate. Returns true if any sort of dirty flag was set
    validateBounds: function() {
      var that = this;
      var i;

      var wasDirtyBefore = false;

      // validate bounds of children if necessary
      if ( this._childBoundsDirty ) {
        wasDirtyBefore = true;

        // have each child validate their own bounds
        i = this._children.length;
        while ( i-- ) {
          this._children[ i ].validateBounds();
        }

        var oldChildBounds = this._childBounds;

        // and recompute our _childBounds
        this._childBounds = Bounds2.NOTHING.copy();

        i = this._children.length;
        while ( i-- ) {
          this._childBounds.includeBounds( this._children[ i ]._bounds );
        }

        // run this before firing the event
        this._childBoundsDirty = false;

        // TODO: don't execute this "if" comparison if there are no listeners?
        if ( !this._childBounds.equals( oldChildBounds ) ) {
          // TODO: consider changing to parameter object (that may be a problem for the GC overhead)
          this.trigger0( 'childBounds' );
        }
      }

      if ( this._localBoundsDirty && !this._localBoundsOverridden ) {
        wasDirtyBefore = true;

        this._localBoundsDirty = false; // we only need this to set local bounds as dirty

        var oldLocalBounds = this._localBounds;
        this._localBounds = this._selfBounds.union( this._childBounds ); // TODO: remove allocation

        // apply clipping to the bounds if we have a clip area (all done in the local coordinate frame)
        if ( this.hasClipArea() ) {
          this._localBounds.constrainBounds( this._clipArea.bounds );
        }

        // TODO: don't execute this "if" comparison if there are no listeners?
        if ( !this._localBounds.equals( oldLocalBounds ) ) {
          this.trigger0( 'localBounds' );

          // sanity check
          this._boundsDirty = true;
        }
      }

      // TODO: layout here?

      if ( this._boundsDirty ) {
        wasDirtyBefore = true;

        // run this before firing the event
        this._boundsDirty = false;

        var oldBounds = this._bounds;

        var newBounds;
        // no need to do the more expensive bounds transformation if we are still axis-aligned
        if ( this._transformBounds && !this._transform.getMatrix().isAxisAligned() ) {
          // mutates the matrix and bounds during recursion
          var matrix = this._transform.getMatrix().copy();
          newBounds = Bounds2.NOTHING.copy();
          // Include each painted self individually, transformed with the exact transform matrix.
          // This is expensive, as we have to do 2 matrix transforms for every descendant.
          this._includeTransformedSubtreeBounds( matrix, newBounds ); // self and children

          if ( this.hasClipArea() ) {
            newBounds.constrainBounds( this._clipArea.getBoundsWithTransform( matrix ) );
          }
        }
        else {
          // converts local to parent bounds. mutable methods used to minimize number of created bounds instances (we create one so we don't change references to the old one)
          newBounds = this.localToParentBounds( this._localBounds ); // TODO: reduce allocation? fully mutable?
        }
        var changed = !newBounds.equals( oldBounds );

        if ( changed ) {
          this._bounds = newBounds;

          i = this._parents.length;
          while ( i-- ) {
            this._parents[ i ].invalidateBounds();
          }

          // TODO: consider changing to parameter object (that may be a problem for the GC overhead)
          this.trigger0( 'bounds' );
        }
      }

      // if there were side-effects, run the validation again until we are clean
      if ( this._childBoundsDirty || this._boundsDirty ) {
        // TODO: if there are side-effects in listeners, this could overflow the stack. we should report an error instead of locking up
        this.validateBounds();
      }

      // double-check that all of our bounds handling has been accurate
      if ( assertSlow ) {
        // new scope for safety
        (function() {
          var epsilon = 0.000001;

          var childBounds = Bounds2.NOTHING.copy();
          _.each( that.children, function( child ) { childBounds.includeBounds( child._bounds ); } );

          var localBounds = that._selfBounds.union( childBounds );

          if ( that.hasClipArea() ) {
            localBounds = localBounds.intersection( that._clipArea.bounds );
          }

          var fullBounds = that.localToParentBounds( localBounds );

          assertSlow && assertSlow( that._childBounds.equalsEpsilon( childBounds, epsilon ), 'Child bounds mismatch after validateBounds: ' +
                                                                                             that._childBounds.toString() + ', expected: ' + childBounds.toString() );
          assertSlow && assertSlow( that._localBoundsOverridden ||
                                    that._transformBounds ||
                                    that._bounds.equalsEpsilon( fullBounds, epsilon ) ||
                                    that._bounds.equalsEpsilon( fullBounds, epsilon ),
            'Bounds mismatch after validateBounds: ' + that._bounds.toString() + ', expected: ' + fullBounds.toString() );
        })();
      }

      return wasDirtyBefore; // whether any dirty flags were set
    },

    // @private: Recursion for accurate transformed bounds handling. Mutates bounds with the added bounds.
    // Mutates the matrix, but mutates it back to the starting point (within floating-point error).
    _includeTransformedSubtreeBounds: function( matrix, bounds ) {
      if ( !this._selfBounds.isEmpty() ) {
        bounds.includeBounds( this.getTransformedSelfBounds( matrix ) );
      }

      var numChildren = this._children.length;
      for ( var i = 0; i < numChildren; i++ ) {
        var child = this._children[ i ];

        matrix.multiplyMatrix( child._transform.getMatrix() );
        child._includeTransformedSubtreeBounds( matrix, bounds );
        matrix.multiplyMatrix( child._transform.getInverse() );
      }

      return bounds;
    },

    // Traverses this subtree and validates bounds only for subtrees that have bounds listeners (trying to exclude as much as possible for performance)
    // This is done so that we can do the minimum bounds validation to prevent any bounds listeners from being triggered in further validateBounds() calls
    // without other Node changes being done. This is required to make the new rendering system work (planned for non-reentrance).
    // @public: NOTE: this should pass by (ignore) any overridden localBounds, to trigger listeners below.
    validateWatchedBounds: function() {
      // Since a bounds listener on one of the roots could invalidate bounds on the other, we need to keep running this until they are all clean.
      // Otherwise, side-effects could occur from bounds validations
      // TODO: consider a way to prevent infinite loops here that occur due to bounds listeners triggering cycles
      while ( this.watchedBoundsScan() ) {}
    },

    // @private: recursive function for validateWatchedBounds. Returned whether any validateBounds() returned true (means we have to traverse again)
    watchedBoundsScan: function() {
      if ( this._boundsEventSelfCount !== 0 ) {
        // we are a root that should be validated. return whether we updated anything
        return this.validateBounds();
      }
      else if ( this._boundsEventCount > 0 && this._childBoundsDirty ) {
        // descendants have watched bounds, traverse!
        var changed = false;
        var numChildren = this._children.length;
        for ( var i = 0; i < numChildren; i++ ) {
          changed = this._children[ i ].watchedBoundsScan() || changed;
        }
        return changed;
      }
      else {
        // if _boundsEventCount is zero, no bounds are watched below us (don't traverse), and it wasn't changed
        return false;
      }
    },

    /*
     * Updates the mouseBounds for the Node. It will include only the specific bounded areas that are relevant for hit-testing
     * mouse events. Thus it:
     * - includes mouseAreas (normal bounds don't)
     * - does not include subtrees that would be pruned in hit-testing
     */
    validateMouseBounds: function( hasListenerEquivalentSelfOrInAncestor ) {
      var that = this;

      // we'll need an updated value for this before deciding whether or not to bail
      hasListenerEquivalentSelfOrInAncestor = hasListenerEquivalentSelfOrInAncestor || this.hasInputListenerEquivalent();

      // Mouse bounds should be valid still if they aren't marked as dirty AND if the "had listener" matches.
      // Thus, even if the mouse bounds aren't marked as dirty, we can still force a refresh (for example: an input listener was added to an ancestor)
      if ( this._mouseBoundsDirty || this._mouseBoundsHadListener !== hasListenerEquivalentSelfOrInAncestor ) {
        // update whether we have a listener equivalent, so we can prune properly

        if ( this.isSubtreePickablePruned( hasListenerEquivalentSelfOrInAncestor ) ) {
          // if this subtree would be pruned, set the mouse bounds to nothing, and bail (skips the entire subtree, since it would never be hit-tested)
          this._mouseBounds.set( Bounds2.NOTHING );
        }
        else {
          // start with the self bounds, then add from there
          this._mouseBounds.set( this._selfBounds );

          // union of all children's mouse bounds
          var i = this._children.length;
          while ( i-- ) {
            var child = this._children[ i ];

            // make sure the child's mouseBounds are up to date
            child.validateMouseBounds( hasListenerEquivalentSelfOrInAncestor );
            that._mouseBounds.includeBounds( child._mouseBounds );
          }

          // do this before the transformation to the parent coordinate frame (the mouseArea is in the local coordinate frame)
          if ( this._mouseArea ) {
            // we accept either Bounds2, or a Shape (in which case, we take the Shape's bounds)
            this._mouseBounds.includeBounds( this._mouseArea.isBounds ? this._mouseArea : this._mouseArea.bounds );
          }

          if ( this.hasClipArea() ) {
            // exclude areas outside of the clipping area's bounds (for efficiency)
            this._mouseBounds.constrainBounds( this._clipArea.bounds );
          }

          // transform it to the parent coordinate frame
          this.transformBoundsFromLocalToParent( this._mouseBounds );
        }

        // update the "dirty" flags
        this._mouseBoundsDirty = false;
        this._mouseBoundsHadListener = hasListenerEquivalentSelfOrInAncestor;
      }
    },

    /*
     * Updates the touchBounds for the Node. It will include only the specific bounded areas that are relevant for hit-testing
     * touch events. Thus it:
     * - includes touchAreas (normal bounds don't)
     * - does not include subtrees that would be pruned in hit-testing
     */
    validateTouchBounds: function( hasListenerEquivalentSelfOrInAncestor ) {
      var that = this;

      // we'll need an updated value for this before deciding whether or not to bail
      hasListenerEquivalentSelfOrInAncestor = hasListenerEquivalentSelfOrInAncestor || this.hasInputListenerEquivalent();

      // Touch bounds should be valid still if they aren't marked as dirty AND if the "had listener" matches.
      // Thus, even if the touch bounds aren't marked as dirty, we can still force a refresh (for example: an input listener was added to an ancestor)
      if ( this._touchBoundsDirty || this._touchBoundsHadListener !== hasListenerEquivalentSelfOrInAncestor ) {
        // update whether we have a listener equivalent, so we can prune properly

        if ( this.isSubtreePickablePruned( hasListenerEquivalentSelfOrInAncestor ) ) {
          // if this subtree would be pruned, set the touch bounds to nothing, and bail (skips the entire subtree, since it would never be hit-tested)
          this._touchBounds.set( Bounds2.NOTHING );
        }
        else {
          // start with the self bounds, then add from there
          this._touchBounds.set( this._selfBounds );

          // union of all children's touch bounds
          var i = this._children.length;
          while ( i-- ) {
            var child = this._children[ i ];

            // make sure the child's touchBounds are up to date
            child.validateTouchBounds( hasListenerEquivalentSelfOrInAncestor );
            that._touchBounds.includeBounds( child._touchBounds );
          }

          // do this before the transformation to the parent coordinate frame (the touchArea is in the local coordinate frame)
          if ( this._touchArea ) {
            // we accept either Bounds2, or a Shape (in which case, we take the Shape's bounds)
            this._touchBounds.includeBounds( this._touchArea.isBounds ? this._touchArea : this._touchArea.bounds );
          }

          if ( this.hasClipArea() ) {
            // exclude areas outside of the clipping area's bounds (for efficiency)
            this._touchBounds.constrainBounds( this._clipArea.bounds );
          }

          // transform it to the parent coordinate frame
          this.transformBoundsFromLocalToParent( this._touchBounds );
        }

        // update the "dirty" flags
        this._touchBoundsDirty = false;
        this._touchBoundsHadListener = hasListenerEquivalentSelfOrInAncestor;
      }
    },

    // mark the bounds of this node as invalid, so it is recomputed before it is accessed again
    invalidateBounds: function() {
      // TODO: sometimes we won't need to invalidate local bounds! it's not too much of a hassle though?
      this._boundsDirty = true;
      this._localBoundsDirty = true;
      this._mouseBoundsDirty = true;
      this._touchBoundsDirty = true;

      // and set flags for all ancestors
      var i = this._parents.length;
      while ( i-- ) {
        this._parents[ i ].invalidateChildBounds();
      }

      // TODO: consider calling invalidateMouseTouchBounds from here? it would mean two traversals, but it may bail out sooner. Hard call.
    },

    // recursively tag all ancestors with _childBoundsDirty
    invalidateChildBounds: function() {
      // don't bother updating if we've already been tagged
      if ( !this._childBoundsDirty || !this._mouseBoundsDirty || !this._touchBoundsDirty ) {
        this._childBoundsDirty = true;
        this._localBoundsDirty = true;
        this._mouseBoundsDirty = true;
        this._touchBoundsDirty = true;
        var i = this._parents.length;
        while ( i-- ) {
          this._parents[ i ].invalidateChildBounds();
        }
      }
    },

    // mark mouse/touch bounds as invalid (can occur from normal bounds invalidation, or from anything that could change pickability)
    // NOTE: we don't have to touch descendants because we also store the last used "under effective listener" value, so "non-dirty"
    // subtrees will still be investigated (or freshly pruned) if the listener status has changed
    invalidateMouseTouchBounds: function() {
      if ( !this._mouseBoundsDirty || !this._touchBoundsDirty ) {
        this._mouseBoundsDirty = true;
        this._touchBoundsDirty = true;
        var i = this._parents.length;
        while ( i-- ) {
          this._parents[ i ].invalidateMouseTouchBounds();
        }
      }
    },

    // called to notify that self rendering will display different paint, with possibly different bounds
    invalidateSelf: function( newBounds ) {
      assert && assert( newBounds.isEmpty() || newBounds.isFinite(), 'Bounds must be empty or finite in invalidateSelf' );

      // if these bounds are different than current self bounds
      if ( !this._selfBounds.equals( newBounds ) ) {
        // set repaint flags
        this._localBoundsDirty = true;
        this.invalidateBounds();

        // record the new bounds
        this._selfBounds = newBounds;

        // fire the event immediately
        this.trigger0( 'selfBounds' );
      }
    },

    isChild: function( potentialChild ) {
      assert && assert( potentialChild && ( potentialChild instanceof Node ), 'isChild needs to be called with a Node' );
      var ourChild = _.contains( this._children, potentialChild );
      var itsParent = _.contains( potentialChild._parents, this );
      assert && assert( ourChild === itsParent );
      return ourChild;
    },

    // the bounds for self content in "local" coordinates.
    getSelfBounds: function() {
      return this._selfBounds;
    },

    // returns a bounding box that should contain all self content in the local coordinate frame (our normal self bounds aren't guaranteed this for Text, etc.)
    getSafeSelfBounds: function() {
      // override this to provide different behavior
      return this._selfBounds;
    },

    getChildBounds: function() {
      this.validateBounds();
      return this._childBounds;
    },

    // local coordinate frame bounds
    getLocalBounds: function() {
      this.validateBounds();
      return this._localBounds;
    },

    // {Bounds2 | null} to override the localBounds. Once this is called, it will always be used for localBounds until this is called again.
    // To revert to having Scenery compute the localBounds, set this to null.
    setLocalBounds: function( localBounds ) {
      assert && assert( localBounds === null || localBounds instanceof Bounds2, 'localBounds override should be set to either null or a Bounds2' );

      if ( localBounds === null ) {
        // we can just ignore this if we weren't actually overriding local bounds before
        if ( this._localBoundsOverridden ) {
          this._localBoundsOverridden = false;
          this.trigger1( 'localBoundsOverride', false );
          this.invalidateBounds();
        }
      }
      else {
        // just an instance check for now. consider equals() in the future depending on cost
        var changed = localBounds !== this._localBounds || !this._localBoundsOverridden;

        if ( changed ) {
          this._localBounds = localBounds;
        }

        if ( !this._localBoundsOverridden ) {
          this._localBoundsOverridden = true; // NOTE: has to be done before invalidating bounds, since this disables localBounds computation
          this.trigger1( 'localBoundsOverride', true );
        }

        if ( changed ) {
          this.invalidateBounds();
        }
      }

      return this; // allow chaining
    },

    // @public, meant to be overridden in sub-types that have more accurate bounds determination (e.g. non-rectangular)
    getTransformedSelfBounds: function( matrix ) {
      // assume that we take up the entire rectangular bounds.
      return this._selfBounds.transformed( matrix );
    },

    // @public, sets whether we will require more accurate (and expensive) bounds computation for this node's transform.
    setTransformBounds: function( transformBounds ) {
      assert && assert( typeof transformBounds === 'boolean', 'transformBounds should be boolean' );

      if ( this._transformBounds !== transformBounds ) {
        this._transformBounds = transformBounds;

        this.invalidateBounds();
      }
    },

    // getter for whether we will transform bounds with rotations and shears when computing bounds
    getTransformBounds: function() {
      return this._transformBounds;
    },

    // the bounds for content in render(), in "parent" coordinates
    getBounds: function() {
      this.validateBounds();
      return this._bounds;
    },

    // like getBounds() in the "parent" coordinate frame, but includes only visible nodes
    getVisibleBounds: function() {
      // defensive copy, since we use mutable modifications below
      var bounds = this._selfBounds.copy();

      var i = this._children.length;
      while ( i-- ) {
        var child = this._children[ i ];
        if ( child.isVisible() ) {
          bounds.includeBounds( child.getVisibleBounds() );
        }
      }

      assert && assert( bounds.isFinite() || bounds.isEmpty(), 'Visible bounds should not be infinite' );
      return this.localToParentBounds( bounds );
    },

    // whether this node effectively behaves as if it has an input listener
    hasInputListenerEquivalent: function() {
      // NOTE: if anything here is added, update when invalidateMouseTouchBounds gets called (since changes to pickability pruning affect mouse/touch bounds)
      return this._inputListeners.length > 0 || this._pickable === true;
    },

    // whether hit-testing for events should be pruned at this node (not even considering this node's self).
    // hasListenerEquivalentSelfOrInAncestor indicates whether this node (or an ancestor) either has input listeners, or has pickable set to true (not the default)
    isSubtreePickablePruned: function( hasListenerEquivalentSelfOrInAncestor ) {
      // NOTE: if anything here is added, update when invalidateMouseTouchBounds gets called (since changes to pickability pruning affect mouse/touch bounds)
      // if invisible: skip it
      // if pickable: false, skip it
      // if pickable: undefined and our pickable count indicates there are no input listeners / pickable: true in our subtree, skip it
      return !this.isVisible() || this._pickable === false || ( this._pickable !== true && !hasListenerEquivalentSelfOrInAncestor && this._subtreePickableCount === 0 );
    },

    trailUnderPointer: function( pointer ) {
      // grab our global reference. this isn't re-entrant, and we don't want to cause allocations here
      var options = trailUnderPointerOptions;

      options.isMouse = !!pointer.isMouse;
      options.isTouch = !!pointer.isTouch;
      options.isPen = !!pointer.isPen;

      return this.trailUnderPoint( pointer.point, options );
    },

    /*
     * Return a trail to the top node (if any, otherwise null) whose self-rendered area contains the
     * point (in parent coordinates).
     *
     * For now, prune anything that is invisible or effectively unpickable
     *
     * When calling, don't pass the recursive flag. It signals that the point passed can be mutated
     */
    trailUnderPoint: function( point, options, recursive, hasListenerEquivalentSelfOrInAncestor ) {
      assert && assert( point, 'trailUnderPointer requires a point' );

      hasListenerEquivalentSelfOrInAncestor = hasListenerEquivalentSelfOrInAncestor || this.hasInputListenerEquivalent();

      // prune if possible (usually invisible, pickable:false, no input listeners that would be triggered by this node or anything under it, etc.)
      if ( this.isSubtreePickablePruned( hasListenerEquivalentSelfOrInAncestor ) ) {
        sceneryLog && sceneryLog.hitTest && sceneryLog.hitTest( this.constructor.name + '#' + this.id + ' isSubtreePickablePruned(' + hasListenerEquivalentSelfOrInAncestor + ')' );
        return null;
      }

      // TODO: consider changing the trailUnderPoint API so that these are fixed (and add an option to override trailUnderPoint handling, like in BAM)
      var useMouseAreas = options && options.isMouse;
      var useTouchAreas = options && ( options.isTouch || options.isPen );

      var pruningBounds;
      // only validate the needed type of bounds. definitely don't do a full 'validateBounds' when testing mouse/touch, since there are large pruned areas,
      // and we want to avoid computing bounds where not needed (think something that is animated and expensive to compute)
      if ( useMouseAreas ) {
        !recursive && this.validateMouseBounds( false ); // update mouse bounds for pruning if we aren't being called from trailUnderPoint (ourself)
        pruningBounds = this._mouseBounds;
      }
      else if ( useTouchAreas ) {
        !recursive && this.validateTouchBounds( false ); // update touch bounds for pruning if we aren't being called from trailUnderPoint (ourself)
        pruningBounds = this._touchBounds;
      }
      else {
        !recursive && this.validateBounds(); // update general bounds for pruning if we aren't being called from trailUnderPoint (ourself)
        pruningBounds = this._bounds;
      }

      // bail quickly if this doesn't hit our computed bounds
      if ( !pruningBounds.containsPoint( point ) ) {
        sceneryLog && sceneryLog.hitTest && sceneryLog.hitTest( this.constructor.name + '#' + this.id + ' pruned: ' + ( useMouseAreas ? 'mouse' : ( useTouchAreas ? 'touch' : 'regular' ) ) );
        return null; // not in our bounds, so this point can't possibly be contained
      }

      // temporary result variable, since it's easier to do this way to free the computed point
      var result = null;

      // point in the local coordinate frame. computed after the main bounds check, so we can bail out there efficiently
      var localPoint = this._transform.getInverse().multiplyVector2( Vector2.createFromPool( point.x, point.y ) );
      // var localPoint = this.parentToLocalPoint( point );

      // if our point is outside of the local-coordinate clipping area, we shouldn't return a hit
      if ( this.hasClipArea() && !this._clipArea.containsPoint( localPoint ) ) {
        sceneryLog && sceneryLog.hitTest && sceneryLog.hitTest( this.constructor.name + '#' + this.id + ' out of clip area' );
        return null;
      }

      sceneryLog && sceneryLog.hitTest && sceneryLog.hitTest( this.constructor.name + '#' + this.id );

      // check children first, since they are rendered later. don't bother checking childBounds, we usually are using mouse/touch.
      // manual iteration here so we can return directly, and so we can iterate backwards (last node is in front)
      for ( var i = this._children.length - 1; i >= 0; i-- ) {
        var child = this._children[ i ];

        sceneryLog && sceneryLog.hitTest && sceneryLog.push();
        var childHit = child.trailUnderPoint( localPoint, options, true, hasListenerEquivalentSelfOrInAncestor );
        sceneryLog && sceneryLog.hitTest && sceneryLog.pop();

        // the child will have the point in its parent's coordinate frame (i.e. this node's frame)
        if ( childHit ) {
          childHit.addAncestor( this, i );
          localPoint.freeToPool();
          return childHit;
        }
      }

      // tests for mouse and touch hit areas before testing containsPointSelf
      if ( useMouseAreas && this._mouseArea ) {
        // NOTE: both Bounds2 and Shape have containsPoint! We use both here!
        result = this._mouseArea.containsPoint( localPoint ) ? new scenery.Trail( this ) : null;
        localPoint.freeToPool();
        sceneryLog && sceneryLog.hitTest && sceneryLog.hitTest( this.constructor.name + '#' + this.id + ' mouse area hit' );
        return result;
      }
      if ( useTouchAreas && this._touchArea ) {
        // NOTE: both Bounds2 and Shape have containsPoint! We use both here!
        result = this._touchArea.containsPoint( localPoint ) ? new scenery.Trail( this ) : null;
        localPoint.freeToPool();
        sceneryLog && sceneryLog.hitTest && sceneryLog.hitTest( this.constructor.name + '#' + this.id + ' touch area hit' );
        return result;
      }

      // didn't hit our children, so check ourself as a last resort. check our selfBounds first, to avoid a potentially more expensive operation
      if ( this._selfBounds.containsPoint( localPoint ) ) {
        if ( this.containsPointSelf( localPoint ) ) {
          localPoint.freeToPool();
          sceneryLog && sceneryLog.hitTest && sceneryLog.hitTest( this.constructor.name + '#' + this.id + ' self hit' );
          return new scenery.Trail( this );
        }
      }

      // signal no hit
      localPoint.freeToPool();
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
      var length = this._children.length;
      for ( var i = 0; i < length; i++ ) {
        this._children[ i ].walkDepthFirst( callback );
      }
    },

    getChildrenWithinBounds: function( bounds ) {
      var result = [];
      var length = this._children.length;
      for ( var i = 0; i < length; i++ ) {
        var child = this._children[ i ];
        if ( !child._bounds.intersection( bounds ).isEmpty() ) {
          result.push( child );
        }
      }
      return result;
    },

    // TODO: set this up with a mix-in for a generic notifier?
    addInputListener: function( listener ) {
      // don't allow listeners to be added multiple times
      if ( _.indexOf( this._inputListeners, listener ) === -1 ) {
        this._inputListeners.push( listener );
        this.changePickableCount( 1 ); // NOTE: this should also trigger invalidation of mouse/touch bounds
      }
      return this;
    },

    removeInputListener: function( listener ) {
      // ensure the listener is in our list
      assert && assert( _.indexOf( this._inputListeners, listener ) !== -1 );

      this._inputListeners.splice( _.indexOf( this._inputListeners, listener ), 1 );
      this.changePickableCount( -1 ); // NOTE: this should also trigger invalidation of mouse/touch bounds
      return this;
    },

    getInputListeners: function() {
      return this._inputListeners.slice( 0 ); // defensive copy
    },

    // TODO: consider renaming to translateBy to match scaleBy
    translate: function( x, y, prependInstead ) {
      if ( typeof x === 'number' ) {
        // translate( x, y, prependInstead )
        if ( !x && !y ) { return; } // bail out if both are zero
        if ( prependInstead ) {
          this.prependTranslation( x, y );
        }
        else {
          this.appendMatrix( Matrix3.translation( x, y ) );
        }
      }
      else {
        // translate( vector, prependInstead )
        var vector = x;
        if ( !vector.x && !vector.y ) { return; } // bail out if both are zero
        this.translate( vector.x, vector.y, y ); // forward to full version
      }
    },

    // scale( s ) is also supported, which will scale both dimensions by the same amount. renamed from 'scale' to satisfy the setter/getter
    scale: function( x, y, prependInstead ) {
      if ( typeof x === 'number' ) {
        if ( y === undefined ) {
          // scale( scale )
          if ( x === 1 ) { return; } // bail out if we are scaling by 1 (identity)
          this.appendMatrix( Matrix3.scaling( x, x ) );
        }
        else {
          // scale( x, y, prependInstead )
          if ( x === 1 && y === 1 ) { return; } // bail out if we are scaling by 1 (identity)
          if ( prependInstead ) {
            this.prependMatrix( Matrix3.scaling( x, y ) );
          }
          else {
            this.appendMatrix( Matrix3.scaling( x, y ) );
          }
        }
      }
      else {
        // scale( vector, prependInstead ) or scale( { x: x, y: y }, prependInstead )
        var vector = x;
        this.scale( vector.x, vector.y, y ); // forward to full version
      }
    },

    // TODO: consider naming to rotateBy to match scaleBy (due to scale property / method name conflict)
    rotate: function( angle, prependInstead ) {
      if ( angle % ( 2 * Math.PI ) === 0 ) { return; } // bail out if our angle is effectively 0
      if ( prependInstead ) {
        this.prependMatrix( Matrix3.rotation2( angle ) );
      }
      else {
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
      return this._transform.getMatrix().m02();
    },

    setX: function( x ) {
      assert && assert( typeof x === 'number' );

      this.translate( x - this.getX(), 0, true );
      return this;
    },

    getY: function() {
      return this._transform.getMatrix().m12();
    },

    setY: function( y ) {
      assert && assert( typeof y === 'number' );

      this.translate( 0, y - this.getY(), true );
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
      }
      else {
        // setScaleMagnitude( vector ), where we set the x-scale to vector.x and y-scale to vector.y
        this.appendMatrix( Matrix3.scaling( a.x / currentScale.x, a.y / currentScale.y ) );
      }
      return this;
    },

    getRotation: function() {
      return this._transform.getMatrix().getRotation();
    },

    setRotation: function( rotation ) {
      assert && assert( typeof rotation === 'number' );

      this.appendMatrix( Matrix3.rotation2( rotation - this.getRotation() ) );
      return this;
    },

    // supports setTranslation( x, y ) or setTranslation( new Vector2( x, y ) ) .. or technically setTranslation( { x: x, y: y } )
    setTranslation: function( a, b ) {
      var m = this._transform.getMatrix();
      var tx = m.m02();
      var ty = m.m12();

      var dx, dy;

      if ( typeof a === 'number' ) {
        dx = a - tx;
        dy = b - ty;
      }
      else {
        dx = a.x - tx;
        dy = a.y - ty;
      }

      this.translate( dx, dy, true );

      return this;
    },

    getTranslation: function() {
      var matrix = this._transform.getMatrix();
      return new Vector2( matrix.m02(), matrix.m12() );
    },

    // append a transformation matrix to our local transform
    appendMatrix: function( matrix ) {
      this._transform.append( matrix );
    },

    // prepend a transformation matrix to our local transform
    prependMatrix: function( matrix ) {
      this._transform.prepend( matrix );
    },

    // prepend an x,y translation to our local transform without allocating a matrix for it, see #119
    prependTranslation: function( x, y ) {
      this._transform.prependTranslation( x, y );
    },

    setMatrix: function( matrix ) {
      this._transform.setMatrix( matrix );
    },

    getMatrix: function() {
      return this._transform.getMatrix();
    },

    // change the actual transform reference (not just the actual transform)
    setTransform: function( transform ) {
      assert && assert( transform.isFinite(), 'Transform should not have infinite/NaN values' );

      if ( this._transform !== transform ) {
        // since our referenced transform doesn't change, we need to trigger the before/after ourselves
        this.beforeTransformChange();

        // swap the transform and move the listener to the new one
        this._transform.removeTransformListener( this._transformListener ); // don't leak memory!
        this._transform = transform;
        this._transform.prependTransformListener( this._transformListener );

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
      //OHTWO TODO: @deprecated, remove
    },

    // called after our transform is changed
    afterTransformChange: function() {
      // NOTE: why is local bounds invalidation needed here?
      this.invalidateBounds();

      this.trigger0( 'transform' );
    },

    // the left bound of this node, in the parent coordinate frame
    getLeft: function() {
      return this.getBounds().minX;
    },

    // shifts this node horizontally so that its left bound (in the parent coordinate frame) is 'left'
    setLeft: function( left ) {
      assert && assert( typeof left === 'number' );

      this.translate( left - this.getLeft(), 0, true );
      return this; // allow chaining
    },

    // the right bound of this node, in the parent coordinate frame
    getRight: function() {
      return this.getBounds().maxX;
    },

    // shifts this node horizontally so that its right bound (in the parent coordinate frame) is 'right'
    setRight: function( right ) {
      assert && assert( typeof right === 'number' );

      this.translate( right - this.getRight(), 0, true );
      return this; // allow chaining
    },

    getCenterX: function() {
      return this.getBounds().getCenterX();
    },

    setCenterX: function( x ) {
      assert && assert( typeof x === 'number' );

      this.translate( x - this.getCenterX(), 0, true );
      return this; // allow chaining
    },

    getCenterY: function() {
      return this.getBounds().getCenterY();
    },

    setCenterY: function( y ) {
      assert && assert( typeof y === 'number' );

      this.translate( 0, y - this.getCenterY(), true );
      return this; // allow chaining
    },

    // the top bound of this node, in the parent coordinate frame
    getTop: function() {
      return this.getBounds().minY;
    },

    // shifts this node vertically so that its top bound (in the parent coordinate frame) is 'top'
    setTop: function( top ) {
      assert && assert( typeof top === 'number' );

      this.translate( 0, top - this.getTop(), true );
      return this; // allow chaining
    },

    // the bottom bound of this node, in the parent coordinate frame
    getBottom: function() {
      return this.getBounds().maxY;
    },

    // shifts this node vertically so that its bottom bound (in the parent coordinate frame) is 'bottom'
    setBottom: function( bottom ) {
      assert && assert( typeof bottom === 'number' );

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
      assert && assert( typeof visible === 'boolean' );

      if ( visible !== this._visible ) {
        // changing visibility can affect pickability pruning, which affects mouse/touch bounds
        this.invalidateMouseTouchBounds();

        this._visible = visible;

        this.trigger0( 'visibility' );
      }
      return this;
    },

    getOpacity: function() {
      return this._opacity;
    },

    setOpacity: function( opacity ) {
      assert && assert( typeof opacity === 'number' );

      var clampedOpacity = clamp( opacity, 0, 1 );
      if ( clampedOpacity !== this._opacity ) {
        this._opacity = clampedOpacity;

        this.trigger0( 'opacity' );
      }
    },

    isPickable: function() {
      return this._pickable;
    },

    setPickable: function( pickable ) {
      assert && assert( pickable === null || typeof pickable === 'boolean' );

      if ( this._pickable !== pickable ) {
        var n = this._pickable === true ? -1 : 0;

        // no paint or invalidation changes for now, since this is only handled for the mouse
        this._pickable = pickable;
        n += this._pickable === true ? 1 : 0;

        if ( n ) {
          this.changePickableCount( n ); // should invalidate mouse/touch bounds, since it changes the pickability
        }

        // TODO: invalidate the cursor somehow? #150
      }
    },

    setCursor: function( cursor ) {
      assert && assert( typeof cursor === 'string' || cursor === null );

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

    setMouseArea: function( area ) {
      assert && assert( area === null || area instanceof Shape || area instanceof Bounds2, 'mouseArea needs to be a kite.Shape, dot.Bounds2, or null' );

      if ( this._mouseArea !== area ) {
        this._mouseArea = area; // TODO: could change what is under the mouse, invalidate!

        this.invalidateMouseTouchBounds();
      }
    },

    getMouseArea: function() {
      return this._mouseArea;
    },

    setTouchArea: function( area ) {
      assert && assert( area === null || area instanceof Shape || area instanceof Bounds2, 'touchArea needs to be a kite.Shape, dot.Bounds2, or null' );

      if ( this._touchArea !== area ) {
        this._touchArea = area; // TODO: could change what is under the touch, invalidate!

        this.invalidateMouseTouchBounds();
      }
    },

    getTouchArea: function() {
      return this._touchArea;
    },

    setClipArea: function( shape ) {
      assert && assert( shape === null || shape instanceof Shape, 'clipArea needs to be a kite.Shape, or null' );

      if ( this._clipArea !== shape ) {
        this._clipArea = shape;

        this.trigger0( 'clip' );

        this.invalidateBounds();
      }
    },

    getClipArea: function() {
      return this._clipArea;
    },

    hasClipArea: function() {
      return this._clipArea !== null;
    },

    supportsCanvas: function() {
      return ( this._rendererBitmask & scenery.bitmaskSupportsCanvas ) !== 0;
    },

    supportsSVG: function() {
      return ( this._rendererBitmask & scenery.bitmaskSupportsSVG ) !== 0;
    },

    supportsDOM: function() {
      return ( this._rendererBitmask & scenery.bitmaskSupportsDOM ) !== 0;
    },

    supportsWebGL: function() {
      return ( this._rendererBitmask & scenery.bitmaskSupportsWebGL ) !== 0;
    },

    supportsPixi: function() {
      return ( this._rendererBitmask & scenery.bitmaskSupportsPixi) !== 0;
    },

    supportsRenderer: function( renderer ) {
      return ( this._rendererBitmask & renderer.bitmask ) !== 0;
    },

    // return a supported renderer (fallback case, not called often)
    pickARenderer: function() {
      if ( this.supportsCanvas() ) {
        return scenery.Renderer.Canvas;
      }
      else if ( this.supportsSVG() ) {
        return scenery.Renderer.SVG;
      }
      else if ( this.supportsDOM() ) {
        return scenery.Renderer.DOM;
      }
      // oi!
    },

    setRendererBitmask: function( bitmask ) {
      if ( bitmask !== this._rendererBitmask ) {
        this._rendererSummary.bitmaskChange( this._rendererBitmask, bitmask );
        this._rendererBitmask = bitmask;
        this.trigger0( 'rendererBitmask' );
      }
    },

    // meant to be overridden
    invalidateSupportedRenderers: function() {

    },

    /*---------------------------------------------------------------------------*
     * Hints
     *----------------------------------------------------------------------------*/

    // provides a rendering hint to use this render whenever possible
    setRenderer: function( renderer ) {
      assert && assert( renderer === null || renderer === 'canvas' || renderer === 'svg' || renderer === 'dom' || renderer === 'webgl' || renderer === 'pixi',
        'Renderer input should be null, or one of: "canvas", "svg", "dom", "webgl" or "pixi".' );

      var newRenderer = 0;
      if ( renderer === 'canvas' ) {
        newRenderer = scenery.Renderer.bitmaskCanvas;
      }
      else if ( renderer === 'svg' ) {
        newRenderer = scenery.Renderer.bitmaskSVG;
      }
      else if ( renderer === 'dom' ) {
        newRenderer = scenery.Renderer.bitmaskDOM;
      }
      else if ( renderer === 'webgl' ) {
        newRenderer = scenery.Renderer.bitmaskWebGL;
      }
      else if ( renderer === 'pixi' ) {
        newRenderer = scenery.Renderer.bitmaskPixi;
      }
      assert && assert( ( renderer === null ) === ( newRenderer === 0 ),
        'We should only end up with no actual renderer if renderer is null' );

      if ( this._hints.renderer !== newRenderer ) {
        this._hints.renderer = newRenderer;

        this.trigger1( 'hint', 'renderer' );
      }
    },

    getRenderer: function() {
      if ( this._hints.renderer === 0 ) {
        return null;
      }
      else if ( this._hints.renderer === scenery.Renderer.bitmaskCanvas ) {
        return 'canvas';
      }
      else if ( this._hints.renderer === scenery.Renderer.bitmaskSVG ) {
        return 'svg';
      }
      else if ( this._hints.renderer === scenery.Renderer.bitmaskDOM ) {
        return 'dom';
      }
      else if ( this._hints.renderer === scenery.Renderer.bitmaskWebGL ) {
        return 'webgl';
      }
      else if ( this._hints.renderer === scenery.Renderer.bitmaskPixi ) {
        return 'pixi';
      }
      assert && assert( false, 'Seems to be an invalid renderer?' );
      return this._hints.renderer;
    },

    hasRenderer: function() {
      return !!this._hints.renderer;
    },

    setRendererOptions: function( options ) {
      // TODO: consider checking options based on the specified 'renderer'?
      // TODO: consider a guard where we check if anything changed
      //OHTWO TODO: Split out all of the renderer options into individual flag ES5'ed getter/setters
      _.extend( this._hints, options );

      this.trigger0( 'hint' );
    },

    getRendererOptions: function() {
      return this._hints;
    },

    hasRendererOptions: function() {
      return !!( this._hints.cssTransform || this._hints.fullResolution );
    },

    setLayerSplit: function( split ) {
      assert && assert( typeof split === 'boolean' );

      if ( split !== this._hints.layerSplit ) {
        this._hints.layerSplit = split;
        this.trigger1( 'hint', 'layerSplit' );
      }
    },

    isLayerSplit: function() {
      return this._hints.layerSplit;
    },

    setUsesOpacity: function( usesOpacity ) {
      assert && assert( typeof usesOpacity === 'boolean' );

      if ( usesOpacity !== this._hints.usesOpacity ) {
        this._hints.usesOpacity = usesOpacity;
        this.trigger1( 'hint', 'usesOpacity' );
      }
    },

    getUsesOpacity: function() {
      return this._hints.usesOpacity;
    },

    /*---------------------------------------------------------------------------*
     * Trail operations
     *----------------------------------------------------------------------------*/

    /**
     * @returns {Trail} - Returns the one Trail that starts from a node with no parents (or if the predicate is present,
     *                    a node that satisfies it), and ends at this node. If more than one Trail would satisfy these
     *                    conditions, an assertion is thrown (please use getTrails() for those cases).
     * @param {function( node ) : boolean} [predicate] - If supplied, we will only return
     */
    getUniqueTrail: function( predicate ) {

      // Without a predicate, we'll be able to bail out the instant we hit a node with 2+ parents, and it makes the
      // logic easier.
      if ( !predicate ) {
        var trail = new scenery.Trail();
        var node = this;

        while ( node ) {
          assert && assert( node._parents.length <= 1,
            'getUniqueTrail found a node with ' + node._parents.length + ' parents.' );

          trail.addAncestor( node );
          node = node._parents[ 0 ]; // should be undefined if there aren't any parents
        }

        return trail;
      }
      // With a predicate, we need to explore multiple parents (since the predicate may filter out all but one)
      else {
        var trails = this.getTrails( predicate );

        assert && assert( trails.length === 1,
          'getUniqueTrail found ' + trails.length + ' matching trails for the predicate' );

        return trails[0];
      }
    },

    /**
     * @returns {Trail[]} - An array of all Trails that start from nodes with no parent (or if a predicate is present,
     *                      those that satisfy the predicate), and ends at this node.
     * @param {function( node ) : boolean} [predicate] - If supplied, we will only return Trails rooted at nodes that
     *                                                   satisfy predicate( node ) == true.
     */
    getTrails: function( predicate ) {
      predicate = predicate || defaultTrailPredicate;

      var trails = [];
      var trail = new scenery.Trail( this );
      scenery.Trail.appendAncestorTrailsWithPredicate( trails, trail, predicate );

      return trails;
    },

    // all nodes in the connected component, returned in an arbitrary order
    getConnectedNodes: function() {
      var result = [];
      var fresh = this._children.concat( this._parents ).concat( this );
      while ( fresh.length ) {
        var node = fresh.pop();
        if ( !_.contains( result, node ) ) {
          result.push( node );
          fresh = fresh.concat( node._children, node._parents );
        }
      }
      return result;
    },

    getTopologicallySortedNodes: function() {
      // see http://en.wikipedia.org/wiki/Topological_sorting
      var edges = {};
      var s = [];
      var l = [];
      var n;
      _.each( this.getConnectedNodes(), function( node ) {
        edges[ node.id ] = {};
        _.each( node.children, function( m ) {
          edges[ node.id ][ m.id ] = true;
        } );
        if ( !node.parents.length ) {
          s.push( node );
        }
      } );
      function handleChild( m ) {
        delete edges[ n.id ][ m.id ];
        if ( _.every( edges, function( children ) { return !children[ m.id ]; } ) ) {
          // there are no more edges to m
          s.push( m );
        }
      }

      while ( s.length ) {
        n = s.pop();
        l.push( n );

        _.each( n.children, handleChild );
      }

      // ensure that there are no edges left, since then it would contain a circular reference
      assert && assert( _.every( edges, function( children ) {
        return _.every( children, function( final ) { return false; } );
      } ), 'circular reference check' );

      return l;
    },

    // verify that this.addChild( child ) it wouldn't cause circular references
    canAddChild: function( child ) {
      if ( this === child || _.contains( this.children, child ) ) {
        return false;
      }

      // see http://en.wikipedia.org/wiki/Topological_sorting
      // TODO: remove duplication with above handling?
      var edges = {};
      var s = [];
      var l = [];
      var n;
      _.each( this.getConnectedNodes().concat( child.getConnectedNodes() ), function( node ) {
        edges[ node.id ] = {};
        _.each( node.children, function( m ) {
          edges[ node.id ][ m.id ] = true;
        } );
        if ( !node.parents.length && node !== child ) {
          s.push( node );
        }
      } );
      edges[ this.id ][ child.id ] = true; // add in our 'new' edge
      function handleChild( m ) {
        delete edges[ n.id ][ m.id ];
        if ( _.every( edges, function( children ) { return !children[ m.id ]; } ) ) {
          // there are no more edges to m
          s.push( m );
        }
      }

      while ( s.length ) {
        n = s.pop();
        l.push( n );

        _.each( n.children, handleChild );

        // handle our new edge
        if ( n === this ) {
          handleChild( child );
        }
      }

      // ensure that there are no edges left, since then it would contain a circular reference
      return _.every( edges, function( children ) {
        return _.every( children, function( final ) { return false; } );
      } );
    },

    // @private
    canvasPaintSelf: function( wrapper ) {
      // To be overridden in paintable node types. Should hook into the drawable's prototype (presumably).
    },

    // @public: Render the self into the canvas wrapper with its local coordinates
    renderToCanvasSelf: function( wrapper ) {
      if ( this.isPainted() && ( this._rendererBitmask & scenery.Renderer.bitmaskCanvas ) ) {
        this.canvasPaintSelf( wrapper );
      }
    },

    // @public: Render this node and its descendants to the Canvas wrapper. matrix is optional
    renderToCanvasSubtree: function( wrapper, matrix ) {
      matrix = matrix || Matrix3.identity();

      wrapper.resetStyles();

      this.renderToCanvasSelf( wrapper );
      for ( var i = 0; i < this._children.length; i++ ) {
        var child = this._children[ i ];

        if ( child.isVisible() ) {
          var requiresScratchCanvas = child._opacity !== 1 || child._clipArea;

          wrapper.context.save();
          matrix.multiplyMatrix( child._transform.getMatrix() );
          matrix.canvasSetTransform( wrapper.context );
          if ( requiresScratchCanvas ) {
            var canvas = document.createElement( 'canvas' );
            canvas.width = wrapper.canvas.width;
            canvas.height = wrapper.canvas.height;
            var context = canvas.getContext( '2d' );
            var childWrapper = new scenery.CanvasContextWrapper( canvas, context );

            matrix.canvasSetTransform( context );

            child.renderToCanvasSubtree( childWrapper, matrix );

            wrapper.context.save();
            if ( child._clipArea ) {
              wrapper.context.beginPath();
              child._clipArea.writeToContext( wrapper.context );
              wrapper.context.clip();
            }
            wrapper.context.setTransform( 1, 0, 0, 1, 0, 0 ); // identity
            wrapper.context.globalAlpha = child._opacity;
            wrapper.context.drawImage( canvas, 0, 0 );
            wrapper.context.restore();
          }
          else {
            child.renderToCanvasSubtree( wrapper, matrix );
          }
          matrix.multiplyMatrix( child._transform.getInverse() );
          wrapper.context.restore();
        }
      }
    },

    // @public @deprecated (API compatibility for now): Render this node to the Canvas (clearing it first)
    renderToCanvas: function( canvas, context, callback, optionalBackgroundColor ) {
      // should basically reset everything (and clear the Canvas)
      canvas.width = canvas.width;

      if ( optionalBackgroundColor ) {
        context.fillStyle = optionalBackgroundColor;
        context.fillRect( 0, 0, canvas.width, canvas.height );
      }

      var wrapper = new scenery.CanvasContextWrapper( canvas, context );

      this.renderToCanvasSubtree( wrapper, Matrix3.identity() );

      callback && callback(); // this was originally asynchronous, so we had a callback
    },

    /*
     * Renders this node to a canvas. If toCanvas( callback ) is used, the canvas will contain the node's
     * entire bounds.
     *
     * callback( canvas, x, y ) is called, where x and y offsets are computed if not specified.
     */
    toCanvas: function( callback, x, y, width, height ) {
      var padding = 2; // padding used if x and y are not set

      // for now, we add an unpleasant hack around Text and safe bounds in general. We don't want to add another Bounds2 object per Node for now.
      var bounds = this.getBounds().union( this.localToParentBounds( this.getSafeSelfBounds() ) );

      x = x !== undefined ? x : Math.ceil( padding - bounds.minX );
      y = y !== undefined ? y : Math.ceil( padding - bounds.minY );
      width = width !== undefined ? width : Math.ceil( bounds.getWidth() + 2 * padding );
      height = height !== undefined ? height : Math.ceil( bounds.getHeight() + 2 * padding );

      var canvas = document.createElement( 'canvas' );
      canvas.width = width;
      canvas.height = height;
      var context = canvas.getContext( '2d' );

      // shift our rendering over by the desired amount
      context.translate( x, y );

      // for API compatibility, we apply our own transform here
      this._transform.getMatrix().canvasAppendTransform( context );

      var wrapper = new scenery.CanvasContextWrapper( canvas, context );

      this.renderToCanvasSubtree( wrapper, Matrix3.translation( x, y ).timesMatrix( this._transform.getMatrix() ) );

      callback( canvas, x, y ); // we used to be asynchronous
    },

    // gives a data URI, with the same parameter handling as Node.toCanvas()
    toDataURL: function( callback, x, y, width, height ) {
      this.toCanvas( function( canvas, x, y ) {
        // this x and y shadow the outside parameters, and will be different if the outside parameters are undefined
        callback( canvas.toDataURL(), x, y );
      }, x, y, width, height );
    },

    // gives an HTMLImageElement with the same parameter handling as Node.toCanvas(). guaranteed to be asynchronous
    toImage: function( callback, x, y, width, height ) {
      this.toDataURL( function( url, x, y ) {
        // this x and y shadow the outside parameters, and will be different if the outside parameters are undefined
        var img = document.createElement( 'img' );
        img.onload = function() {
          callback( img, x, y );
          try {
            delete img.onload;
          }
          catch( e ) {} // fails on Safari 5.1
        };
        img.src = url;
      }, x, y, width, height );
    },

    // will call callback( node )
    toImageNodeAsynchronous: function( callback, x, y, width, height ) {
      this.toImage( function( image, x, y ) {
        callback( new scenery.Node( {
          children: [
            new scenery.Image( image, { x: -x, y: -y } )
          ]
        } ) );
      }, x, y, width, height );
    },

    // fully synchronous, but returns a node that can only be rendered in Canvas
    toCanvasNodeSynchronous: function( x, y, width, height ) {
      var result;
      this.toCanvas( function( canvas, x, y ) {
        result = new scenery.Node( {
          children: [
            new scenery.Image( canvas, { x: -x, y: -y } )
          ]
        } );
      }, x, y, width, height );
      assert && assert( result, 'toCanvasNodeSynchronous requires that the node can be rendered only using Canvas' );
      return result;
    },

    // synchronous, but Image will not have the correct bounds immediately (that will be asynchronous)
    toDataURLNodeSynchronous: function( x, y, width, height ) {
      var result;
      this.toDataURL( function( dataURL, x, y ) {
        result = new scenery.Node( {
          children: [
            new scenery.Image( dataURL, { x: -x, y: -y } )
          ]
        } );
      }, x, y, width, height );
      assert && assert( result, 'toDataURLNodeSynchronous requires that the node can be rendered only using Canvas' );
      return result;
    },

    /*---------------------------------------------------------------------------*
     * Instance handling
     *----------------------------------------------------------------------------*/

    // @private
    getInstances: function() {
      return this._instances;
    },

    // @private
    addInstance: function( instance ) {
      assert && assert( instance instanceof scenery.Instance );
      this._instances.push( instance );
      if ( this._instances.length === 1 ) {
        this.firstInstanceAdded();
      }
    },

    // @private
    removeInstance: function( instance ) {
      assert && assert( instance instanceof scenery.Instance );
      var index = _.indexOf( this._instances, instance );
      assert && assert( index !== -1, 'Cannot remove a Instance from a Node if it was not there' );
      this._instances.splice( index, 1 );
      if ( this._instances.length === 0 ) {
        this.lastInstanceRemoved();
      }
    },

    firstInstanceAdded: function() {
      // no-op, meant to be overridden in the prototype chain by Paintable
    },

    lastInstanceRemoved: function() {
      // no-op, meant to be overridden in the prototype chain by Paintable
    },

    /*---------------------------------------------------------------------------*
     * Coordinate transform methods
     *----------------------------------------------------------------------------*/

    // apply this node's transform to the point
    localToParentPoint: function( point ) {
      return this._transform.transformPosition2( point );
    },

    // apply this node's transform to the bounds
    localToParentBounds: function( bounds ) {
      return this._transform.transformBounds2( bounds );
    },

    // apply the inverse of this node's transform to the point
    parentToLocalPoint: function( point ) {
      return this._transform.inversePosition2( point );
    },

    // apply the inverse of this node's transform to the bounds
    parentToLocalBounds: function( bounds ) {
      return this._transform.inverseBounds2( bounds );
    },

    // mutable optimized form of localToParentBounds
    transformBoundsFromLocalToParent: function( bounds ) {
      return bounds.transform( this._transform.getMatrix() );
    },

    // mutable optimized form of parentToLocalBounds
    transformBoundsFromParentToLocal: function( bounds ) {
      return bounds.transform( this._transform.getInverse() );
    },

    // returns the matrix (fresh copy) that transforms points from the local coordinate frame into the global coordinate frame
    getLocalToGlobalMatrix: function() {
      var node = this;

      // we need to apply the transformations in the reverse order, so we temporarily store them
      var matrices = [];

      // concatenation like this has been faster than getting a unique trail, getting its transform, and applying it
      while ( node ) {
        matrices.push( node._transform.getMatrix() );
        assert && assert( node._parents[ 1 ] === undefined, 'getLocalToGlobalMatrix unable to work for DAG' );
        node = node._parents[ 0 ];
      }

      var matrix = Matrix3.identity(); // will be modified in place

      // iterate from the back forwards (from the root node to here)
      for ( var i = matrices.length - 1; i >= 0; i-- ) {
        matrix.multiplyMatrix( matrices[ i ] );
      }

      // NOTE: always return a fresh copy, getGlobalToLocalMatrix depends on it to minimize instance usage!
      return matrix;
    },

    // equivalent to getUniqueTrail().getTransform(), but faster.
    getUniqueTransform: function() {
      return new Transform3( this.getLocalToGlobalMatrix() );
    },

    // returns the matrix (fresh copy) that transforms points in the global coordinate frame into the local coordinate frame
    getGlobalToLocalMatrix: function() {
      return this.getLocalToGlobalMatrix().invert();
    },

    // apply this node's transform (and then all of its parents' transforms) to the point
    localToGlobalPoint: function( point ) {
      var node = this;
      var resultPoint = point.copy();
      while ( node ) {
        // in-place multiplication
        node._transform.getMatrix().multiplyVector2( resultPoint );
        assert && assert( node._parents[ 1 ] === undefined, 'localToGlobalPoint unable to work for DAG' );
        node = node._parents[ 0 ];
      }
      return resultPoint;
    },

    globalToLocalPoint: function( point ) {
      var node = this;
      // TODO: performance: test whether it is faster to get a total transform and then invert (won't compute individual inverses)

      // we need to apply the transformations in the reverse order, so we temporarily store them
      var transforms = [];
      while ( node ) {
        transforms.push( node._transform );
        assert && assert( node._parents[ 1 ] === undefined, 'globalToLocalPoint unable to work for DAG' );
        node = node._parents[ 0 ];
      }

      // iterate from the back forwards (from the root node to here)
      var resultPoint = point.copy();
      for ( var i = transforms.length - 1; i >= 0; i-- ) {
        // in-place multiplication
        transforms[ i ].getInverse().multiplyVector2( resultPoint );
      }
      return resultPoint;
    },

    // apply this node's transform (and then all of its parents' transforms) to the bounds
    localToGlobalBounds: function( bounds ) {
      // apply the bounds transform only once, so we can minimize the expansion encountered from multiple rotations
      // it also seems to be a bit faster this way
      return bounds.transformed( this.getLocalToGlobalMatrix() );
    },

    globalToLocalBounds: function( bounds ) {
      // apply the bounds transform only once, so we can minimize the expansion encountered from multiple rotations
      return bounds.transformed( this.getGlobalToLocalMatrix() );
    },

    // like localToGlobalPoint, but without applying this node's transform
    parentToGlobalPoint: function( point ) {
      assert && assert( this.parents.length <= 1, 'parentToGlobalPoint unable to work for DAG' );
      return this.parents.length ? this.parents[ 0 ].localToGlobalPoint( point ) : point;
    },

    // like localToGlobalBounds, but without applying this node's transform
    parentToGlobalBounds: function( bounds ) {
      assert && assert( this.parents.length <= 1, 'parentToGlobalBounds unable to work for DAG' );
      return this.parents.length ? this.parents[ 0 ].localToGlobalBounds( bounds ) : bounds;
    },

    globalToParentPoint: function( point ) {
      assert && assert( this.parents.length <= 1, 'globalToParentPoint unable to work for DAG' );
      return this.parents.length ? this.parents[ 0 ].globalToLocalPoint( point ) : point;
    },

    globalToParentBounds: function( bounds ) {
      assert && assert( this.parents.length <= 1, 'globalToParentBounds unable to work for DAG' );
      return this.parents.length ? this.parents[ 0 ].globalToLocalBounds( bounds ) : bounds;
    },

    // get the Bounds2 of this node in the global coordinate frame.  Does not work for DAG.
    getGlobalBounds: function() {
      assert && assert( this.parents.length <= 1, 'globalBounds unable to work for DAG' );
      return this.parentToGlobalBounds( this.getBounds() );
    },

    // get the Bounds2 of any other node by converting to the global coordinate frame.  Does not work for DAG.
    boundsOf: function( node ) {
      return this.globalToLocalBounds( node.getGlobalBounds() );
    },

    // get the Bounds2 of this node in the coordinate frame of the parameter node. Does not work for DAG cases.
    boundsTo: function( node ) {
      return node.globalToLocalBounds( this.getGlobalBounds() );
    },

    /*---------------------------------------------------------------------------*
     * Drawable handling
     *----------------------------------------------------------------------------*/

    // will notify the drawable of visual state changes while it is attached
    attachDrawable: function( drawable ) {
      this._drawables.push( drawable );
      drawable.onAttach( this );
      return this; // allow chaining
    },

    // will not notify the drawable of visual state changes after it is detached
    detachDrawable: function( drawable ) {
      drawable.onDetach( this );
      var index = _.indexOf( this._drawables, drawable );

      assert && assert( index >= 0, 'Invalid operation: trying to detach a non-referenced drawable' );

      this._drawables.splice( index, 1 ); // TODO: replace with a remove() function
      return this;
    },

    /*---------------------------------------------------------------------------*
     * Flags
     *----------------------------------------------------------------------------*/

    // whether for layer fitting we should use "safe" bounds, instead of the bounds used for layout
    requiresSafeBounds: false,

    /*---------------------------------------------------------------------------*
     * ES5 get/set
     *----------------------------------------------------------------------------*/

    set layerSplit( value ) { this.setLayerSplit( value ); },
    get layerSplit() { return this.isLayerSplit(); },

    set renderer( value ) { this.setRenderer( value ); },
    get renderer() { return this.getRenderer(); },

    set rendererOptions( value ) { this.setRendererOptions( value ); },
    get rendererOptions() { return this.getRendererOptions(); },

    set usesOpacity( value ) { this.setUsesOpacity( value ); },
    get usesOpacity() { return this.getUsesOpacity(); },

    set cursor( value ) { this.setCursor( value ); },
    get cursor() { return this.getCursor(); },

    set mouseArea( value ) { this.setMouseArea( value ); },
    get mouseArea() { return this.getMouseArea(); },

    set touchArea( value ) { this.setTouchArea( value ); },
    get touchArea() { return this.getTouchArea(); },

    set clipArea( value ) { this.setClipArea( value ); },
    get clipArea() { return this.getClipArea(); },

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
    get localBounds() { return this.getLocalBounds(); },
    set localBounds( value ) { return this.setLocalBounds( value ); },
    set transformBounds( value ) { return this.setTransformBounds( value ); },
    get transformBounds() { return this.getTransformBounds(); },
    get globalBounds() { return this.getGlobalBounds(); },
    get visibleBounds() { return this.getVisibleBounds(); },
    get id() { return this.getId(); },

    // getInstances is marked as @private.  TODO: Perhaps this should be too?
    get instances() { return this.getInstances(); },

    mutate: function( options ) {
      if ( !options ) {
        return this;
      }

      if ( assert ) {
        assert && assert( _.filter( [ 'translation', 'x', 'left', 'right', 'centerX', 'centerTop', 'rightTop', 'leftCenter', 'center', 'rightCenter', 'leftBottom', 'centerBottom', 'rightBottom' ], function( key ) { return options[ key ] !== undefined; } ).length <= 1,
          'More than one mutation on this Node set the x component, check ' + Object.keys( options ).join( ',' ) );

        assert && assert( _.filter( [ 'translation', 'y', 'top', 'bottom', 'centerY', 'centerTop', 'rightTop', 'leftCenter', 'center', 'rightCenter', 'leftBottom', 'centerBottom', 'rightBottom' ], function( key ) { return options[ key ] !== undefined; } ).length <= 1,
          'More than one mutation on this Node set the y component, check ' + Object.keys( options ).join( ',' ) );
      }

      var node = this;

      _.each( this._mutatorKeys, function( key ) {
        if ( options[ key ] !== undefined ) {
          var descriptor = Object.getOwnPropertyDescriptor( Node.prototype, key );

          // if the key refers to a function that is not ES5 writable, it will execute that function with the single argument
          if ( descriptor && typeof descriptor.value === 'function' ) {
            node[ key ]( options[ key ] );
          }
          else {
            node[ key ] = options[ key ];
          }
        }
      } );

      return this; // allow chaining
    },

    getDebugHTMLExtras: function() {
      return ''; // override for extra information in the debugging output
    },

    toString: function( spaces, includeChildren ) {
      spaces = spaces || '';
      var props = this.getPropString( spaces + '  ', includeChildren === undefined ? true : includeChildren );
      return spaces + this.getBasicConstructor( props ? ( '\n' + props + '\n' + spaces ) : '' );
    },

    getBasicConstructor: function( propLines ) {
      return 'new scenery.Node( {' + propLines + '} )';
    },

    getPropString: function( spaces, includeChildren ) {
      var result = '';

      function addProp( key, value, nowrap ) {
        if ( result ) {
          result += ',\n';
        }
        if ( !nowrap && typeof value === 'string' ) {
          result += spaces + key + ': \'' + value + '\'';
        }
        else {
          result += spaces + key + ': ' + value;
        }
      }

      if ( this._children.length && includeChildren ) {
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
      if ( this.pickable !== null ) { addProp( 'pickable', this.pickable ); }
      if ( this.opacity !== 1 ) { addProp( 'opacity', this.opacity ); }

      if ( !this.transform.isIdentity() ) {
        var m = this.transform.getMatrix();
        addProp( 'matrix', 'dot.Matrix3.createFromPool(' +
                           m.m00() + ', ' + m.m01() + ', ' + m.m02() + ', ' +
                           m.m10() + ', ' + m.m11() + ', ' + m.m12() + ', ' +
                           m.m20() + ', ' + m.m21() + ', ' + m.m22() + ' )', true );
      }

      if ( this.renderer ) {
        addProp( 'renderer', this.renderer.name );
        if ( this.rendererOptions ) {
          // addProp( 'rendererOptions', JSON.stringify( this.rendererOptions ), true );
        }
      }

      if ( this._hints.layerSplit ) {
        addProp( 'layerSplit', true );
      }

      return result;
    },

    /*---------------------------------------------------------------------------*
     * Compatibility with old events API (now using axon.Events)
     *----------------------------------------------------------------------------*/

    addEventListener: function( eventName, listener ) {
      // can't guarantee static with old usage
      return this.on( eventName, listener );
    },

    removeEventListener: function( eventName, listener ) {
      // can't guarantee static with old usage
      return this.off( eventName, listener );
    },

    containsEventListener: function( eventName, listener ) {
      return this.hasListener( eventName, listener );
    },

    onEventListenerAdded: function( eventName, listener ) {
      if ( eventName in eventsRequiringBoundsValidation ) {
        this.changeBoundsEventCount( 1 );
        this._boundsEventSelfCount++;
      }
    },

    onEventListenerRemoved: function( eventName, listener ) {
      if ( eventName in eventsRequiringBoundsValidation ) {
        this.changeBoundsEventCount( -1 );
        this._boundsEventSelfCount--;
      }
    }

  }, Events.prototype, {
    on: function onOverride( eventName, listener ) {
      Events.prototype.on.call( this, eventName, listener );
      this.onEventListenerAdded( eventName, listener );
    },

    onStatic: function onStaticOverride( eventName, listener ) {
      Events.prototype.onStatic.call( this, eventName, listener );
      this.onEventListenerAdded( eventName, listener );
    },

    off: function offOverride( eventName, listener ) {
      var index = Events.prototype.off.call( this, eventName, listener );
      assert && assert( index >= 0, 'Node.off was called but no listener was removed' );
      this.onEventListenerRemoved( eventName, listener );
      return index;
    },

    offStatic: function offStaticOverride( eventName, listener ) {
      var index = Events.prototype.offStatic.call( this, eventName, listener );
      assert && assert( index >= 0, 'Node.offStatic was called but no listener was removed' );
      this.onEventListenerRemoved( eventName, listener );
      return index;
    }
  } ) );


  /*
   * Convenience locations
   * upper is in terms of the visual layout in Scenery and other programs, so the minY is the "upper", and minY is the "lower"
   *
   *             left (x)     centerX        right
   *          ---------------------------------------
   * top  (y) | leftTop     centerTop     rightTop
   * centerY  | leftCenter  center        rightCenter
   * bottom   | leftBottom  centerBottom  rightBottom
   */

  // assumes the getterMethod is the same for Node and Bounds2
  function addBoundsVectorGetterSetter( getterMethod, setterMethod, propertyName ) {
    Node.prototype[ getterMethod ] = function() {
      return this.getBounds()[ getterMethod ]();
    };

    Node.prototype[ setterMethod ] = function( value ) {
      assert && assert( value instanceof Vector2 );

      this.translate( value.minus( this[ getterMethod ]() ), true );
      return this; // allow chaining
    };

    // ES5 getter and setter
    Object.defineProperty( Node.prototype, propertyName, {
      set: Node.prototype[ setterMethod ],
      get: Node.prototype[ getterMethod ]
    } );
  }

  // arguments are more explicit so text-searches will hopefully identify this code.
  addBoundsVectorGetterSetter( 'getLeftTop', 'setLeftTop', 'leftTop' );
  addBoundsVectorGetterSetter( 'getCenterTop', 'setCenterTop', 'centerTop' );
  addBoundsVectorGetterSetter( 'getRightTop', 'setRightTop', 'rightTop' );
  addBoundsVectorGetterSetter( 'getLeftCenter', 'setLeftCenter', 'leftCenter' );
  addBoundsVectorGetterSetter( 'getCenter', 'setCenter', 'center' );
  addBoundsVectorGetterSetter( 'getRightCenter', 'setRightCenter', 'rightCenter' );
  addBoundsVectorGetterSetter( 'getLeftBottom', 'setLeftBottom', 'leftBottom' );
  addBoundsVectorGetterSetter( 'getCenterBottom', 'setCenterBottom', 'centerBottom' );
  addBoundsVectorGetterSetter( 'getRightBottom', 'setRightBottom', 'rightBottom' );

  /*
   * This is an array of property (setter) names for Node.mutate(), which are also used when creating nodes with parameter objects.
   *
   * E.g. new scenery.Node( { x: 5, rotation: 20 } ) will create a Path, and apply setters in the order below (node.x = 5; node.rotation = 20)
   *
   * The order below is important! Don't change this without knowing the implications.
   * NOTE: translation-based mutators come before rotation/scale, since typically we think of their operations occuring "after" the rotation / scaling
   * NOTE: left/right/top/bottom/centerX/centerY are at the end, since they rely potentially on rotation / scaling changes of bounds that may happen beforehand
   */
  Node.prototype._mutatorKeys = [
    'children', 'cursor', 'visible', 'pickable', 'opacity', 'matrix', 'translation', 'x', 'y', 'rotation', 'scale',
    'leftTop', 'centerTop', 'rightTop', 'leftCenter', 'center', 'rightCenter', 'leftBottom', 'centerBottom', 'rightBottom',
    'left', 'right', 'top', 'bottom', 'centerX', 'centerY', 'renderer', 'rendererOptions',
    'layerSplit', 'usesOpacity', 'mouseArea', 'touchArea', 'clipArea', 'transformBounds', 'focusable', 'focusIndicator',
    'textDescription'
  ];

  return Node;
} );
