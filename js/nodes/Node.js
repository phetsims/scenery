// Copyright 2002-2014, University of Colorado Boulder

/**
 * A node for the Scenery scene graph. Supports general directed acyclic graphics (DAGs).
 * Handles multiple layers with assorted types (Canvas 2D, SVG, DOM, WebGL, etc.).
 *
 * See http://phetsims.github.io/scenery/doc/#node
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
  var Renderer = require( 'SCENERY/display/Renderer' );
  var RendererSummary = require( 'SCENERY/util/RendererSummary' );
  require( 'SCENERY/util/CanvasContextWrapper' );
  // commented out so Require.js doesn't balk at the circular dependency
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

  function defaultLeafTrailPredicate( node ) {
    return node._children.length === 0;
  }

  function hasRootedDisplayPredicate( node ) {
    return node._rootedDisplays.length > 0;
  }

  var scratchBounds2 = Bounds2.NOTHING.copy(); // mutable {Bounds2} used temporarily in methods
  var scratchMatrix3 = new Matrix3();

  /*
   * See http://phetsims.github.io/scenery/doc/#node-options
   *
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
    // supertype call to axon.Events (should just initialize a few properties here, notably _eventListeners and _staticEventListeners)
    Events.call( this );

    // NOTE: All member properties with names starting with '_' are assumed to be @private!

    // @private {number} - Assigns a unique ID to this node (allows trails to get a unique list of IDs)
    this._id = globalIdCounter++;

    // @protected {Array.<Instance>} - All of the Instances tracking this Node
    this._instances = [];

    // @protected {Array.<AccessibleInstance>} - Empty unless the node contains some accessible instance.
    this._accessibleInstances = [];

    // @protected {Array.<Display>} - All displays where this node is the root.
    this._rootedDisplays = [];

    // @protected {Array.<Drawable>} - Drawable states that need to be updated on mutations. Generally added by SVG and
    // DOM elements that need to closely track state (possibly by Canvas to maintain dirty state).
    this._drawables = [];

    // @private {boolean} - Whether this node (and its children) will be visible when the scene is updated. Visible
    // nodes by default will not be pickable either.
    this._visible = true;

    // @private {number} - Opacity, in the range from 0 (fully transparent) to 1 (fully opaque).
    this._opacity = 1;

    // @private Whether this node (and its subtree) will allow hit-testing (and thus user interaction). Notably:
    // pickable: null  - default. Node is only pickable if it (or an ancestor/descendant) has either an input listener or pickable: true set
    // pickable: false - Node (and subtree) is pickable, just like if there is an input listener
    // pickable: true  - Node is unpickable (only has an effect when underneath a node with an input listener / pickable: true set)
    this._pickable = null;

    // @private - This node and all children will be clipped by this shape (in addition to any other clipping shapes).
    // {Shape|null} The shape should be in the local coordinate frame.
    this._clipArea = null;

    // @private - Areas for hit intersection. If set on a Node, no descendants can handle events.
    this._mouseArea = null; // {Shape|Bounds2} for mouse position in the local coordinate frame
    this._touchArea = null; // {Shape|Bounds2} for touch and pen position in the local coordinate frame

    // @private {string} - The CSS cursor to be displayed over this node. null should be the default (inherit) value.
    this._cursor = null;

    // @private {boolean} - Whether this Node should be accessible via tab ordering. Defaults to false.
    this._focusable = false;

    // @private @deprecated {string} - 'cursor' or 'rectangle' at the moment. WARNING: in active development!
    this._focusIndicator = 'rectangle';

    // @private {null|Object} - If non-null, this node will be represented in the parallel DOM by the accessible content.
    // The accessibleContent object will be of the form:
    // {
    //   createPeer: function( {AccessibleInstance} ): {AccessiblePeer},
    //   [focusHighlight]: {Bounds2|Shape|Node}
    // }
    this._accessibleContent = null;

    // @private {Array.<Node> | null} - If provided, it will override the focus order between children (and optionally
    // descendants). If not provided, the focus order will default to the rendering order (first children first, last
    // children last) determined by the children array.
    this._accessibleOrder = null;

    // @public (scenery-internal) - Not for public use, but used directly internally for performance.
    this._children = []; // {Array.<Node>} - Ordered array of child nodes.
    this._parents = []; // {Array.<Node>} - Unordered array of parent nodes.

    // @private @deprecated
    this._peers = []; // array of peer factories: { element: ..., options: ... }, where element can be an element or a string
    this._liveRegions = []; // array of live region instances

    // @private {boolean} - Whether we will do more accurate (and tight) bounds computations for rotations and shears.
    this._transformBounds = false;

    /*
     * Set up the transform reference. we add a listener so that the transform itself can be modified directly
     * by reference, triggering the event notifications for Scenery The reference to the Transform3 will never change.
     */
    this._transform = new Transform3(); // @private {Transform3}
    this._transformListener = this.onTransformChange.bind( this ); // @private {Function}
    this._transform.on( 'change', this._transformListener );

    /*
     * Maxmimum dimensions for the node's local bounds before a corrective scaling factor is applied to maintain size.
     * The maximum dimensions are always compared to local bounds, and applied "before" the node's transform.
     * Whenever the local bounds or maximum dimensions of this Node change and it has at least one maximum dimension
     * (width or height), an ideal scale is computed (either the smallest scale for our local bounds to fit the
     * dimension constraints, OR 1, whichever is lower). Then the Node's transform will be scaled (prepended) with
     * a scale adjustment of ( idealScale / alreadyAppliedScaleFactor ).
     * In the simpe case where the Node isn't otherwise transformed, this will apply and update the Node's scale so that
     * the node matches the maximum dimensions, while never scaling over 1. Note that manually applying transforms to
     * the Node is fine, but may make the node's width greater than the maximum width.
     * NOTE: If a dimension constraint is null, no resizing will occur due to it. If both maxWidth and maxHeight are null,
     * no scale adjustment will be applied.
     *
     * Also note that setting maxWidth/maxHeight is like adding a local bounds listener (will trigger validation of
     * bounds during the updateDisplay step). NOTE: this means updates to the transform (on a local bounds change) will
     * happen when bounds are validated (validateBounds()), which does not happen synchronously on a child's size
     * change. It does happen at least once in updateDisplay() before rendering, and calling validateBounds() can force
     * a re-check and transform.
     */
    this._maxWidth = null; // @private {number|null}
    this._maxHeight = null; // @private {number|null}
    this._appliedScaleFactor = 1; // @private {number} - Scale applied due to the maximum dimension constraints.

    // @private {Array.<Function>} - For user input handling (mouse/touch).
    this._inputListeners = [];

    // @private {Bounds2} - [mutable] Bounds for this node and its children in the "parent" coordinate frame.
    this._bounds = Bounds2.NOTHING.copy();

    // @private {Bounds2} - [mutable] Bounds for this node and its children in the "local" coordinate frame.
    this._localBounds = Bounds2.NOTHING.copy();

    // @private {Bounds2} - [mutable] Bounds just for this node, in the "local" coordinate frame.
    this._selfBounds = Bounds2.NOTHING.copy();

    // @private {Bounds2} - [mutable] Bounds just for children of this node (and sub-trees), in the "local" coordinate frame.
    this._childBounds = Bounds2.NOTHING.copy();

    // @private {boolean} - Whether our localBounds have been set (with the ES5 setter/setLocalBounds()) to a custom
    // overridden value. If true, then localBounds itself will not be updated, but will instead always be the
    // overridden value.
    this._localBoundsOverridden = false;

    this._boundsDirty = true; // @private {boolean} - Whether bounds needs to be recomputed to be valid.
    this._localBoundsDirty = true; // @private {boolean} - Whether localBounds needs to be recomputed to be valid.
    this._childBoundsDirty = true; // @private {boolean} - Whether childBounds needs to be recomputed to be valid.

    if ( assert ) {
      // for assertions later to ensure that we are using the same Bounds2 copies as before
      this._originalBounds = this._bounds;
      this._originalLocalBounds = this._localBounds;
      this._originalSelfBounds = this._selfBounds;
      this._originalChildBounds = this._childBounds;
    }

    // Similar to bounds, but includes any mouse/touch areas respectively, and excludes areas that would be pruned in
    // hit-testing. They are validated separately (independent from normal bounds validation), but should now always be
    // non-null (since we now properly handle pruning).
    this._mouseBounds = Bounds2.NOTHING.copy(); // @private {Bounds2} - [mutable] Hit bounds for mouse input
    this._touchBounds = Bounds2.NOTHING.copy(); // @private {Bounds2} - [mutable] Hit bounds for touch input
    this._mouseBoundsDirty = true; // @private {boolean} - Whether the bounds are marked as dirty
    this._touchBoundsDirty = true; // @private {boolean} - Whether the bounds are marked as dirty
    // Dirty flags for mouse/touch bounds. Since we only walk the dirty flags up ancestors, we need a way to
    // re-evaluate descendants when the existence of effective listeners changes.
    this._mouseBoundsHadListener = false; // @private {boolean}
    this._touchBoundsHadListener = false; // @private {boolean}

    // @public (scenery-internal) {Object} - Where rendering-specific settings are stored. They are generally modified
    // internally, so there is no ES5 setter for hints.
    this._hints = {
      // {number} - What type of renderer should be forced for this node. Uses the internal bitmask structure declared
      //            in scenery.js and Renderer.js.
      renderer: 0,

      // {boolean} - Whether it is ancitipated that opacity will be switched on. If so, having this set to true will
      //             make switching back-and-forth between opacity:1 and other opacities much faster.
      usesOpacity: false,

      // {boolean} - Whether layers should be split before and after this node.
      layerSplit: false,

      // {boolean} - Whether this node and its subtree should handle transforms by using a CSS transform of a div.
      cssTransform: false,

      // {boolean} - When rendered as Canvas, whether we should use full (device) resolution on retina-like devices.
      //             TODO: ensure that this is working? 0.2 may have caused a regression.
      fullResolution: false,

      // {boolean} - Whether SVG (or other) content should be excluded from the DOM tree when invisible
      //             (instead of just being hidden)
      excludeInvisible: false,

      // {number|null} - If non-null, a multiplier to the detected pixel-to-pixel scaling of the WebGL Canvas
      webglScale: null
    };

    // @public (scenery-internal) {number} - The subtree pickable count is #pickable:true + #inputListeners, since we
    // can prune subtrees with a pickable count of 0.
    this._subtreePickableCount = 0;

    // @public (scenery-internal) {number} - A bitmask which specifies which renderers this node (and only this node,
    // not its subtree) supports.
    this._rendererBitmask = Renderer.bitmaskNodeDefault;

    // @public (scenery-internal) {RendererSummary} - A bitmask-like summary of what renderers and options are supported
    // by this node and all of its descendants
    this._rendererSummary = new RendererSummary( this );

    /*
     * So we can traverse only the subtrees that require bounds validation for events firing.
     * This is a sum of the number of events requiring bounds validation on this Node, plus the number of children whose
     * count is non-zero.
     * NOTE: this means that if A has a child B, and B has a boundsEventCount of 5, it only contributes 1 to A's count.
     * This allows us to have changes localized (increasing B's count won't change A or any of A's ancestors), and
     * guarantees that we will know whether a subtree has bounds listeners. Also important: decreasing B's
     * boundsEventCount down to 0 will allow A to decrease its count by 1, without having to check its other children
     * (if we were just using a boolean value, this operation would require A to check if any OTHER children besides
     * B had bounds listeners)
     */
    this._boundsEventCount = 0; // @private {number}
    // @private {number} - This signals that we can validateBounds() on this subtree and we don't have to traverse further
    this._boundsEventSelfCount = 0;

    if ( options ) {
      this.mutate( options );
    }

    phetAllocation && phetAllocation( 'Node' );
  };
  var Node = scenery.Node;

  inherit( Object, Node, extend( {
    /**
     * Inserts a child node at a specific index, see http://phetsims.github.io/scenery/doc/#node-insertChild
     * @public
     *
     * NOTE: overridden by Leaf for some subtypes
     *
     * @param {number} index
     * @param {Node} node
     */
    insertChild: function( index, node ) {
      assert && assert( node !== null && node !== undefined, 'insertChild cannot insert a null/undefined child' );
      assert && assert( !_.contains( this._children, node ), 'Parent already contains child' );
      assert && assert( node !== this, 'Cannot add self as a child' );

      // needs to be early to prevent re-entrant children modifications
      this.changePickableCount( node._subtreePickableCount );
      this.changeBoundsEventCount( node._boundsEventCount > 0 ? 1 : 0 );
      this._rendererSummary.summaryChange( RendererSummary.bitmaskAll, node._rendererSummary.bitmask );

      node._parents.push( this );
      this._children.splice( index, 0, node );

      if ( !node._rendererSummary.isNotAccessible() ) {
        var trails = node.getTrails( hasRootedDisplayPredicate );
        for ( var i = 0; i < trails.length; i++ ) {
          var trail = trails[ i ];
          var rootedDisplays = trail.rootNode()._rootedDisplays;
          for ( var j = 0; j < rootedDisplays.length; j++ ) {
            rootedDisplays[ j ].addAccessibleTrail( trail );
          }
        }
      }

      node.invalidateBounds();

      // like calling this.invalidateBounds(), but we already marked all ancestors with dirty child bounds
      this._boundsDirty = true;

      this.trigger2( 'childInserted', node, index );

      return this; // allow chaining
    },

    /**
     * Appends a child node to our list of children, see http://phetsims.github.io/scenery/doc/#node-addChild
     * @public
     *
     * @param {Node} node
     */
    addChild: function( node ) {
      this.insertChild( this._children.length, node );

      return this; // allow chaining
    },

    /**
     * Removes a child node from our list of children, see http://phetsims.github.io/scenery/doc/#node-removeChild
     * Will fail an assertion if the node is not currently one of our children
     * @public
     *
     * @param {Node} node
     */
    removeChild: function( node ) {
      assert && assert( node && node instanceof Node, 'Need to call node.removeChild() with a Node.' );
      assert && assert( this.isChild( node ), 'Attempted to removeChild with a node that was not a child.' );

      var indexOfChild = _.indexOf( this._children, node );

      this.removeChildWithIndex( node, indexOfChild );

      return this; // allow chaining
    },

    /**
     * Removes a child node at a specific index (node.children[ index ]) from our list of children.
     * Will fail if the index is out of bounds.
     * @public
     *
     * @param {number} index
     */
    removeChildAt: function( index ) {
      assert && assert( index >= 0 );
      assert && assert( index < this._children.length );

      var node = this._children[ index ];

      this.removeChildWithIndex( node, index );

      return this; // allow chaining
    },

    /**
     * Internal method for removing a node (always has the Node and index).
     * @private
     *
     * NOTE: overridden by Leaf for some subtypes
     *
     * @param {Node} node - The child node to remove from this node (it's parent)
     * @param {number} indexOfChild - Should satisfy this.children[ indexOfChild ] === node
     */
    removeChildWithIndex: function( node, indexOfChild ) {
      assert && assert( node && node instanceof Node, 'Need to call node.removeChildWithIndex() with a Node.' );
      assert && assert( this.isChild( node ), 'Attempted to removeChild with a node that was not a child.' );
      assert && assert( this._children[ indexOfChild ] === node, 'Incorrect index for removeChildWithIndex' );

      var indexOfParent = _.indexOf( node._parents, this );

      // NOTE: Potentially removes bounds listeners here!
      if ( !node._rendererSummary.isNotAccessible() ) {
        var trails = node.getTrails( hasRootedDisplayPredicate );
        for ( var i = 0; i < trails.length; i++ ) {
          var trail = trails[ i ];
          var rootedDisplays = trail.rootNode()._rootedDisplays;
          for ( var j = 0; j < rootedDisplays.length; j++ ) {
            rootedDisplays[ j ].removeAccessibleTrail( trail );
          }
        }
      }

      // needs to be early to prevent re-entrant children modifications
      this.changePickableCount( -node._subtreePickableCount );
      this.changeBoundsEventCount( node._boundsEventCount > 0 ? -1 : 0 );
      this._rendererSummary.summaryChange( node._rendererSummary.bitmask, RendererSummary.bitmaskAll );

      node._parents.splice( indexOfParent, 1 );
      this._children.splice( indexOfChild, 1 );

      this.invalidateBounds();
      this._childBoundsDirty = true; // force recomputation of child bounds after removing a child

      this.trigger2( 'childRemoved', node, indexOfChild );
    },

    /**
     * Removes all children from this Node.
     * @public
     */
    removeAllChildren: function() {
      this.setChildren( [] );

      return this; // allow chaining
    },

    /**
     * Sets the children of the Node to be equivalent to the passed-in array of Nodes. Does this by removing all current
     * children, and adding in children from the array.
     * @public
     *
     * @param {Array.<Node>} children
     */
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

      return this; // allow chaining
    },
    set children( value ) { this.setChildren( value ); },

    /**
     * Returns a defensive copy of our children.
     * @public
     *
     * @returns {Array.<Node>}
     */
    getChildren: function() {
      // TODO: ensure we are not triggering this in Scenery code when not necessary!
      return this._children.slice( 0 ); // create a defensive copy
    },
    get children() { return this.getChildren(); },

    /**
     * Returns a count of children, without needing to make a defensive copy.
     * @public
     *
     * @returns {number}
     */
    getChildrenCount: function() {
      return this._children.length;
    },

    /**
     * Returns a defensive copy of our parents.
     * @public
     *
     * @returns {Array.<Node>}
     */
    getParents: function() {
      return this._parents.slice( 0 ); // create a defensive copy
    },
    get parents() { return this.getParents(); },

    /**
     * Returns a single parent if it exists, otherwise null (no parents), or an assertion failure (multiple parents).
     * @public
     *
     * @returns {Node|null}
     */
    getParent: function() {
      assert && assert( this._parents.length <= 1, 'Cannot call getParent on a node with multiple parents' );
      return this._parents.length ? this._parents[ 0 ] : null;
    },

    /**
     * Gets the child at a specific index into the children array.
     * @public
     *
     * @param {number} index
     * @returns {Node}
     */
    getChildAt: function( index ) {
      return this._children[ index ];
    },

    /**
     * Finds the index of a parent Node in the parents array.
     * @public
     *
     * @param {Node} parent - Should be a parent of this node.
     * @returns {number} - An index such that this.parents[ index ] === parent
     */
    indexOfParent: function( parent ) {
      return _.indexOf( this._parents, parent );
    },

    /**
     * Finds the index of a child Node in the children array.
     * @public
     *
     * @param {Node} child - Should be a child of this node.
     * @returns {number} - An index such that this.children[ index ] === child
     */
    indexOfChild: function( child ) {
      return _.indexOf( this._children, child );
    },

    /**
     * Moves this node to the front (end) of all of its parents children array.
     * @public
     */
    moveToFront: function() {
      var self = this;
      _.each( this._parents.slice( 0 ), function( parent ) {
        parent.moveChildToFront( self );
      } );

      return this; // allow chaining
    },

    /**
     * Moves one of our children to the front (end) of our children array.
     * @public
     *
     * @param {Node} child - Our child to move to the front.
     */
    moveChildToFront: function( child ) {
      if ( this.indexOfChild( child ) !== this._children.length - 1 ) {
        this.removeChild( child );
        this.addChild( child );
      }

      return this; // allow chaining
    },

    /**
     * Moves this node to the back (front) of all of its parents children array.
     * @public
     */
    moveToBack: function() {
      var self = this;
      _.each( this._parents.slice( 0 ), function( parent ) {
        parent.moveChildToBack( self );
      } );

      return this; // allow chaining
    },

    /**
     * Moves one of our children to the back (front) of our children array.
     * @public
     *
     * @param {Node} child - Our child to move to the back.
     */
    moveChildToBack: function( child ) {
      if ( this.indexOfChild( child ) !== 0 ) {
        this.removeChild( child );
        this.insertChild( 0, child );
      }

      return this; // allow chaining
    },

    /**
     * Removes this node from all of its parents.
     * @public
     */
    detach: function() {
      var that = this;
      _.each( this._parents.slice( 0 ), function( parent ) {
        parent.removeChild( that );
      } );

      return this; // allow chaining
    },

    /**
     * Propagate the pickable count change down to our ancestors.
     * @private (Scenery-internal)
     *
     * @param {number} n - The delta of how many pickable counts have been added/removed
     */
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

    /**
     * Update our event count, usually by 1 or -1. See documentation on _boundsEventCount in constructor.
     * @private
     *
     * @param {number} n - How to increment/decrement the bounds event listener count
     */
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

    /**
     * @deprecated?
     * currently, there is no way to remove peers. if a string is passed as the element pattern, it will be turned into
     * an element
     */
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

    /**
     * Ensures that cached bounds stored on this node (and all children) are accurate. Returns true if any sort of dirty
     * flag was set before this was called.
     * @public
     *
     * @returns {boolean} - Was something potentially updated?
     */
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

        // and recompute our _childBounds
        var oldChildBounds = scratchBounds2.set( this._childBounds ); // store old value in a temporary Bounds2
        this._childBounds.set( Bounds2.NOTHING ); // initialize to a value that can be unioned with includeBounds()

        i = this._children.length;
        while ( i-- ) {
          this._childBounds.includeBounds( this._children[ i ]._bounds );
        }

        // run this before firing the event
        this._childBoundsDirty = false;

        if ( !this._childBounds.equals( oldChildBounds ) ) {
          // notifies only on an actual change
          this.trigger0( 'childBounds' );
        }
      }

      if ( this._localBoundsDirty && !this._localBoundsOverridden ) {
        wasDirtyBefore = true;

        this._localBoundsDirty = false; // we only need this to set local bounds as dirty

        var oldLocalBounds = scratchBounds2.set( this._localBounds ); // store old value in a temporary Bounds2

        // local bounds are a union between our self bounds and child bounds
        this._localBounds.set( this._selfBounds ).includeBounds( this._childBounds );

        // apply clipping to the bounds if we have a clip area (all done in the local coordinate frame)
        if ( this.hasClipArea() ) {
          this._localBounds.constrainBounds( this._clipArea.bounds );
        }

        if ( !this._localBounds.equals( oldLocalBounds ) ) {
          this.trigger0( 'localBounds' );

          // sanity check
          this._boundsDirty = true;
        }

        // adjust our transform to match maximum bounds if necessary on a local bounds change
        if ( this._maxWidth !== null || this._maxHeight !== null ) {
          this.updateMaxDimension( this._localBounds );
        }
      }

      // TODO: layout here?

      if ( this._boundsDirty ) {
        wasDirtyBefore = true;

        // run this before firing the event
        this._boundsDirty = false;

        var oldBounds = scratchBounds2.set( this._bounds ); // store old value in a temporary Bounds2

        // no need to do the more expensive bounds transformation if we are still axis-aligned
        if ( this._transformBounds && !this._transform.getMatrix().isAxisAligned() ) {
          // mutates the matrix and bounds during recursion

          var matrix = scratchMatrix3.set( this.getMatrix() ); // calls below mutate this matrix
          this._bounds.set( Bounds2.NOTHING );
          // Include each painted self individually, transformed with the exact transform matrix.
          // This is expensive, as we have to do 2 matrix transforms for every descendant.
          this._includeTransformedSubtreeBounds( matrix, this._bounds ); // self and children

          if ( this.hasClipArea() ) {
            this._bounds.constrainBounds( this._clipArea.getBoundsWithTransform( matrix ) );
          }
        }
        else {
          // converts local to parent bounds. mutable methods used to minimize number of created bounds instances
          // (we create one so we don't change references to the old one)
          this._bounds.set( this._localBounds );
          this.transformBoundsFromLocalToParent( this._bounds );
        }

        if ( !this._bounds.equals( oldBounds ) ) {
          // if we have a bounds change, we need to invalidate our parents so they can be recomputed
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
        // TODO: if there are side-effects in listeners, this could overflow the stack. we should report an error
        // instead of locking up
        this.validateBounds();
      }

      if ( assert ) {
        assert( this._originalBounds === this._bounds, 'Reference for _bounds changed!' );
        assert( this._originalLocalBounds === this._localBounds, 'Reference for _localBounds changed!' );
        assert( this._originalSelfBounds === this._selfBounds, 'Reference for _selfBounds changed!' );
        assert( this._originalChildBounds === this._childBounds, 'Reference for _childBounds changed!' );
      }

      // double-check that all of our bounds handling has been accurate
      if ( assertSlow ) {
        // new scope for safety
        (function() {
          var epsilon = 0.000001;

          var childBounds = Bounds2.NOTHING.copy();
          _.each( that._children, function( child ) { childBounds.includeBounds( child._bounds ); } );

          var localBounds = that._selfBounds.union( childBounds );

          if ( that.hasClipArea() ) {
            localBounds = localBounds.intersection( that._clipArea.bounds );
          }

          var fullBounds = that.localToParentBounds( localBounds );

          assertSlow && assertSlow( that._childBounds.equalsEpsilon( childBounds, epsilon ),
            'Child bounds mismatch after validateBounds: ' +
            that._childBounds.toString() + ', expected: ' + childBounds.toString() );
          assertSlow && assertSlow( that._localBoundsOverridden ||
                                    that._transformBounds ||
                                    that._bounds.equalsEpsilon( fullBounds, epsilon ) ||
                                    that._bounds.equalsEpsilon( fullBounds, epsilon ),
            'Bounds mismatch after validateBounds: ' + that._bounds.toString() +
            ', expected: ' + fullBounds.toString() );
        })();
      }

      return wasDirtyBefore; // whether any dirty flags were set
    },

    /**
     * Recursion for accurate transformed bounds handling. Mutates bounds with the added bounds.
     * Mutates the matrix (parameter), but mutates it back to the starting point (within floating-point error).
     * @private
     *
     * @param {Matrix3} matrix
     * @param {Bounds2} bounds
     */
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

    /**
     * Traverses this subtree and validates bounds only for subtrees that have bounds listeners (trying to exclude as
     * much as possible for performance). This is done so that we can do the minimum bounds validation to prevent any
     * bounds listeners from being triggered in further validateBounds() calls without other Node changes being done.
     * This is required for Display's atomic (non-reentrant) updateDisplay(), so that we don't accidentally trigger
     * bounds listeners while computing bounds during updateDisplay().
     * @public (scenery-internal)
     *
     * NOTE: this should pass by (ignore) any overridden localBounds, to trigger listeners below.
     */
    validateWatchedBounds: function() {
      // Since a bounds listener on one of the roots could invalidate bounds on the other, we need to keep running this
      // until they are all clean. Otherwise, side-effects could occur from bounds validations
      // TODO: consider a way to prevent infinite loops here that occur due to bounds listeners triggering cycles
      while ( this.watchedBoundsScan() ) {
        // do nothing
      }
    },

    /**
     * Recursive function for validateWatchedBounds. Returned whether any validateBounds() returned true (means we have
     * to traverse again)
     * @public (scenery-internal)
     *
     * @returns {boolean} - Whether there could have been any changes.
     */
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
     * Updates the mouseBounds for the Node. It will include only the specific bounded areas that are relevant for
     * hit-testing mouse events. Thus it:
     * - includes mouseAreas (normal bounds don't)
     * - does not include subtrees that would be pruned in hit-testing
     * @public (scenery-internal)
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
     * @public (scenery-internal)
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

    /**
     * Marks the bounds of this node as invalid, so they are recomputed before being accessed again.
     * @public
     */
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

    /**
     * Recursively tag all ancestors with _childBoundsDirty
     * @public (scenery-internal)
     */
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

    /**
     * Mark mouse/touch bounds as invalid (can occur from normal bounds invalidation, or from anything that could change
     * pickability).
     * @public (scenery-internal)
     *
     * NOTE: we don't have to touch descendants because we also store the last used "under effective listener" value, so
     * "non-dirty" subtrees will still be investigated (or freshly pruned) if the listener status has changed.
     */
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

    /**
     * Should be called to notify that our selfBounds needs to change to this new value.
     * @public
     *
     * @param {Bounds2} newSelfBounds
     */
    invalidateSelf: function( newSelfBounds ) {
      assert && assert( newSelfBounds.isEmpty() || newSelfBounds.isFinite(), 'Bounds must be empty or finite in invalidateSelf' );

      // if these bounds are different than current self bounds
      if ( !this._selfBounds.equals( newSelfBounds ) ) {
        // set repaint flags
        this._localBoundsDirty = true;
        this.invalidateBounds();

        // record the new bounds
        this._selfBounds.set( newSelfBounds );

        // fire the event immediately
        this.trigger0( 'selfBounds' );
      }
    },

    /**
     * Returns whether a Node is a child of this node.
     * @public
     *
     * @param {Node} potentialChild
     * @returns {boolean} - Whether potentialChild is actually our child.
     */
    isChild: function( potentialChild ) {
      assert && assert( potentialChild && ( potentialChild instanceof Node ), 'isChild needs to be called with a Node' );
      var ourChild = _.contains( this._children, potentialChild );
      var itsParent = _.contains( potentialChild._parents, this );
      assert && assert( ourChild === itsParent );
      return ourChild;
    },

    /**
     * Returns our selfBounds (the bounds for this Node's content in the local coordinates, excluding anything from our
     * children and descendants).
     * @public
     *
     * NOTE: Do NOT mutate the returned value!
     *
     * @returns {Bounds2}
     */
    getSelfBounds: function() {
      return this._selfBounds;
    },
    get selfBounds() { return this.getSelfBounds(); },

    /**
     * Returns a bounding box that should contain all self content in the local coordinate frame (our normal self bounds
     * aren't guaranteed this for Text, etc.)
     * @public
     *
     * Override this to provide different behavior.
     *
     * @returns {Bounds2}
     */
    getSafeSelfBounds: function() {
      return this._selfBounds;
    },

    /**
     * Returns the bounding box that should contain all content of our children in our local coordinate frame. Does not
     * include our "self" bounds.
     * @public
     *
     * NOTE: Do NOT mutate the returned value!
     *
     * @returns {Bounds2}
     */
    getChildBounds: function() {
      this.validateBounds();
      return this._childBounds;
    },
    get childBounds() { return this.getChildBounds(); },

    /**
     * Returns the bounding box that should contain all content of our children AND our self in our local coordinate
     * frame.
     * @public
     *
     * NOTE: Do NOT mutate the returned value!
     *
     * @returns {Bounds2}
     */
    getLocalBounds: function() {
      this.validateBounds();
      return this._localBounds;
    },
    get localBounds() { return this.getLocalBounds(); },

    /**
     * Allows overriding the value of localBounds (and thus changing things like 'bounds' that depend on localBounds).
     * If it's set to a non-null value, that value will always be used for localBounds until this function is called
     * again. To revert to having Scenery compute the localBounds, set this to null.
     * @public
     *
     * @param {Bounds2|null} localBounds
     */
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
        var changed = !localBounds.equals( this._localBounds ) || !this._localBoundsOverridden;

        if ( changed ) {
          this._localBounds.set( localBounds );
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
    set localBounds( value ) { return this.setLocalBounds( value ); },

    /**
     * Meant to be overridden in sub-types that have more accurate bounds determination for when we are transformed.
     * Usually rotation is significant here, so that transformed bounds for non-rectangular shapes will be different.
     * @public
     *
     * @param {Matrix3} matrix
     * @returns {Bounds2}
     */
    getTransformedSelfBounds: function( matrix ) {
      // assume that we take up the entire rectangular bounds.
      return this._selfBounds.transformed( matrix );
    },

    /**
     * Sets the flag that determines whether we will require more accurate (and expensive) bounds computation for this
     * node's transform.
     * @public
     *
     * If set to false (default), Scenery will get the bounds of content, and then if rotated will determine the on-axis
     * bounds that completely cover the rotated bounds (potentially larger than actual content).
     * If set to true, Scenery will try to get the bounds of the actual rotated/transformed content.
     *
     * A good example of when this is necessary is if there are a bunch of nested children that each have pi/4 rotations.
     *
     * @param {boolean} transformBounds - Whether accurate transform bounds should be used.
     */
    setTransformBounds: function( transformBounds ) {
      assert && assert( typeof transformBounds === 'boolean', 'transformBounds should be boolean' );

      if ( this._transformBounds !== transformBounds ) {
        this._transformBounds = transformBounds;

        this.invalidateBounds();
      }

      return this; // allow chaining
    },
    set transformBounds( value ) { return this.setTransformBounds( value ); },

    /**
     * Returns whether accurate transformation bounds are used in bounds computation (see setTransformBounds).
     * @public
     *
     * @returns {boolean}
     */
    getTransformBounds: function() {
      return this._transformBounds;
    },
    get transformBounds() { return this.getTransformBounds(); },

    /**
     * Returns the bounding box of this Node and all of its sub-trees (in the "parent" coordinate frame).
     *
     * NOTE: Do NOT mutate the returned value!
     *
     * @returns {Bounds2}
     */
    getBounds: function() {
      this.validateBounds();
      return this._bounds;
    },
    get bounds() { return this.getBounds(); },

    /**
     * Like getLocalBounds() in the "local" coordinate frame, but includes only visible nodes.
     * @public
     *
     * @returns {Bounds2}
     */
    getVisibleLocalBounds: function() {
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
      return bounds;
    },
    get visibleLocalBounds() { return this.getVisibleLocalBounds(); },

    /**
     * Like getBounds() in the "parent" coordinate frame, but includes only visible nodes
     * @public
     *
     * @returns {Bounds2}
     */
    getVisibleBounds: function() {
      return this.getVisibleLocalBounds().transform( this.getMatrix() );
    },
    get visibleBounds() { return this.getVisibleBounds(); },

    /**
     * Whether this node effectively behaves as if it has an input listener.
     * @public (scenery-internal)
     *
     * @returns {boolean}
     */
    hasInputListenerEquivalent: function() {
      // NOTE: if anything here is added, update when invalidateMouseTouchBounds gets called (since changes to pickability pruning affect mouse/touch bounds)
      return this._inputListeners.length > 0 || this._pickable === true;
    },

    /**
     * Whether hit-testing for events should be pruned at this node (not even considering this node's self).
     * @public (scenery-internal)
     *
     * @param {boolean} hasListenerEquivalentSelfOrInAncestor - Indicates whether this node (or an ancestor) either has
     *                                                          input listeners, or has pickable set to true (which is
     *                                                          not the default).
     * @returns {boolean}
     */
    //
    //
    isSubtreePickablePruned: function( hasListenerEquivalentSelfOrInAncestor ) {
      // NOTE: if anything here is added, update when invalidateMouseTouchBounds gets called (since changes to pickability pruning affect mouse/touch bounds)
      // if invisible: skip it
      // if pickable: false, skip it
      // if pickable: undefined and our pickable count indicates there are no input listeners / pickable: true in our subtree, skip it
      return !this.isVisible() || this._pickable === false || ( this._pickable !== true && !hasListenerEquivalentSelfOrInAncestor && this._subtreePickableCount === 0 );
    },

    /**
     * Hit-tests what is under the pointer, and returns a {Trail} to that node (or null if there is no matching node).
     * @public
     *
     * @param {Pointer} pointer
     * @returns {Trail|null}
     */
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
     * @public
     *
     * For now, prune anything that is invisible or effectively unpickable
     *
     * @param {Vector2} point
     * @param {Object} options - Checks for options.isMouse/isTouch/isPen currently
     * @param {boolean} [recursive] - Don't pass when calling, signals that the point passed can be mutated
     * @param {boolean} [hasListenerEquivalentSelfOrInAncestor] - Don't pass when calling, used for recursion
     * @returns {Trail|null}
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

    /**
     * Returns whether a point (in parent coordinates) is contained in this node's sub-tree.
     * @public
     *
     * @param {Vector2} point
     * @returns {boolean} - Whether the point is contained.
     */
    containsPoint: function( point ) {
      return this.trailUnderPoint( point ) !== null;
    },

    /**
     * Override this for computation of whether a point is inside our self content (defaults to selfBounds check).
     * @protected
     *
     * @param {Vector2} point - Considered to be in the local coordinate frame
     * @returns {boolean}
     */
    containsPointSelf: function( point ) {
      // if self bounds are not null default to checking self bounds
      return this._selfBounds.containsPoint( point );
    },

    /**
     * Returns whether this node's selfBounds is intersected by the specified bounds.
     * @public
     *
     * @param {Bounds2} bounds - Bounds to test, assumed to be in the local coordinate frame.
     * @returns {boolean}
     */
    intersectsBoundsSelf: function( bounds ) {
      // if self bounds are not null, child should override this
      return this._selfBounds.intersectsBounds( bounds );
    },

    /**
     * Whether this Node itself is painted (displays something itself). Meant to be overridden.
     * @public
     *
     * @returns {boolean}
     */
    isPainted: function() {
      return false;
    },

    /**
     * Whether this Node's selfBounds are considered to be valid (always containing the displayed self content
     * of this node). Meant to be overridden in subtypes when this can change (e.g. Text).
     * @public
     *
     * If this value would potentially change, please trigger the event 'selfBoundsValid'.
     *
     * @returns {boolean}
     */
    areSelfBoundsValid: function() {
      return true;
    },

    /**
     * Returns whether this node has any parents at all.
     * @public
     *
     * @returns {boolean}
     */
    hasParent: function() {
      return this._parents.length !== 0;
    },

    /**
     * Returns whether this node has any children at all.
     * @public
     *
     * @returns {boolean}
     */
    hasChildren: function() {
      return this._children.length > 0;
    },

    /**
     * Calls the callback on nodes recursively in a depth-first manner.
     * @public
     *
     * @param {Function} callback
     */
    walkDepthFirst: function( callback ) {
      callback( this );
      var length = this._children.length;
      for ( var i = 0; i < length; i++ ) {
        this._children[ i ].walkDepthFirst( callback );
      }
    },

    /**
     * Returns a list of child nodes that intersect the passed in bounds.
     * @public
     *
     * @param {Bounds2} bounds - In the local coordinate frame
     * @returns {Array.<Node>}
     */
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

    /**
     * Adds an input listener.
     * @public
     *
     * @param {Object} listener
     */
    addInputListener: function( listener ) {
      // don't allow listeners to be added multiple times
      if ( _.indexOf( this._inputListeners, listener ) === -1 ) {
        this._inputListeners.push( listener );
        this.changePickableCount( 1 ); // NOTE: this should also trigger invalidation of mouse/touch bounds
      }
      return this;
    },

    /**
     * Removes an input listener that was previously added with addInputListener.
     * @public
     *
     * @param {Object} listener
     */
    removeInputListener: function( listener ) {
      // ensure the listener is in our list
      assert && assert( _.indexOf( this._inputListeners, listener ) !== -1 );

      this._inputListeners.splice( _.indexOf( this._inputListeners, listener ), 1 );
      this.changePickableCount( -1 ); // NOTE: this should also trigger invalidation of mouse/touch bounds
      return this;
    },

    /**
     * Returns a copy of all of our input listeners.
     * @public
     *
     * @returns {Array.<Object>}
     */
    getInputListeners: function() {
      return this._inputListeners.slice( 0 ); // defensive copy
    },

    /**
     * Changes the transform of this node by adding a transform. The default "appends" the transform, so that it will
     * appear to happen to the node before the rest of the transform would apply, but if "prepended", the rest of the
     * transform would apply first.
     * @public
     *
     * As an example, if a Node is centered at (0,0) and scaled by 2:
     * translate( 100, 0 ) would cause the center of the node (in the parent coordinate frame) to be at (200,0).
     * translate( 100, 0, true ) would cause the center of the node (in the parent coordinate frame) to be at (100,0).
     *
     * Allowed call signatures:
     * translate( x {number}, y {number} )
     * translate( x {number}, y {number}, prependInstead {boolean} )
     * translate( vector {Vector2} )
     * translate( vector {Vector2}, prependInstead {boolean} )
     *
     * @param {number} x - The x coordinate
     * @param {number} y - The y coordinate
     * @param {Vector2} vector - If present, the y coordinate (required if x is a number)
     * @param {boolean} [prependInstead] - Whether the transform should be prepended (defaults to false)
     */
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

    /**
     * Scales the node's transform. The default "appends" the transform, so that it will
     * appear to happen to the node before the rest of the transform would apply, but if "prepended", the rest of the
     * transform would apply first.
     * @public
     *
     * As an example, if a Node is translated to (100,0):
     * scale( 2 ) will leave the node translated at (100,0), but it will be twice as big around its origin at that location.
     * scale( 2, true ) will shift the node to (200,0).
     *
     * Allowed call signatures:
     * scale( s {number} )
     * scale( s {number}, prependInstead {boolean} )
     * scale( sx {number}, sy {number} )
     * scale( sx {number}, sy {number}, prependInstead {boolean} )
     *
     * @param {number} s - Scales in both the X and Y directions
     * @param {number} sx - Scales in the X direction
     * @param {number} sy - Scales in the Y direction
     * @param {boolean} [prependInstead] - Whether the transform should be prepended (defaults to false)
     */
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

    /**
     * Rotates the node's transform. The default "appends" the transform, so that it will
     * appear to happen to the node before the rest of the transform would apply, but if "prepended", the rest of the
     * transform would apply first.
     * @public
     *
     * As an example, if a Node is translated to (100,0):
     * rotate( Math.PI ) will rotate the node around (100,0)
     * rotate( Math.PI, true ) will rotate the node around the origin, moving it to (-100,0)
     *
     * @param {number} angle - The angle (in radians) to rotate by
     * @param {boolean} [prependInstead] - Whether the transform should be prepended (defaults to false)
     */
    rotate: function( angle, prependInstead ) {
      if ( angle % ( 2 * Math.PI ) === 0 ) { return; } // bail out if our angle is effectively 0
      if ( prependInstead ) {
        this.prependMatrix( Matrix3.rotation2( angle ) );
      }
      else {
        this.appendMatrix( Matrix3.rotation2( angle ) );
      }
    },

    /**
     * Rotates the node's transform around a specific point (in the parent coordinate frame) by prepending the transform.
     * @public
     *
     * @param {Vector2} point - In the parent coordinate frame
     * @param {number} angle - In radians
     *
     * TODO: determine whether this should use the appendMatrix method
     */
    rotateAround: function( point, angle ) {
      var matrix = Matrix3.translation( -point.x, -point.y );
      matrix = Matrix3.rotation2( angle ).timesMatrix( matrix );
      matrix = Matrix3.translation( point.x, point.y ).timesMatrix( matrix );
      this.prependMatrix( matrix );
    },

    /**
     * Shifts the x coordinate (in the parent coordinate frame) of where the node's origin is transformed to.
     * @public
     *
     * @param {number} x
     */
    setX: function( x ) {
      assert && assert( typeof x === 'number' );

      this.translate( x - this.getX(), 0, true );
      return this;
    },
    set x( value ) { this.setX( value ); },

    /**
     * Returns the x coordinate (in the parent coorindate frame) of where the node's origin is transformed to.
     * @public
     *
     * @returns {number}
     */
    getX: function() {
      return this._transform.getMatrix().m02();
    },
    get x() { return this.getX(); },

    /**
     * Shifts the y coordinate (in the parent coordinate frame) of where the node's origin is transformed to.
     * @public
     *
     * @param {number} y
     */
    setY: function( y ) {
      assert && assert( typeof y === 'number' );

      this.translate( 0, y - this.getY(), true );
      return this;
    },
    set y( value ) { this.setY( value ); },

    /**
     * Returns the y coordinate (in the parent coorindate frame) of where the node's origin is transformed to.
     * @public
     *
     * @returns {number}
     */
    getY: function() {
      return this._transform.getMatrix().m12();
    },
    get y() { return this.getY(); },

    /**
     * Typically without rotations or negative parameters, this sets the scale for each axis. In its more general form,
     * it modifies the node's transform so that:
     * - Transforming (1,0) with our transform will result in a vector with magnitude abs( x-scale-magnitude )
     * - Transforming (0,1) with our transform will result in a vector with magnitude abs( y-scale-magnitude )
     * - If parameters are negative, it will flip orientation in that direct.
     * @public
     *
     * Allowed call signatures:
     * setScaleMagnitude( s )
     * setScaleMagnitude( sx, sy )
     * setScaleMagnitude( vector )
     *
     * @param {number} s - Scale for both axes
     * @param {number} sx - Scale for the X axis
     * @param {number} sy - Scale for the Y axis
     * @param {Vector2} vector - Scale for the x/y axes in the vector's components.
     */
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

    /**
     * Returns a vector with an entry for each axis, e.g. (5,2) for an affine matrix with rows ((5,0,0),(0,2,0),(0,0,1)).
     * @public
     *
     * It is equivalent to:
     * ( T(1,0).magnitude(), T(0,1).magnitude() ) where T() transforms points with our transform.
     *
     * @returns {Vector2}
     */
    getScaleVector: function() {
      return this._transform.getMatrix().getScaleVector();
    },

    /**
     * Rotates this node's transform so that a unit (1,0) vector would be rotated by this node's transform by the
     * specified amount.
     * @public
     *
     * @param {number} rotation - In radians
     */
    setRotation: function( rotation ) {
      assert && assert( typeof rotation === 'number' );

      this.appendMatrix( Matrix3.rotation2( rotation - this.getRotation() ) );
      return this;
    },
    set rotation( value ) { this.setRotation( value ); },

    /**
     * Returns the rotation (in radians) that would be applied to a unit (1,0) vector when transformed with this Node's
     * transform.
     * @public
     *
     * @returns {number}
     */
    getRotation: function() {
      return this._transform.getMatrix().getRotation();
    },
    get rotation() { return this.getRotation(); },

    /**
     * Modifies the translation of this Node's transform so that the node's local-coordinate origin will be transformed
     * to the passed-in x/y.
     * @public
     *
     * Allowed call signatures:
     * setTranslation( x, y )
     * setTranslation( vector )
     *
     * @param {number} x - X translation
     * @param {number} y - Y translation
     * @param {Vector2} vector - Vector with x/y translation in components
     */
    setTranslation: function( a, b ) {
      var m = this._transform.getMatrix();
      var tx = m.m02();
      var ty = m.m12();

      var dx;
      var dy;

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
    set translation( value ) { this.setTranslation( value ); },

    /**
     * Returns a vector of where this Node's local-coordinate origin will be transformed by it's own transform.
     * @public
     *
     * @returns {Vector2}
     */
    getTranslation: function() {
      var matrix = this._transform.getMatrix();
      return new Vector2( matrix.m02(), matrix.m12() );
    },
    get translation() { return this.getTranslation(); },

    /**
     * Appends a transformation matrix to this Node's transform. Appending means this transform is conceptually applied
     * first before the rest of the Node's current transform (i.e. applied in the local coordinate frame).
     * @public
     *
     * @param {Matrix3} matrix
     */
    appendMatrix: function( matrix ) {
      this._transform.append( matrix );
    },

    /**
     * Prepends a transformation matrix to this Node's transform. Prepending means this transform is conceptually applied
     * after the rest of the Node's current transform (i.e. applied in the parent coordinate frame).
     * @public
     *
     * @param {Matrix3} matrix
     */
    prependMatrix: function( matrix ) {
      this._transform.prepend( matrix );
    },

    /**
     * Prepends an (x,y) translation to our Node's transform in an efficient manner without allocating a matrix.
     * see https://github.com/phetsims/scenery/issues/119
     * @public
     *
     * @param {number} x
     * @param {number} y
     */
    prependTranslation: function( x, y ) {
      assert && assert( typeof x === 'number', 'x not a number' );
      assert && assert( typeof y === 'number', 'y not a number' );
      assert && assert( isFinite( x ), 'x not finite' );
      assert && assert( isFinite( y ), 'y not finite' );

      if ( !x && !y ) { return; } // bail out if both are zero

      this._transform.prependTranslation( x, y );
    },

    /**
     * Changes this Node's transform to match the passed-in transformation matrix.
     * @public
     *
     * @param {Matrix3} matrix
     */
    setMatrix: function( matrix ) {
      this._transform.setMatrix( matrix );
    },
    set matrix( value ) { this.setMatrix( value ); },

    /**
     * Returns a Matrix3 representing our Node's transform.
     * @public
     *
     * NOTE: Do not mutate the returned matrix.
     *
     * @returns {Matrix3}
     */
    getMatrix: function() {
      return this._transform.getMatrix();
    },
    get matrix() { return this.getMatrix(); },

    /**
     * Returns a reference to our Node's transform
     * @public
     *
     * @returns {Transform3}
     */
    getTransform: function() {
      // for now, return an actual copy. we can consider listening to changes in the future
      return this._transform;
    },
    get transform() { return this.getTransform(); },

    /**
     * Resets our Node's transform to an identity transform (i.e. no transform is applied).
     * @public
     */
    resetTransform: function() {
      this.setMatrix( Matrix3.IDENTITY );
    },

    /**
     * Callback function that should be called when our transform is changed.
     * @private
     */
    onTransformChange: function() {
      // NOTE: why is local bounds invalidation needed here?
      this.invalidateBounds();

      this.trigger0( 'transform' );
    },

    /**
     * Updates our node's scale and applied scale factor if we need to change our scale to fit within the maximum
     * dimensions (maxWidth and maxHeight). See documentation in constructor for detailed behavior.
     * @private
     *
     * @param {Bounds2} localBounds
     */
    updateMaxDimension: function( localBounds ) {
      var currentScale = this._appliedScaleFactor;
      var idealScale = 1;

      if ( this._maxWidth !== null ) {
        var width = localBounds.width;
        if ( width > this._maxWidth ) {
          idealScale = Math.min( idealScale, this._maxWidth / width );
        }
      }

      if ( this._maxHeight !== null ) {
        var height = localBounds.height;
        if ( height > this._maxHeight ) {
          idealScale = Math.min( idealScale, this._maxHeight / height );
        }
      }

      var scaleAdjustment = idealScale / currentScale;
      if ( scaleAdjustment !== 1 ) {
        this.scale( scaleAdjustment );

        this._appliedScaleFactor = idealScale;
      }
    },

    /**
     * Increments/decrements bounds "listener" count based on the values of maxWidth/maxHeight before and after.
     * null is like no listener, non-null is like having a listener, so we increment for null => non-null, and
     * decrement for non-null => null.
     * @private
     *
     * @param {null | number} beforeMaxLength
     * @param {null | number} afterMaxLength
     */
    onMaxDimensionChange: function( beforeMaxLength, afterMaxLength ) {
      if ( beforeMaxLength === null && afterMaxLength !== null ) {
        this.changeBoundsEventCount( 1 );
        this._boundsEventSelfCount++;
      }
      else if ( beforeMaxLength !== null && afterMaxLength === null ) {
        this.changeBoundsEventCount( -1 );
        this._boundsEventSelfCount--;
      }
    },

    /**
     * Sets the maximum width of the Node (see constructor for documentation on how maximum dimensions work).
     * @public
     *
     * @param {number|null} maxWidth
     */
    setMaxWidth: function( maxWidth ) {
      assert && assert( maxWidth === null || typeof maxWidth === 'number',
        'maxWidth should be null (no constraint) or a number' );

      if ( this._maxWidth !== maxWidth ) {
        // update synthetic bounds listener count (to ensure our bounds are validated at the start of updateDisplay)
        this.onMaxDimensionChange( this._maxWidth, maxWidth );

        this._maxWidth = maxWidth;

        this.updateMaxDimension( this._localBounds );
      }
    },
    set maxWidth( value ) { this.setMaxWidth( value ); },

    /**
     * Returns the maximum width (if any) of the Node.
     * @public
     *
     * @returns {number|null}
     */
    getMaxWidth: function() {
      return this._maxWidth;
    },
    get maxWidth() { return this.getMaxWidth(); },

    /**
     * Sets the maximum height of the Node (see constructor for documentation on how maximum dimensions work).
     * @public
     *
     * @param {number|null} maxHeight
     */
    setMaxHeight: function( maxHeight ) {
      assert && assert( maxHeight === null || typeof maxHeight === 'number',
        'maxHeight should be null (no constraint) or a number' );

      if ( this._maxHeight !== maxHeight ) {
        // update synthetic bounds listener count (to ensure our bounds are validated at the start of updateDisplay)
        this.onMaxDimensionChange( this._maxHeight, maxHeight );

        this._maxHeight = maxHeight;

        this.updateMaxDimension( this._localBounds );
      }
    },
    set maxHeight( value ) { this.setMaxHeight( value ); },

    /**
     * Returns the maximum height (if any) of the Node.
     * @public
     *
     * @returns {number|null}
     */
    getMaxHeight: function() {
      return this._maxHeight;
    },
    get maxHeight() { return this.getMaxHeight(); },

    /**
     * Shifts this node horizontally so that its left bound (in the parent coordinate frame) is equal to the passed-in
     * 'left' X value.
     * @public
     *
     * @param {number} left
     */
    setLeft: function( left ) {
      assert && assert( typeof left === 'number' );

      this.translate( left - this.getLeft(), 0, true );
      return this; // allow chaining
    },
    set left( value ) { this.setLeft( value ); },

    /**
     * Returns the X value of the left side of the bounding box of this node (in the parent coordinate frame).
     * @public
     *
     * @returns {number}
     */
    getLeft: function() {
      return this.getBounds().minX;
    },
    get left() { return this.getLeft(); },

    /**
     * Shifts this node horizontally so that its right bound (in the parent coordinate frame) is equal to the passed-in
     * 'right' X value.
     * @public
     *
     * @param {number} right
     */
    setRight: function( right ) {
      assert && assert( typeof right === 'number' );

      this.translate( right - this.getRight(), 0, true );
      return this; // allow chaining
    },
    set right( value ) { this.setRight( value ); },

    /**
     * Returns the X value of the right side of the bounding box of this node (in the parent coordinate frame).
     * @public
     *
     * @returns {number}
     */
    getRight: function() {
      return this.getBounds().maxX;
    },
    get right() { return this.getRight(); },

    /**
     * Shifts this node horizontally so that its horizontal center (in the parent coordinate frame) is equal to the
     * passed-in center X value.
     * @public
     *
     * @param {number} x
     */
    setCenterX: function( x ) {
      assert && assert( typeof x === 'number' );

      this.translate( x - this.getCenterX(), 0, true );
      return this; // allow chaining
    },
    set centerX( value ) { this.setCenterX( value ); },

    /**
     * Returns the X value of this node's horizontal center (in the parent coordinate frame)
     * @public
     *
     * @returns {number}
     */
    getCenterX: function() {
      return this.getBounds().getCenterX();
    },
    get centerX() { return this.getCenterX(); },

    /**
     * Shifts this node vertically so that its vertical center (in the parent coordinate frame) is equal to the
     * passed-in center Y value.
     * @public
     *
     * @param {number} y
     */
    setCenterY: function( y ) {
      assert && assert( typeof y === 'number' );

      this.translate( 0, y - this.getCenterY(), true );
      return this; // allow chaining
    },
    set centerY( value ) { this.setCenterY( value ); },

    /**
     * Returns the Y value of this node's vertical center (in the parent coordinate frame)
     * @public
     *
     * @returns {number}
     */
    getCenterY: function() {
      return this.getBounds().getCenterY();
    },
    get centerY() { return this.getCenterY(); },

    /**
     * Shifts this node vertically so that its top (in the parent coordinate frame) is equal to the passed-in Y value.
     * @public
     *
     * NOTE: top is the lowest Y value in our bounds.
     *
     * @param {number} top
     */
    setTop: function( top ) {
      assert && assert( typeof top === 'number' );

      this.translate( 0, top - this.getTop(), true );
      return this; // allow chaining
    },
    set top( value ) { this.setTop( value ); },

    /**
     * Returns the lowest Y value of this node's bounding box (in the parent coordinate frame).
     * @public
     *
     * @returns {number}
     */
    getTop: function() {
      return this.getBounds().minY;
    },
    get top() { return this.getTop(); },

    /**
     * Shifts this node vertically so that its bottom (in the parent coordinate frame) is equal to the passed-in Y value.
     * @public
     *
     * NOTE: top is the highest Y value in our bounds.
     *
     * @param {number} top
     */
    setBottom: function( bottom ) {
      assert && assert( typeof bottom === 'number' );

      this.translate( 0, bottom - this.getBottom(), true );
      return this; // allow chaining
    },
    set bottom( value ) { this.setBottom( value ); },

    /**
     * Returns the highest Y value of this node's bounding box (in the parent coordinate frame).
     * @public
     *
     * @returns {number}
     */
    getBottom: function() {
      return this.getBounds().maxY;
    },
    get bottom() { return this.getBottom(); },

    /**
     * Returns the width of this node's bounding box (in the parent coordinate frame).
     * @public
     *
     * @returns {number}
     */
    getWidth: function() {
      return this.getBounds().getWidth();
    },
    get width() { return this.getWidth(); },

    /**
     * Returns the height of this node's bounding box (in the parent coordinate frame).
     * @public
     *
     * @returns {number}
     */
    getHeight: function() {
      return this.getBounds().getHeight();
    },
    get height() { return this.getHeight(); },

    /**
     * Returns the unique integral ID for this node.
     * @public
     *
     * @returns {number}
     */
    getId: function() {
      return this._id;
    },
    get id() { return this.getId(); },

    /**
     * Sets whether this node is visible.
     * @public
     *
     * @param {boolean} visible
     */
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
    set visible( value ) { this.setVisible( value ); },

    /**
     * Returns whether this node is visible.
     * @public
     *
     * @returns {boolean}
     */
    isVisible: function() {
      return this._visible;
    },
    get visible() { return this.isVisible(); },

    /**
     * Sets the opacity of this node (and its sub-tree), where 0 is fully transparent, and 1 is fully opaque.
     * @public
     *
     * @param {number} opacity
     */
    setOpacity: function( opacity ) {
      assert && assert( typeof opacity === 'number' );

      var clampedOpacity = clamp( opacity, 0, 1 );
      if ( clampedOpacity !== this._opacity ) {
        this._opacity = clampedOpacity;

        this.trigger0( 'opacity' );
      }
    },
    set opacity( value ) { this.setOpacity( value ); },

    /**
     * Returns the opacity of this node.
     * @public
     *
     * @returns {number}
     */
    getOpacity: function() {
      return this._opacity;
    },
    get opacity() { return this.getOpacity(); },

    /**
     * Sets the pickability of this node (see the constructor for more detailed documentation).
     * @public
     *
     * @param {boolean|null} pickable
     */
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
    set pickable( value ) { this.setPickable( value ); },

    /**
     * Returns the pickability of this node.
     * @public
     *
     * @returns {boolean|null}
     */
    isPickable: function() {
      return this._pickable;
    },
    get pickable() { return this.isPickable(); },

    /**
     * Sets the CSS cursor string that should be used when the mouse is over this node. null is the default, and
     * indicates that ancestor nodes (or the browser default) should be used.
     * @public
     *
     * @param {string} cursor - A CSS cursor string, like 'pointer', or 'none'
     */
    setCursor: function( cursor ) {
      assert && assert( typeof cursor === 'string' || cursor === null );

      // TODO: consider a mapping of types to set reasonable defaults
      /*
       auto default none inherit help pointer progress wait crosshair text vertical-text alias copy move no-drop not-allowed
       e-resize n-resize w-resize s-resize nw-resize ne-resize se-resize sw-resize ew-resize ns-resize nesw-resize nwse-resize
       context-menu cell col-resize row-resize all-scroll url( ... ) --> does it support data URLs?
       */

      // allow the 'auto' cursor type to let the ancestors or scene pick the cursor type
      this._cursor = cursor === 'auto' ? null : cursor;
    },
    set cursor( value ) { this.setCursor( value ); },

    /**
     * Returns the CSS cursor string for this node.
     * @public
     *
     * @returns {string}
     */
    getCursor: function() {
      return this._cursor;
    },
    get cursor() { return this.getCursor(); },

    /**
     * Sets the hit-tested mouse area for this node (see constructor for more advanced documentation). Use null for the
     * default behavior.
     * @public
     *
     * @param {Bounds2|Shape|null} area
     */
    setMouseArea: function( area ) {
      assert && assert( area === null || area instanceof Shape || area instanceof Bounds2, 'mouseArea needs to be a kite.Shape, dot.Bounds2, or null' );

      if ( this._mouseArea !== area ) {
        this._mouseArea = area; // TODO: could change what is under the mouse, invalidate!

        this.invalidateMouseTouchBounds();
      }
    },
    set mouseArea( value ) { this.setMouseArea( value ); },

    /**
     * Returns the hit-tested mouse area for this node.
     * @public
     *
     * @returns {Bounds2|Shape|null}
     */
    getMouseArea: function() {
      return this._mouseArea;
    },
    get mouseArea() { return this.getMouseArea(); },

    /**
     * Sets the hit-tested touch area for this node (see constructor for more advanced documentation). Use null for the
     * default behavior.
     * @public
     *
     * @param {Bounds2|Shape|null} area
     */
    setTouchArea: function( area ) {
      assert && assert( area === null || area instanceof Shape || area instanceof Bounds2, 'touchArea needs to be a kite.Shape, dot.Bounds2, or null' );

      if ( this._touchArea !== area ) {
        this._touchArea = area; // TODO: could change what is under the touch, invalidate!

        this.invalidateMouseTouchBounds();
      }
    },
    set touchArea( value ) { this.setTouchArea( value ); },

    /**
     * Returns the hit-tested touch area for this node.
     * @public
     *
     * @returns {Bounds2|Shape|null}
     */
    getTouchArea: function() {
      return this._touchArea;
    },
    get touchArea() { return this.getTouchArea(); },

    /**
     * Sets a clipped shape where only content in our local coordinate frame that is inside the clip area will be shown
     * (anything outside is fully transparent).
     * @public
     *
     * @param {Shape|null} shape
     */
    setClipArea: function( shape ) {
      assert && assert( shape === null || shape instanceof Shape, 'clipArea needs to be a kite.Shape, or null' );

      if ( this._clipArea !== shape ) {
        this._clipArea = shape;

        this.trigger0( 'clip' );

        this.invalidateBounds();
      }
    },
    set clipArea( value ) { this.setClipArea( value ); },

    /**
     * Returns the clipped area for this node.
     * @public
     *
     * @returns {Shape|null}
     */
    getClipArea: function() {
      return this._clipArea;
    },
    get clipArea() { return this.getClipArea(); },

    /**
     * Returns whether this node has a clip area.
     * @public
     *
     * @returns {boolean}
     */
    hasClipArea: function() {
      return this._clipArea !== null;
    },

    /**
     * Sets whether this node should be focusable for keyboard support.
     * @public
     *
     * @param {boolean} focusable
     */
    setFocusable: function( focusable ) {
      if ( this._focusable !== focusable ) {
        this._focusable = focusable;

        this.trigger0( 'focusable' );
      }
    },
    set focusable( value ) { this.setFocusable( value ); },

    /**
     * Returns whether this node is focusable.
     * @public
     *
     * @returns {boolean}
     */
    getFocusable: function() {
      return this._focusable;
    },
    get focusable() { return this.getFocusable(); },

    /**
     * Sets the focus indicator for the node ('rectangle' or 'cursor').
     * @public
     *
     * @param {string} focusIndicator
     */
    setFocusIndicator: function( focusIndicator ) {
      if ( this._focusIndicator !== focusIndicator ) {
        this._focusIndicator = focusIndicator;

        this.trigger0( 'focusIndicator' );
      }
    },
    set focusIndicator( value ) { this.setFocusIndicator( value ); },

    /**
     * Returns the focus indicator string for this node.
     * @public
     *
     * @returns {string}
     */
    getFocusIndicator: function() {
      return this._focusIndicator;
    },
    get focusIndicator() { return this.getFocusIndicator(); },

    /**
     * Sets the accessible focus order for this node. This includes not only focussed items, but elements that can be
     * placed in the parallel DOM. If provided, it will override the focus order between children (and
     * optionally descendants). If not provided, the focus order will default to the rendering order (first children
     * first, last children last), determined by the children array.
     * @public
     *
     * @param {Array.<Node>|null} accessibleOrder
     */
    setAccessibleOrder: function( accessibleOrder ) {
      assert && assert( accessibleOrder === null || accessibleOrder instanceof Array );

      if ( this._accessibleOrder !== accessibleOrder ) {
        this._accessibleOrder = accessibleOrder;

        var trails = this.getTrails( hasRootedDisplayPredicate );
        for ( var i = 0; i < trails.length; i++ ) {
          var trail = trails[ i ];
          var rootedDisplays = trail.rootNode()._rootedDisplays;
          for ( var j = 0; j < rootedDisplays.length; j++ ) {
            rootedDisplays[ j ].changedAccessibleOrder( trail );
          }
        }

        this.trigger0( 'accessibleOrder' );
      }
    },
    set accessibleOrder( value ) { this.setAccessibleOrder( value ); },

    /**
     * Returns the accessible (focus) order for this node.
     * @public
     *
     * @returns {Array.<Node>|null}
     */
    getAccessibleOrder: function() {
      return this._accessibleOrder;
    },
    get accessibleOrder() { return this.getAccessibleOrder(); },

    /**
     * Sets the accessible content for a Node. See constructor for more information.
     * @public
     *
     * @param {null|Object} accessibleContent
     */
    setAccessibleContent: function( accessibleContent ) {
      assert && assert( accessibleContent === null || accessibleContent instanceof Object );

      if ( this._accessibleContent !== accessibleContent ) {
        var oldAccessibleContent = this._accessibleContent;
        this._accessibleContent = accessibleContent;

        var trails = this.getTrails( hasRootedDisplayPredicate );
        for ( var i = 0; i < trails.length; i++ ) {
          var trail = trails[ i ];
          var rootedDisplays = trail.rootNode()._rootedDisplays;
          for ( var j = 0; j < rootedDisplays.length; j++ ) {
            rootedDisplays[ j ].changedAccessibleContent( trail, oldAccessibleContent, accessibleContent );
          }
        }

        this.trigger0( 'accessibleContent' );
      }
    },
    set accessibleContent( value ) { this.setAccessibleContent( value ); },

    /**
     * Returns the accessible content for this node.
     * @public
     *
     * @returns {Array.<Node>|null}
     */
    getAccessibleContent: function() {
      return this._accessibleContent;
    },
    get accessibleContent() { return this.getAccessibleContent(); },

    // @deprecated
    supportsCanvas: function() {
      return ( this._rendererBitmask & Renderer.bitmaskCanvas ) !== 0;
    },

    // @deprecated
    supportsSVG: function() {
      return ( this._rendererBitmask & Renderer.bitmaskSVG ) !== 0;
    },

    // @deprecated
    supportsDOM: function() {
      return ( this._rendererBitmask & Renderer.bitmaskDOM ) !== 0;
    },

    // @deprecated
    supportsWebGL: function() {
      return ( this._rendererBitmask & Renderer.bitmaskWebGL ) !== 0;
    },

    // @deprecated
    supportsRenderer: function( renderer ) {
      return ( this._rendererBitmask & renderer.bitmask ) !== 0;
    },

    /**
     * Sets what self renderers (and other bitmask flags) are supported by this node.
     * @protected
     *
     * @param {number} bitmask
     */
    setRendererBitmask: function( bitmask ) {
      if ( bitmask !== this._rendererBitmask ) {
        this._rendererBitmask = bitmask;

        this._rendererSummary.selfChange();
        this.trigger0( 'rendererBitmask' );
      }
    },

    /**
     * Meant to be overridden, so that it can be called to ensure that the renderer bitmask will be up-to-date.
     * @protected
     */
    invalidateSupportedRenderers: function() {

    },

    /*---------------------------------------------------------------------------*
     * Hints
     *----------------------------------------------------------------------------*/

    /**
     * Sets a preferred renderer for this node and its sub-tree. Scenery will attempt to use this renderer under here
     * unless it isn't supported, OR another preferred renderer is set as a closer ancestor. Acceptable values are:
     * - null (default, no preference)
     * - 'canvas'
     * - 'svg'
     * - 'dom'
     * - 'webgl'
     * @public
     *
     * @param {string|null} renderer
     */
    setRenderer: function( renderer ) {
      assert && assert( renderer === null || renderer === 'canvas' || renderer === 'svg' || renderer === 'dom' || renderer === 'webgl',
        'Renderer input should be null, or one of: "canvas", "svg", "dom" or "webgl".' );

      var newRenderer = 0;
      if ( renderer === 'canvas' ) {
        newRenderer = Renderer.bitmaskCanvas;
      }
      else if ( renderer === 'svg' ) {
        newRenderer = Renderer.bitmaskSVG;
      }
      else if ( renderer === 'dom' ) {
        newRenderer = Renderer.bitmaskDOM;
      }
      else if ( renderer === 'webgl' ) {
        newRenderer = Renderer.bitmaskWebGL;
      }
      assert && assert( ( renderer === null ) === ( newRenderer === 0 ),
        'We should only end up with no actual renderer if renderer is null' );

      if ( this._hints.renderer !== newRenderer ) {
        this._hints.renderer = newRenderer;

        this.trigger1( 'hint', 'renderer' );
      }
    },
    set renderer( value ) { this.setRenderer( value ); },

    /**
     * Returns the preferred renderer (if any) of this node, as a string.
     * @public
     *
     * @returns {string|null}
     */
    getRenderer: function() {
      if ( this._hints.renderer === 0 ) {
        return null;
      }
      else if ( this._hints.renderer === Renderer.bitmaskCanvas ) {
        return 'canvas';
      }
      else if ( this._hints.renderer === Renderer.bitmaskSVG ) {
        return 'svg';
      }
      else if ( this._hints.renderer === Renderer.bitmaskDOM ) {
        return 'dom';
      }
      else if ( this._hints.renderer === Renderer.bitmaskWebGL ) {
        return 'webgl';
      }
      assert && assert( false, 'Seems to be an invalid renderer?' );
      return this._hints.renderer;
    },
    get renderer() { return this.getRenderer(); },

    /**
     * Returns whether there is a preferred renderer for this node.
     * @public
     *
     * @returns {boolean}
     */
    hasRenderer: function() {
      return !!this._hints.renderer;
    },

    // @deprecated
    setRendererOptions: function( options ) {
      // TODO: consider checking options based on the specified 'renderer'?
      // TODO: consider a guard where we check if anything changed
      //OHTWO TODO: Split out all of the renderer options into individual flag ES5'ed getter/setters
      _.extend( this._hints, options );

      this.trigger0( 'hint' );
    },
    set rendererOptions( value ) { this.setRendererOptions( value ); },

    // @deprecated
    getRendererOptions: function() {
      return this._hints;
    },
    get rendererOptions() { return this.getRendererOptions(); },

    /**
     * Sets whether or not Scenery will try to put this node (and its descendants) into a separate SVG/Canvas/WebGL/etc.
     * layer, different from other siblings or other nodes. Can be used for performance purposes.
     * @public
     *
     * @param {boolean} split
     */
    setLayerSplit: function( split ) {
      assert && assert( typeof split === 'boolean' );

      if ( split !== this._hints.layerSplit ) {
        this._hints.layerSplit = split;
        this.trigger1( 'hint', 'layerSplit' );
      }
    },
    set layerSplit( value ) { this.setLayerSplit( value ); },

    /**
     * Returns whether the layerSplit performance flag is set.
     * @public
     *
     * @returns {boolean}
     */
    isLayerSplit: function() {
      return this._hints.layerSplit;
    },
    get layerSplit() { return this.isLayerSplit(); },

    /**
     * Sets whether or not Scenery will take into account that this Node plans to use opacity. Can have performance
     * gains if there need to be multiple layers for this node's descendants.
     * @public
     *
     * @param {boolean} usesOpacity
     */
    setUsesOpacity: function( usesOpacity ) {
      assert && assert( typeof usesOpacity === 'boolean' );

      if ( usesOpacity !== this._hints.usesOpacity ) {
        this._hints.usesOpacity = usesOpacity;
        this.trigger1( 'hint', 'usesOpacity' );
      }
    },
    set usesOpacity( value ) { this.setUsesOpacity( value ); },

    /**
     * Returns whether the usesOpacity performance flag is set.
     * @public
     *
     * @returns {boolean}
     */
    getUsesOpacity: function() {
      return this._hints.usesOpacity;
    },
    get usesOpacity() { return this.getUsesOpacity(); },

    /**
     * Sets a flag for whether whether the contents of this Node and its children should be displayed in a separate
     * DOM element that is transformed with CSS transforms. It can have potential speedups, since the browser may not
     * have to rerasterize contents when it is animated.
     * @public
     *
     * @param {boolean} cssTransform
     */
    setCSSTransform: function( cssTransform ) {
      assert && assert( typeof cssTransform === 'boolean' );

      if ( cssTransform !== this._hints.cssTransform ) {
        this._hints.cssTransform = cssTransform;
        this.trigger1( 'hint', 'cssTransform' );
      }
    },
    set cssTransform( value ) { this.setCSSTransform( value ); },

    /**
     * Returns wehther the cssTransform performance flag is set.
     * @public
     *
     * @returns {boolean}
     */
    isCSSTransformed: function() {
      return this._hints.cssTransform;
    },
    get cssTransform() { return this._hints.cssTransform; },

    /**
     * Sets a performance flag for whether layers/DOM elements should be excluded (or included) when things are
     * invisible. The default is false, and invisible content is in the DOM, but hidden.
     * @public
     *
     * @param {boolean} excludeInvisible
     */
    setExcludeInvisible: function( excludeInvisible ) {
      assert && assert( typeof excludeInvisible === 'boolean' );

      if ( excludeInvisible !== this._hints.excludeInvisible ) {
        this._hints.excludeInvisible = excludeInvisible;
        this.trigger1( 'hint', 'excludeInvisible' );
      }
    },
    set excludeInvisible( value ) { this.setExcludeInvisible( value ); },

    /**
     * Returns whether the excludeInvisible performance flag is set.
     * @public
     *
     * @returns {boolean}
     */
    isExcludeInvisible: function() {
      return this._hints.excludeInvisible;
    },
    get excludeInvisible() { return this.isExcludeInvisible(); },

    /**
     * Sets whether there is a custom WebGL scale applied to the Canvas, and if so what scale.
     * @public
     *
     * @param {number|null} webglScale
     */
    setWebGLScale: function( webglScale ) {
      assert && assert( webglScale === null || typeof webglScale === 'number' );

      if ( webglScale !== this._hints.webglScale ) {
        this._hints.webglScale = webglScale;
        this.trigger1( 'hint', 'webglScale' );
      }
    },
    set webglScale( value ) { this.setWebGLScale( value ); },

    /**
     * Returns the value of the webglScale performance flag.
     * @public
     *
     * @returns {number|null}
     */
    getWebGLScale: function() {
      return this._hints.webglScale;
    },
    get webglScale() { return this.getWebGLScale(); },

    /*---------------------------------------------------------------------------*
     * Trail operations
     *----------------------------------------------------------------------------*/

    /**
     * Returns the one Trail that starts from a node with no parents (or if the predicate is present, a node that
     * satisfies it), and ends at this node. If more than one Trail would satisfy these conditions, an assertion is
     * thrown (please use getTrails() for those cases).
     * @public
     *
     * @param {function( node ) : boolean} [predicate] - If supplied, we will only return trails rooted at a node that
     *                                                   satisfies predicate( node ) == true
     * @returns {Trail}
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

        return trails[ 0 ];
      }
    },

    /**
     * Returns a Trail rooted at rootNode and ends at this node. Throws an assertion if the number of trails that match
     * this condition isn't exactly 1.
     * @public
     *
     * @param {Node} rootNode
     * @returns {Trail}
     */
    getUniqueTrailTo: function( rootNode ) {
      return this.getUniqueTrail( function( node ) {
        return rootNode === node;
      } );
    },

    /**
     * Returns an array of all Trails that start from nodes with no parent (or if a predicate is present, those that
     * satisfy the predicate), and ends at this node.
     * @public
     *
     * @param {function( node ) : boolean} [predicate] - If supplied, we will only return Trails rooted at nodes that
     *                                                   satisfy predicate( node ) == true.
     * @returns {Array.<Trail>}
     */
    getTrails: function( predicate ) {
      predicate = predicate || defaultTrailPredicate;

      var trails = [];
      var trail = new scenery.Trail( this );
      scenery.Trail.appendAncestorTrailsWithPredicate( trails, trail, predicate );

      return trails;
    },

    /**
     * Returns an array of all Trails rooted at rootNode and end at this node.
     * @public

     * @param {Node} rootNode
     * @returns {Array.<Trail>}
     */
    getTrailsTo: function( rootNode ) {
      return this.getTrails( function( node ) {
        return node === rootNode;
      } );
    },

    /**
     * Returns an array of all Trails rooted at this node and end with nodes with no children (or if a predicate is
     * present, those that satisfy the predicate).
     * @public
     *
     * @param {function( node ) : boolean} [predicate] - If supplied, we will only return Trails ending at nodes that
     *                                                   satisfy predicate( node ) == true.
     * @returns {Array.<Trail>}
     */
    getLeafTrails: function( predicate ) {
      predicate = predicate || defaultLeafTrailPredicate;

      var trails = [];
      var trail = new scenery.Trail( this );
      scenery.Trail.appendDescendantTrailsWithPredicate( trails, trail, predicate );

      return trails;
    },

    /**
     * Returns an array of all Trails rooted at this node and end with leafNode.
     * @public
     *
     * @param {Node} leafNode
     * @returns {Array.<Trail>}
     */
    getLeafTrailsTo: function( leafNode ) {
      return this.getLeafTrails( function( node ) {
        return node === leafNode;
      } );
    },

    /**
     * Returns a Trail rooted at this node and ending at a node that has no children (or if a predicate is provided, a
     * node that satisfies the predicate). If more than one trail matches this description, an assertion will be fired.
     * @public
     *
     * @param {function( node ) : boolean} [predicate] - If supplied, we will return a Trail that ends with a node that
     *                                                   satisfies predicate( node ) == true
     * @returns {Trail}
     */
    getUniqueLeafTrail: function( predicate ) {
      var trails = this.getLeafTrails( predicate );

      assert && assert( trails.length === 1,
        'getUniqueLeafTrail found ' + trails.length + ' matching trails for the predicate' );

      return trails[ 0 ];
    },

    /**
     * Returns a Trail rooted at this node and ending at leafNode. If more than one trail matches this description,
     * an assertion will be fired.
     * @public
     *
     * @param {Node} leafNode
     * @returns {Trail}
     */
    getUniqueLeafTrailTo: function( leafNode ) {
      return this.getUniqueLeafTrail( function( node ) {
        return node === leafNode;
      } );
    },

    /*
     * Returns all nodes in the connected component, returned in an arbitrary order, including nodes that are ancestors
     * of this node.
     * @public
     *
     * @returns {Array.<Node>}
     */
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

    /**
     * Returns a recursive data structure that represents the nested ordering of accessible content for this Node's
     * subtree. Each "Item" will have the type { trail: {Trail}, children: {Array.<Item>} }, forming a tree-like
     * structure.
     * @public
     *
     * @returns {Array.<Item>}
     */
    getNestedAccessibleOrder: function() {
      var currentTrail = new scenery.Trail( this );
      var pruneStack = []; // {Array.<Node>} - A list of nodes to prune

      // {Array.<Item>} - The main result we will be returning. It is the top-level array where child items will be
      // inserted.
      var result = [];

      // {Array.<Array.<Item>>} A stack of children arrays, where we should be inserting items into the top array.
      // We will start out with the result, and as nested levels are added, the children arrays of those items will be
      // pushed and poppped, so that the top array on this stack is where we should insert our next child item.
      var nestedChildStack = [ result ];

      function addTrailsForNode( node, overridePruning ) {
        // If subtrees were specified with accessibleOrder, they should be skipped from the ordering of ancestor subtrees,
        // otherwise we could end up having multiple references to the same trail (which should be disallowed).
        var pruneCount = 0;
        // count the number of times our node appears in the pruneStack
        _.each( pruneStack, function( pruneNode ) {
          if ( node === pruneNode ) {
            pruneCount++;
          }
        } );

        // If overridePruning is set, we ignore one reference to our node in the prune stack. If there are two copies,
        // however, it means a node was specified in a accessibleOrder that already needs to be pruned (so we skip it instead
        // of creating duplicate references in the tab order).
        if ( pruneCount > 1 || ( pruneCount === 1 && !overridePruning ) ) {
          return;
        }

        // Pushing item and its children array, if accessible
        if ( node.accessibleContent ) {
          var item = {
            trail: currentTrail.copy(),
            children: []
          };
          nestedChildStack[ nestedChildStack.length - 1 ].push( item );
          nestedChildStack.push( item.children );
        }

        // Pushing pruned nodes to the stack (if ordered), AND visiting trails to ordered nodes.
        if ( node._accessibleOrder ) {
          // push specific focused nodes to the stack
          pruneStack = pruneStack.concat( node._accessibleOrder );

          _.each( node._accessibleOrder, function( descendant ) {
            // Find all descendant references to the node.
            // NOTE: We are not reordering trails (due to descendant constraints) if there is more than one instance for
            // this descendant node.
            _.each( node.getLeafTrailsTo( descendant ), function( descendantTrail ) {
              descendantTrail.removeAncestor(); // strip off 'node', so that we handle only children

              // same as the normal order, but adding a full trail (since we may be referencing a descendant node)
              currentTrail.addDescendantTrail( descendantTrail );
              addTrailsForNode( descendant, true ); // 'true' overrides one reference in the prune stack (added above)
              currentTrail.removeDescendantTrail( descendantTrail );
            } );
          } );
        }

        // Visit everything. If there is an accessibleOrder, those trails were already visited, and will be excluded.
        var numChildren = node._children.length;
        for ( var i = 0; i < numChildren; i++ ) {
          var child = node._children[ i ];

          currentTrail.addDescendant( child, i );
          addTrailsForNode( child, false );
          currentTrail.removeDescendant();
        }

        // Popping pruned nodes from the stack (if ordered)
        if ( node._accessibleOrder ) {
          // pop focused nodes from the stack (that were added above)
          _.each( node._accessibleOrder, function( descendant ) {
            pruneStack.pop();
          } );
        }

        // Popping children array if accessible
        if ( node.accessibleContent ) {
          nestedChildStack.pop();
        }
      }

      addTrailsForNode( this, false );

      return result;
    },

    /**
     * Returns all nodes that are connected to this node, sorted in topological order.
     * @public
     *
     * @returns {Array.<Node>}
     */
    getTopologicallySortedNodes: function() {
      // see http://en.wikipedia.org/wiki/Topological_sorting
      var edges = {};
      var s = [];
      var l = [];
      var n;
      _.each( this.getConnectedNodes(), function( node ) {
        edges[ node.id ] = {};
        _.each( node._children, function( m ) {
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

        _.each( n._children, handleChild );
      }

      // ensure that there are no edges left, since then it would contain a circular reference
      assert && assert( _.every( edges, function( children ) {
        return _.every( children, function( final ) { return false; } );
      } ), 'circular reference check' );

      return l;
    },

    /**
     * Returns whether this.addChild( child ) will not cause circular references.
     * @public
     *
     * @param {Node} child
     * @returns {boolean}
     */
    canAddChild: function( child ) {
      if ( this === child || _.contains( this._children, child ) ) {
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
        _.each( node._children, function( m ) {
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

        _.each( n._children, handleChild );

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

    /**
     * To be overridden in paintable node types. Should hook into the drawable's prototype (presumably).
     * @protected
     *
     * @param {CanvasContextWrapper} wrapper
     */
    canvasPaintSelf: function( wrapper ) {

    },

    /**
     * Renders this Node only (its self) into the Canvas wrapper, in its local coordinate frame.
     * @public
     *
     * @param {CanvasContextWrapper} wrapper
     */
    renderToCanvasSelf: function( wrapper ) {
      if ( this.isPainted() && ( this._rendererBitmask & Renderer.bitmaskCanvas ) ) {
        this.canvasPaintSelf( wrapper );
      }
    },

    /**
     * Renders this node and its descendants into the Canvas wrapper.
     * @public
     *
     * @param {CanvasContextWrapper} wrapper
     * @param {Matrix3} [matrix] - Optional transform to be applied
     */
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

    /**
     * @deprecated
     * Render this node to the Canvas (clearing it first)
     * @public
     *
     * @param {HTMLCanvasElement} canvas
     * @param {CanvasRenderingContext2D} context
     * @param {Function} callback - Called with no arguments
     * @param {string} [backgroundColor]
     */
    // @public @deprecated (API compatibility for now): Render this node to the Canvas (clearing it first)
    renderToCanvas: function( canvas, context, callback, backgroundColor ) {
      // should basically reset everything (and clear the Canvas)
      canvas.width = canvas.width;

      if ( backgroundColor ) {
        context.fillStyle = backgroundColor;
        context.fillRect( 0, 0, canvas.width, canvas.height );
      }

      var wrapper = new scenery.CanvasContextWrapper( canvas, context );

      this.renderToCanvasSubtree( wrapper, Matrix3.identity() );

      callback && callback(); // this was originally asynchronous, so we had a callback
    },

    /*
     * Renders this node to a canvas. If toCanvas( callback ) is used, the canvas will contain the node's
     * entire bounds (if no x/y/width/height is provided)
     * @public
     *
     * @param {Function} callback - callback( canvas, x, y ) is called, where x,y are computed if not specified.
     * @param {number} [x] - The X offset for where the upper-left of the content drawn into the Canvas
     * @param {number} [y] - The Y offset for where the upper-left of the content drawn into the Canvas
     * @param {number} [width] - The width of the Canvas output
     * @param {number} [height] - The height of the Canvas output
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

    /**
     * Renders this node to a Canvas, then calls the callback with the data URI from it.
     * @public
     *
     * @param {Function} callback - callback( dataURI {string}, x, y ) is called, where x,y are computed if not specified.
     * @param {number} [x] - The X offset for where the upper-left of the content drawn into the Canvas
     * @param {number} [y] - The Y offset for where the upper-left of the content drawn into the Canvas
     * @param {number} [width] - The width of the Canvas output
     * @param {number} [height] - The height of the Canvas output
     */
    toDataURL: function( callback, x, y, width, height ) {
      this.toCanvas( function( canvas, x, y ) {
        // this x and y shadow the outside parameters, and will be different if the outside parameters are undefined
        callback( canvas.toDataURL(), x, y );
      }, x, y, width, height );
    },

    /**
     * Calls the callback with an HTMLImageElement that contains this Node's subtree's visual form.
     * Will always be asynchronous.
     * @public
     *
     * @param {Function} callback - callback( image {HTMLImageElement}, x, y ) is called
     * @param {number} [x] - The X offset for where the upper-left of the content drawn into the Canvas
     * @param {number} [y] - The Y offset for where the upper-left of the content drawn into the Canvas
     * @param {number} [width] - The width of the Canvas output
     * @param {number} [height] - The height of the Canvas output
     */
    toImage: function( callback, x, y, width, height ) {
      this.toDataURL( function( url, x, y ) {
        // this x and y shadow the outside parameters, and will be different if the outside parameters are undefined
        var img = document.createElement( 'img' );
        img.onload = function() {
          callback( img, x, y );
          try {
            delete img.onload;
          }
          catch( e ) {
            // do nothing
          } // fails on Safari 5.1
        };
        img.src = url;
      }, x, y, width, height );
    },

    /**
     * Calls the callback with an Image node that contains this Node's subtree's visual form. This is always
     * asynchronous, but the resulting image node can be used with any back-end (Canvas/WebGL/SVG/etc.)
     * @public
     *
     * @param {Function} callback - callback( imageNode {Image} ) is called
     * @param {number} [x] - The X offset for where the upper-left of the content drawn into the Canvas
     * @param {number} [y] - The Y offset for where the upper-left of the content drawn into the Canvas
     * @param {number} [width] - The width of the Canvas output
     * @param {number} [height] - The height of the Canvas output
     */
    toImageNodeAsynchronous: function( callback, x, y, width, height ) {
      this.toImage( function( image, x, y ) {
        callback( new scenery.Node( {
          children: [
            new scenery.Image( image, { x: -x, y: -y } )
          ]
        } ) );
      }, x, y, width, height );
    },

    /**
     * Calls the callback with an Image node that contains this Node's subtree's visual form. This is always
     * synchronous, but the resulting image node can ONLY used with Canvas/WebGL (NOT SVG).
     * @public
     *
     * @param {Function} callback - callback( imageNode {Image} ) is called
     * @param {number} [x] - The X offset for where the upper-left of the content drawn into the Canvas
     * @param {number} [y] - The Y offset for where the upper-left of the content drawn into the Canvas
     * @param {number} [width] - The width of the Canvas output
     * @param {number} [height] - The height of the Canvas output
     */
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

    /**
     * Calls the callback with a Node that contains this Node's subtree's visual form. This is always
     * synchronous, but the resulting node will not have the correct bounds immediately (that will be asynchronous).
     * @public
     *
     * TODO: set initialWidth/initialHeight so that we have the bounds immediately?
     *
     * @param {Function} callback - callback( imageNode {Image} ) is called
     * @param {number} [x] - The X offset for where the upper-left of the content drawn into the Canvas
     * @param {number} [y] - The Y offset for where the upper-left of the content drawn into the Canvas
     * @param {number} [width] - The width of the Canvas output
     * @param {number} [height] - The height of the Canvas output
     */
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

    /**
     * Returns a reference to the instances array.
     * @public (scenery-internal)
     *
     * @returns {Array.<Instance>}
     */
    getInstances: function() {
      return this._instances;
    },
    get instances() { return this.getInstances(); },

    /**
     * Adds an Instance reference to our array.
     * @public (scenery-internal)
     *
     * @param {Instance} instance
     */
    addInstance: function( instance ) {
      assert && assert( instance instanceof scenery.Instance );
      this._instances.push( instance );
    },

    /**
     * Removes an Instance reference from our array.
     * @public (scenery-internal)
     *
     * @param {Instance} instance
     */
    removeInstance: function( instance ) {
      assert && assert( instance instanceof scenery.Instance );
      var index = _.indexOf( this._instances, instance );
      assert && assert( index !== -1, 'Cannot remove a Instance from a Node if it was not there' );
      this._instances.splice( index, 1 );
    },

    /*---------------------------------------------------------------------------*
     * Accessible Instance handling
     *----------------------------------------------------------------------------*/

    /**
     * Returns a reference to the accessible instances array.
     * @public (scenery-internal)
     *
     * @returns {Array.<AccessibleInstance>}
     */
    getAccessibleInstances: function() {
      return this._accessibleInstances;
    },
    get accessibleInstances() { return this.getAccessibleInstances(); },

    /**
     * Adds an AccessibleInstance reference to our array.
     * @public (scenery-internal)
     *
     * @param {AccessibleInstance} accessibleInstance
     */
    addAccessibleInstance: function( accessibleInstance ) {
      assert && assert( accessibleInstance instanceof scenery.AccessibleInstance );
      this._accessibleInstances.push( accessibleInstance );
    },

    /**
     * Removes an AccessibleInstance reference from our array.
     * @public (scenery-internal)
     *
     * @param {AccessibleInstance} accessibleInstance
     */
    removeAccessibleInstance: function( accessibleInstance ) {
      assert && assert( accessibleInstance instanceof scenery.AccessibleInstance );
      var index = _.indexOf( this._accessibleInstances, accessibleInstance );
      assert && assert( index !== -1, 'Cannot remove an AccessibleInstance from a Node if it was not there' );
      this._accessibleInstances.splice( index, 1 );
    },

    /*---------------------------------------------------------------------------*
     * Display handling
     *----------------------------------------------------------------------------*/

    /**
     * Returns a reference to the display array.
     * @public (scenery-internal)
     *
     * @returns {Array.<Display>}
     */
    getRootedDisplays: function() {
      return this._rootedDisplays;
    },
    get rootedDisplays() { return this.getRootedDisplays(); },

    /**
     * Adds an display reference to our array.
     * @public (scenery-internal)
     *
     * @param {Display} display
     */
    addRootedDisplay: function( display ) {
      assert && assert( display instanceof scenery.Display );
      this._rootedDisplays.push( display );
    },

    /**
     * Removes a Display reference from our array.
     * @public (scenery-internal)
     *
     * @param {Display} display
     */
    removeRootedDisplay: function( display ) {
      assert && assert( display instanceof scenery.Display );
      var index = _.indexOf( this._rootedDisplays, display );
      assert && assert( index !== -1, 'Cannot remove a Display from a Node if it was not there' );
      this._rootedDisplays.splice( index, 1 );
    },

    /*---------------------------------------------------------------------------*
     * Coordinate transform methods
     *----------------------------------------------------------------------------*/

    /**
     * Returns a point transformed from our local coordinate frame into our parent coordinate frame. Applies our node's
     * transform to it.
     * @public
     *
     * @param {Vector2} point
     * @returns {Vector2}
     */
    localToParentPoint: function( point ) {
      return this._transform.transformPosition2( point );
    },

    /**
     * Returns bounds transformed from our local coordinate frame into our parent coordinate frame. If it includes a
     * rotation, the resulting bounding box will include every point that could have been in the original bounding box
     * (and it can be expanded).
     * @public
     *
     * @param   unds2} bounds
     * @returns {Bounds2}
     */
    localToParentBounds: function( bounds ) {
      return this._transform.transformBounds2( bounds );
    },

    /**
     * Returns a point transformed from our parent coordinate frame into our local coordinate frame. Applies the inverse
     * of our node's transform to it.
     * @public
     *
     * @param {Vector2} point
     * @returns {Vector2}
     */
    parentToLocalPoint: function( point ) {
      return this._transform.inversePosition2( point );
    },

    /**
     * Returns bounds transformed from our parent coordinate frame into our local coordinate frame. If it includes a
     * rotation, the resulting bounding box will include every point that could have been in the original bounding box
     * (and it can be expanded).
     * @public
     *
     * @param {Bounds2} bounds
     * @returns {Bounds2}
     */
    parentToLocalBounds: function( bounds ) {
      return this._transform.inverseBounds2( bounds );
    },

    /**
     * A mutable-optimized form of localToParentBounds() that will modify the provided bounds, transforming it from our
     * local coordinate frame to our parent coordinate frame.
     * @public
     *
     * @param {Bounds2} bounds
     * @returns {Bounds2} - The same bounds object.
     */
    transformBoundsFromLocalToParent: function( bounds ) {
      return bounds.transform( this._transform.getMatrix() );
    },

    /**
     * A mutable-optimized form of parentToLocalBounds() that will modify the provided bounds, transforming it from our
     * parent coordinate frame to our local coordinate frame.
     * @public
     *
     * @param {Bounds2} bounds
     * @returns {Bounds2} - The same bounds object.
     */
    transformBoundsFromParentToLocal: function( bounds ) {
      return bounds.transform( this._transform.getInverse() );
    },

    /**
     * Returns a new matrix (fresh copy) that would transform points from our local coordinate frame to the global
     * coordinate frame.
     * @public
     *
     * NOTE: If there are multiple instances of this node (e.g. this or one ancestor has two parents), it will fail
     * with an assertion (since the transform wouldn't be uniquely defined).
     *
     * @returns {Matrix3}
     */
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

    /**
     * Returns a Transform3 that would transform things from our local coordinate frame to the global coordinate frame.
     * Equivalent to getUniqueTrail().getTransform(), but faster.
     * @public
     *
     * NOTE: If there are multiple instances of this node (e.g. this or one ancestor has two parents), it will fail
     * with an assertion (since the transform wouldn't be uniquely defined).
     *
     * @returns {Transform3}
     */
    getUniqueTransform: function() {
      return new Transform3( this.getLocalToGlobalMatrix() );
    },

    /**
     * Returns a new matrix (fresh copy) that would transform points from the global coordinate frame to our local
     * coordinate frame.
     * @public
     *
     * NOTE: If there are multiple instances of this node (e.g. this or one ancestor has two parents), it will fail
     * with an assertion (since the transform wouldn't be uniquely defined).
     *
     * @returns {Matrix3}
     */
    getGlobalToLocalMatrix: function() {
      return this.getLocalToGlobalMatrix().invert();
    },

    /**
     * Transforms a point from our local coordinate frame to the global coordinate frame.
     * @public
     *
     * NOTE: If there are multiple instances of this node (e.g. this or one ancestor has two parents), it will fail
     * with an assertion (since the transform wouldn't be uniquely defined).
     *
     * @param {Vector2} point
     * @returns {Vector2}
     */
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

    /**
     * Transforms a point from the global coordinate frame to our local coordinate frame.
     * @public
     *
     * NOTE: If there are multiple instances of this node (e.g. this or one ancestor has two parents), it will fail
     * with an assertion (since the transform wouldn't be uniquely defined).
     *
     * @param {Vector2} point
     * @returns {Vector2}
     */
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

    /**
     * Transforms bounds from our local coordinate frame to the global coordinate frame. If it includes a
     * rotation, the resulting bounding box will include every point that could have been in the original bounding box
     * (and it can be expanded).
     * @public
     *
     * NOTE: If there are multiple instances of this node (e.g. this or one ancestor has two parents), it will fail
     * with an assertion (since the transform wouldn't be uniquely defined).
     *
     * @param {Bounds2} bounds
     * @returns {Bounds2}
     */
    localToGlobalBounds: function( bounds ) {
      // apply the bounds transform only once, so we can minimize the expansion encountered from multiple rotations
      // it also seems to be a bit faster this way
      return bounds.transformed( this.getLocalToGlobalMatrix() );
    },

    /**
     * Transforms bounds from the global coordinate frame to our local coordinate frame. If it includes a
     * rotation, the resulting bounding box will include every point that could have been in the original bounding box
     * (and it can be expanded).
     * @public
     *
     * NOTE: If there are multiple instances of this node (e.g. this or one ancestor has two parents), it will fail
     * with an assertion (since the transform wouldn't be uniquely defined).
     *
     * @param {Bounds2} bounds
     * @returns {Bounds2}
     */
    globalToLocalBounds: function( bounds ) {
      // apply the bounds transform only once, so we can minimize the expansion encountered from multiple rotations
      return bounds.transformed( this.getGlobalToLocalMatrix() );
    },

    /**
     * Transforms a point from our parent coordinate frame to the global coordinate frame.
     * @public
     *
     * NOTE: If there are multiple instances of this node (e.g. this or one ancestor has two parents), it will fail
     * with an assertion (since the transform wouldn't be uniquely defined).
     *
     * @param {Vector2} point
     * @returns {Vector2}
     */
    parentToGlobalPoint: function( point ) {
      assert && assert( this.parents.length <= 1, 'parentToGlobalPoint unable to work for DAG' );
      return this.parents.length ? this.parents[ 0 ].localToGlobalPoint( point ) : point;
    },

    /**
     * Transforms bounds from our parent coordinate frame to the global coordinate frame. If it includes a
     * rotation, the resulting bounding box will include every point that could have been in the original bounding box
     * (and it can be expanded).
     * @public
     *
     * NOTE: If there are multiple instances of this node (e.g. this or one ancestor has two parents), it will fail
     * with an assertion (since the transform wouldn't be uniquely defined).
     *
     * @param {Bounds2} bounds
     * @returns {Bounds2}
     */
    parentToGlobalBounds: function( bounds ) {
      assert && assert( this.parents.length <= 1, 'parentToGlobalBounds unable to work for DAG' );
      return this.parents.length ? this.parents[ 0 ].localToGlobalBounds( bounds ) : bounds;
    },

    /**
     * Transforms a point from the global coordinate frame to our parent coordinate frame.
     * @public
     *
     * NOTE: If there are multiple instances of this node (e.g. this or one ancestor has two parents), it will fail
     * with an assertion (since the transform wouldn't be uniquely defined).
     *
     * @param {Vector2} point
     * @returns {Vector2}
     */
    globalToParentPoint: function( point ) {
      assert && assert( this.parents.length <= 1, 'globalToParentPoint unable to work for DAG' );
      return this.parents.length ? this.parents[ 0 ].globalToLocalPoint( point ) : point;
    },

    /**
     * Transforms bounds from the global coordinate frame to our parent coordinate frame. If it includes a
     * rotation, the resulting bounding box will include every point that could have been in the original bounding box
     * (and it can be expanded).
     * @public
     *
     * NOTE: If there are multiple instances of this node (e.g. this or one ancestor has two parents), it will fail
     * with an assertion (since the transform wouldn't be uniquely defined).
     *
     * @param {Bounds2} bounds
     * @returns {Bounds2}
     */
    globalToParentBounds: function( bounds ) {
      assert && assert( this.parents.length <= 1, 'globalToParentBounds unable to work for DAG' );
      return this.parents.length ? this.parents[ 0 ].globalToLocalBounds( bounds ) : bounds;
    },

    /**
     * Returns a bounding box for this Node (and its sub-tree) in the global coordinate frame.
     * @public
     *
     * NOTE: If there are multiple instances of this node (e.g. this or one ancestor has two parents), it will fail
     * with an assertion (since the transform wouldn't be uniquely defined).
     *
     * @returns {Bounds2}
     */
    getGlobalBounds: function() {
      assert && assert( this.parents.length <= 1, 'globalBounds unable to work for DAG' );
      return this.parentToGlobalBounds( this.getBounds() );
    },
    get globalBounds() { return this.getGlobalBounds(); },

    /**
     * Returns the bounds of any other node in our local coordinate frame.
     *
     * NOTE: If this node or the passed in node have multiple instances (e.g. this or one ancestor has two parents), it will fail
     * with an assertion.
     *
     * TODO: Possible to be well-defined and have multiple instances of each.
     *
     * @param {Node} node
     * @returns {Bounds2}
     */
    boundsOf: function( node ) {
      return this.globalToLocalBounds( node.getGlobalBounds() );
    },

    /**
     * Returns the bounds of this node in another node's local coordinate frame.
     *
     * NOTE: If this node or the passed in node have multiple instances (e.g. this or one ancestor has two parents), it will fail
     * with an assertion.
     *
     * TODO: Possible to be well-defined and have multiple instances of each.
     *
     * @param {Node} node
     * @returns {Bounds2}
     */
    boundsTo: function( node ) {
      return node.globalToLocalBounds( this.getGlobalBounds() );
    },

    /*---------------------------------------------------------------------------*
     * Drawable handling
     *----------------------------------------------------------------------------*/

    /**
     * Adds the drawable to our list of drawables to notify of visual changes.
     * @public (scenery-internal)
     *
     * @param {Drawable} drawable
     */
    attachDrawable: function( drawable ) {
      this._drawables.push( drawable );
      return this; // allow chaining
    },

    /**
     * Removes the drawable from our list of drawables to notify of visual changes.
     * @public (scenery-internal)
     *
     * @param {Drawable} drawable
     */
    detachDrawable: function( drawable ) {
      var index = _.indexOf( this._drawables, drawable );

      assert && assert( index >= 0, 'Invalid operation: trying to detach a non-referenced drawable' );

      this._drawables.splice( index, 1 ); // TODO: replace with a remove() function
      return this;
    },

    /**
     * Scans the options object for key names that correspond to ES5 setters or other setter functions, and calls those
     * with the values.
     * @public
     *
     * For example:
     *
     * node.mutate( { top: 0, left: 5 } );
     *
     * will be equivalent to:
     *
     * node.left = 5;
     * node.top = 0;
     *
     * In particular, note that the order is different. Mutators will be applied in the order of _mutatorKeys, which can
     * be added to by subtypes.
     *
     * Additionally, some keys are actually direct function names, like 'scale'. mutate( { scale: 2 } ) will call
     * node.scale( 2 ) instead of activating an ES5 setter directly.
     *
     * @param {Object} [options]
     */
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

    /**
     * Override for extra information in the debugging output
     * @protected (scenery-internal)
     */
    getDebugHTMLExtras: function() {
      return '';
    },

    /**
     * Returns a debugging string that is an attempted serialization of this node's sub-tree.
     * @public
     *
     * @param {string} spaces - Whitespace to add
     * @param {boolean} [includeChildren]
     */
    toString: function( spaces, includeChildren ) {
      spaces = spaces || '';
      var props = this.getPropString( spaces + '  ', includeChildren === undefined ? true : includeChildren );
      return spaces + this.getBasicConstructor( props ? ( '\n' + props + '\n' + spaces ) : '' );
    },

    /**
     * Returns a constructor template for toString(). Meant to be overridden by subtypes.
     * @protected (scenery-internal)
     *
     * @param {string} propLines - What is included.
     */
    getBasicConstructor: function( propLines ) {
      return 'new scenery.Node( {' + propLines + '} )';
    },

    /**
     * Returns the property object string for use with toString(). Meant to be overridden to add subtype-specific types.
     * @protected (scenery-internal)
     *
     * @param {string} spaces - Whitespace to add
     * @param {boolean} [includeChildren]
     */
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
        addProp( 'renderer', this.renderer );
        if ( this.rendererOptions ) {
          // addProp( 'rendererOptions', JSON.stringify( this.rendererOptions ), true );
        }
      }

      if ( this._hints.layerSplit ) {
        addProp( 'layerSplit', true );
      }

      return result;
    },

    /**
     * Performs checks to see if the internal state of Instance references is correct at a certain point in/after the
     * Display's updateDisplay().
     * @private
     */
    auditInstanceSubtreeForDisplay: function( display ) {
      if ( assertSlow ) {
        var numInstances = this._instances.length;
        for ( var i = 0; i < numInstances; i++ ) {
          var instance = this._instances[ i ];
          if ( instance.display === display ) {
            assertSlow( instance.trail.isValid(),
              'Invalid trail on Instance: ' + instance.toString() + ' with trail ' + instance.trail.toString() );
          }
        }

        // audit all of the children
        this.children.forEach( function( child ) {
          child.auditInstanceSubtreeForDisplay( display );
        } );
      }
    },

    /*---------------------------------------------------------------------------*
     * Compatibility with old events API (now using axon.Events)
     *----------------------------------------------------------------------------*/

    /**
     * @deprecated, please use node.on( eventName, listener) instead.
     * Adds a listener for a specific event name.
     * @public
     *
     * @param {string} eventName
     * @param {Function} listener
     */
    addEventListener: function( eventName, listener ) {
      // can't guarantee static with old usage
      return this.on( eventName, listener );
    },

    /**
     * @deprecated, please use node.off( eventName, listener) instead.
     * Removes a listener for a specific event name.
     * @public
     *
     * @param {string} eventName
     * @param {Function} listener
     */
    removeEventListener: function( eventName, listener ) {
      // can't guarantee static with old usage
      return this.off( eventName, listener );
    },

    /**
     * @deprecated, please use node.hasListener( eventName, listener) instead.
     * Checks for the presence of a listener for a specific event name.
     * @public
     *
     * @param {string} eventName
     * @param {Function} listener
     */
    containsEventListener: function( eventName, listener ) {
      return this.hasListener( eventName, listener );
    },

    /**
     * Tracks when an event listener is added, so that we can prune hit testing for performance.
     * @private
     *
     * @param {string} eventName
     * @param {Function} listener
     */
    onEventListenerAdded: function( eventName, listener ) {
      if ( eventName in eventsRequiringBoundsValidation ) {
        this.changeBoundsEventCount( 1 );
        this._boundsEventSelfCount++;
      }
    },

    /**
     * Tracks when an event listener is removed, so that we can prune hit testing for performance.
     * @private
     *
     * @param {string} eventName
     * @param {Function} listener
     */
    onEventListenerRemoved: function( eventName, listener ) {
      if ( eventName in eventsRequiringBoundsValidation ) {
        this.changeBoundsEventCount( -1 );
        this._boundsEventSelfCount--;
      }
    }

  }, Events.prototype, {
    /**
     * Adds a listener for a specific event name. Overridden so we can track specific types of listeners.
     * @public
     *
     * @param {string} eventName
     * @param {Function} listener
     */
    on: function onOverride( eventName, listener ) {
      Events.prototype.on.call( this, eventName, listener );
      this.onEventListenerAdded( eventName, listener );
    },

    /**
     * Adds a listener for a specific event name, that guarantees it won't trigger changes to the listener list when
     * the listener is called. Overridden so we can track specific types of listeners.
     * @public
     *
     * @param {string} eventName
     * @param {Function} listener
     */
    onStatic: function onStaticOverride( eventName, listener ) {
      Events.prototype.onStatic.call( this, eventName, listener );
      this.onEventListenerAdded( eventName, listener );
    },

    /**
     * Removes a listener for a specific event name. Overridden so we can track specific types of listeners.
     * @public
     *
     * @param {string} eventName
     * @param {Function} listener
     */
    off: function offOverride( eventName, listener ) {
      var index = Events.prototype.off.call( this, eventName, listener );
      assert && assert( index >= 0, 'Node.off was called but no listener was removed' );
      this.onEventListenerRemoved( eventName, listener );
      return index;
    },

    /**
     * Removes a listener for a specific event name, that guarantees it won't trigger changes to the listener list when
     * the listener is called. Overridden so we can track specific types of listeners.
     * @public
     *
     * @param {string} eventName
     * @param {Function} listener
     */
    offStatic: function offStaticOverride( eventName, listener ) {
      var index = Events.prototype.offStatic.call( this, eventName, listener );
      assert && assert( index >= 0, 'Node.offStatic was called but no listener was removed' );
      this.onEventListenerRemoved( eventName, listener );
      return index;
    }
  } ) );


  /*
   * Convenience locations
   * @public
   *
   * Upper is in terms of the visual layout in Scenery and other programs, so the minY is the "upper", and minY is the "lower"
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

  // @public
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
   * This is an array of property (setter) names for Node.mutate(), which are also used when creating nodes with
   * parameter objects.
   * @protected
   *
   * E.g. new scenery.Node( { x: 5, rotation: 20 } ) will create a Path, and apply setters in the order below
   * (node.x = 5; node.rotation = 20)
   *
   * The order below is important! Don't change this without knowing the implications.
   * NOTE: translation-based mutators come before rotation/scale, since typically we think of their operations occuring
   * "after" the rotation / scaling
   * NOTE: left/right/top/bottom/centerX/centerY are at the end, since they rely potentially on rotation / scaling
   * changes of bounds that may happen beforehand
   */
  Node.prototype._mutatorKeys = [
    'children', 'cursor', 'visible', 'pickable', 'opacity', 'matrix', 'translation', 'x', 'y', 'rotation', 'scale', 'maxWidth', 'maxHeight',
    'leftTop', 'centerTop', 'rightTop', 'leftCenter', 'center', 'rightCenter', 'leftBottom', 'centerBottom', 'rightBottom',
    'left', 'right', 'top', 'bottom', 'centerX', 'centerY', 'renderer', 'rendererOptions',
    'layerSplit', 'usesOpacity', 'cssTransform', 'excludeInvisible', 'webglScale', 'mouseArea', 'touchArea', 'clipArea',
    'transformBounds', 'focusable', 'focusIndicator', 'accessibleContent', 'accessibleOrder', 'textDescription'
  ];

  return Node;
} );
