// Copyright 2015-2016, University of Colorado Boulder

/**
 * Focus highlight overlay for accessible displays.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var FocusHighlightFromNode = require( 'SCENERY/accessibility/FocusHighlightFromNode' );
  var FocusHighlightPath = require( 'SCENERY/accessibility/FocusHighlightPath' );
  var inherit = require( 'PHET_CORE/inherit' );
  var Node = require( 'SCENERY/nodes/Node' );
  var scenery = require( 'SCENERY/scenery' );
  var Shape = require( 'KITE/Shape' );
  var TransformTracker = require( 'SCENERY/util/TransformTracker' );

  // colors for the focus highlights, can be changed for different application backgrounds or color profiles, see
  // the setters and getters below for these values.
  var outerHighlightColor = FocusHighlightPath.OUTER_FOCUS_COLOR;
  var innerHighlightColor = FocusHighlightPath.INNER_FOCUS_COLOR;

  var innerGroupHighlightColor = FocusHighlightPath.INNER_LIGHT_GROUP_FOCUS_COLOR;
  var outerGroupHighlightColor = FocusHighlightPath.OUTER_LIGHT_GROUP_FOCUS_COLOR;

  /**
   * @constructor
   *
   * @param {Display} display
   * @param {Node} focusRootNode - the root node of our display
   */
  function FocusOverlay( display, focusRootNode ) {
    this.display = display; // @private {Display}
    this.focusRootNode = focusRootNode; // @private {Node} - The root Node of our child display

    // @private {Trail|null} - trail to the node with focus, modified when focus changes
    this.trail = null;

    // @private {Node|null} - node with focus, modified when focus changes
    this.node = null;

    // @private {string|null} - signifies method of representing focus, 'bounds'|'node'|'shape'|'invisible', modified
    // when focus changes
    this.mode = null;

    // @private {string|null} - signifies method off representing group focus, 'bounds'|'node', modified when
    // focus changes
    this.groupMode = null;

    // @private {Node|null} - the group highlight node around an ancestor of this.node when focus changes,
    // see Accessibility.setGroupFocusHighlight for more information on the group focus highlight, modified when
    // focus changes
    this.groupHighlightNode = null;

    // @private {TransformTracker|null} - tracks transformations to the focused node and the node with a group
    // focus highlight, modified when focus changes
    this.transformTracker = null;
    this.groupTransformTracker = null;

    // @private {Node|null} - If a node is using a custom focus highlight, a reference is kept so that it can be
    // removed from the overlay when node focus changes.
    this.nodeFocusHighlight = null;

    // @private {boolean} - if true, the next update() will trigger an update to the highlight's transform
    this.transformDirty = true;

    // @private {Node} - The main node for the highlight. It will be transformed.
    this.highlightNode = new Node();
    this.focusRootNode.addChild( this.highlightNode );

    // @private {Display} - display that manages all focus highlights
    this.focusDisplay = new scenery.Display( this.focusRootNode, {
      width: this.width,
      height: this.height,
      allowWebGL: display._allowWebGL,
      allowCSSHacks: false,
      accessibility: false,
      isApplication: false,
      interactive: false
    } );

    // @private {HTMLElement}
    this.domElement = this.focusDisplay.domElement;
    this.domElement.style.pointerEvents = 'none';

    // Used as the focus highlight when the overlay is passed a shape
    this.shapeFocusHighlightPath = new FocusHighlightPath( null );
    this.boundsFocusHighlightPath = new FocusHighlightFromNode( null, {
      useLocalBounds: true
    } );

    // @private {FocusHighlightPath} - Focus highlight for 'groups' of Nodes. When descendant node has focus, ancestor
    // with groupFocusHighlight flag will have this extra focus highlight surround its local bounds
    this.groupFocusHighlightPath = new FocusHighlightFromNode( null, {
      useLocalBounds: true,
      useGroupDilation: true,
      outerLineWidth: FocusHighlightPath.GROUP_OUTER_LINE_WIDTH,
      innerLineWidth: FocusHighlightPath.GROUP_INNER_LINE_WIDTH,
      innerStroke: FocusHighlightPath.FOCUS_COLOR
    } );

    this.highlightNode.addChild( this.shapeFocusHighlightPath );
    this.highlightNode.addChild( this.boundsFocusHighlightPath );
    this.focusRootNode.addChild( this.groupFocusHighlightPath );

    // @private - Listeners bound once, so we can access them for removal.
    this.boundsListener = this.onBoundsChange.bind( this );
    this.transformListener = this.onTransformChange.bind( this );
    this.focusListener = this.onFocusChange.bind( this );
    this.focusHighlightListener = this.onFocusHighlightChange.bind( this );

    scenery.Display.focusProperty.link( this.focusListener );
  }

  scenery.register( 'FocusOverlay', FocusOverlay );

  inherit( Object, FocusOverlay, {
    dispose: function() {
      if ( this.hasHighlight() ) {
        this.deactivateHighlight();
      }

      scenery.Display.focusProperty.unlink( this.focusListener );
    },

    hasHighlight: function() {
      return !!this.trail;
    },

    /**
     * Activates the highlight, choosing a mode for whether the highlight will be a shape, node, or bounds.
     * @private
     *
     * @param {Trail} trail - The focused trail to highlight. It assumes that this trail is in this display.
     */
    activateHighlight: function( trail ) {
      this.trail = trail;
      this.node = trail.lastNode();
      var focusHighlight = this.node.focusHighlight;

      // we may or may not track this trail depending on whether the focus highlight surrounds the trail's leaf node or
      // a different node
      var trailToTrack = trail;

      // Invisible mode - no focus highlight; this is only for testing mode, when Nodes rarely have bounds.
      if ( focusHighlight === 'invisible' ) {
        this.mode = 'invisible';
      }
      // Shape mode
      else if ( focusHighlight instanceof Shape ) {
        this.mode = 'shape';

        this.shapeFocusHighlightPath.visible = true;
        this.shapeFocusHighlightPath.setShape( focusHighlight );
      }
      // Node mode
      else if ( focusHighlight instanceof Node ) {
        this.mode = 'node';

        // if using a focus highlight from another node, we will track that node's transform instead of the focused node
        if ( focusHighlight instanceof FocusHighlightFromNode ) {
          trailToTrack = focusHighlight.getUniqueHighlightTrail();
        }

        // store the focus highlight so that it can be removed later
        this.nodeFocusHighlight = focusHighlight;

        // If focusHighlightLayerable, then the focusHighlight is just a node in the scene graph, so set it visible
        if ( this.node.focusHighlightLayerable ) {
          this.nodeFocusHighlight.visible = true;
        }
        else {
          this.nodeFocusHighlight.visible = true;

          // Use the node itself as the highlight
          this.highlightNode.addChild( this.nodeFocusHighlight );
        }
      }
      // Bounds mode
      else {
        this.mode = 'bounds';

        this.boundsFocusHighlightPath.setShapeFromNode( this.node );

        this.boundsFocusHighlightPath.visible = true;
        this.node.onStatic( 'localBounds', this.boundsListener );

        this.onBoundsChange();
      }

      // handle any changes to the focus highlight while the node has focus
      this.node.onStatic( 'focusHighlightChanged', this.focusHighlightListener );

      this.transformTracker = new TransformTracker( trailToTrack, {
        isStatic: true
      } );
      this.transformTracker.addListener( this.transformListener );

      // handle group focus highlights
      this.activateGroupHighlights();

      // update highlight colors if necessary
      this.updateHighlightColors();

      this.transformDirty = true;
    },

    /**
     * Deactivates the current highlight, disposing and removing listeners as necessary.
     * @private
     */
    deactivateHighlight: function() {
      if ( this.mode === 'shape' ) {
        this.shapeFocusHighlightPath.visible = false;
      }
      else if ( this.mode === 'node' ) {

        // If focusHighlightLayerable, then the focusHighlight is just a node in the scene graph, so set it invisible
        if ( this.node.focusHighlightLayerable ) {
          this.node.focusHighlight.visible = false;
        }
        else {
          this.highlightNode.removeChild( this.nodeFocusHighlight );
          this.nodeFocusHighlight = null;
        }
      }
      else if ( this.mode === 'bounds' ) {
        this.boundsFocusHighlightPath.visible = false;
        this.node.offStatic( 'localBounds', this.boundsListener );
      }

      // remove listener that updated focus highlight while this node had focus
      this.node.offStatic( 'focusHighlightChanged', this.focusHighlightListener );

      // remove all 'group' focus highlights
      this.deactivateGroupHighlights();

      this.trail = null;
      this.node = null;
      this.mode = null;
      this.transformTracker.removeListener( this.transformListener );
      this.transformTracker.dispose();
    },

    /**
     * Activate all 'group' focus highlights by searching for ancestor nodes from the node that has focus
     * and adding a rectangle around it if it has a "groupFocusHighlight". A group highlight will only appear around
     * the closest ancestor that has a one.
     *
     * @private
     */
    activateGroupHighlights: function() {

      var trail = this.trail;
      for ( var i = 0; i < trail.length; i++ ) {
        var node = trail.nodes[ i ];
        var highlight = node.groupFocusHighlight;
        if ( highlight ) {

          // update transform tracker
          var trailToParent = trail.upToNode( node );
          this.groupTransformTracker = new TransformTracker( trailToParent );
          this.groupTransformTracker.addListener( this.transformListener );

          if ( typeof highlight === 'boolean' ) {

            // add a bounding rectangle around the node that uses group highlights
            this.groupFocusHighlightPath.setShapeFromNode( node );
            this.groupFocusHighlightPath.visible = true;

            this.groupHighlightNode = this.groupFocusHighlightPath;
            this.groupMode = 'bounds';
          }
          else if ( highlight instanceof Node ) {
            this.groupHighlightNode = highlight;
            this.focusRootNode.addChild( highlight );

            this.groupMode = 'node';
          }

          // Only closest ancestor with group highlight will get the group highlight
          break;
        }
      }
    },

    /**
     * Update focus highlight colors. This is a no-op if we are in 'node' mode, or if none of the highlight colors
     * have changed.
     *
     * TODO: Support updating focus highlight strokes in 'node' mode as well?
     */
    updateHighlightColors: function() {

      if ( this.mode === 'shape' ) {
        if ( this.shapeFocusHighlightPath.innerHighlightColor !== FocusOverlay.innerHighlightColor ) {
          this.shapeFocusHighlightPath.setInnerHighlightColor( FocusOverlay.innerHighlightColor );
        }
        if ( this.shapeFocusHighlightPath.outerHighlightColor !== FocusOverlay.outerHighlightColor ) {
          this.shapeFocusHighlightPath.setOuterHighlightColor( FocusOverlay.outerHighlightColor );
        }
      }
      else if ( this.mode === 'bounds' ) {
        if ( this.boundsFocusHighlightPath.innerHighlightColor !== FocusOverlay.innerHighlightColor ) {
          this.boundsFocusHighlightPath.setInnerHighlightColor( FocusOverlay.innerHighlightColor );
        }
        if ( this.boundsFocusHighlightPath.outerHighlightColor !== FocusOverlay.outerHighlightColor ) {
          this.boundsFocusHighlightPath.setOuterHighlightColor( FocusOverlay.outerHighlightColor );
        }
      }

      // if a group focus highlight is active, update strokes
      if ( this.groupMode ) {
        if ( this.groupFocusHighlightPath.innerHighlightColor !== FocusOverlay.innerGroupHighlightColor ) {
          this.groupFocusHighlightPath.setInnerHighlightColor( FocusOverlay.innerGroupHighlightColor );
        }
        if ( this.groupFocusHighlightPath.outerHighlightColor !== FocusOverlay.outerGroupHighlightColor ) {
          this.groupFocusHighlightPath.setOuterHighlightColor( FocusOverlay.outerGroupHighlightColor );
        }
      }
    },

    /**
     * Remove all group focus highlights by making them invisible, or removing them from the root of this overlay,
     * depending on mode.
     * @private
     */
    deactivateGroupHighlights: function() {
      if ( this.groupMode ) {
        if ( this.groupMode === 'bounds' ) {
          this.groupFocusHighlightPath.visible = false;
        }
        else if ( this.groupMode === 'node' ) {
          this.focusRootNode.removeChild( this.groupHighlightNode );
        }

        this.groupMode = null;
        this.groupHighlightNode = null;
        this.groupTransformTracker.removeListener( this.transformListener );
        this.groupTransformTracker.dispose();
      }
    },

    // Called from FocusOverlay after transforming the highlight. Only called when the transform changes.
    afterTransform: function() {
      if ( this.mode === 'shape' ) {
        this.shapeFocusHighlightPath.updateLineWidth();
      }
      else if ( this.mode === 'bounds' ) {
        this.boundsFocusHighlightPath.updateLineWidth();
      }
      else if ( this.mode === 'node' && this.node.focusHighlight.updateLineWidth ) {

        // Update the transform based on the transform of the node that the focusHighlight is highlighting.
        this.node.focusHighlight.updateLineWidth( this.node );
      }
    },

    onTransformChange: function() {
      this.transformDirty = true;
    },

    // Called when bounds change on our node when we are in "Bounds" mode
    onBoundsChange: function() {
      this.boundsFocusHighlightPath.setShapeFromNode( this.node );
    },

    // Called when the main Scenery focus pair (Display,Trail) changes.
    onFocusChange: function( focus ) {
      var newTrail = ( focus && focus.display === this.display ) ? focus.trail : null;

      if ( this.hasHighlight() ) {
        this.deactivateHighlight();
      }

      if ( newTrail ) {
        this.activateHighlight( newTrail );
      }
    },

    /**
     * If the focused node has an updated focus highlight, we must do all the work of highlight deactivation/activation
     * as if the application focus changed. If focus highlight mode changed, we need to add/remove static listeners,
     * add/remove highlight children, and so on. Called when focus highlight changes, but should only ever be
     * necessary when the node has focus.
     */
    onFocusHighlightChange: function() {
      assert && assert( this.node.focused, 'update should only be necessary if node already has focus' );
      this.onFocusChange( scenery.Display.focus );
    },

    update: function() {
      // Transform the highlight to match the position of the node
      if ( this.hasHighlight() && this.transformDirty ) {
        this.transformDirty = false;

        this.highlightNode.setMatrix( this.transformTracker.matrix );
        this.groupHighlightNode && this.groupHighlightNode.setMatrix( this.groupTransformTracker.matrix );

        this.afterTransform();
      }

      if ( !this.display.size.equals( this.focusDisplay.size ) ) {
        this.focusDisplay.setWidthHeight( this.display.width, this.display.height );
      }
      this.focusDisplay.updateDisplay();
    }

  }, {

    /**
     * Set the inner color of all focus highlights.
     * @public
     *
     * @param {PaintDef} color
     */
    setInnerHighlightColor: function( color ) {
      innerHighlightColor = color;
    },
    set innerHighlightColor( color ) { this.setInnerHighlightColor( color ); },

    /**
     * Get the inner color of all focus highlights.
     * @public
     *
     * @returns {PaintDef}
     */
    getInnerHighlightColor: function() {
      return innerHighlightColor;
    },
    get innerHighlightColor() { return this.getInnerHighlightColor(); },

    /**
     * Set the outer color of all focus highlights.
     * @public
     *
     * @param {PaintDef} color
     */
    setOuterHilightColor: function( color ) {
      outerHighlightColor = color;
    },
    set outerHighlightColor( color ) { this.setOuterHilightColor( color ); },

    /**
     * Get the outer color of all focus highlights.
     * @public
     *
     * @returns {PaintDef} color
     */
    getOuterHighlightColor: function() {
      return outerHighlightColor;
    },
    get outerHighlightColor() { return this.getOuterHighlightColor(); },

    /**
     * Set the inner color of all group focus highlights.
     * @public
     *
     * @param {PaintDef} color
     */
    setInnerGroupHighlightColor: function( color ) {
      innerGroupHighlightColor = color;
    },
    set innerGroupHighlightColor( color ) { this.setInnerGroupHighlightColor( color ); },

    /**
     * Get the inner color of all group focus highlights
     * @public
     *
     * @returns {PaintDef} color
     */
    getInnerGroupHighlightColor: function() {
      return innerGroupHighlightColor;
    },
    get innerGroupHighlightColor() { return this.getInnerGroupHighlightColor(); },

    /**
     * Set the outer color of all group focus highlight.
     * @public
     *
     * @param {PaintDef} color
     */
    setOuterGroupHighlightColor: function( color ) {
      outerGroupHighlightColor = color;
    },
    set outerGroupHighlightColor( color ) { this.setOuterGroupHighlightColor( color ); },

    /**
     * Get the outer color of all group focus highlights.
     * @public
     *
     * @returns {PaintDef} color
     */
    getOuterGroupHighlightColor: function() {
      return outerGroupHighlightColor;
    },
    get outerGroupHighlightColor() { return this.getOuterGroupHighlightColor(); }

  } );

  return FocusOverlay;
} );