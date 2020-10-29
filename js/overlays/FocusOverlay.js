// Copyright 2015-2020, University of Colorado Boulder

/**
 * Focus highlight overlay for accessible displays.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Shape from '../../../kite/js/Shape.js';
import FocusHighlightFromNode from '../accessibility/FocusHighlightFromNode.js';
import FocusHighlightPath from '../accessibility/FocusHighlightPath.js';
import Display from '../display/Display.js';
import Node from '../nodes/Node.js';
import scenery from '../scenery.js';
import TransformTracker from '../util/TransformTracker.js';

// colors for the focus highlights, can be changed for different application backgrounds or color profiles, see
// the setters and getters below for these values.
let outerHighlightColor = FocusHighlightPath.OUTER_FOCUS_COLOR;
let innerHighlightColor = FocusHighlightPath.INNER_FOCUS_COLOR;

let innerGroupHighlightColor = FocusHighlightPath.INNER_LIGHT_GROUP_FOCUS_COLOR;
let outerGroupHighlightColor = FocusHighlightPath.OUTER_LIGHT_GROUP_FOCUS_COLOR;

class FocusOverlay {
  /**
   * @param {Display} display
   * @param {Node} focusRootNode - the root node of our display
   */
  constructor( display, focusRootNode ) {
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
    // see ParallelDOM.setGroupFocusHighlight for more information on the group focus highlight, modified when
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
    this.focusDisplay = new Display( this.focusRootNode, {
      width: this.width,
      height: this.height,
      allowWebGL: display.isWebGLAllowed(),
      allowCSSHacks: false,
      accessibility: false,
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

    Display.focusProperty.link( this.focusListener );
  }

  /**
   * Releases references
   * @public
   */
  dispose() {
    if ( this.hasHighlight() ) {
      this.deactivateHighlight();
    }

    Display.focusProperty.unlink( this.focusListener );
  }

  /**
   * @public
   *
   * @returns {boolean}
   */
  hasHighlight() {
    return !!this.trail;
  }

  /**
   * Activates the highlight, choosing a mode for whether the highlight will be a shape, node, or bounds.
   * @private
   *
   * @param {Trail} trail - The focused trail to highlight. It assumes that this trail is in this display.
   */
  activateHighlight( trail ) {
    this.trail = trail;
    this.node = trail.lastNode();
    const focusHighlight = this.node.focusHighlight;

    // we may or may not track this trail depending on whether the focus highlight surrounds the trail's leaf node or
    // a different node
    let trailToTrack = trail;

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
      if ( focusHighlight.transformSourceNode ) {
        trailToTrack = focusHighlight.getUniqueHighlightTrail();
      }

      // store the focus highlight so that it can be removed later
      this.nodeFocusHighlight = focusHighlight;

      assert && assert( this.nodeFocusHighlight.shape !== null,
        'The shape of the Node focusHighlight should be set by now. Does it have bounds?' );

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
      this.node.localBoundsProperty.lazyLink( this.boundsListener );

      this.onBoundsChange();
    }

    // handle any changes to the focus highlight while the node has focus
    this.node.focusHighlightChangedEmitter.addListener( this.focusHighlightListener );

    this.transformTracker = new TransformTracker( trailToTrack, {
      isStatic: true
    } );
    this.transformTracker.addListener( this.transformListener );

    // handle group focus highlights
    this.activateGroupHighlights();

    // update highlight colors if necessary
    this.updateHighlightColors();

    this.transformDirty = true;
  }

  /**
   * Deactivates the current highlight, disposing and removing listeners as necessary.
   * @private
   */
  deactivateHighlight() {
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
      this.node.localBoundsProperty.unlink( this.boundsListener );
    }

    // remove listener that updated focus highlight while this node had focus
    this.node.focusHighlightChangedEmitter.removeListener( this.focusHighlightListener );

    // remove all 'group' focus highlights
    this.deactivateGroupHighlights();

    this.trail = null;
    this.node = null;
    this.mode = null;
    this.transformTracker.removeListener( this.transformListener );
    this.transformTracker.dispose();
  }

  /**
   * Activate all 'group' focus highlights by searching for ancestor nodes from the node that has focus
   * and adding a rectangle around it if it has a "groupFocusHighlight". A group highlight will only appear around
   * the closest ancestor that has a one.
   * @private
   */
  activateGroupHighlights() {

    const trail = this.trail;
    for ( let i = 0; i < trail.length; i++ ) {
      const node = trail.nodes[ i ];
      const highlight = node.groupFocusHighlight;
      if ( highlight ) {

        // update transform tracker
        const trailToParent = trail.upToNode( node );
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
  }

  /**
   * Update focus highlight colors. This is a no-op if we are in 'node' mode, or if none of the highlight colors
   * have changed.
   * @private
   *
   * TODO: Support updating focus highlight strokes in 'node' mode as well?
   */
  updateHighlightColors() {

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
  }

  /**
   * Remove all group focus highlights by making them invisible, or removing them from the root of this overlay,
   * depending on mode.
   * @private
   */
  deactivateGroupHighlights() {
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
  }

  /**
   * Called from FocusOverlay after transforming the highlight. Only called when the transform changes.
   * @private
   */
  afterTransform() {
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
  }

  /**
   * @private
   */
  onTransformChange() {
    this.transformDirty = true;
  }

  /**
   * Called when bounds change on our node when we are in "Bounds" mode
   * @private
   */
  onBoundsChange() {
    this.boundsFocusHighlightPath.setShapeFromNode( this.node );
  }

  /**
   * Called when the main Scenery focus pair (Display,Trail) changes.
   * @private
   *
   * @param {Focus} focus
   */
  onFocusChange( focus ) {
    const newTrail = ( focus && focus.display === this.display ) ? focus.trail : null;

    if ( this.hasHighlight() ) {
      this.deactivateHighlight();
    }

    if ( newTrail ) {
      this.activateHighlight( newTrail );
    }
  }

  /**
   * If the focused node has an updated focus highlight, we must do all the work of highlight deactivation/activation
   * as if the application focus changed. If focus highlight mode changed, we need to add/remove static listeners,
   * add/remove highlight children, and so on. Called when focus highlight changes, but should only ever be
   * necessary when the node has focus.
   * @private
   */
  onFocusHighlightChange() {
    assert && assert( this.node.focused, 'update should only be necessary if node already has focus' );
    this.onFocusChange( Display.focus );
  }

  /**
   * @public
   */
  update() {
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

  /**
   * Set the inner color of all focus highlights.
   * @public
   *
   * @param {PaintDef} color
   */
  static setInnerHighlightColor( color ) {
    innerHighlightColor = color;
  }
  static set innerHighlightColor( color ) { this.setInnerHighlightColor( color ); }

  /**
   * Get the inner color of all focus highlights.
   * @public
   *
   * @returns {PaintDef}
   */
  static getInnerHighlightColor() {
    return innerHighlightColor;
  }
  static get innerHighlightColor() { return this.getInnerHighlightColor(); } // eslint-disable-line bad-sim-text

  /**
   * Set the outer color of all focus highlights.
   * @public
   *
   * @param {PaintDef} color
   */
  static setOuterHilightColor( color ) {
    outerHighlightColor = color;
  }
  static set outerHighlightColor( color ) { this.setOuterHilightColor( color ); }

  /**
   * Get the outer color of all focus highlights.
   * @public
   *
   * @returns {PaintDef} color
   */
  static getOuterHighlightColor() {
    return outerHighlightColor;
  }
  static get outerHighlightColor() { return this.getOuterHighlightColor(); } // eslint-disable-line bad-sim-text

  /**
   * Set the inner color of all group focus highlights.
   * @public
   *
   * @param {PaintDef} color
   */
  static setInnerGroupHighlightColor( color ) {
    innerGroupHighlightColor = color;
  }
  static set innerGroupHighlightColor( color ) { this.setInnerGroupHighlightColor( color ); }

  /**
   * Get the inner color of all group focus highlights
   * @public
   *
   * @returns {PaintDef} color
   */
  static getInnerGroupHighlightColor() {
    return innerGroupHighlightColor;
  }
  static get innerGroupHighlightColor() { return this.getInnerGroupHighlightColor(); } // eslint-disable-line bad-sim-text

  /**
   * Set the outer color of all group focus highlight.
   * @public
   *
   * @param {PaintDef} color
   */
  static setOuterGroupHighlightColor( color ) {
    outerGroupHighlightColor = color;
  }
  static set outerGroupHighlightColor( color ) { this.setOuterGroupHighlightColor( color ); }

  /**
   * Get the outer color of all group focus highlights.
   * @public
   *
   * @returns {PaintDef} color
   */
  static getOuterGroupHighlightColor() {
    return outerGroupHighlightColor;
  }
  static get outerGroupHighlightColor() { return this.getOuterGroupHighlightColor(); } // eslint-disable-line bad-sim-text
}

scenery.register( 'FocusOverlay', FocusOverlay );
export default FocusOverlay;