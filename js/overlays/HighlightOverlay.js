// Copyright 2015-2021, University of Colorado Boulder

/**
 * An overlay that implements highlights for a Display. This is responsible for drawing the highlights and
 * observing Properties or Emitters that dictate when highlights should become active. A highlight surrounds a Node
 * to indicate that it is in focus or relevant.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import BooleanProperty from '../../../axon/js/BooleanProperty.js';
import Shape from '../../../kite/js/Shape.js';
import merge from '../../../phet-core/js/merge.js';
import FocusHighlightFromNode from '../accessibility/FocusHighlightFromNode.js';
import FocusHighlightPath from '../accessibility/FocusHighlightPath.js';
import FocusManager from '../accessibility/FocusManager.js';
import ActivatedReadingBlockHighlight from '../accessibility/voicing/ActivatedReadingBlockHighlight.js';
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

class HighlightOverlay {

  /**
   * @param {Display} display
   * @param {Node} focusRootNode - the root node of our display
   * @param {Object} [options]
   */
  constructor( display, focusRootNode, options ) {

    options = merge( {

      // {BooleanProperty} - controls whether highlights related to DOM focus are visible
      pdomFocusHighlightsVisibleProperty: new BooleanProperty( true ),

      // {BooleanProperty} - controls whether highlights related to Interactive Highlights are visible
      interactiveHighlightsVisibleProperty: new BooleanProperty( false ),

      // {BooleanProperty - controls whether highlights associated with ReadingBlocks (of the Voicing feature set)
      // are shown when pointerFocusProperty changes
      readingBlockHighlightsVisibleProperty: new BooleanProperty( false )
    }, options );

    this.display = display; // @private {Display}
    this.focusRootNode = focusRootNode; // @private {Node} - The root Node of our child display

    // @private {Trail|null} - trail to the node with focus, modified when focus changes
    this.trail = null;

    // @private {Node|null} - node with focus, modified when focus changes
    this.node = null;

    // @private {Node|Shape|string|null} - A references to the highlight from the Node that is highlighted.
    this.activeHighlight = null;

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
    this.nodeModeHighlight = null;

    // @private {boolean} - If true, the active highlight is in "node" mode and is layered in the scene graph. This
    // field lets us deactivate the highlight appropriately when it is in that state.
    this.nodeModeHighlightLayered = false;

    // @private {boolean} - if true, the next update() will trigger an update to the highlight's transform
    this.transformDirty = true;

    // @private {Node} - The main node for the highlight. It will be transformed.
    this.highlightNode = new Node();
    this.focusRootNode.addChild( this.highlightNode );

    // @private {Node} - The main Node for the ReadingBlock highlight, while ReadingBlock content
    // is being spoken by speech synthesis.
    this.readingBlockHighlightNode = new Node();
    this.focusRootNode.addChild( this.readingBlockHighlightNode );

    // @private {null|Node|Shape} - A reference to the Node that is added when a custom node is specified as the
    // active highlight for the ReadingBlock. Stored so that we can remove it when deactivating reading
    // block highlights.
    this.addedReadingBlockHighlight = null;

    // @private {null|Node} - A reference to the Node that is a ReadingBlock which the Voicing framework
    // is currently speaking about.
    this.activeReadingBlockNode = null;

    // @private {Trail} - Trail to the ReadingBlock Node with an active highlight around it
    // while the voicingManager is speaking its content.
    this.readingBlockTrail = null;

    // @private {boolean} - Whether or not the transform applied to the readinBlockHighlightNode
    // is out of date.
    this.readingBlockTransformDirty = true;

    // @private {TransformTracker} - The TransformTracker used to observe changes to the transform of the Node with
    // Reading Block focus, so that the highlight can match the ReadingBlock.
    this.readingBlockTransformTracker = null;

    // @public - control if highlights are visible on this overlay
    this.pdomFocusHighlightsVisibleProperty = options.pdomFocusHighlightsVisibleProperty;
    this.interactiveHighlightsVisibleProperty = options.interactiveHighlightsVisibleProperty;
    this.readingBlockHighlightsVisibleProperty = options.readingBlockHighlightsVisibleProperty;

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

    this.highlightNode.addChild( this.shapeFocusHighlightPath );
    this.highlightNode.addChild( this.boundsFocusHighlightPath );

    // @private {FocusHighlightPath} - Focus highlight for 'groups' of Nodes. When descendant node has focus, ancestor
    // with groupFocusHighlight flag will have this extra focus highlight surround its local bounds
    this.groupFocusHighlightPath = new FocusHighlightFromNode( null, {
      useLocalBounds: true,
      useGroupDilation: true,
      outerLineWidth: FocusHighlightPath.GROUP_OUTER_LINE_WIDTH,
      innerLineWidth: FocusHighlightPath.GROUP_INNER_LINE_WIDTH,
      innerStroke: FocusHighlightPath.FOCUS_COLOR
    } );

    // @private {Node} - a parent Node for group focus highlights so visibility of all group highlights can easily
    // becontrolled
    this.groupFocusHighlightParent = new Node( {
      children: [ this.groupFocusHighlightPath ]
    } );
    this.focusRootNode.addChild( this.groupFocusHighlightParent );

    // @private {Node} - The highlight shown around ReadingBlock Nodes while the voicingManager is speaking.
    this.readingBlockHighlightPath = new ActivatedReadingBlockHighlight( null );
    this.readingBlockHighlightNode.addChild( this.readingBlockHighlightPath );

    // @private - Listeners bound once, so we can access them for removal.
    this.boundsListener = this.onBoundsChange.bind( this );
    this.transformListener = this.onTransformChange.bind( this );
    this.domFocusListener = this.onFocusChange.bind( this );
    this.readingBlockTransformListener = this.onReadingBlockTransformChange.bind( this );
    this.focusHighlightListener = this.onFocusHighlightChange.bind( this );
    this.mouseHighlightListener = this.onMouseHighlightChange.bind( this );
    this.focusHighlightsVisibleListener = this.onFocusHighlightsVisibleChange.bind( this );
    this.voicingHighlightsVisibleListener = this.onVoicingHighlightsVisibleChange.bind( this );
    this.pointerFocusListener = this.onPointerFocusChange.bind( this );
    this.lockedPointerFocusListener = this.onLockedPointerFocusChange.bind( this );
    this.readingBlockFocusListener = this.onReadingBlockFocusChange.bind( this );
    this.readingBlockHighlightChangeListener = this.onReadingBlockHighlightChange.bind( this );

    FocusManager.pdomFocusProperty.link( this.domFocusListener );
    display.focusManager.pointerFocusProperty.link( this.pointerFocusListener );
    display.focusManager.readingBlockFocusProperty.link( this.readingBlockFocusListener );

    display.focusManager.lockedPointerFocusProperty.link( this.lockedPointerFocusListener );

    this.pdomFocusHighlightsVisibleProperty.link( this.focusHighlightsVisibleListener );
    this.interactiveHighlightsVisibleProperty.link( this.voicingHighlightsVisibleListener );
  }

  /**
   * Releases references
   * @public
   */
  dispose() {
    if ( this.hasHighlight() ) {
      this.deactivateHighlight();
    }

    FocusManager.pdomFocusProperty.unlink( this.domFocusListener );
    this.pdomFocusHighlightsVisibleProperty.unlink( this.focusHighlightsVisibleListener );
    this.interactiveHighlightsVisibleProperty.unlink( this.voicingHighlightsVisibleListener );

    this.display.focusManager.pointerFocusProperty.unlink( this.pointerFocusListener );
    this.display.focusManager.readingBlockFocusProperty.unlink( this.readingBlockFocusListener );

    this.focusDisplay.dispose();
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
   * Returns true if there is an active highlight around a ReadingBlock while the
   * voicingManager is speaking its Voicing content.
   * @public
   *
   * @returns {boolean}
   */
  hasReadingBlockHighlight() {
    return !!this.readingBlockTrail;
  }

  /**
   * Activates the highlight, choosing a mode for whether the highlight will be a shape, node, or bounds.
   * @private
   *
   * @param {Trail} trail - The focused trail to highlight. It assumes that this trail is in this display.
   * @param {Node} node - Node receiving the highlight
   * @param {Node|Shape|null} nodeHighlight - the highlight to use
   * @param {boolean} layerable - Is the highlight layerable in the scene graph?
   * @param {BooleanProperty} visibleProperty - Property controlling the visibility for the provided highlight
   */
  activateHighlight( trail, node, nodeHighlight, layerable, visibleProperty ) {
    this.trail = trail;
    this.node = node;

    const highlight = nodeHighlight;
    this.activeHighlight = highlight;

    // we may or may not track this trail depending on whether the focus highlight surrounds the trail's leaf node or
    // a different node
    let trailToTrack = trail;

    // Invisible mode - no focus highlight; this is only for testing mode, when Nodes rarely have bounds.
    if ( highlight === 'invisible' ) {
      this.mode = 'invisible';
    }
    // Shape mode
    else if ( highlight instanceof Shape ) {
      this.mode = 'shape';

      this.shapeFocusHighlightPath.visible = true;
      this.shapeFocusHighlightPath.setShape( highlight );
    }
    // Node mode
    else if ( highlight instanceof Node ) {
      this.mode = 'node';

      // if using a focus highlight from another node, we will track that node's transform instead of the focused node
      if ( highlight.transformSourceNode ) {
        trailToTrack = highlight.getUniqueHighlightTrail( this.trail );
      }

      // store the focus highlight so that it can be removed later
      this.nodeModeHighlight = highlight;

      assert && assert( this.nodeModeHighlight.shape !== null,
        'The shape of the Node highlight should be set by now. Does it have bounds?' );

      if ( layerable ) {

        // flag so that we know how to deactivate in this case
        this.nodeModeHighlightLayered = true;

        // the focusHighlight is just a node in the scene graph, so set it visible - visibility of other highlights
        // controlled by visibility of parent Nodes but that cannot be done in this case because the highlight
        // can be anywhere in the scene graph, so have to check pdomFocusHighlightsVisibleProperty
        this.nodeModeHighlight.visible = visibleProperty.get();
      }
      else {

        // the node is already in the scene graph, so this will set visibility
        // for all instances.
        this.nodeModeHighlight.visible = true;

        // Use the node itself as the highlight
        this.highlightNode.addChild( this.nodeModeHighlight );
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
   * Activate a focus highlight, activating the highlight and adding a listener that will update the highlight whenever
   * the Node's focusHighlight changes
   * @private
   *
   * @param {Trail} trail
   * @param {Node} node
   */
  activateFocusHighlight( trail, node ) {
    this.activateHighlight( trail, node, node.focusHighlight, node.focusHighlightLayerable, this.pdomFocusHighlightsVisibleProperty );

    // handle any changes to the focus highlight while the node has focus
    node.focusHighlightChangedEmitter.addListener( this.focusHighlightListener );
  }

  /**
   * Activate a mouse highlight, activating the highlight and adding a listener that will update the highlight changes
   * while it is active.
   * @private
   *
   * @param {Trail} trail
   * @param {Node} node
   */
  activateMouseHighlight( trail, node ) {

    // if mouseHighlightLayerable is not set, default to focusHighlightLayerable
    const layerable = node.mouseHighlightLayerable === null ? node.focusHighlightLayerable :
                      node.mouseHighlightLayerable;

    this.activateHighlight(
      trail,
      node,
      node.mouseHighlight || node.focusHighlight,
      layerable,
      this.interactiveHighlightsVisibleProperty
    );

    // handle changes to the highlight while it is active
    node.mouseHighlightChangedEmitter.addListener( this.mouseHighlightListener );
  }

  /**
   * Activate the Reading Block highlight. This highlight is separate from others in the overlay and will always
   * surround the Bounds of the focused Node. It is shown in response to certain input on Nodes with Voicing while
   * the voicingManager is speaking.
   *
   * Note that customizations for this highlight are not supported at this time, that could be added in the future if
   * we need.
   * @private
   *
   * @param {Trail} trail
   */
  activateReadingBlockHighlight( trail ) {
    this.readingBlockTrail = trail;

    this.activeReadingBlockNode = trail.lastNode();
    assert && assert( this.activeReadingBlockNode.isReadingBlock,
      'should not activate a reading block highlight for a Node that is not a ReadingBlock' );

    const readingBlockHighlight = this.activeReadingBlockNode.readingBlockActiveHighlight;

    this.addedReadingBlockHighlight = readingBlockHighlight;

    if ( readingBlockHighlight instanceof Shape ) {
      this.readingBlockHighlightPath.setShapeFromNode( readingBlockHighlight );
      this.readingBlockHighlightPath.visible = true;
    }
    else if ( readingBlockHighlight instanceof Node ) {

      // node mode
      this.readingBlockHighlightNode.addChild( readingBlockHighlight );
    }
    else {

      // bounds mode
      this.readingBlockHighlightPath.setShapeFromNode( this.activeReadingBlockNode );
      this.readingBlockHighlightPath.visible = true;
    }

    // update the highlight if the transform for the Node ever changes
    this.readingBlockTransformTracker = new TransformTracker( this.readingBlockTrail, {
      isStatic: true
    } );
    this.readingBlockTransformTracker.addListener( this.readingBlockTransformListener );

    // update the highlight if it is changed on the Node while active
    this.activeReadingBlockNode.readingBlockActiveHighlightChangedEmitter.addListener( this.readingBlockHighlightChangeListener );

    this.readingBlockTransformDirty = true;
  }

  /**
   * Deactivate the speaking highlight by making it invisible.
   * @private
   */
  deactivateReadingBlockHighlight() {
    this.readingBlockHighlightPath.visible = false;

    if ( this.addedReadingBlockHighlight instanceof Node ) {
      this.readingBlockHighlightNode.removeChild( this.addedReadingBlockHighlight );
    }

    this.readingBlockTransformTracker.removeListener( this.readingBlockTransformListener );
    this.readingBlockTransformTracker.dispose();
    this.readingBlockTransformTracker = null;

    this.activeReadingBlockNode.readingBlockActiveHighlightChangedEmitter.removeListener( this.readingBlockHighlightChangeListener );

    this.activeReadingBlockNode = null;
    this.readingBlockTrail = null;
    this.addedReadingBlockHighlight = null;
  }

  /**
   * Deactivates the all active highlights, disposing and removing listeners as necessary.
   * @private
   */
  deactivateHighlight() {

    if ( this.mode === 'shape' ) {
      this.shapeFocusHighlightPath.visible = false;
    }
    else if ( this.mode === 'node' ) {

      // If layered, client has put the Node where they want in the scene graph and we cannot remove it
      if ( this.nodeModeHighlightLayered ) {
        this.nodeModeHighlightLayered = false;
      }
      else {
        this.highlightNode.removeChild( this.nodeModeHighlight );
      }

      // node focus highlight can be cleared now that it has been removed
      this.nodeModeHighlight.visible = false;
      this.nodeModeHighlight = null;
    }
    else if ( this.mode === 'bounds' ) {
      this.boundsFocusHighlightPath.visible = false;
      this.node.localBoundsProperty.unlink( this.boundsListener );
    }

    // remove listeners that redraw the highlight if a type of highlight changes on the Node
    if ( this.node.focusHighlightChangedEmitter.hasListener( this.focusHighlightListener ) ) {
      this.node.focusHighlightChangedEmitter.removeListener( this.focusHighlightListener );
    }
    if ( this.node.isMouseHighlighting && this.node.mouseHighlightChangedEmitter.hasListener( this.mouseHighlightListener ) ) {
      this.node.mouseHighlightChangedEmitter.removeListener( this.mouseHighlightListener );
    }

    // remove all 'group' focus highlights
    this.deactivateGroupHighlights();

    this.trail = null;
    this.node = null;
    this.mode = null;
    this.activeHighlight = null;
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
          this.groupFocusHighlightParent.addChild( highlight );

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
      if ( this.shapeFocusHighlightPath.innerHighlightColor !== HighlightOverlay.innerHighlightColor ) {
        this.shapeFocusHighlightPath.setInnerHighlightColor( HighlightOverlay.innerHighlightColor );
      }
      if ( this.shapeFocusHighlightPath.outerHighlightColor !== HighlightOverlay.outerHighlightColor ) {
        this.shapeFocusHighlightPath.setOuterHighlightColor( HighlightOverlay.outerHighlightColor );
      }
    }
    else if ( this.mode === 'bounds' ) {
      if ( this.boundsFocusHighlightPath.innerHighlightColor !== HighlightOverlay.innerHighlightColor ) {
        this.boundsFocusHighlightPath.setInnerHighlightColor( HighlightOverlay.innerHighlightColor );
      }
      if ( this.boundsFocusHighlightPath.outerHighlightColor !== HighlightOverlay.outerHighlightColor ) {
        this.boundsFocusHighlightPath.setOuterHighlightColor( HighlightOverlay.outerHighlightColor );
      }
    }

    // if a group focus highlight is active, update strokes
    if ( this.groupMode ) {
      if ( this.groupFocusHighlightPath.innerHighlightColor !== HighlightOverlay.innerGroupHighlightColor ) {
        this.groupFocusHighlightPath.setInnerHighlightColor( HighlightOverlay.innerGroupHighlightColor );
      }
      if ( this.groupFocusHighlightPath.outerHighlightColor !== HighlightOverlay.outerGroupHighlightColor ) {
        this.groupFocusHighlightPath.setOuterHighlightColor( HighlightOverlay.outerGroupHighlightColor );
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
        this.groupFocusHighlightParent.removeChild( this.groupHighlightNode );
      }

      this.groupMode = null;
      this.groupHighlightNode = null;
      this.groupTransformTracker.removeListener( this.transformListener );
      this.groupTransformTracker.dispose();
    }
  }

  /**
   * Called from HighlightOverlay after transforming the highlight. Only called when the transform changes.
   * @private
   */
  afterTransform() {
    if ( this.mode === 'shape' ) {
      this.shapeFocusHighlightPath.updateLineWidth();
    }
    else if ( this.mode === 'bounds' ) {
      this.boundsFocusHighlightPath.updateLineWidth();
    }
    else if ( this.mode === 'node' && this.activeHighlight.updateLineWidth ) {

      // Update the transform based on the transform of the node that the focusHighlight is highlighting.
      this.activeHighlight.updateLineWidth( this.node );
    }
  }

  /**
   * @private
   */
  onTransformChange() {
    this.transformDirty = true;
  }

  /**
   * Mark that the transform for the ReadingBlock highlight is out of date and needs
   * to be recalculated next update.
   * @private
   */
  onReadingBlockTransformChange() {
    this.readingBlockTransformDirty = true;
  }

  /**
   * Called when bounds change on our node when we are in "Bounds" mode
   * @private
   */
  onBoundsChange() {
    this.boundsFocusHighlightPath.setShapeFromNode( this.node );
  }

  /**
   * Called when the main Scenery focus pair (Display,Trail) changes. The Trail points to the Node that has
   * focus and a highlight will appear around this Node if focus highlights are visible.
   * @private
   *
   * @param {Focus} focus
   */
  onFocusChange( focus ) {
    const newTrail = ( focus && focus.display === this.display ) ? focus.trail : null;

    if ( this.hasHighlight() ) {
      this.deactivateHighlight();
    }

    if ( newTrail && this.pdomFocusHighlightsVisibleProperty.value ) {
      const node = newTrail.lastNode();

      this.activateFocusHighlight( newTrail, node );
    }
    else if ( this.display.focusManager.pointerFocusProperty.value && this.interactiveHighlightsVisibleProperty.value ) {
      this.updateMouseHighlight( this.display.focusManager.pointerFocusProperty.value );
    }
  }

  /**
   * Called when the pointerFocusProperty changes. pointerFocusProperty will have the Trail to the
   * Node that composes Voicing and is under the Pointer. A highlight will appear around this Node if
   * voicing highlights are visible.
   *
   * As of 8/11/21 we also decided that Interactive Highlights should also never be shown while
   * PDOM highlights are visible, to avoid confusing cases where the Interactive Highlight
   * can appear while the DOM focus highlight is active and conveying information. In the future
   * we might make it so that both can be visible at the same time, but that will require
   * changing the look of one of the highlights so it is clear they are distinct.
   * @private
   *
   * @param {Focus} focus
   */
  onPointerFocusChange( focus ) {
    if ( !this.display.focusManager.lockedPointerFocusProperty.value &&
         !this.display.focusManager.pdomFocusHighlightsVisibleProperty.value ) {
      this.updateMouseHighlight( focus );
    }
  }

  /**
   * Redraws the mouse highlight. There are cases where we want to do this regardless of whether the pointer focus
   * is locked, such as when the mouse highlight changes changes for a Node that is activated for mouse highlighting.
   * @private
   *
   * @param {Focus} focus
   */
  updateMouseHighlight( focus ) {
    const newTrail = ( focus && focus.display === this.display ) ? focus.trail : null;
    if ( this.hasHighlight() ) {
      this.deactivateHighlight();
    }

    let activated = false;
    if ( newTrail ) {
      const node = newTrail.lastNode();

      if ( ( node.isReadingBlock && this.readingBlockHighlightsVisibleProperty.value ) || ( !node.isReadingBlock && this.interactiveHighlightsVisibleProperty.value ) ) {
        this.activateMouseHighlight( newTrail, node );
        activated = true;
      }
    }

    if ( !activated && FocusManager.pdomFocus && this.pdomFocusHighlightsVisibleProperty.value ) {
      this.onFocusChange( FocusManager.pdomFocus );
    }
  }

  /**
   * Called whenever the lockedPointerFocusProperty changes. If the lockedPointerFocusProperty changes we probably
   * have to update the highlight because interaction with a Node that uses MouseHighlighting just ended.
   * @private
   *
   * @param {Focus|null} focus
   */
  onLockedPointerFocusChange( focus ) {
    this.updateMouseHighlight( focus || this.display.focusManager.pointerFocusProperty.value );
  }

  /**
   * Responsible for deactivating the Reading Block highlight when the display.focusManager.readingBlockFocusProperty changes.
   * The Reading Block waits to activate until the voicingManager starts speaking because there is often a stop speaking
   * event that comes right after the speaker starts to interrupt the previous utterance.
   * @private
   *
   * @param {Focus|null} focus
   */
  onReadingBlockFocusChange( focus ) {
    if ( this.hasReadingBlockHighlight() ) {
      this.deactivateReadingBlockHighlight();
    }

    const newTrail = ( focus && focus.display === this.display ) ? focus.trail : null;
    if ( newTrail ) {
      this.activateReadingBlockHighlight( newTrail );
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
    this.onFocusChange( FocusManager.pdomFocus );
  }

  /**
   * If the Node has pointer focus and the mouse highlight changes, we must do all of the work to reapply the
   * highlight as if the value of the focusProperty changed.
   * @private
   */
  onMouseHighlightChange() {

    if ( assert ) {
      const lockedPointerFocus = this.display.focusManager.lockedPointerFocusProperty.value;
      assert( this.node.mouseActivated || ( lockedPointerFocus && lockedPointerFocus.trail.lastNode() === this.node ),
        'Update should only be necessary if Node is activated with a Pointer or pointer focus is locked during interaction' );
    }

    this.updateMouseHighlight( this.display.focusManager.lockedPointerFocusProperty.value );
  }

  /**
   * Redraw the highlight for the ReadingBlock if it changes while the reading block highlight is already
   * active for a Node.
   * @private
   */
  onReadingBlockHighlightChange() {
    assert && assert( this.activeReadingBlockNode, 'Update should only be necessary when there is an active ReadingBlock Node' );
    assert && assert( this.activeReadingBlockNode.readingBlockActivated, 'Update should only be necessary while the ReadingBlock is activated' );
    this.onReadingBlockFocusChange( this.display.focusManager.readingBlockFocusProperty.value );
  }

  /**
   * When focus highlight visibility changes, deactivate highlights or reactivate the highlight around the Node
   * with focus.
   * @private
   */
  onFocusHighlightsVisibleChange() {
    this.onFocusChange( FocusManager.pdomFocus );
  }

  /**
   * When voicing highlight visibility changes, deactivate highlights or reactivate the highlight around the Node
   * with focus. Note that when voicing is disabled we will never set the pointerFocusProperty to prevent
   * extra work, so this function shouldn't do much. But it is here to complete the API.
   * @private
   */
  onVoicingHighlightsVisibleChange() {
    this.onPointerFocusChange( this.display.focusManager.pointerFocusProperty.value );
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
    if ( this.hasReadingBlockHighlight() && this.readingBlockTransformDirty ) {
      this.readingBlockTransformDirty = false;
      this.readingBlockHighlightNode.setMatrix( this.readingBlockTransformTracker.matrix );
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

scenery.register( 'HighlightOverlay', HighlightOverlay );
export default HighlightOverlay;