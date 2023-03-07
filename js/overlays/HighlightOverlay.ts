// Copyright 2015-2023, University of Colorado Boulder

/**
 * An overlay that implements highlights for a Display. This is responsible for drawing the highlights and
 * observing Properties or Emitters that dictate when highlights should become active. A highlight surrounds a Node
 * to indicate that it is in focus or relevant.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import BooleanProperty from '../../../axon/js/BooleanProperty.js';
import { Shape } from '../../../kite/js/imports.js';
import optionize from '../../../phet-core/js/optionize.js';
import { ActivatedReadingBlockHighlight, Display, Focus, FocusHighlightFromNode, FocusHighlightPath, FocusManager, Node, scenery, TOverlay, TPaint, Trail, TransformTracker } from '../imports.js';
import { InteractiveHighlightingNode } from '../accessibility/voicing/InteractiveHighlighting.js';
import { ReadingBlockNode } from '../accessibility/voicing/ReadingBlock.js';
import TProperty from '../../../axon/js/TProperty.js';

// colors for the focus highlights, can be changed for different application backgrounds or color profiles, see
// the setters and getters below for these values.
let outerHighlightColor: TPaint = FocusHighlightPath.OUTER_FOCUS_COLOR;
let innerHighlightColor: TPaint = FocusHighlightPath.INNER_FOCUS_COLOR;

let innerGroupHighlightColor: TPaint = FocusHighlightPath.INNER_LIGHT_GROUP_FOCUS_COLOR;
let outerGroupHighlightColor: TPaint = FocusHighlightPath.OUTER_LIGHT_GROUP_FOCUS_COLOR;

// Type for the "mode" of a particular highlight, signifying behavior for handling the active highlight.
type HighlightMode = null | 'bounds' | 'node' | 'shape' | 'invisible';

// Highlights displayed by the overlay support these types. Highlight behavior works like the following:
// - If value is null, the highlight will use default stylings of FocusHighlightPath and surround the Node with focus.
// - If value is a Shape the Shape is set to a FocusHighlightPath with default stylings in the global coordinate frame.
// - If you provide a Node it is your responsibility to position it in the global coordinate frame.
// - If the value is 'invisible' no highlight will be displayed at all.
export type Highlight = Node | Shape | null | 'invisible';

export type HighlightOverlayOptions = {

  // Controls whether highlights related to DOM focus are visible
  pdomFocusHighlightsVisibleProperty?: TProperty<boolean>;

  // Controls whether highlights related to Interactive Highlights are visible
  interactiveHighlightsVisibleProperty?: TProperty<boolean>;

  // Controls whether highlights associated with ReadingBlocks (of the Voicing feature set)
  // are shown when pointerFocusProperty changes
  readingBlockHighlightsVisibleProperty?: TProperty<boolean>;
};

export default class HighlightOverlay implements TOverlay {

  private readonly display: Display;

  // The root Node of our child display
  private readonly focusRootNode: Node;

  // Trail to the node with focus, modified when focus changes
  private trail: Trail | null = null;

  // Node with focus, modified when focus changes
  private node: Node | null = null;

  // A references to the highlight from the Node that is highlighted.
  private activeHighlight: Highlight = null;

  // Signifies method of representing focus, 'bounds'|'node'|'shape'|'invisible', modified
  // when focus changes
  private mode: HighlightMode = null;

  // Signifies method off representing group focus, 'bounds'|'node', modified when
  // focus changes
  private groupMode: HighlightMode = null;

  // The group highlight node around an ancestor of this.node when focus changes, see ParallelDOM.setGroupFocusHighlight
  // for more information on the group focus highlight, modified when focus changes
  private groupHighlightNode: Node | null = null;

  // Tracks transformations to the focused node and the node with a group focus highlight, modified when focus changes
  private transformTracker: TransformTracker | null = null;
  private groupTransformTracker: TransformTracker | null = null;

  // If a node is using a custom focus highlight, a reference is kept so that it can be removed from the overlay when
  // node focus changes.
  private nodeModeHighlight: Node | null = null;

  // If true, the active highlight is in "node" mode and is layered in the scene graph. This field lets us deactivate
  // the highlight appropriately when it is in that state.
  private nodeModeHighlightLayered = false;

  // If true, the next update() will trigger an update to the highlight's transform
  private transformDirty = true;

  // The main node for the highlight. It will be transformed.
  private readonly highlightNode = new Node();

  // The main Node for the ReadingBlock highlight, while ReadingBlock content is being spoken by speech synthesis.
  private readonly readingBlockHighlightNode = new Node();

  // A reference to the Node that is added when a custom node is specified as the active highlight for the
  // ReadingBlock. Stored so that we can remove it when deactivating reading block highlights.
  private addedReadingBlockHighlight: Highlight = null;

  // A reference to the Node that is a ReadingBlock which the Voicing framework is currently speaking about.
  private activeReadingBlockNode: null | ReadingBlockNode = null;

  // Trail to the ReadingBlock Node with an active highlight around it while the voicingManager is speaking its content.
  private readingBlockTrail: null | Trail = null;

  // Whether the transform applied to the readinBlockHighlightNode is out of date.
  private readingBlockTransformDirty = true;

  // The TransformTracker used to observe changes to the transform of the Node with Reading Block focus, so that
  // the highlight can match the ReadingBlock.
  private readingBlockTransformTracker: null | TransformTracker = null;

  // See HighlightOverlayOptions for documentation.
  private readonly pdomFocusHighlightsVisibleProperty: TProperty<boolean>;

  // See HighlightOverlayOptions for documentation.
  private readonly interactiveHighlightsVisibleProperty: TProperty<boolean>;

  // See HighlightOverlayOptions for documentation.
  private readonly readingBlockHighlightsVisibleProperty: TProperty<boolean>;

  // Display that manages all highlights
  private readonly focusDisplay: Display;

  // HTML element of the display
  public readonly domElement: HTMLElement;

  // Used as the focus highlight when the overlay is passed a shape
  private readonly shapeFocusHighlightPath: FocusHighlightPath;

  // Used as the default case for the highlight when the highlight value is null
  private readonly boundsFocusHighlightPath: FocusHighlightFromNode;

  // Focus highlight for 'groups' of Nodes. When descendant node has focus, ancestor with groupFocusHighlight flag will
  // have this extra focus highlight surround its local bounds
  private readonly groupFocusHighlightPath: FocusHighlightFromNode;

  // A parent Node for group focus highlights so visibility of all group highlights can easily be controlled
  private readonly groupFocusHighlightParent: Node;

  // The highlight shown around ReadingBlock Nodes while the voicingManager is speaking.
  private readonly readingBlockHighlightPath: ActivatedReadingBlockHighlight;

  private readonly boundsListener: () => void;
  private readonly transformListener: () => void;
  private readonly domFocusListener: ( focus: Focus | null ) => void;
  private readonly readingBlockTransformListener: () => void;
  private readonly focusHighlightListener: () => void;
  private readonly interactiveHighlightListener: () => void;
  private readonly focusHighlightsVisibleListener: () => void;
  private readonly voicingHighlightsVisibleListener: () => void;
  private readonly pointerFocusListener: ( focus: Focus | null ) => void;
  private readonly lockedPointerFocusListener: ( focus: Focus | null ) => void;
  private readonly readingBlockFocusListener: ( focus: Focus | null ) => void;
  private readonly readingBlockHighlightChangeListener: () => void;

  public constructor( display: Display, focusRootNode: Node, providedOptions?: HighlightOverlayOptions ) {

    const options = optionize<HighlightOverlayOptions>()( {

      // Controls whether highlights related to DOM focus are visible
      pdomFocusHighlightsVisibleProperty: new BooleanProperty( true ),

      // Controls whether highlights related to Interactive Highlights are visible
      interactiveHighlightsVisibleProperty: new BooleanProperty( false ),

      // Controls whether highlights associated with ReadingBlocks (of the Voicing feature set) are shown when
      // pointerFocusProperty changes
      readingBlockHighlightsVisibleProperty: new BooleanProperty( false )
    }, providedOptions );

    this.display = display;
    this.focusRootNode = focusRootNode;

    this.focusRootNode.addChild( this.highlightNode );
    this.focusRootNode.addChild( this.readingBlockHighlightNode );

    this.pdomFocusHighlightsVisibleProperty = options.pdomFocusHighlightsVisibleProperty;
    this.interactiveHighlightsVisibleProperty = options.interactiveHighlightsVisibleProperty;
    this.readingBlockHighlightsVisibleProperty = options.readingBlockHighlightsVisibleProperty;

    this.focusDisplay = new Display( this.focusRootNode, {
      allowWebGL: display.isWebGLAllowed(),
      allowCSSHacks: false,
      accessibility: false,
      interactive: false
    } );

    this.domElement = this.focusDisplay.domElement;
    this.domElement.style.pointerEvents = 'none';

    this.shapeFocusHighlightPath = new FocusHighlightPath( null );
    this.boundsFocusHighlightPath = new FocusHighlightFromNode( null, {
      useLocalBounds: true
    } );

    this.highlightNode.addChild( this.shapeFocusHighlightPath );
    this.highlightNode.addChild( this.boundsFocusHighlightPath );

    this.groupFocusHighlightPath = new FocusHighlightFromNode( null, {
      useLocalBounds: true,
      useGroupDilation: true,
      outerLineWidth: FocusHighlightPath.GROUP_OUTER_LINE_WIDTH,
      innerLineWidth: FocusHighlightPath.GROUP_INNER_LINE_WIDTH,
      innerStroke: FocusHighlightPath.OUTER_FOCUS_COLOR
    } );

    this.groupFocusHighlightParent = new Node( {
      children: [ this.groupFocusHighlightPath ]
    } );
    this.focusRootNode.addChild( this.groupFocusHighlightParent );

    this.readingBlockHighlightPath = new ActivatedReadingBlockHighlight( null );
    this.readingBlockHighlightNode.addChild( this.readingBlockHighlightPath );

    // Listeners bound once, so we can access them for removal.
    this.boundsListener = this.onBoundsChange.bind( this );
    this.transformListener = this.onTransformChange.bind( this );
    this.domFocusListener = this.onFocusChange.bind( this );
    this.readingBlockTransformListener = this.onReadingBlockTransformChange.bind( this );
    this.focusHighlightListener = this.onFocusHighlightChange.bind( this );
    this.interactiveHighlightListener = this.onInteractiveHighlightChange.bind( this );
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
   */
  public dispose(): void {
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
   * Returns whether or not this HighlightOverlay is displaying some highlight.
   */
  public hasHighlight(): boolean {
    return !!this.trail;
  }

  /**
   * Returns true if there is an active highlight around a ReadingBlock while the voicingManager is speaking its
   * Voicing content.
   */
  public hasReadingBlockHighlight(): boolean {
    return !!this.readingBlockTrail;
  }

  /**
   * Activates the highlight, choosing a mode for whether the highlight will be a shape, node, or bounds.
   *
   * @param trail - The focused trail to highlight. It assumes that this trail is in this display.
   * @param node - Node receiving the highlight
   * @param nodeHighlight - the highlight to use
   * @param layerable - Is the highlight layerable in the scene graph?
   * @param visibleProperty - Property controlling the visibility for the provided highlight
   */
  private activateHighlight( trail: Trail, node: Node, nodeHighlight: Highlight, layerable: boolean, visibleProperty: TProperty<boolean> ): void {
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
      if ( highlight instanceof FocusHighlightPath ) {
        const highlightPath = highlight;
        assert && assert( highlight.shape !== null, 'The shape of the Node highlight should be set by now. Does it have bounds?' );

        if ( highlightPath.transformSourceNode ) {
          trailToTrack = highlight.getUniqueHighlightTrail( this.trail );
        }
      }

      // store the focus highlight so that it can be removed later
      this.nodeModeHighlight = highlight;

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
   */
  private activateFocusHighlight( trail: Trail, node: Node ): void {
    this.activateHighlight( trail, node, node.focusHighlight, node.focusHighlightLayerable, this.pdomFocusHighlightsVisibleProperty );

    // handle any changes to the focus highlight while the node has focus
    node.focusHighlightChangedEmitter.addListener( this.focusHighlightListener );
  }

  /**
   * Activate an interactive highlight, activating the highlight and adding a listener that will update the highlight
   * changes while it is active.
   */
  private activateInteractiveHighlight( trail: Trail, node: InteractiveHighlightingNode ): void {

    this.activateHighlight(
      trail,
      node,
      node.interactiveHighlight || node.focusHighlight,
      node.interactiveHighlightLayerable,
      this.interactiveHighlightsVisibleProperty
    );

    // handle changes to the highlight while it is active - Since the highlight can fall back to the focus highlight
    // watch for updates to redraw when that highlight changes as well
    node.interactiveHighlightChangedEmitter.addListener( this.interactiveHighlightListener );
    node.focusHighlightChangedEmitter.addListener( this.interactiveHighlightListener );
  }

  /**
   * Activate the Reading Block highlight. This highlight is separate from others in the overlay and will always
   * surround the Bounds of the focused Node. It is shown in response to certain input on Nodes with Voicing while
   * the voicingManager is speaking.
   *
   * Note that customizations for this highlight are not supported at this time, that could be added in the future if
   * we need.
   */
  private activateReadingBlockHighlight( trail: Trail ): void {
    this.readingBlockTrail = trail;

    const readingBlockNode = trail.lastNode() as ReadingBlockNode;
    assert && assert( readingBlockNode.isReadingBlock,
      'should not activate a reading block highlight for a Node that is not a ReadingBlock' );
    this.activeReadingBlockNode = readingBlockNode;

    const readingBlockHighlight = this.activeReadingBlockNode.readingBlockActiveHighlight;

    this.addedReadingBlockHighlight = readingBlockHighlight;

    if ( readingBlockHighlight === 'invisible' ) {
      // nothing to draw
    }
    else if ( readingBlockHighlight instanceof Shape ) {
      this.readingBlockHighlightPath.setShape( readingBlockHighlight );
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
   */
  private deactivateReadingBlockHighlight(): void {
    this.readingBlockHighlightPath.visible = false;

    if ( this.addedReadingBlockHighlight instanceof Node ) {
      this.readingBlockHighlightNode.removeChild( this.addedReadingBlockHighlight );
    }

    assert && assert( this.readingBlockTransformTracker, 'How can we deactivate the TransformTracker if it wasnt assigned.' );
    const transformTracker = this.readingBlockTransformTracker!;
    transformTracker.removeListener( this.readingBlockTransformListener );
    transformTracker.dispose();
    this.readingBlockTransformTracker = null;

    assert && assert( this.activeReadingBlockNode, 'How can we deactivate the activeReadingBlockNode if it wasnt assigned.' );
    this.activeReadingBlockNode!.readingBlockActiveHighlightChangedEmitter.removeListener( this.readingBlockHighlightChangeListener );

    this.activeReadingBlockNode = null;
    this.readingBlockTrail = null;
    this.addedReadingBlockHighlight = null;
  }

  /**
   * Deactivates the all active highlights, disposing and removing listeners as necessary.
   */
  private deactivateHighlight(): void {
    assert && assert( this.node, 'Need an active Node to deactivate highlights' );
    const activeNode = this.node!;

    if ( this.mode === 'shape' ) {
      this.shapeFocusHighlightPath.visible = false;
    }
    else if ( this.mode === 'node' ) {
      assert && assert( this.nodeModeHighlight, 'How can we deactivate if nodeModeHighlight is not assigned' );
      const nodeModeHighlight = this.nodeModeHighlight!;

      // If layered, client has put the Node where they want in the scene graph and we cannot remove it
      if ( this.nodeModeHighlightLayered ) {
        this.nodeModeHighlightLayered = false;
      }
      else {
        this.highlightNode.removeChild( nodeModeHighlight );
      }

      // node focus highlight can be cleared now that it has been removed
      nodeModeHighlight.visible = false;
      this.nodeModeHighlight = null;
    }
    else if ( this.mode === 'bounds' ) {
      this.boundsFocusHighlightPath.visible = false;
      activeNode.localBoundsProperty.unlink( this.boundsListener );
    }

    // remove listeners that redraw the highlight if a type of highlight changes on the Node
    if ( activeNode.focusHighlightChangedEmitter.hasListener( this.focusHighlightListener ) ) {
      activeNode.focusHighlightChangedEmitter.removeListener( this.focusHighlightListener );
    }

    const activeInteractiveHighlightingNode = activeNode as InteractiveHighlightingNode;
    if ( activeInteractiveHighlightingNode.isInteractiveHighlighting ) {
      if ( activeInteractiveHighlightingNode.interactiveHighlightChangedEmitter.hasListener( this.interactiveHighlightListener ) ) {
        activeInteractiveHighlightingNode.interactiveHighlightChangedEmitter.removeListener( this.interactiveHighlightListener );
      }
      if ( activeInteractiveHighlightingNode.focusHighlightChangedEmitter.hasListener( this.interactiveHighlightListener ) ) {
        activeInteractiveHighlightingNode.focusHighlightChangedEmitter.removeListener( this.interactiveHighlightListener );
      }
    }

    // remove all 'group' focus highlights
    this.deactivateGroupHighlights();

    this.trail = null;
    this.node = null;
    this.mode = null;
    this.activeHighlight = null;
    this.transformTracker!.removeListener( this.transformListener );
    this.transformTracker!.dispose();
    this.transformTracker = null;
  }

  /**
   * Activate all 'group' focus highlights by searching for ancestor nodes from the node that has focus
   * and adding a rectangle around it if it has a "groupFocusHighlight". A group highlight will only appear around
   * the closest ancestor that has a one.
   */
  private activateGroupHighlights(): void {

    assert && assert( this.trail, 'must have an active trail to activate group highlights' );
    const trail = this.trail!;
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
   *
   * TODO: Support updating focus highlight strokes in 'node' mode as well?
   */
  private updateHighlightColors(): void {

    if ( this.mode === 'shape' ) {
      if ( this.shapeFocusHighlightPath.innerHighlightColor !== HighlightOverlay.getInnerHighlightColor() ) {
        this.shapeFocusHighlightPath.setInnerHighlightColor( HighlightOverlay.getInnerHighlightColor() );
      }
      if ( this.shapeFocusHighlightPath.outerHighlightColor !== HighlightOverlay.getOuterHighlightColor() ) {
        this.shapeFocusHighlightPath.setOuterHighlightColor( HighlightOverlay.getOuterHighlightColor() );
      }
    }
    else if ( this.mode === 'bounds' ) {
      if ( this.boundsFocusHighlightPath.innerHighlightColor !== HighlightOverlay.getInnerHighlightColor() ) {
        this.boundsFocusHighlightPath.setInnerHighlightColor( HighlightOverlay.getInnerHighlightColor() );
      }
      if ( this.boundsFocusHighlightPath.outerHighlightColor !== HighlightOverlay.getOuterHighlightColor() ) {
        this.boundsFocusHighlightPath.setOuterHighlightColor( HighlightOverlay.getOuterHighlightColor() );
      }
    }

    // if a group focus highlight is active, update strokes
    if ( this.groupMode ) {
      if ( this.groupFocusHighlightPath.innerHighlightColor !== HighlightOverlay.getInnerGroupHighlightColor() ) {
        this.groupFocusHighlightPath.setInnerHighlightColor( HighlightOverlay.getInnerGroupHighlightColor() );
      }
      if ( this.groupFocusHighlightPath.outerHighlightColor !== HighlightOverlay.getOuterGroupHighlightColor() ) {
        this.groupFocusHighlightPath.setOuterHighlightColor( HighlightOverlay.getOuterGroupHighlightColor() );
      }
    }
  }

  /**
   * Remove all group focus highlights by making them invisible, or removing them from the root of this overlay,
   * depending on mode.
   */
  private deactivateGroupHighlights(): void {
    if ( this.groupMode ) {
      if ( this.groupMode === 'bounds' ) {
        this.groupFocusHighlightPath.visible = false;
      }
      else if ( this.groupMode === 'node' ) {
        assert && assert( this.groupHighlightNode, 'Need a groupHighlightNode to deactivate this mode' );
        this.groupFocusHighlightParent.removeChild( this.groupHighlightNode! );
      }

      this.groupMode = null;
      this.groupHighlightNode = null;

      assert && assert( this.groupTransformTracker, 'Need a groupTransformTracker to dispose' );
      this.groupTransformTracker!.removeListener( this.transformListener );
      this.groupTransformTracker!.dispose();
      this.groupTransformTracker = null;
    }
  }

  /**
   * Called from HighlightOverlay after transforming the highlight. Only called when the transform changes.
   */
  private afterTransform(): void {
    if ( this.mode === 'shape' ) {
      this.shapeFocusHighlightPath.updateLineWidth();
    }
    else if ( this.mode === 'bounds' ) {
      this.boundsFocusHighlightPath.updateLineWidth();
    }
    else if ( this.mode === 'node' && this.activeHighlight instanceof FocusHighlightPath && this.activeHighlight.updateLineWidth ) {

      // Update the transform based on the transform of the node that the focusHighlight is highlighting.
      assert && assert( this.node, 'Need an active Node to update line width' );
      this.activeHighlight.updateLineWidth( this.node! );
    }
  }

  /**
   * Every time the transform changes on the target Node signify that updates are necessary, see the usage of the
   * TransformTrackers.
   */
  private onTransformChange(): void {
    this.transformDirty = true;
  }

  /**
   * Mark that the transform for the ReadingBlock highlight is out of date and needs to be recalculated next update.
   */
  private onReadingBlockTransformChange(): void {
    this.readingBlockTransformDirty = true;
  }

  /**
   * Called when bounds change on our node when we are in "Bounds" mode
   */
  private onBoundsChange(): void {
    assert && assert( this.node, 'Must have an active node when bounds are changing' );
    this.boundsFocusHighlightPath.setShapeFromNode( this.node! );
  }

  /**
   * Called when the main Scenery focus pair (Display,Trail) changes. The Trail points to the Node that has
   * focus and a highlight will appear around this Node if focus highlights are visible.
   */
  private onFocusChange( focus: Focus | null ): void {
    const newTrail = ( focus && focus.display === this.display ) ? focus.trail : null;

    if ( this.hasHighlight() ) {
      this.deactivateHighlight();
    }

    if ( newTrail && this.pdomFocusHighlightsVisibleProperty.value ) {
      const node = newTrail.lastNode();

      this.activateFocusHighlight( newTrail, node );
    }
    else if ( this.display.focusManager.pointerFocusProperty.value && this.interactiveHighlightsVisibleProperty.value ) {
      this.updateInteractiveHighlight( this.display.focusManager.pointerFocusProperty.value );
    }
  }

  /**
   * Called when the pointerFocusProperty changes. pointerFocusProperty will have the Trail to the
   * Node that composes Voicing and is under the Pointer. A highlight will appear around this Node if
   * voicing highlights are visible.
   */
  private onPointerFocusChange( focus: Focus | null ): void {

    // updateInteractiveHighlight will only activate the highlight if pdomFocusHighlightsVisibleProperty is false,
    // but check here as well so that we don't do work to deactivate highlights only to immediately reactivate them
    if ( !this.display.focusManager.lockedPointerFocusProperty.value &&
         !this.display.focusManager.pdomFocusHighlightsVisibleProperty.value ) {
      this.updateInteractiveHighlight( focus );
    }
  }

  /**
   * Redraws the highlight. There are cases where we want to do this regardless of whether the pointer focus
   * is locked, such as when the highlight changes changes for a Node that is activated for highlighting.
   *
   * As of 8/11/21 we also decided that Interactive Highlights should also never be shown while
   * PDOM highlights are visible, to avoid confusing cases where the Interactive Highlight
   * can appear while the DOM focus highlight is active and conveying information. In the future
   * we might make it so that both can be visible at the same time, but that will require
   * changing the look of one of the highlights so it is clear they are distinct.
   */
  private updateInteractiveHighlight( focus: Focus | null ): void {
    const newTrail = ( focus && focus.display === this.display ) ? focus.trail : null;

    // always clear the highlight if it is being removed
    if ( this.hasHighlight() ) {
      this.deactivateHighlight();
    }

    // only activate a new highlight if PDOM focus highlights are not displayed, see JSDoc
    let activated = false;
    if ( newTrail && !this.display.focusManager.pdomFocusHighlightsVisibleProperty.value ) {
      const node = newTrail.lastNode() as ReadingBlockNode;

      if ( ( node.isReadingBlock && this.readingBlockHighlightsVisibleProperty.value ) || ( !node.isReadingBlock && this.interactiveHighlightsVisibleProperty.value ) ) {
        this.activateInteractiveHighlight( newTrail, node );
        activated = true;
      }
    }

    if ( !activated && FocusManager.pdomFocus && this.pdomFocusHighlightsVisibleProperty.value ) {
      this.onFocusChange( FocusManager.pdomFocus );
    }
  }

  /**
   * Called whenever the lockedPointerFocusProperty changes. If the lockedPointerFocusProperty changes we probably
   * have to update the highlight because interaction with a Node that uses InteractiveHighlighting just ended.
   */
  private onLockedPointerFocusChange( focus: Focus | null ): void {
    this.updateInteractiveHighlight( focus || this.display.focusManager.pointerFocusProperty.value );
  }

  /**
   * Responsible for deactivating the Reading Block highlight when the display.focusManager.readingBlockFocusProperty changes.
   * The Reading Block waits to activate until the voicingManager starts speaking because there is often a stop speaking
   * event that comes right after the speaker starts to interrupt the previous utterance.
   */
  private onReadingBlockFocusChange( focus: Focus | null ): void {
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
   */
  private onFocusHighlightChange(): void {
    assert && assert( this.node && this.node.focused, 'update should only be necessary if node already has focus' );
    this.onFocusChange( FocusManager.pdomFocus );
  }

  /**
   * If the Node has pointer focus and the interacive highlight changes, we must do all of the work to reapply the
   * highlight as if the value of the focusProperty changed.
   */
  private onInteractiveHighlightChange(): void {

    if ( assert ) {
      const interactiveHighlightNode = this.node as InteractiveHighlightingNode;
      const lockedPointerFocus = this.display.focusManager.lockedPointerFocusProperty.value;
      assert( interactiveHighlightNode || ( lockedPointerFocus && lockedPointerFocus.trail.lastNode() === this.node ),
        'Update should only be necessary if Node is activated with a Pointer or pointer focus is locked during interaction' );
    }

    this.updateInteractiveHighlight( this.display.focusManager.lockedPointerFocusProperty.value );
  }

  /**
   * Redraw the highlight for the ReadingBlock if it changes while the reading block highlight is already
   * active for a Node.
   */
  private onReadingBlockHighlightChange(): void {
    assert && assert( this.activeReadingBlockNode, 'Update should only be necessary when there is an active ReadingBlock Node' );
    assert && assert( this.activeReadingBlockNode!.readingBlockActivated, 'Update should only be necessary while the ReadingBlock is activated' );
    this.onReadingBlockFocusChange( this.display.focusManager.readingBlockFocusProperty.value );
  }

  /**
   * When focus highlight visibility changes, deactivate highlights or reactivate the highlight around the Node
   * with focus.
   */
  private onFocusHighlightsVisibleChange(): void {
    this.onFocusChange( FocusManager.pdomFocus );
  }

  /**
   * When voicing highlight visibility changes, deactivate highlights or reactivate the highlight around the Node
   * with focus. Note that when voicing is disabled we will never set the pointerFocusProperty to prevent
   * extra work, so this function shouldn't do much. But it is here to complete the API.
   */
  private onVoicingHighlightsVisibleChange(): void {
    this.onPointerFocusChange( this.display.focusManager.pointerFocusProperty.value );
  }

  /**
   * Called by Display, updates this overlay in the Display.updateDisplay call.
   */
  public update(): void {

    // Transform the highlight to match the position of the node
    if ( this.hasHighlight() && this.transformDirty ) {
      this.transformDirty = false;

      assert && assert( this.transformTracker, 'The transformTracker must be available on update if transform is dirty' );
      this.highlightNode.setMatrix( this.transformTracker!.matrix );

      if ( this.groupHighlightNode ) {
        assert && assert( this.groupTransformTracker, 'The groupTransformTracker must be available on update if transform is dirty' );
        this.groupHighlightNode.setMatrix( this.groupTransformTracker!.matrix );
      }

      this.afterTransform();
    }
    if ( this.hasReadingBlockHighlight() && this.readingBlockTransformDirty ) {
      this.readingBlockTransformDirty = false;

      assert && assert( this.readingBlockTransformTracker, 'The groupTransformTracker must be available on update if transform is dirty' );
      this.readingBlockHighlightNode.setMatrix( this.readingBlockTransformTracker!.matrix );
    }

    if ( !this.display.size.equals( this.focusDisplay.size ) ) {
      this.focusDisplay.setWidthHeight( this.display.width, this.display.height );
    }
    this.focusDisplay.updateDisplay();
  }

  /**
   * Set the inner color of all focus highlights.
   */
  public static setInnerHighlightColor( color: TPaint ): void {
    innerHighlightColor = color;
  }

  /**
   * Get the inner color of all focus highlights.
   */
  public static getInnerHighlightColor(): TPaint {
    return innerHighlightColor;
  }

  /**
   * Set the outer color of all focus highlights.
   */
  public static setOuterHilightColor( color: TPaint ): void {
    outerHighlightColor = color;
  }

  /**
   * Get the outer color of all focus highlights.
   */
  public static getOuterHighlightColor(): TPaint {
    return outerHighlightColor;
  }

  /**
   * Set the inner color of all group focus highlights.
   */
  public static setInnerGroupHighlightColor( color: TPaint ): void {
    innerGroupHighlightColor = color;
  }

  /**
   * Get the inner color of all group focus highlights
   */
  public static getInnerGroupHighlightColor(): TPaint {
    return innerGroupHighlightColor;
  }

  /**
   * Set the outer color of all group focus highlight.
   */
  public static setOuterGroupHighlightColor( color: TPaint ): void {
    outerGroupHighlightColor = color;
  }

  /**
   * Get the outer color of all group focus highlights.
   */
  public static getOuterGroupHighlightColor(): TPaint {
    return outerGroupHighlightColor;
  }
}

scenery.register( 'HighlightOverlay', HighlightOverlay );
