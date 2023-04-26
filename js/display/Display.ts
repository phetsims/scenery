// Copyright 2013-2023, University of Colorado Boulder

/**
 * A persistent display of a specific Node and its descendants, which is updated at discrete points in time.
 *
 * Use display.getDOMElement or display.domElement to retrieve the Display's DOM representation.
 * Use display.updateDisplay() to trigger the visual update in the Display's DOM element.
 *
 * A standard way of using a Display with Scenery is to:
 * 1. Create a Node that will be the root
 * 2. Create a Display, referencing that node
 * 3. Make changes to the scene graph
 * 4. Call display.updateDisplay() to draw the scene graph into the Display
 * 5. Go to (3)
 *
 * Common ways to simplify the change/update loop would be to:
 * - Use Node-based events. Initialize it with Display.initializeEvents(), then
 *   add input listeners to parts of the scene graph (see Node.addInputListener).
 * - Execute code (and update the display afterwards) by using Display.updateOnRequestAnimationFrame.
 *
 * Internal documentation:
 *
 * Lifecycle information:
 *   Instance (create,dispose)
 *     - out of update:            Stateless stub is created synchronously when a Node's children are added where we
 *                                 have no relevant Instance.
 *     - start of update:          Creates first (root) instance if it doesn't exist (stateless stub).
 *     - synctree:                 Create descendant instances under stubs, fills in state, and marks removed subtree
 *                                 roots for disposal.
 *     - update instance disposal: Disposes root instances that were marked. This also disposes all descendant
 *                                 instances, and for every instance,
 *                                 it disposes the currently-attached drawables.
 *   Drawable (create,dispose)
 *     - synctree:                 Creates all drawables where necessary. If it replaces a self/group/shared drawable on
 *                                 the instance,
 *                                 that old drawable is marked for disposal.
 *     - update instance disposal: Any drawables attached to disposed instances are disposed themselves (see Instance
 *                                 lifecycle).
 *     - update drawable disposal: Any marked drawables that were replaced or removed from an instance (it didn't
 *                                 maintain a reference) are disposed.
 *
 *   add/remove drawables from blocks:
 *     - stitching changes pending "parents", marks for block update
 *     - backbones marked for disposal (e.g. instance is still there, just changed to not have a backbone) will mark
 *         drawables for block updates
 *     - add/remove drawables phase updates drawables that were marked
 *     - disposed backbone instances will only remove drawables if they weren't marked for removal previously (e.g. in
 *         case we are from a removed instance)
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Emitter from '../../../axon/js/Emitter.js';
import StrictOmit from '../../../phet-core/js/types/StrictOmit.js';
import TProperty from '../../../axon/js/TProperty.js';
import stepTimer from '../../../axon/js/stepTimer.js';
import TinyProperty from '../../../axon/js/TinyProperty.js';
import Bounds2 from '../../../dot/js/Bounds2.js';
import Dimension2 from '../../../dot/js/Dimension2.js';
import { Matrix3Type } from '../../../dot/js/Matrix3.js';
import Vector2 from '../../../dot/js/Vector2.js';
import escapeHTML from '../../../phet-core/js/escapeHTML.js';
import optionize from '../../../phet-core/js/optionize.js';
import platform from '../../../phet-core/js/platform.js';
import PhetioObject from '../../../tandem/js/PhetioObject.js';
import Tandem from '../../../tandem/js/Tandem.js';
import AriaLiveAnnouncer from '../../../utterance-queue/js/AriaLiveAnnouncer.js';
import UtteranceQueue from '../../../utterance-queue/js/UtteranceQueue.js';
import { BackboneDrawable, Block, CanvasBlock, CanvasNodeBoundsOverlay, ChangeInterval, Color, DOMBlock, DOMDrawable, Drawable, Features, FittedBlockBoundsOverlay, FocusManager, FullScreen, globalKeyStateTracker, HighlightOverlay, HitAreaOverlay, Input, InputOptions, Instance, KeyboardUtils, Node, PDOMInstance, PDOMSiblingStyle, PDOMTree, PDOMUtils, PointerAreaOverlay, PointerOverlay, Renderer, scenery, scenerySerialize, SelfDrawable, TInputListener, TOverlay, Trail, Utils, WebGLBlock } from '../imports.js';
import TEmitter from '../../../axon/js/TEmitter.js';
import SafariWorkaroundOverlay from '../overlays/SafariWorkaroundOverlay.js';

export type DisplayOptions = {
  // Initial (or override) display width
  width?: number;

  // Initial (or override) display height
  height?: number;

  // Applies CSS styles to the root DOM element that make it amenable to interactive content
  allowCSSHacks?: boolean;

  // Whether we allow the display to put a rectangle in front of everything that subtly shifts every frame, in order to
  // force repaints for https://github.com/phetsims/geometric-optics-basics/issues/31.
  allowSafariRedrawWorkaround?: boolean;

  // Usually anything displayed outside our dom element is hidden with CSS overflow.
  allowSceneOverflow?: boolean;

  // What cursor is used when no other cursor is specified
  defaultCursor?: string;

  // Initial background color
  backgroundColor?: Color | string | null;

  // Whether WebGL will preserve the drawing buffer
  // WARNING!: This can significantly reduce performance if set to true.
  preserveDrawingBuffer?: boolean;

  // Whether WebGL is enabled at all for drawables in this Display
  // Makes it possible to disable WebGL for ease of testing on non-WebGL platforms, see #289
  allowWebGL?: boolean;

  // Enables accessibility features
  accessibility?: boolean;

  // {boolean} - Enables Interactive Highlights in the HighlightOverlay. These are highlights that surround
  // interactive components when using mouse or touch which improves low vision access.
  supportsInteractiveHighlights?: boolean;

  // Whether mouse/touch/keyboard inputs are enabled (if input has been added).
  interactive?: boolean;

  // If true, input event listeners will be attached to the Display's DOM element instead of the window.
  // Normally, attaching listeners to the window is preferred (it will see mouse moves/ups outside of the browser
  // window, allowing correct button tracking), however there may be instances where a global listener is not
  // preferred.
  listenToOnlyElement?: boolean;

  // Forwarded to Input: If true, most event types will be batched until otherwise triggered.
  batchDOMEvents?: boolean;

  // If true, the input event location (based on the top-left of the browser tab's viewport, with no
  // scaling applied) will be used. Usually, this is not a safe assumption, so when false the location of the
  // display's DOM element will be used to get the correct event location. There is a slight performance hit to
  // doing so, thus this option is provided if the top-left location can be guaranteed.
  // NOTE: Rotation of the Display's DOM element (e.g. with a CSS transform) will result in an incorrect event
  //       mapping, as getBoundingClientRect() can't work with this. getBoxQuads() should fix this when browser
  //       support is available.
  assumeFullWindow?: boolean;

  // Whether Scenery will try to aggressively re-create WebGL Canvas/context instead of waiting for
  // a context restored event. Sometimes context losses can occur without a restoration afterwards, but this can
  // jump-start the process.
  // See https://github.com/phetsims/scenery/issues/347.
  aggressiveContextRecreation?: boolean;

  // Whether the `passive` flag should be set when adding and removing DOM event listeners.
  // See https://github.com/phetsims/scenery/issues/770 for more details.
  // If it is true or false, that is the value of the passive flag that will be used. If it is null, the default
  // behavior of the browser will be used.
  //
  // Safari doesn't support touch-action: none, so we NEED to not use passive events (which would not allow
  // preventDefault to do anything, so drags actually can scroll the sim).
  // Chrome also did the same "passive by default", but because we have `touch-action: none` in place, it doesn't
  // affect us, and we can potentially get performance improvements by allowing passive events.
  // See https://github.com/phetsims/scenery/issues/770 for more information.
  passiveEvents?: boolean | null;

  // Whether, if no WebGL antialiasing is detected, the backing scale can be increased to provide some
  // antialiasing benefit. See https://github.com/phetsims/scenery/issues/859.
  allowBackingScaleAntialiasing?: boolean;

  // An HTMLElement used to contain the contents of the Display
  container?: HTMLElement;

  // phet-io
  tandem?: Tandem;
};

const CUSTOM_CURSORS = {
  'scenery-grab-pointer': [ 'grab', '-moz-grab', '-webkit-grab', 'pointer' ],
  'scenery-grabbing-pointer': [ 'grabbing', '-moz-grabbing', '-webkit-grabbing', 'pointer' ]
} as Record<string, string[]>;

let globalIdCounter = 1;

export default class Display {

  // unique ID for the display instance, (scenery-internal), and useful for debugging with multiple displays.
  public readonly id: number;

  // The (integral, > 0) dimensions of the Display's DOM element (only updates the DOM element on updateDisplay())
  public readonly sizeProperty: TProperty<Dimension2>;

  // data structure for managing aria-live alerts the this Display instance
  public descriptionUtteranceQueue: UtteranceQueue;

  // Manages the various types of Focus that can go through the Display, as well as Properties
  // controlling which forms of focus should be displayed in the HighlightOverlay.
  public focusManager: FocusManager;

  // (phet-io,scenery) - Will be filled in with a phet.scenery.Input if event handling is enabled
  public _input: Input | null;

  // (scenery-internal) Whether accessibility is enabled for this particular display.
  public readonly _accessible: boolean;

  // (scenery-internal)
  public readonly _preserveDrawingBuffer: boolean;

  // (scenery-internal) map from Node ID to Instance, for fast lookup
  public _sharedCanvasInstances: Record<number, Instance>;

  // (scenery-internal) - We have a monotonically-increasing frame ID, generally for use with a pattern
  // where we can mark objects with this to note that they are either up-to-date or need refreshing due to this
  // particular frame (without having to clear that information after use). This is incremented every frame
  public _frameId: number;

  // (scenery-internal)
  public _aggressiveContextRecreation: boolean;
  public _allowBackingScaleAntialiasing: boolean;

  private readonly _allowWebGL: boolean;
  private readonly _allowCSSHacks: boolean;
  private readonly _allowSceneOverflow: boolean;
  private readonly _defaultCursor: string;

  private readonly _rootNode: Node;
  private _rootBackbone: BackboneDrawable | null; // to be filled in later
  private readonly _domElement: HTMLElement;
  private _baseInstance: Instance | null; // will be filled with the root Instance

  // Used to check against new size to see what we need to change
  private _currentSize: Dimension2;

  private _dirtyTransformRoots: Instance[];
  private _dirtyTransformRootsWithoutPass: Instance[];
  private _instanceRootsToDispose: Instance[];

  // At the end of Display.update, reduceReferences will be called on all of these. It's meant to
  // catch various objects that would usually have update() called, but if they are invisible or otherwise not updated
  // for performance, they may need to release references another way instead.
  // See https://github.com/phetsims/energy-forms-and-changes/issues/356
  private _reduceReferencesNeeded: { reduceReferences: () => void }[];

  private _drawablesToDispose: Drawable[];

  // Block changes are handled by changing the "pending" block/backbone on drawables. We
  // want to change them all after the main stitch process has completed, so we can guarantee that a single drawable is
  // removed from its previous block before being added to a new one. This is taken care of in an updateDisplay pass
  // after syncTree / stitching.
  private _drawablesToChangeBlock: Drawable[];

  // Drawables have two implicit linked-lists, "current" and "old". syncTree modifies the
  // "current" linked-list information so it is up-to-date, but needs to use the "old" information also. We move
  // updating the "current" => "old" linked-list information until after syncTree and stitching is complete, and is
  // taken care of in an updateDisplay pass.
  private _drawablesToUpdateLinks: Drawable[];

  // We store information on {ChangeInterval}s that records change interval
  // information, that may contain references. We don't want to leave those references dangling after we don't need
  // them, so they are recorded and cleaned in one of updateDisplay's phases.
  private _changeIntervalsToDispose: ChangeInterval[];

  private _lastCursor: string | null;
  private _currentBackgroundCSS: string | null;

  private _backgroundColor: Color | string | null;

  // Used for shortcut animation frame functions
  private _requestAnimationFrameID: number;

  // Listeners that will be called for every event.
  private _inputListeners: TInputListener[];

  // Whether mouse/touch/keyboard inputs are enabled (if input has been added). Simulation will still step.
  private _interactive: boolean;

  // Passed through to Input
  private _listenToOnlyElement: boolean;
  private _batchDOMEvents: boolean;
  private _assumeFullWindow: boolean;
  private _passiveEvents: boolean | null;

  // Overlays currently being displayed.
  private _overlays: TOverlay[];

  private _pointerOverlay: PointerOverlay | null;
  private _pointerAreaOverlay: PointerAreaOverlay | null;
  private _hitAreaOverlay: HitAreaOverlay | null;
  private _canvasAreaBoundsOverlay: CanvasNodeBoundsOverlay | null;
  private _fittedBlockBoundsOverlay: FittedBlockBoundsOverlay | null;

  // @assertion-only - Whether we are running the paint phase of updateDisplay() for this Display.
  private _isPainting?: boolean;

  // @assertion-only
  public _isDisposing?: boolean;

  // @assertion-only Whether disposal has started (but not finished)
  public _isDisposed?: boolean;

  // If accessible
  private _focusRootNode?: Node;
  private _focusOverlay?: HighlightOverlay;

  // (scenery-internal, if accessible)
  public _rootPDOMInstance?: PDOMInstance;

  // (if accessible)
  private _boundHandleFullScreenNavigation?: ( domEvent: KeyboardEvent ) => void;

  // If logging performance
  private perfSyncTreeCount?: number;
  private perfStitchCount?: number;
  private perfIntervalCount?: number;
  private perfDrawableBlockChangeCount?: number;
  private perfDrawableOldIntervalCount?: number;
  private perfDrawableNewIntervalCount?: number;

  /**
   * Constructs a Display that will show the rootNode and its subtree in a visual state. Default options provided below
   *
   * @param rootNode - Displays this node and all of its descendants
   * @param [providedOptions]
   */
  public constructor( rootNode: Node, providedOptions?: DisplayOptions ) {
    assert && assert( rootNode, 'rootNode is a required parameter' );

    //OHTWO TODO: hybrid batching (option to batch until an event like 'up' that might be needed for security issues)

    const options = optionize<DisplayOptions, StrictOmit<DisplayOptions, 'container'>>()( {
      // {number} - Initial display width
      width: ( providedOptions && providedOptions.container && providedOptions.container.clientWidth ) || 640,

      // {number} - Initial display height
      height: ( providedOptions && providedOptions.container && providedOptions.container.clientHeight ) || 480,

      // {boolean} - Applies CSS styles to the root DOM element that make it amenable to interactive content
      allowCSSHacks: true,

      allowSafariRedrawWorkaround: false,

      // {boolean} - Usually anything displayed outside of our dom element is hidden with CSS overflow
      allowSceneOverflow: false,

      // {string} - What cursor is used when no other cursor is specified
      defaultCursor: 'default',

      // {ColorDef} - Intial background color
      backgroundColor: null,

      // {boolean} - Whether WebGL will preserve the drawing buffer
      preserveDrawingBuffer: false,

      // {boolean} - Whether WebGL is enabled at all for drawables in this Display
      allowWebGL: true,

      // {boolean} - Enables accessibility features
      accessibility: true,

      // {boolean} - See declaration.
      supportsInteractiveHighlights: false,

      // {boolean} - Whether mouse/touch/keyboard inputs are enabled (if input has been added).
      interactive: true,

      // {boolean} - If true, input event listeners will be attached to the Display's DOM element instead of the window.
      // Normally, attaching listeners to the window is preferred (it will see mouse moves/ups outside of the browser
      // window, allowing correct button tracking), however there may be instances where a global listener is not
      // preferred.
      listenToOnlyElement: false,

      // {boolean} - Forwarded to Input: If true, most event types will be batched until otherwise triggered.
      batchDOMEvents: false,

      // {boolean} - If true, the input event location (based on the top-left of the browser tab's viewport, with no
      // scaling applied) will be used. Usually, this is not a safe assumption, so when false the location of the
      // display's DOM element will be used to get the correct event location. There is a slight performance hit to
      // doing so, thus this option is provided if the top-left location can be guaranteed.
      // NOTE: Rotation of the Display's DOM element (e.g. with a CSS transform) will result in an incorrect event
      //       mapping, as getBoundingClientRect() can't work with this. getBoxQuads() should fix this when browser
      //       support is available.
      assumeFullWindow: false,

      // {boolean} - Whether Scenery will try to aggressively re-create WebGL Canvas/context instead of waiting for
      // a context restored event. Sometimes context losses can occur without a restoration afterwards, but this can
      // jump-start the process.
      // See https://github.com/phetsims/scenery/issues/347.
      aggressiveContextRecreation: true,

      // {boolean|null} - Whether the `passive` flag should be set when adding and removing DOM event listeners.
      // See https://github.com/phetsims/scenery/issues/770 for more details.
      // If it is true or false, that is the value of the passive flag that will be used. If it is null, the default
      // behavior of the browser will be used.
      //
      // Safari doesn't support touch-action: none, so we NEED to not use passive events (which would not allow
      // preventDefault to do anything, so drags actually can scroll the sim).
      // Chrome also did the same "passive by default", but because we have `touch-action: none` in place, it doesn't
      // affect us, and we can potentially get performance improvements by allowing passive events.
      // See https://github.com/phetsims/scenery/issues/770 for more information.
      passiveEvents: platform.safari ? false : null,

      // {boolean} - Whether, if no WebGL antialiasing is detected, the backing scale can be increased so as to
      //             provide some antialiasing benefit. See https://github.com/phetsims/scenery/issues/859.
      allowBackingScaleAntialiasing: true,

      // phet-io
      tandem: Tandem.OPTIONAL
    }, providedOptions );

    this.id = globalIdCounter++;

    this._accessible = options.accessibility;
    this._preserveDrawingBuffer = options.preserveDrawingBuffer;
    this._allowWebGL = options.allowWebGL;
    this._allowCSSHacks = options.allowCSSHacks;
    this._allowSceneOverflow = options.allowSceneOverflow;

    this._defaultCursor = options.defaultCursor;

    this.sizeProperty = new TinyProperty( new Dimension2( options.width, options.height ) );

    this._currentSize = new Dimension2( -1, -1 );
    this._rootNode = rootNode;
    this._rootNode.addRootedDisplay( this );
    this._rootBackbone = null; // to be filled in later
    this._domElement = options.container ?
                       BackboneDrawable.repurposeBackboneContainer( options.container ) :
                       BackboneDrawable.createDivBackbone();

    this._sharedCanvasInstances = {};
    this._baseInstance = null; // will be filled with the root Instance
    this._frameId = 0;
    this._dirtyTransformRoots = [];
    this._dirtyTransformRootsWithoutPass = [];
    this._instanceRootsToDispose = [];
    this._reduceReferencesNeeded = [];
    this._drawablesToDispose = [];
    this._drawablesToChangeBlock = [];
    this._drawablesToUpdateLinks = [];
    this._changeIntervalsToDispose = [];
    this._lastCursor = null;
    this._currentBackgroundCSS = null;
    this._backgroundColor = null;
    this._requestAnimationFrameID = 0;
    this._input = null;
    this._inputListeners = [];
    this._interactive = options.interactive;
    this._listenToOnlyElement = options.listenToOnlyElement;
    this._batchDOMEvents = options.batchDOMEvents;
    this._assumeFullWindow = options.assumeFullWindow;
    this._passiveEvents = options.passiveEvents;
    this._aggressiveContextRecreation = options.aggressiveContextRecreation;
    this._allowBackingScaleAntialiasing = options.allowBackingScaleAntialiasing;
    this._overlays = [];
    this._pointerOverlay = null;
    this._pointerAreaOverlay = null;
    this._hitAreaOverlay = null;
    this._canvasAreaBoundsOverlay = null;
    this._fittedBlockBoundsOverlay = null;

    if ( assert ) {
      this._isPainting = false;
      this._isDisposing = false;
      this._isDisposed = false;
    }

    this.applyCSSHacks();

    this.setBackgroundColor( options.backgroundColor );

    const ariaLiveAnnouncer = new AriaLiveAnnouncer();
    this.descriptionUtteranceQueue = new UtteranceQueue( ariaLiveAnnouncer, {
      initialize: this._accessible,
      featureSpecificAnnouncingControlPropertyName: 'descriptionCanAnnounceProperty'
    } );

    if ( platform.safari && options.allowSafariRedrawWorkaround ) {
      this.addOverlay( new SafariWorkaroundOverlay( this ) );
    }

    this.focusManager = new FocusManager();

    // Features that require the HighlightOverlay
    if ( this._accessible || options.supportsInteractiveHighlights ) {
      this._focusRootNode = new Node();
      this._focusOverlay = new HighlightOverlay( this, this._focusRootNode, {
        pdomFocusHighlightsVisibleProperty: this.focusManager.pdomFocusHighlightsVisibleProperty,
        interactiveHighlightsVisibleProperty: this.focusManager.interactiveHighlightsVisibleProperty,
        readingBlockHighlightsVisibleProperty: this.focusManager.readingBlockHighlightsVisibleProperty
      } );
      this.addOverlay( this._focusOverlay );
    }

    if ( this._accessible ) {
      this._rootPDOMInstance = PDOMInstance.pool.create( null, this, new Trail() );
      sceneryLog && sceneryLog.PDOMInstance && sceneryLog.PDOMInstance(
        `Display root instance: ${this._rootPDOMInstance.toString()}` );
      PDOMTree.rebuildInstanceTree( this._rootPDOMInstance );

      // add the accessible DOM as a child of this DOM element
      assert && assert( this._rootPDOMInstance.peer, 'Peer should be created from createFromPool' );
      this._domElement.appendChild( this._rootPDOMInstance.peer!.primarySibling! );

      const ariaLiveContainer = ariaLiveAnnouncer.ariaLiveContainer;

      // add aria-live elements to the display
      this._domElement.appendChild( ariaLiveContainer );

      // set `user-select: none` on the aria-live container to prevent iOS text selection issue, see
      // https://github.com/phetsims/scenery/issues/1006
      ariaLiveContainer.style[ Features.userSelect ] = 'none';

      // Prevent focus from being lost in FullScreen mode, listener on the globalKeyStateTracker
      // because tab navigation may happen before focus is within the PDOM. See handleFullScreenNavigation
      // for more.
      this._boundHandleFullScreenNavigation = this.handleFullScreenNavigation.bind( this );
      globalKeyStateTracker.keydownEmitter.addListener( this._boundHandleFullScreenNavigation );
    }
  }

  public getDOMElement(): HTMLElement {
    return this._domElement;
  }

  public get domElement(): HTMLElement { return this.getDOMElement(); }

  /**
   * Updates the display's DOM element with the current visual state of the attached root node and its descendants
   */
  public updateDisplay(): void {
    // @ts-expect-error scenery namespace
    if ( sceneryLog && scenery.isLoggingPerformance() ) {
      this.perfSyncTreeCount = 0;
      this.perfStitchCount = 0;
      this.perfIntervalCount = 0;
      this.perfDrawableBlockChangeCount = 0;
      this.perfDrawableOldIntervalCount = 0;
      this.perfDrawableNewIntervalCount = 0;
    }

    if ( assert ) {
      Display.assertSubtreeDisposed( this._rootNode );
    }

    sceneryLog && sceneryLog.Display && sceneryLog.Display( `updateDisplay frame ${this._frameId}` );
    sceneryLog && sceneryLog.Display && sceneryLog.push();

    const firstRun = !!this._baseInstance;

    // check to see whether contents under pointers changed (and if so, send the enter/exit events) to
    // maintain consistent state
    if ( this._input ) {
      // TODO: Should this be handled elsewhere?
      this._input.validatePointers();
    }

    if ( this._accessible ) {

      // update positioning of focusable peer siblings so they are discoverable on mobile assistive devices
      this._rootPDOMInstance!.peer!.updateSubtreePositioning();
    }

    // validate bounds for everywhere that could trigger bounds listeners. we want to flush out any changes, so that we can call validateBounds()
    // from code below without triggering side effects (we assume that we are not reentrant).
    this._rootNode.validateWatchedBounds();

    if ( assertSlow ) { this._accessible && this._rootPDOMInstance!.auditRoot(); }

    if ( assertSlow ) { this._rootNode._picker.audit(); }

    // @ts-expect-error TODO Instance
    this._baseInstance = this._baseInstance || Instance.createFromPool( this, new Trail( this._rootNode ), true, false );
    this._baseInstance!.baseSyncTree();
    if ( firstRun ) {
      // @ts-expect-error TODO instance
      this.markTransformRootDirty( this._baseInstance!, this._baseInstance!.isTransformed ); // marks the transform root as dirty (since it is)
    }

    // update our drawable's linked lists where necessary
    while ( this._drawablesToUpdateLinks.length ) {
      this._drawablesToUpdateLinks.pop()!.updateLinks();
    }

    // clean change-interval information from instances, so we don't leak memory/references
    while ( this._changeIntervalsToDispose.length ) {
      this._changeIntervalsToDispose.pop()!.dispose();
    }

    this._rootBackbone = this._rootBackbone || this._baseInstance!.groupDrawable;
    assert && assert( this._rootBackbone, 'We are guaranteed a root backbone as the groupDrawable on the base instance' );
    assert && assert( this._rootBackbone === this._baseInstance!.groupDrawable, 'We don\'t want the base instance\'s groupDrawable to change' );


    if ( assertSlow ) { this._rootBackbone!.audit( true, false, true ); } // allow pending blocks / dirty

    sceneryLog && sceneryLog.Display && sceneryLog.Display( 'drawable block change phase' );
    sceneryLog && sceneryLog.Display && sceneryLog.push();
    while ( this._drawablesToChangeBlock.length ) {
      const changed = this._drawablesToChangeBlock.pop()!.updateBlock();
      // @ts-expect-error scenery namespace
      if ( sceneryLog && scenery.isLoggingPerformance() && changed ) {
        this.perfDrawableBlockChangeCount!++;
      }
    }
    sceneryLog && sceneryLog.Display && sceneryLog.pop();

    if ( assertSlow ) { this._rootBackbone!.audit( false, false, true ); } // allow only dirty
    if ( assertSlow ) { this._baseInstance!.audit( this._frameId, false ); }

    // pre-repaint phase: update relative transform information for listeners (notification) and precomputation where desired
    this.updateDirtyTransformRoots();
    // pre-repaint phase update visibility information on instances
    this._baseInstance!.updateVisibility( true, true, true, false );
    if ( assertSlow ) { this._baseInstance!.auditVisibility( true ); }

    if ( assertSlow ) { this._baseInstance!.audit( this._frameId, true ); }

    sceneryLog && sceneryLog.Display && sceneryLog.Display( 'instance root disposal phase' );
    sceneryLog && sceneryLog.Display && sceneryLog.push();
    // dispose all of our instances. disposing the root will cause all descendants to also be disposed.
    // will also dispose attached drawables (self/group/etc.)
    while ( this._instanceRootsToDispose.length ) {
      this._instanceRootsToDispose.pop()!.dispose();
    }
    sceneryLog && sceneryLog.Display && sceneryLog.pop();

    if ( assertSlow ) { this._rootNode.auditInstanceSubtreeForDisplay( this ); } // make sure trails are valid

    sceneryLog && sceneryLog.Display && sceneryLog.Display( 'drawable disposal phase' );
    sceneryLog && sceneryLog.Display && sceneryLog.push();
    // dispose all of our other drawables.
    while ( this._drawablesToDispose.length ) {
      this._drawablesToDispose.pop()!.dispose();
    }
    sceneryLog && sceneryLog.Display && sceneryLog.pop();

    if ( assertSlow ) { this._baseInstance!.audit( this._frameId, false ); }

    if ( assert ) {
      assert( !this._isPainting, 'Display was already updating paint, may have thrown an error on the last update' );
      this._isPainting = true;
    }

    // repaint phase
    //OHTWO TODO: can anything be updated more efficiently by tracking at the Display level? Remember, we have recursive updates so things get updated in the right order!
    sceneryLog && sceneryLog.Display && sceneryLog.Display( 'repaint phase' );
    sceneryLog && sceneryLog.Display && sceneryLog.push();
    this._rootBackbone!.update();
    sceneryLog && sceneryLog.Display && sceneryLog.pop();

    if ( assert ) {
      this._isPainting = false;
    }

    if ( assertSlow ) { this._rootBackbone!.audit( false, false, false ); } // allow nothing
    if ( assertSlow ) { this._baseInstance!.audit( this._frameId, false ); }

    this.updateCursor();
    this.updateBackgroundColor();

    this.updateSize();

    if ( this._overlays.length ) {
      let zIndex = this._rootBackbone!.lastZIndex!;
      for ( let i = 0; i < this._overlays.length; i++ ) {
        // layer the overlays properly
        const overlay = this._overlays[ i ];
        overlay.domElement.style.zIndex = '' + ( zIndex++ );

        overlay.update();
      }
    }

    // After our update and disposals, we want to eliminate any memory leaks from anything that wasn't updated.
    while ( this._reduceReferencesNeeded.length ) {
      this._reduceReferencesNeeded.pop()!.reduceReferences();
    }

    this._frameId++;

    // @ts-expect-error TODO scenery namespace
    if ( sceneryLog && scenery.isLoggingPerformance() ) {
      const syncTreeMessage = `syncTree count: ${this.perfSyncTreeCount}`;
      if ( this.perfSyncTreeCount! > 500 ) {
        sceneryLog.PerfCritical && sceneryLog.PerfCritical( syncTreeMessage );
      }
      else if ( this.perfSyncTreeCount! > 100 ) {
        sceneryLog.PerfMajor && sceneryLog.PerfMajor( syncTreeMessage );
      }
      else if ( this.perfSyncTreeCount! > 20 ) {
        sceneryLog.PerfMinor && sceneryLog.PerfMinor( syncTreeMessage );
      }
      else if ( this.perfSyncTreeCount! > 0 ) {
        sceneryLog.PerfVerbose && sceneryLog.PerfVerbose( syncTreeMessage );
      }

      const drawableBlockCountMessage = `drawable block changes: ${this.perfDrawableBlockChangeCount} for` +
                                        ` -${this.perfDrawableOldIntervalCount
                                        } +${this.perfDrawableNewIntervalCount}`;
      if ( this.perfDrawableBlockChangeCount! > 200 ) {
        sceneryLog.PerfCritical && sceneryLog.PerfCritical( drawableBlockCountMessage );
      }
      else if ( this.perfDrawableBlockChangeCount! > 60 ) {
        sceneryLog.PerfMajor && sceneryLog.PerfMajor( drawableBlockCountMessage );
      }
      else if ( this.perfDrawableBlockChangeCount! > 10 ) {
        sceneryLog.PerfMinor && sceneryLog.PerfMinor( drawableBlockCountMessage );
      }
      else if ( this.perfDrawableBlockChangeCount! > 0 ) {
        sceneryLog.PerfVerbose && sceneryLog.PerfVerbose( drawableBlockCountMessage );
      }
    }

    PDOMTree.auditPDOMDisplays( this.rootNode );

    sceneryLog && sceneryLog.Display && sceneryLog.pop();
  }

  // Used for Studio Autoselect to determine the leafiest PhET-iO Element under the mouse
  public getPhetioElementAt( point: Vector2 ): PhetioObject | null {
    const node = this._rootNode.getPhetioMouseHit( point );
    return node && node.isPhetioInstrumented() ? node : null;
  }

  private updateSize(): void {
    let sizeDirty = false;
    //OHTWO TODO: if we aren't clipping or setting background colors, can we get away with having a 0x0 container div and using absolutely-positioned children?
    if ( this.size.width !== this._currentSize.width ) {
      sizeDirty = true;
      this._currentSize.width = this.size.width;
      this._domElement.style.width = `${this.size.width}px`;
    }
    if ( this.size.height !== this._currentSize.height ) {
      sizeDirty = true;
      this._currentSize.height = this.size.height;
      this._domElement.style.height = `${this.size.height}px`;
    }
    if ( sizeDirty && !this._allowSceneOverflow ) {
      // to prevent overflow, we add a CSS clip
      //TODO: 0px => 0?
      this._domElement.style.clip = `rect(0px,${this.size.width}px,${this.size.height}px,0px)`;
    }
  }

  /**
   * Whether WebGL is allowed to be used in drawables for this Display
   */
  public isWebGLAllowed(): boolean {
    return this._allowWebGL;
  }

  public get webglAllowed(): boolean { return this.isWebGLAllowed(); }

  public getRootNode(): Node {
    return this._rootNode;
  }

  public get rootNode(): Node { return this.getRootNode(); }

  public getRootBackbone(): BackboneDrawable {
    assert && assert( this._rootBackbone );
    return this._rootBackbone!;
  }

  public get rootBackbone(): BackboneDrawable { return this.getRootBackbone(); }

  /**
   * The dimensions of the Display's DOM element
   */
  public getSize(): Dimension2 {
    return this.sizeProperty.value;
  }

  public get size(): Dimension2 { return this.getSize(); }

  public getBounds(): Bounds2 {
    return this.size.toBounds();
  }

  public get bounds(): Bounds2 { return this.getBounds(); }

  /**
   * Changes the size that the Display's DOM element will be after the next updateDisplay()
   */
  public setSize( size: Dimension2 ): void {
    assert && assert( size.width % 1 === 0, 'Display.width should be an integer' );
    assert && assert( size.width > 0, 'Display.width should be greater than zero' );
    assert && assert( size.height % 1 === 0, 'Display.height should be an integer' );
    assert && assert( size.height > 0, 'Display.height should be greater than zero' );

    this.sizeProperty.value = size;
  }

  /**
   * Changes the size that the Display's DOM element will be after the next updateDisplay()
   */
  public setWidthHeight( width: number, height: number ): void {
    this.setSize( new Dimension2( width, height ) );
  }

  /**
   * The width of the Display's DOM element
   */
  public getWidth(): number {
    return this.size.width;
  }

  public get width(): number { return this.getWidth(); }

  public set width( value: number ) { this.setWidth( value ); }

  /**
   * Sets the width that the Display's DOM element will be after the next updateDisplay(). Should be an integral value.
   */
  public setWidth( width: number ): this {

    if ( this.getWidth() !== width ) {
      this.setSize( new Dimension2( width, this.getHeight() ) );
    }

    return this;
  }

  /**
   * The height of the Display's DOM element
   */
  public getHeight(): number {
    return this.size.height;
  }

  public get height(): number { return this.getHeight(); }

  public set height( value: number ) { this.setHeight( value ); }

  /**
   * Sets the height that the Display's DOM element will be after the next updateDisplay(). Should be an integral value.
   */
  public setHeight( height: number ): this {

    if ( this.getHeight() !== height ) {
      this.setSize( new Dimension2( this.getWidth(), height ) );
    }

    return this;
  }

  /**
   * Will be applied to the root DOM element on updateDisplay(), and no sooner.
   */
  public setBackgroundColor( color: Color | string | null ): this {
    assert && assert( color === null || typeof color === 'string' || color instanceof Color );

    this._backgroundColor = color;

    return this;
  }

  public set backgroundColor( value: Color | string | null ) { this.setBackgroundColor( value ); }

  public get backgroundColor(): Color | string | null { return this.getBackgroundColor(); }

  public getBackgroundColor(): Color | string | null {
    return this._backgroundColor;
  }

  public get interactive(): boolean { return this._interactive; }

  public set interactive( value: boolean ) {
    if ( this._accessible && value !== this._interactive ) {
      this._rootPDOMInstance!.peer!.recursiveDisable( !value );
    }

    this._interactive = value;
    if ( !this._interactive && this._input ) {
      this._input.interruptPointers();
      this._input.clearBatchedEvents();
      this._input.removeTemporaryPointers();
      this._rootNode.interruptSubtreeInput();
      this.interruptInput();
    }
  }

  /**
   * Adds an overlay to the Display. Each overlay should have a .domElement (the DOM element that will be used for
   * display) and an .update() method.
   */
  public addOverlay( overlay: TOverlay ): void {
    this._overlays.push( overlay );
    this._domElement.appendChild( overlay.domElement );

    // ensure that the overlay is hidden from screen readers, all accessible content should be in the dom element
    // of the this._rootPDOMInstance
    overlay.domElement.setAttribute( 'aria-hidden', 'true' );
  }

  /**
   * Removes an overlay from the display.
   */
  public removeOverlay( overlay: TOverlay ): void {
    this._domElement.removeChild( overlay.domElement );
    this._overlays.splice( _.indexOf( this._overlays, overlay ), 1 );
  }

  /**
   * Get the root accessible DOM element which represents this display and provides semantics for assistive
   * technology. If this Display is not accessible, returns null.
   */
  public getPDOMRootElement(): HTMLElement | null {
    return this._accessible ? this._rootPDOMInstance!.peer!.primarySibling : null;
  }

  public get pdomRootElement(): HTMLElement | null { return this.getPDOMRootElement(); }

  /**
   * Has this Display enabled accessibility features like PDOM creation and support.
   */
  public isAccessible(): boolean {
    return this._accessible;
  }

  /**
   * Implements a workaround that prevents DOM focus from leaving the Display in FullScreen mode. There is
   * a bug in some browsers where DOM focus can be permanently lost if tabbing out of the FullScreen element,
   * see https://github.com/phetsims/scenery/issues/883.
   */
  private handleFullScreenNavigation( domEvent: KeyboardEvent ): void {
    assert && assert( this.pdomRootElement, 'There must be a PDOM to support keyboard navigation' );

    if ( FullScreen.isFullScreen() && KeyboardUtils.isKeyEvent( domEvent, KeyboardUtils.KEY_TAB ) ) {
      const rootElement = this.pdomRootElement;
      const nextElement = domEvent.shiftKey ? PDOMUtils.getPreviousFocusable( rootElement || undefined ) :
                          PDOMUtils.getNextFocusable( rootElement || undefined );
      if ( nextElement === domEvent.target ) {
        domEvent.preventDefault();
      }
    }
  }

  /**
   * Returns the bitmask union of all renderers (canvas/svg/dom/webgl) that are used for display, excluding
   * BackboneDrawables (which would be DOM).
   */
  public getUsedRenderersBitmask(): number {
    function renderersUnderBackbone( backbone: BackboneDrawable ): number {
      let bitmask = 0;
      _.each( backbone.blocks, block => {
        if ( block instanceof DOMBlock && block.domDrawable instanceof BackboneDrawable ) {
          bitmask = bitmask | renderersUnderBackbone( block.domDrawable );
        }
        else {
          bitmask = bitmask | block.renderer;
        }
      } );
      return bitmask;
    }

    // only return the renderer-specific portion (no other hints, etc)
    return renderersUnderBackbone( this._rootBackbone! ) & Renderer.bitmaskRendererArea;
  }

  /**
   * Called from Instances that will need a transform update (for listeners and precomputation). (scenery-internal)
   *
   * @param instance
   * @param passTransform - Whether we should pass the first transform root when validating transforms (should
   * be true if the instance is transformed)
   */
  public markTransformRootDirty( instance: Instance, passTransform: boolean ): void {
    passTransform ? this._dirtyTransformRoots.push( instance ) : this._dirtyTransformRootsWithoutPass.push( instance );
  }

  private updateDirtyTransformRoots(): void {
    sceneryLog && sceneryLog.transformSystem && sceneryLog.transformSystem( 'updateDirtyTransformRoots' );
    sceneryLog && sceneryLog.transformSystem && sceneryLog.push();
    while ( this._dirtyTransformRoots.length ) {
      this._dirtyTransformRoots.pop()!.relativeTransform.updateTransformListenersAndCompute( false, false, this._frameId, true );
    }
    while ( this._dirtyTransformRootsWithoutPass.length ) {
      this._dirtyTransformRootsWithoutPass.pop()!.relativeTransform.updateTransformListenersAndCompute( false, false, this._frameId, false );
    }
    sceneryLog && sceneryLog.transformSystem && sceneryLog.pop();
  }

  /**
   * (scenery-internal)
   */
  public markDrawableChangedBlock( drawable: Drawable ): void {
    sceneryLog && sceneryLog.Display && sceneryLog.Display( `markDrawableChangedBlock: ${drawable.toString()}` );
    this._drawablesToChangeBlock.push( drawable );
  }

  /**
   * Marks an item for later reduceReferences() calls at the end of Display.update().
   * (scenery-internal)
   */
  public markForReducedReferences( item: { reduceReferences: () => void } ): void {
    assert && assert( !!item.reduceReferences );

    this._reduceReferencesNeeded.push( item );
  }

  /**
   * (scenery-internal)
   */
  public markInstanceRootForDisposal( instance: Instance ): void {
    sceneryLog && sceneryLog.Display && sceneryLog.Display( `markInstanceRootForDisposal: ${instance.toString()}` );
    this._instanceRootsToDispose.push( instance );
  }

  /**
   * (scenery-internal)
   */
  public markDrawableForDisposal( drawable: Drawable ): void {
    sceneryLog && sceneryLog.Display && sceneryLog.Display( `markDrawableForDisposal: ${drawable.toString()}` );
    this._drawablesToDispose.push( drawable );
  }

  /**
   * (scenery-internal)
   */
  public markDrawableForLinksUpdate( drawable: Drawable ): void {
    this._drawablesToUpdateLinks.push( drawable );
  }

  /**
   * Add a {ChangeInterval} for the "remove change interval info" phase (we don't want to leak memory/references)
   * (scenery-internal)
   */
  public markChangeIntervalToDispose( changeInterval: ChangeInterval ): void {
    this._changeIntervalsToDispose.push( changeInterval );
  }

  private updateBackgroundColor(): void {
    assert && assert( this._backgroundColor === null ||
                      typeof this._backgroundColor === 'string' ||
                      this._backgroundColor instanceof Color );

    const newBackgroundCSS = this._backgroundColor === null ?
                             '' :
                             ( ( this._backgroundColor as Color ).toCSS ?
                               ( this._backgroundColor as Color ).toCSS() :
                               this._backgroundColor as string );
    if ( newBackgroundCSS !== this._currentBackgroundCSS ) {
      this._currentBackgroundCSS = newBackgroundCSS;

      this._domElement.style.backgroundColor = newBackgroundCSS;
    }
  }

  /*---------------------------------------------------------------------------*
   * Cursors
   *----------------------------------------------------------------------------*/

  private updateCursor(): void {
    if ( this._input && this._input.mouse && this._input.mouse.point ) {
      if ( this._input.mouse.cursor ) {
        sceneryLog && sceneryLog.Cursor && sceneryLog.Cursor( `set on pointer: ${this._input.mouse.cursor}` );
        this.setSceneCursor( this._input.mouse.cursor );
        return;
      }

      //OHTWO TODO: For a display, just return an instance and we can avoid the garbage collection/mutation at the cost of the linked-list traversal instead of an array
      const mouseTrail = this._rootNode.trailUnderPointer( this._input.mouse );

      if ( mouseTrail ) {
        for ( let i = mouseTrail.getCursorCheckIndex(); i >= 0; i-- ) {
          const node = mouseTrail.nodes[ i ];
          const cursor = node.getEffectiveCursor();

          if ( cursor ) {
            sceneryLog && sceneryLog.Cursor && sceneryLog.Cursor( `${cursor} on ${node.constructor.name}#${node.id}` );
            this.setSceneCursor( cursor );
            return;
          }
        }
      }

      sceneryLog && sceneryLog.Cursor && sceneryLog.Cursor( `--- for ${mouseTrail ? mouseTrail.toString() : '(no hit)'}` );
    }

    // fallback case
    this.setSceneCursor( this._defaultCursor );
  }

  /**
   * Sets the cursor to be displayed when over the Display.
   */
  private setElementCursor( cursor: string ): void {
    this._domElement.style.cursor = cursor;

    // In some cases, Chrome doesn't seem to respect the cursor set on the Display's domElement. If we are using the
    // full window, we can apply the workaround of controlling the body's style.
    // See https://github.com/phetsims/scenery/issues/983
    if ( this._assumeFullWindow ) {
      document.body.style.cursor = cursor;
    }
  }

  private setSceneCursor( cursor: string ): void {
    if ( cursor !== this._lastCursor ) {
      this._lastCursor = cursor;
      const customCursors = CUSTOM_CURSORS[ cursor ];
      if ( customCursors ) {
        // go backwards, so the most desired cursor sticks
        for ( let i = customCursors.length - 1; i >= 0; i-- ) {
          this.setElementCursor( customCursors[ i ] );
        }
      }
      else {
        this.setElementCursor( cursor );
      }
    }
  }

  private applyCSSHacks(): void {
    // to use CSS3 transforms for performance, hide anything outside our bounds by default
    if ( !this._allowSceneOverflow ) {
      this._domElement.style.overflow = 'hidden';
    }

    // forward all pointer events
    // @ts-expect-error legacy
    this._domElement.style.msTouchAction = 'none';

    // don't allow browser to switch between font smoothing methods for text (see https://github.com/phetsims/scenery/issues/431)
    Features.setStyle( this._domElement, Features.fontSmoothing, 'antialiased' );

    if ( this._allowCSSHacks ) {
      // Prevents selection cursor issues in Safari, see https://github.com/phetsims/scenery/issues/476
      document.onselectstart = () => false;

      // prevent any default zooming behavior from a trackpad on IE11 and Edge, all should be handled by scenery - must
      // be on the body, doesn't prevent behavior if on the display div
      // @ts-expect-error legacy
      document.body.style.msContentZooming = 'none';

      // some css hacks (inspired from https://github.com/EightMedia/hammer.js/blob/master/hammer.js).
      // modified to only apply the proper prefixed version instead of spamming all of them, and doesn't use jQuery.
      Features.setStyle( this._domElement, Features.userDrag, 'none' );
      Features.setStyle( this._domElement, Features.userSelect, 'none' );
      Features.setStyle( this._domElement, Features.touchAction, 'none' );
      Features.setStyle( this._domElement, Features.touchCallout, 'none' );
      Features.setStyle( this._domElement, Features.tapHighlightColor, 'rgba(0,0,0,0)' );
    }
  }

  public canvasDataURL( callback: ( str: string ) => void ): void {
    this.canvasSnapshot( ( canvas: HTMLCanvasElement ) => {
      callback( canvas.toDataURL() );
    } );
  }

  /**
   * Renders what it can into a Canvas (so far, Canvas and SVG layers work fine)
   */
  public canvasSnapshot( callback: ( canvas: HTMLCanvasElement, imageData: ImageData ) => void ): void {
    const canvas = document.createElement( 'canvas' );
    canvas.width = this.size.width;
    canvas.height = this.size.height;

    const context = canvas.getContext( '2d' )!;

    //OHTWO TODO: allow actual background color directly, not having to check the style here!!!
    this._rootNode.renderToCanvas( canvas, context, () => {
      callback( canvas, context.getImageData( 0, 0, canvas.width, canvas.height ) );
    }, this.domElement.style.backgroundColor );
  }

  /**
   * TODO: reduce code duplication for handling overlays
   */
  public setPointerDisplayVisible( visibility: boolean ): void {
    const hasOverlay = !!this._pointerOverlay;

    if ( visibility !== hasOverlay ) {
      if ( !visibility ) {
        this.removeOverlay( this._pointerOverlay! );
        this._pointerOverlay!.dispose();
        this._pointerOverlay = null;
      }
      else {
        this._pointerOverlay = new PointerOverlay( this, this._rootNode );
        this.addOverlay( this._pointerOverlay );
      }
    }
  }

  /**
   * TODO: reduce code duplication for handling overlays
   */
  public setPointerAreaDisplayVisible( visibility: boolean ): void {
    const hasOverlay = !!this._pointerAreaOverlay;

    if ( visibility !== hasOverlay ) {
      if ( !visibility ) {
        this.removeOverlay( this._pointerAreaOverlay! );
        this._pointerAreaOverlay!.dispose();
        this._pointerAreaOverlay = null;
      }
      else {
        this._pointerAreaOverlay = new PointerAreaOverlay( this, this._rootNode );
        this.addOverlay( this._pointerAreaOverlay );
      }
    }
  }

  /**
   * TODO: reduce code duplication for handling overlays
   */
  public setHitAreaDisplayVisible( visibility: boolean ): void {
    const hasOverlay = !!this._hitAreaOverlay;

    if ( visibility !== hasOverlay ) {
      if ( !visibility ) {
        this.removeOverlay( this._hitAreaOverlay! );
        this._hitAreaOverlay!.dispose();
        this._hitAreaOverlay = null;
      }
      else {
        this._hitAreaOverlay = new HitAreaOverlay( this, this._rootNode );
        this.addOverlay( this._hitAreaOverlay );
      }
    }
  }

  /**
   * TODO: reduce code duplication for handling overlays
   */
  public setCanvasNodeBoundsVisible( visibility: boolean ): void {
    const hasOverlay = !!this._canvasAreaBoundsOverlay;

    if ( visibility !== hasOverlay ) {
      if ( !visibility ) {
        this.removeOverlay( this._canvasAreaBoundsOverlay! );
        this._canvasAreaBoundsOverlay!.dispose();
        this._canvasAreaBoundsOverlay = null;
      }
      else {
        this._canvasAreaBoundsOverlay = new CanvasNodeBoundsOverlay( this, this._rootNode );
        this.addOverlay( this._canvasAreaBoundsOverlay );
      }
    }
  }

  /**
   * TODO: reduce code duplication for handling overlays
   */
  public setFittedBlockBoundsVisible( visibility: boolean ): void {
    const hasOverlay = !!this._fittedBlockBoundsOverlay;

    if ( visibility !== hasOverlay ) {
      if ( !visibility ) {
        this.removeOverlay( this._fittedBlockBoundsOverlay! );
        this._fittedBlockBoundsOverlay!.dispose();
        this._fittedBlockBoundsOverlay = null;
      }
      else {
        this._fittedBlockBoundsOverlay = new FittedBlockBoundsOverlay( this, this._rootNode );
        this.addOverlay( this._fittedBlockBoundsOverlay );
      }
    }
  }

  /**
   * Sets up the Display to resize to whatever the window inner dimensions will be.
   */
  public resizeOnWindowResize(): void {
    const resizer = () => {
      this.setWidthHeight( window.innerWidth, window.innerHeight ); // eslint-disable-line bad-sim-text
    };
    window.addEventListener( 'resize', resizer );
    resizer();
  }

  /**
   * Updates on every request animation frame. If stepCallback is passed in, it is called before updateDisplay() with
   * stepCallback( timeElapsedInSeconds )
   */
  public updateOnRequestAnimationFrame( stepCallback?: ( dt: number ) => void ): void {
    // keep track of how much time elapsed over the last frame
    let lastTime = 0;
    let timeElapsedInSeconds = 0;

    const self = this; // eslint-disable-line @typescript-eslint/no-this-alias
    ( function step() {
      // @ts-expect-error LEGACY --- it would know to update just the DOM element's location if it's the second argument
      self._requestAnimationFrameID = window.requestAnimationFrame( step, self._domElement );

      // calculate how much time has elapsed since we rendered the last frame
      const timeNow = Date.now();
      if ( lastTime !== 0 ) {
        timeElapsedInSeconds = ( timeNow - lastTime ) / 1000.0;
      }
      lastTime = timeNow;

      // step the timer that drives any time dependent updates of the Display
      stepTimer.emit( timeElapsedInSeconds );

      stepCallback && stepCallback( timeElapsedInSeconds );
      self.updateDisplay();
    } )();
  }

  public cancelUpdateOnRequestAnimationFrame(): void {
    window.cancelAnimationFrame( this._requestAnimationFrameID );
  }

  /**
   * Initializes event handling, and connects the browser's input event handlers to notify this Display of events.
   *
   * NOTE: This can be reversed with detachEvents().
   */
  public initializeEvents( options?: InputOptions ): void {
    assert && assert( !this._input, 'Events cannot be attached twice to a display (for now)' );

    // TODO: refactor here
    const input = new Input( this, !this._listenToOnlyElement, this._batchDOMEvents, this._assumeFullWindow, this._passiveEvents, options );
    this._input = input;

    input.connectListeners();
  }

  /**
   * Detach already-attached input event handling (from initializeEvents()).
   */
  public detachEvents(): void {
    assert && assert( this._input, 'detachEvents() should be called only when events are attached' );

    this._input!.disconnectListeners();
    this._input = null;
  }


  /**
   * Adds an input listener.
   */
  public addInputListener( listener: TInputListener ): this {
    assert && assert( !_.includes( this._inputListeners, listener ), 'Input listener already registered on this Display' );

    // don't allow listeners to be added multiple times
    if ( !_.includes( this._inputListeners, listener ) ) {
      this._inputListeners.push( listener );
    }
    return this;
  }

  /**
   * Removes an input listener that was previously added with addInputListener.
   */
  public removeInputListener( listener: TInputListener ): this {
    // ensure the listener is in our list
    assert && assert( _.includes( this._inputListeners, listener ) );

    this._inputListeners.splice( _.indexOf( this._inputListeners, listener ), 1 );

    return this;
  }

  /**
   * Returns whether this input listener is currently listening to this Display.
   *
   * More efficient than checking display.inputListeners, as that includes a defensive copy.
   */
  public hasInputListener( listener: TInputListener ): boolean {
    for ( let i = 0; i < this._inputListeners.length; i++ ) {
      if ( this._inputListeners[ i ] === listener ) {
        return true;
      }
    }
    return false;
  }

  /**
   * Returns a copy of all of our input listeners.
   */
  public getInputListeners(): TInputListener[] {
    return this._inputListeners.slice( 0 ); // defensive copy
  }

  public get inputListeners(): TInputListener[] { return this.getInputListeners(); }

  /**
   * Interrupts all input listeners that are attached to this Display.
   */
  public interruptInput(): this {
    const listenersCopy = this.inputListeners;

    for ( let i = 0; i < listenersCopy.length; i++ ) {
      const listener = listenersCopy[ i ];

      listener.interrupt && listener.interrupt();
    }

    return this;
  }

  /**
   * (scenery-internal)
   */
  public ensureNotPainting(): void {
    assert && assert( !this._isPainting,
      'This should not be run in the call tree of updateDisplay(). If you see this, it is likely that either the ' +
      'last updateDisplay() had a thrown error and it is trying to be run again (in which case, investigate that ' +
      'error), OR code was run/triggered from inside an updateDisplay() that has the potential to cause an infinite ' +
      'loop, e.g. CanvasNode paintCanvas() call manipulating another Node, or a bounds listener that Scenery missed.' );
  }

  /**
   * Triggers a loss of context for all WebGL blocks.
   *
   * NOTE: Should generally only be used for debugging.
   */
  public loseWebGLContexts(): void {
    ( function loseBackbone( backbone: BackboneDrawable ) {
      if ( backbone.blocks ) {
        backbone.blocks.forEach( ( block: Block ) => {
          const gl = ( block as unknown as WebGLBlock ).gl;
          if ( gl ) {
            Utils.loseContext( gl );
          }

          //TODO: pattern for this iteration
          for ( let drawable = block.firstDrawable; drawable !== null; drawable = drawable.nextDrawable ) {
            loseBackbone( drawable );
            if ( drawable === block.lastDrawable ) { break; }
          }
        } );
      }
    } )( this._rootBackbone! );
  }

  /**
   * Makes this Display available for inspection.
   */
  public inspect(): void {
    localStorage.scenerySnapshot = JSON.stringify( scenerySerialize( this ) );
  }

  /**
   * Returns an HTML fragment that includes a large amount of debugging information, including a view of the
   * instance tree and drawable tree.
   */
  public getDebugHTML(): string {
    const headerStyle = 'font-weight: bold; font-size: 120%; margin-top: 5px;';

    let depth = 0;

    let result = '';

    result += `<div style="${headerStyle}">Display (${this.id}) Summary</div>`;
    result += `${this.size.toString()} frame:${this._frameId} input:${!!this._input} cursor:${this._lastCursor}<br/>`;

    function nodeCount( node: Node ): number {
      let count = 1; // for us
      for ( let i = 0; i < node.children.length; i++ ) {
        count += nodeCount( node.children[ i ] );
      }
      return count;
    }

    result += `Nodes: ${nodeCount( this._rootNode )}<br/>`;

    function instanceCount( instance: Instance ): number {
      let count = 1; // for us
      for ( let i = 0; i < instance.children.length; i++ ) {
        count += instanceCount( instance.children[ i ] );
      }
      return count;
    }

    result += this._baseInstance ? ( `Instances: ${instanceCount( this._baseInstance )}<br/>` ) : '';

    function drawableCount( drawable: Drawable ): number {
      let count = 1; // for us
      if ( ( drawable as unknown as BackboneDrawable ).blocks ) {
        // we're a backbone
        _.each( ( drawable as unknown as BackboneDrawable ).blocks, childDrawable => {
          count += drawableCount( childDrawable );
        } );
      }
      else if ( ( drawable as unknown as Block ).firstDrawable && ( drawable as unknown as Block ).lastDrawable ) {
        // we're a block
        for ( let childDrawable = ( drawable as unknown as Block ).firstDrawable; childDrawable !== ( drawable as unknown as Block ).lastDrawable; childDrawable = childDrawable.nextDrawable ) {
          count += drawableCount( childDrawable );
        }
        count += drawableCount( ( drawable as unknown as Block ).lastDrawable! );
      }
      return count;
    }

    // @ts-expect-error TODO BackboneDrawable
    result += this._rootBackbone ? ( `Drawables: ${drawableCount( this._rootBackbone )}<br/>` ) : '';

    const drawableCountMap: Record<string, number> = {}; // {string} drawable constructor name => {number} count of seen
    // increment the count in our map
    function countRetainedDrawable( drawable: Drawable ): void {
      const name = drawable.constructor.name;
      if ( drawableCountMap[ name ] ) {
        drawableCountMap[ name ]++;
      }
      else {
        drawableCountMap[ name ] = 1;
      }
    }

    function retainedDrawableCount( instance: Instance ): number {
      let count = 0;
      if ( instance.selfDrawable ) {
        countRetainedDrawable( instance.selfDrawable );
        count++;
      }
      if ( instance.groupDrawable ) {
        countRetainedDrawable( instance.groupDrawable );
        count++;
      }
      if ( instance.sharedCacheDrawable ) {
        // @ts-expect-error TODO Instance
        countRetainedDrawable( instance.sharedCacheDrawable );
        count++;
      }
      for ( let i = 0; i < instance.children.length; i++ ) {
        count += retainedDrawableCount( instance.children[ i ] );
      }
      return count;
    }

    result += this._baseInstance ? ( `Retained Drawables: ${retainedDrawableCount( this._baseInstance )}<br/>` ) : '';
    for ( const drawableName in drawableCountMap ) {
      result += `&nbsp;&nbsp;&nbsp;&nbsp;${drawableName}: ${drawableCountMap[ drawableName ]}<br/>`;
    }

    function blockSummary( block: Block ): string {
      // ensure we are a block
      if ( !block.firstDrawable || !block.lastDrawable ) {
        return '';
      }

      // @ts-expect-error TODO display stuff
      const hasBackbone = block.domDrawable && block.domDrawable.blocks;

      let div = `<div style="margin-left: ${depth * 20}px">`;

      div += block.toString();
      if ( !hasBackbone ) {
        div += ` (${block.drawableCount} drawables)`;
      }

      div += '</div>';

      depth += 1;
      if ( hasBackbone ) {
        // @ts-expect-error TODO display stuff
        for ( let k = 0; k < block.domDrawable.blocks.length; k++ ) {
          // @ts-expect-error TODO display stuff
          div += blockSummary( block.domDrawable.blocks[ k ] );
        }
      }
      depth -= 1;

      return div;
    }

    if ( this._rootBackbone ) {
      result += `<div style="${headerStyle}">Block Summary</div>`;
      for ( let i = 0; i < this._rootBackbone.blocks.length; i++ ) {
        result += blockSummary( this._rootBackbone.blocks[ i ] );
      }
    }

    function instanceSummary( instance: Instance ): string {
      let iSummary = '';

      function addQualifier( text: string ): void {
        iSummary += ` <span style="color: #008">${text}</span>`;
      }

      const node = instance.node!;

      iSummary += instance.id;
      iSummary += ` ${node.constructor.name ? node.constructor.name : '?'}`;
      iSummary += ` <span style="font-weight: ${node.isPainted() ? 'bold' : 'normal'}">${node.id}</span>`;
      iSummary += node.getDebugHTMLExtras();

      if ( !node.visible ) {
        addQualifier( 'invis' );
      }
      if ( !instance.visible ) {
        addQualifier( 'I-invis' );
      }
      if ( !instance.relativeVisible ) {
        addQualifier( 'I-rel-invis' );
      }
      if ( !instance.selfVisible ) {
        addQualifier( 'I-self-invis' );
      }
      if ( !instance.fittability.ancestorsFittable ) {
        addQualifier( 'nofit-ancestor' );
      }
      if ( !instance.fittability.selfFittable ) {
        addQualifier( 'nofit-self' );
      }
      if ( node.pickable === true ) {
        addQualifier( 'pickable' );
      }
      if ( node.pickable === false ) {
        addQualifier( 'unpickable' );
      }
      if ( instance.trail!.isPickable() ) {
        addQualifier( '<span style="color: #808">hits</span>' );
      }
      if ( node.getEffectiveCursor() ) {
        addQualifier( `effectiveCursor:${node.getEffectiveCursor()}` );
      }
      if ( node.clipArea ) {
        addQualifier( 'clipArea' );
      }
      if ( node.mouseArea ) {
        addQualifier( 'mouseArea' );
      }
      if ( node.touchArea ) {
        addQualifier( 'touchArea' );
      }
      if ( node.getInputListeners().length ) {
        addQualifier( 'inputListeners' );
      }
      if ( node.getRenderer() ) {
        addQualifier( `renderer:${node.getRenderer()}` );
      }
      if ( node.isLayerSplit() ) {
        addQualifier( 'layerSplit' );
      }
      if ( node.opacity < 1 ) {
        addQualifier( `opacity:${node.opacity}` );
      }
      if ( node.disabledOpacity < 1 ) {
        addQualifier( `disabledOpacity:${node.disabledOpacity}` );
      }

      if ( node._boundsEventCount > 0 ) {
        addQualifier( `<span style="color: #800">boundsListen:${node._boundsEventCount}:${node._boundsEventSelfCount}</span>` );
      }

      let transformType = '';
      switch( node.transform.getMatrix().type ) {
        case Matrix3Type.IDENTITY:
          transformType = '';
          break;
        case Matrix3Type.TRANSLATION_2D:
          transformType = 'translated';
          break;
        case Matrix3Type.SCALING:
          transformType = 'scale';
          break;
        case Matrix3Type.AFFINE:
          transformType = 'affine';
          break;
        case Matrix3Type.OTHER:
          transformType = 'other';
          break;
        default:
          throw new Error( `invalid matrix type: ${node.transform.getMatrix().type}` );
      }
      if ( transformType ) {
        iSummary += ` <span style="color: #88f" title="${node.transform.getMatrix().toString().replace( '\n', '&#10;' )}">${transformType}</span>`;
      }

      iSummary += ` <span style="color: #888">[Trail ${instance.trail!.indices.join( '.' )}]</span>`;
      // iSummary += ` <span style="color: #c88">${str( instance.state )}</span>`;
      iSummary += ` <span style="color: #8c8">${node._rendererSummary.bitmask.toString( 16 )}${node._rendererBitmask !== Renderer.bitmaskNodeDefault ? ` (${node._rendererBitmask.toString( 16 )})` : ''}</span>`;

      return iSummary;
    }

    function drawableSummary( drawable: Drawable ): string {
      let drawableString = drawable.toString();
      if ( drawable.visible ) {
        drawableString = `<strong>${drawableString}</strong>`;
      }
      if ( drawable.dirty ) {
        drawableString += ( drawable.dirty ? ' <span style="color: #c00;">[x]</span>' : '' );
      }
      if ( !drawable.fittable ) {
        drawableString += ( drawable.dirty ? ' <span style="color: #0c0;">[no-fit]</span>' : '' );
      }
      return drawableString;
    }

    function printInstanceSubtree( instance: Instance ): void {
      let div = `<div style="margin-left: ${depth * 20}px">`;

      function addDrawable( name: string, drawable: Drawable ): void {
        div += ` <span style="color: #888">${name}:${drawableSummary( drawable )}</span>`;
      }

      div += instanceSummary( instance );

      instance.selfDrawable && addDrawable( 'self', instance.selfDrawable );
      instance.groupDrawable && addDrawable( 'group', instance.groupDrawable );
      // @ts-expect-error TODO Instance
      instance.sharedCacheDrawable && addDrawable( 'sharedCache', instance.sharedCacheDrawable );

      div += '</div>';
      result += div;

      depth += 1;
      _.each( instance.children, childInstance => {
        printInstanceSubtree( childInstance );
      } );
      depth -= 1;
    }

    if ( this._baseInstance ) {
      result += `<div style="${headerStyle}">Root Instance Tree</div>`;
      printInstanceSubtree( this._baseInstance );
    }

    _.each( this._sharedCanvasInstances, instance => {
      result += `<div style="${headerStyle}">Shared Canvas Instance Tree</div>`;
      printInstanceSubtree( instance );
    } );

    function printDrawableSubtree( drawable: Drawable ): void {
      let div = `<div style="margin-left: ${depth * 20}px">`;

      div += drawableSummary( drawable );
      if ( ( drawable as unknown as SelfDrawable ).instance ) {
        div += ` <span style="color: #0a0;">(${( drawable as unknown as SelfDrawable ).instance.trail.toPathString()})</span>`;
        div += `&nbsp;&nbsp;&nbsp;${instanceSummary( ( drawable as unknown as SelfDrawable ).instance )}`;
      }
      else if ( ( drawable as unknown as BackboneDrawable ).backboneInstance ) {
        div += ` <span style="color: #a00;">(${( drawable as unknown as BackboneDrawable ).backboneInstance.trail.toPathString()})</span>`;
        div += `&nbsp;&nbsp;&nbsp;${instanceSummary( ( drawable as unknown as BackboneDrawable ).backboneInstance )}`;
      }

      div += '</div>';
      result += div;

      if ( ( drawable as unknown as BackboneDrawable ).blocks ) {
        // we're a backbone
        depth += 1;
        _.each( ( drawable as unknown as BackboneDrawable ).blocks, childDrawable => {
          printDrawableSubtree( childDrawable );
        } );
        depth -= 1;
      }
      else if ( ( drawable as unknown as Block ).firstDrawable && ( drawable as unknown as Block ).lastDrawable ) {
        // we're a block
        depth += 1;
        for ( let childDrawable = ( drawable as unknown as Block ).firstDrawable; childDrawable !== ( drawable as unknown as Block ).lastDrawable; childDrawable = childDrawable.nextDrawable ) {
          printDrawableSubtree( childDrawable );
        }
        printDrawableSubtree( ( drawable as unknown as Block ).lastDrawable! ); // wasn't hit in our simplified (and safer) loop
        depth -= 1;
      }
    }

    if ( this._rootBackbone ) {
      result += '<div style="font-weight: bold;">Root Drawable Tree</div>';
      // @ts-expect-error TODO BackboneDrawable
      printDrawableSubtree( this._rootBackbone );
    }

    //OHTWO TODO: add shared cache drawable trees

    return result;
  }

  /**
   * Returns the getDebugHTML() information, but wrapped into a full HTML page included in a data URI.
   */
  public getDebugURI(): string {
    return `data:text/html;charset=utf-8,${encodeURIComponent(
      `${'<!DOCTYPE html>' +
      '<html lang="en">' +
      '<head><title>Scenery Debug Snapshot</title></head>' +
      '<body style="font-size: 12px;">'}${this.getDebugHTML()}</body>` +
      '</html>'
    )}`;
  }

  /**
   * Attempts to open a popup with the getDebugHTML() information.
   */
  public popupDebug(): void {
    window.open( this.getDebugURI() );
  }

  /**
   * Attempts to open an iframe popup with the getDebugHTML() information in the same window. This is similar to
   * popupDebug(), but should work in browsers that block popups, or prevent that type of data URI being opened.
   */
  public iframeDebug(): void {
    const iframe = document.createElement( 'iframe' );
    iframe.width = '' + window.innerWidth; // eslint-disable-line bad-sim-text
    iframe.height = '' + window.innerHeight; // eslint-disable-line bad-sim-text
    iframe.style.position = 'absolute';
    iframe.style.left = '0';
    iframe.style.top = '0';
    iframe.style.zIndex = '10000';
    document.body.appendChild( iframe );

    iframe.contentWindow!.document.open();
    iframe.contentWindow!.document.write( this.getDebugHTML() );
    iframe.contentWindow!.document.close();

    iframe.contentWindow!.document.body.style.background = 'white';

    const closeButton = document.createElement( 'button' );
    closeButton.style.position = 'absolute';
    closeButton.style.top = '0';
    closeButton.style.right = '0';
    closeButton.style.zIndex = '10001';
    document.body.appendChild( closeButton );

    closeButton.textContent = 'close';

    // A normal 'click' event listener doesn't seem to be working. This is less-than-ideal.
    [ 'pointerdown', 'click', 'touchdown' ].forEach( eventType => {
      closeButton.addEventListener( eventType, () => {
        document.body.removeChild( iframe );
        document.body.removeChild( closeButton );
      }, true );
    } );
  }

  public getPDOMDebugHTML(): string {
    let result = '';

    const headerStyle = 'font-weight: bold; font-size: 120%; margin-top: 5px;';
    const indent = '&nbsp;&nbsp;&nbsp;&nbsp;';

    result += `<div style="${headerStyle}">Accessible Instances</div><br>`;

    recurse( this._rootPDOMInstance!, '' );

    function recurse( instance: PDOMInstance, indentation: string ): void {
      result += `${indentation + escapeHTML( `${instance.isRootInstance ? '' : instance.node!.tagName} ${instance.toString()}` )}<br>`;
      instance.children.forEach( ( child: PDOMInstance ) => {
        recurse( child, indentation + indent );
      } );
    }

    result += `<br><div style="${headerStyle}">Parallel DOM</div><br>`;

    let parallelDOM = this._rootPDOMInstance!.peer!.primarySibling!.outerHTML;
    parallelDOM = parallelDOM.replace( /></g, '>\n<' );
    const lines = parallelDOM.split( '\n' );

    let indentation = '';
    for ( let i = 0; i < lines.length; i++ ) {
      const line = lines[ i ];
      const isEndTag = line.startsWith( '</' );

      if ( isEndTag ) {
        indentation = indentation.slice( indent.length );
      }
      result += `${indentation + escapeHTML( line )}<br>`;
      if ( !isEndTag ) {
        indentation += indent;
      }
    }
    return result;
  }

  /**
   * Will attempt to call callback( {string} dataURI ) with the rasterization of the entire Display's DOM structure,
   * used for internal testing. Will call-back null if there was an error
   *
   * Only tested on recent Chrome and Firefox, not recommended for general use. Guaranteed not to work for IE <= 10.
   *
   * See https://github.com/phetsims/scenery/issues/394 for some details.
   */
  public foreignObjectRasterization( callback: ( url: string | null ) => void ): void {
    // Scan our drawable tree for Canvases. We'll rasterize them here (to data URLs) so we can replace them later in
    // the HTML tree (with images) before putting that in the foreignObject. That way, we can actually display
    // things rendered in Canvas in our rasterization.
    const canvasUrlMap: Record<string, string> = {};

    let unknownIds = 0;

    function addCanvas( canvas: HTMLCanvasElement ): void {
      if ( !canvas.id ) {
        canvas.id = `unknown-canvas-${unknownIds++}`;
      }
      canvasUrlMap[ canvas.id ] = canvas.toDataURL();
    }

    function scanForCanvases( drawable: Drawable ): void {
      if ( drawable instanceof BackboneDrawable ) {
        // we're a backbone
        _.each( drawable.blocks, childDrawable => {
          scanForCanvases( childDrawable );
        } );
      }
      else if ( drawable instanceof Block && drawable.firstDrawable && drawable.lastDrawable ) {
        // we're a block
        for ( let childDrawable = drawable.firstDrawable; childDrawable !== drawable.lastDrawable; childDrawable = childDrawable.nextDrawable ) {
          scanForCanvases( childDrawable );
        }
        scanForCanvases( drawable.lastDrawable ); // wasn't hit in our simplified (and safer) loop

        if ( ( drawable instanceof CanvasBlock || drawable instanceof WebGLBlock ) && drawable.canvas && drawable.canvas instanceof window.HTMLCanvasElement ) {
          addCanvas( drawable.canvas );
        }
      }

      if ( DOMDrawable && drawable instanceof DOMDrawable ) {
        if ( drawable.domElement instanceof window.HTMLCanvasElement ) {
          addCanvas( drawable.domElement );
        }
        Array.prototype.forEach.call( drawable.domElement.getElementsByTagName( 'canvas' ), canvas => {
          addCanvas( canvas );
        } );
      }
    }

    // @ts-expect-error TODO BackboneDrawable
    scanForCanvases( this._rootBackbone! );

    // Create a new document, so that we can (1) serialize it to XHTML, and (2) manipulate it independently.
    // Inspired by http://cburgmer.github.io/rasterizeHTML.js/
    const doc = document.implementation.createHTMLDocument( '' );
    doc.documentElement.innerHTML = this.domElement.outerHTML;
    doc.documentElement.setAttribute( 'xmlns', doc.documentElement.namespaceURI! );

    // Hide the PDOM
    doc.documentElement.appendChild( document.createElement( 'style' ) ).innerHTML = `.${PDOMSiblingStyle.ROOT_CLASS_NAME} { display:none; } `;

    // Replace each <canvas> with an <img> that has src=canvas.toDataURL() and the same style
    let displayCanvases: HTMLElement[] | HTMLCollection = doc.documentElement.getElementsByTagName( 'canvas' );
    displayCanvases = Array.prototype.slice.call( displayCanvases ); // don't use a live HTMLCollection copy!
    for ( let i = 0; i < displayCanvases.length; i++ ) {
      const displayCanvas = displayCanvases[ i ];

      const cssText = displayCanvas.style.cssText;

      const displayImg = doc.createElement( 'img' );
      const src = canvasUrlMap[ displayCanvas.id ];
      assert && assert( src, 'Must have missed a toDataURL() on a Canvas' );

      displayImg.src = src;
      displayImg.setAttribute( 'style', cssText );

      displayCanvas.parentNode!.replaceChild( displayImg, displayCanvas );
    }

    const displayWidth = this.width;
    const displayHeight = this.height;
    const completeFunction = () => {
      Display.elementToSVGDataURL( doc.documentElement, displayWidth, displayHeight, callback );
    };

    // Convert each <image>'s xlink:href so that it's a data URL with the relevant data, e.g.
    // <image ... xlink:href="http://localhost:8080/scenery-phet/images/batteryDCell.png?bust=1476308407988"/>
    // gets replaced with a data URL.
    // See https://github.com/phetsims/scenery/issues/573
    let replacedImages = 0; // Count how many images get replaced. We'll decrement with each finished image.
    let hasReplacedImages = false; // Whether any images are replaced
    const displaySVGImages = Array.prototype.slice.call( doc.documentElement.getElementsByTagName( 'image' ) );
    for ( let j = 0; j < displaySVGImages.length; j++ ) {
      const displaySVGImage = displaySVGImages[ j ];
      const currentHref = displaySVGImage.getAttribute( 'xlink:href' );
      if ( currentHref.slice( 0, 5 ) !== 'data:' ) {
        replacedImages++;
        hasReplacedImages = true;

        ( () => { // eslint-disable-line @typescript-eslint/no-loop-func
          // Closure variables need to be stored for each individual SVG image.
          const refImage = new window.Image();
          const svgImage = displaySVGImage;

          refImage.onload = () => {
            // Get a Canvas
            const refCanvas = document.createElement( 'canvas' );
            refCanvas.width = refImage.width;
            refCanvas.height = refImage.height;
            const refContext = refCanvas.getContext( '2d' )!;

            // Draw the (now loaded) image into the Canvas
            refContext.drawImage( refImage, 0, 0 );

            // Replace the <image>'s href with the Canvas' data.
            svgImage.setAttribute( 'xlink:href', refCanvas.toDataURL() );

            // If it's the last replaced image, go to the next step
            if ( --replacedImages === 0 ) {
              completeFunction();
            }

            assert && assert( replacedImages >= 0 );
          };
          refImage.onerror = () => {
            // NOTE: not much we can do, leave this element alone.

            // If it's the last replaced image, go to the next step
            if ( --replacedImages === 0 ) {
              completeFunction();
            }

            assert && assert( replacedImages >= 0 );
          };

          // Kick off loading of the image.
          refImage.src = currentHref;
        } )();
      }
    }

    // If no images are replaced, we need to call our callback through this route.
    if ( !hasReplacedImages ) {
      completeFunction();
    }
  }

  public popupRasterization(): void {
    this.foreignObjectRasterization( url => {
      if ( url ) {
        window.open( url );
      }
    } );
  }

  /**
   * Will return null if the string of indices isn't part of the PDOMInstance tree
   */
  public getTrailFromPDOMIndicesString( indicesString: string ): Trail | null {

    // No PDOMInstance tree if the display isn't accessible
    if ( !this._rootPDOMInstance ) {
      return null;
    }

    let instance = this._rootPDOMInstance;
    const indexStrings = indicesString.split( PDOMUtils.PDOM_UNIQUE_ID_SEPARATOR );
    for ( let i = 0; i < indexStrings.length; i++ ) {
      const digit = Number( indexStrings[ i ] );
      instance = instance.children[ digit ];
      if ( !instance ) {
        return null;
      }
    }

    return ( instance && instance.trail ) ? instance.trail : null;
  }

  /**
   * Releases references.
   *
   * TODO: this dispose function is not complete.
   */
  public dispose(): void {
    if ( assert ) {
      assert( !this._isDisposing );
      assert( !this._isDisposed );

      this._isDisposing = true;
    }

    if ( this._input ) {
      this.detachEvents();
    }
    this._rootNode.removeRootedDisplay( this );

    if ( this._accessible ) {
      assert && assert( this._boundHandleFullScreenNavigation, '_boundHandleFullScreenNavigation was not added to the keyStateTracker' );
      globalKeyStateTracker.keydownEmitter.removeListener( this._boundHandleFullScreenNavigation! );
      this._rootPDOMInstance!.dispose();
    }

    this._focusOverlay && this._focusOverlay.dispose();

    this.sizeProperty.dispose();

    // Will immediately dispose recursively, all Instances AND their attached drawables, which will include the
    // rootBackbone.
    this._baseInstance && this._baseInstance.dispose();

    this.descriptionUtteranceQueue.dispose();

    this.focusManager && this.focusManager.dispose();

    if ( assert ) {
      this._isDisposing = false;
      this._isDisposed = true;
    }
  }

  /**
   * Takes a given DOM element, and asynchronously renders it to a string that is a data URL representing an SVG
   * file.
   *
   * @param domElement
   * @param width - The width of the output SVG
   * @param height - The height of the output SVG
   * @param callback - Called as callback( url: {string} ), where the URL will be the encoded SVG file.
   */
  public static elementToSVGDataURL( domElement: HTMLElement, width: number, height: number, callback: ( url: string | null ) => void ): void {
    const canvas = document.createElement( 'canvas' );
    const context = canvas.getContext( '2d' )!;
    canvas.width = width;
    canvas.height = height;

    // Serialize it to XHTML that can be used in foreignObject (HTML can't be)
    const xhtml = new window.XMLSerializer().serializeToString( domElement );

    // Create an SVG container with a foreignObject.
    const data = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}">` +
                 '<foreignObject width="100%" height="100%">' +
                 `<div xmlns="http://www.w3.org/1999/xhtml">${
                   xhtml
                 }</div>` +
                 '</foreignObject>' +
                 '</svg>';

    // Load an <img> with the SVG data URL, and when loaded draw it into our Canvas
    const img = new window.Image();
    img.onload = () => {
      context.drawImage( img, 0, 0 );
      callback( canvas.toDataURL() ); // Endpoint here
    };
    img.onerror = () => {
      callback( null );
    };

    // We can't btoa() arbitrary unicode, so we need another solution,
    // see https://developer.mozilla.org/en-US/docs/Web/API/WindowBase64/Base64_encoding_and_decoding#The_.22Unicode_Problem.22
    // @ts-expect-error - Exterior lib
    const uint8array = new window.TextEncoderLite( 'utf-8' ).encode( data );
    // @ts-expect-error - Exterior lib
    const base64 = window.fromByteArray( uint8array );

    // turn it to base64 and wrap it in the data URL format
    img.src = `data:image/svg+xml;base64,${base64}`;
  }

  /**
   * Returns true when NO nodes in the subtree are disposed.
   */
  private static assertSubtreeDisposed( node: Node ): void {
    assert && assert( !node.isDisposed, 'Disposed nodes should not be included in a scene graph to display.' );

    if ( assert ) {
      for ( let i = 0; i < node.children.length; i++ ) {
        Display.assertSubtreeDisposed( node.children[ i ] );
      }
    }
  }

  /**
   * Adds an input listener to be fired for ANY Display
   */
  public static addInputListener( listener: TInputListener ): void {
    assert && assert( !_.includes( Display.inputListeners, listener ), 'Input listener already registered' );

    // don't allow listeners to be added multiple times
    if ( !_.includes( Display.inputListeners, listener ) ) {
      Display.inputListeners.push( listener );
    }
  }

  /**
   * Removes an input listener that was previously added with Display.addInputListener.
   */
  public static removeInputListener( listener: TInputListener ): void {
    // ensure the listener is in our list
    assert && assert( _.includes( Display.inputListeners, listener ) );

    Display.inputListeners.splice( _.indexOf( Display.inputListeners, listener ), 1 );
  }

  /**
   * Interrupts all input listeners that are attached to all Displays.
   */
  public static interruptInput(): void {
    const listenersCopy = Display.inputListeners.slice( 0 );

    for ( let i = 0; i < listenersCopy.length; i++ ) {
      const listener = listenersCopy[ i ];

      listener.interrupt && listener.interrupt();
    }
  }

  // Fires when we detect an input event that would be considered a "user gesture" by Chrome, so
  // that we can trigger browser actions that are only allowed as a result.
  // See https://github.com/phetsims/scenery/issues/802 and https://github.com/phetsims/vibe/issues/32 for more
  // information.
  public static userGestureEmitter: TEmitter;

  // Listeners that will be called for every event on ANY Display, see
  // https://github.com/phetsims/scenery/issues/1149. Do not directly modify this!
  public static inputListeners: TInputListener[];
}

scenery.register( 'Display', Display );

Display.userGestureEmitter = new Emitter();
Display.inputListeners = [];
