// Copyright 2013-2021, University of Colorado Boulder

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
import stepTimer from '../../../axon/js/stepTimer.js';
import TinyProperty from '../../../axon/js/TinyProperty.js';
import Dimension2 from '../../../dot/js/Dimension2.js';
import Matrix3 from '../../../dot/js/Matrix3.js';
import escapeHTML from '../../../phet-core/js/escapeHTML.js';
import merge from '../../../phet-core/js/merge.js';
import platform from '../../../phet-core/js/platform.js';
import Tandem from '../../../tandem/js/Tandem.js';
import AriaLiveAnnouncer from '../../../utterance-queue/js/AriaLiveAnnouncer.js';
import UtteranceQueue from '../../../utterance-queue/js/UtteranceQueue.js';
import FocusManager from '../accessibility/FocusManager.js';
import globalKeyStateTracker from '../accessibility/globalKeyStateTracker.js';
import KeyboardUtils from '../accessibility/KeyboardUtils.js';
import PDOMInstance from '../accessibility/pdom/PDOMInstance.js';
import PDOMSiblingStyle from '../accessibility/pdom/PDOMSiblingStyle.js';
import PDOMTree from '../accessibility/pdom/PDOMTree.js';
import PDOMUtils from '../accessibility/pdom/PDOMUtils.js';
import Input from '../input/Input.js';
import Node from '../nodes/Node.js';
import CanvasNodeBoundsOverlay from '../overlays/CanvasNodeBoundsOverlay.js';
import FittedBlockBoundsOverlay from '../overlays/FittedBlockBoundsOverlay.js';
import HighlightOverlay from '../overlays/HighlightOverlay.js';
import HitAreaOverlay from '../overlays/HitAreaOverlay.js';
import PointerAreaOverlay from '../overlays/PointerAreaOverlay.js';
import PointerOverlay from '../overlays/PointerOverlay.js';
import scenery from '../scenery.js';
import Color from '../util/Color.js';
import Features from '../util/Features.js';
import FullScreen from '../util/FullScreen.js';
import Trail from '../util/Trail.js';
import Utils from '../util/Utils.js';
import BackboneDrawable from './BackboneDrawable.js';
import ChangeInterval from './ChangeInterval.js';
import DOMBlock from './DOMBlock.js';
import Drawable from './Drawable.js';
import DOMDrawable from './drawables/DOMDrawable.js';
import Instance from './Instance.js';
import Renderer from './Renderer.js';

class Display {
  /**
   * Constructs a Display that will show the rootNode and its subtree in a visual state. Default options provided below
   *
   * @param {Node} rootNode - Displays this node and all of its descendants
   * @param {Object} [options] - Valid parameters in the parameter object:
   * {
   *   allowSceneOverflow: false,           // Usually anything displayed outside of this $main (DOM/CSS3 transformed SVG) is hidden with CSS overflow
   *   allowCSSHacks: true,                 // Applies styling that prevents mobile browser graphical issues
   *   width: <current main width>,         // Override the main container's width
   *   height: <current main height>,       // Override the main container's height
   *   preserveDrawingBuffer: false,        // Whether WebGL Canvases should preserve their drawing buffer.
   *                                        //   WARNING!: This can significantly reduce performance if set to true.
   *   allowWebGL: true,                    // Boolean flag that indicates whether scenery is allowed to use WebGL for rendering
   *                                        // Makes it possible to disable WebGL for ease of testing on non-WebGL platforms, see #289
   *   accessibility: true                  // Whether accessibility enhancements is enabled
   *   interactive: true                    // Whether mouse/touch/keyboard inputs are enabled (if input has been added). Simulation will still step.
   */
  constructor( rootNode, options ) {
    assert && assert( rootNode, 'rootNode is a required parameter' );

    //OHTWO TODO: hybrid batching (option to batch until an event like 'up' that might be needed for security issues)

    options = merge( {
      // {number} - Initial display width
      width: ( options && options.container && options.container.clientWidth ) || 640,

      // {number} - Initial display height
      height: ( options && options.container && options.container.clientHeight ) || 480,

      // {boolean} - Applies CSS styles to the root DOM element that make it amenable to interactive content
      allowCSSHacks: true,

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
    }, options );

    // @public (scenery-internal) {boolean} - Whether accessibility is enabled for this particular display.
    this._accessible = options.accessibility;

    // @public (scenery-internal) {boolean}
    this._preserveDrawingBuffer = options.preserveDrawingBuffer;

    // @private {boolean}
    this._allowWebGL = options.allowWebGL;
    this._allowCSSHacks = options.allowCSSHacks;
    this._allowSceneOverflow = options.allowSceneOverflow;

    // @private {string}
    this._defaultCursor = options.defaultCursor;

    // @public {Property.<Dimension2>} The (integral, > 0) dimensions of the Display's DOM element (only updates the DOM
    // element on updateDisplay())
    this.sizeProperty = new TinyProperty( new Dimension2( options.width, options.height ) );

    // @private {Dimension2} - used to check against new size to see what we need to change
    this._currentSize = new Dimension2( -1, -1 );

    // @private {Node}
    this._rootNode = rootNode;
    this._rootNode.addRootedDisplay( this );

    // @private {BackboneDrawable|null} - to be filled in later
    this._rootBackbone = null;

    // @private {Element}
    this._domElement = options.container ?
                       BackboneDrawable.repurposeBackboneContainer( options.container ) :
                       BackboneDrawable.createDivBackbone();

    // @public (scenery-internal) {Object.<nodeID:number,Instance>} - map from Node ID to Instance, for fast lookup
    this._sharedCanvasInstances = {};

    // @private {Instance|null} - will be filled with the root Instance
    this._baseInstance = null;

    // @public (scenery-internal) {number} - We have a monotonically-increasing frame ID, generally for use with a pattern
    // where we can mark objects with this to note that they are either up-to-date or need refreshing due to this
    // particular frame (without having to clear that information after use). This is incremented every frame
    this._frameId = 0;

    // @private {Array.<Instance>}
    this._dirtyTransformRoots = [];
    this._dirtyTransformRootsWithoutPass = [];

    // @private {Array.<Instance>}
    this._instanceRootsToDispose = [];

    // @private {Array.<*>} - At the end of Display.update, reduceReferences will be called on all of these. It's meant to
    // catch various objects that would usually have update() called, but if they are invisible or otherwise not updated
    // for performance, they may need to release references another way instead.
    // See https://github.com/phetsims/energy-forms-and-changes/issues/356
    this._reduceReferencesNeeded = [];

    // @private {Array.<Drawable>}
    this._drawablesToDispose = [];

    // @private {Array.<Drawable>} - Block changes are handled by changing the "pending" block/backbone on drawables. We
    // want to change them all after the main stitch process has completed, so we can guarantee that a single drawable is
    // removed from its previous block before being added to a new one. This is taken care of in an updateDisplay pass
    // after syncTree / stitching.
    this._drawablesToChangeBlock = [];

    // @private {Array.<Drawable>} - Drawables have two implicit linked-lists, "current" and "old". syncTree modifies the
    // "current" linked-list information so it is up-to-date, but needs to use the "old" information also. We move
    // updating the "current" => "old" linked-list information until after syncTree and stitching is complete, and is
    // taken care of in an updateDisplay pass.
    this._drawablesToUpdateLinks = [];

    // @private {Array.<ChangeInterval>} - We store information on {ChangeInterval}s that records change interval
    // information, that may contain references. We don't want to leave those references dangling after we don't need
    // them, so they are recorded and cleaned in one of updateDisplay's phases.
    this._changeIntervalsToDispose = [];

    // @private {string|null}
    this._lastCursor = null;
    this._currentBackgroundCSS = null;

    // @private {ColorDef}
    this._backgroundColor = null;

    // @private {number} - Used for shortcut animation frame functions
    this._requestAnimationFrameID = 0;

    // @public (phet-io,scenery) - Will be filled in with a scenery.Input if event handling is enabled
    this._input = null;

    // @private {Array.<Object>} - Listeners that will be called for every event.
    this._inputListeners = [];

    // @private {boolean} - Whether mouse/touch/keyboard inputs are enabled (if input has been added). Simulation will still step.
    this._interactive = options.interactive;

    // @private {boolean} - Passed through to Input
    this._listenToOnlyElement = options.listenToOnlyElement;
    this._batchDOMEvents = options.batchDOMEvents;
    this._assumeFullWindow = options.assumeFullWindow;

    // @private {boolean|null}
    this._passiveEvents = options.passiveEvents;

    // @public (scenery-internal) {boolean}
    this._aggressiveContextRecreation = options.aggressiveContextRecreation;

    // @public (scenery-internal) {boolean}
    this._allowBackingScaleAntialiasing = options.allowBackingScaleAntialiasing;

    // @private {Array.<*>} - Overlays currently being displayed.
    // API expected:
    //   .domElement
    //   .update()
    this._overlays = [];

    // @private {PointerOverlay|null}
    this._pointerOverlay = null;

    // @private {PointerAreaOverlay|null}
    this._pointerAreaOverlay = null;

    // @private {HitAreaOverlay|null}
    this._hitAreaOverlay = null;

    // @private {CanvasNodeBoundsOverlay|null}
    this._canvasAreaBoundsOverlay = null;

    // @private {FittedBlockBoundsOverlay|null}
    this._fittedBlockBoundsOverlay = null;

    if ( assert ) {
      // @private @assertion-only {boolean} - Whether we are running the paint phase of updateDisplay() for this Display.
      this._isPainting = false;

      // @public @assertion-only {boolean}
      this._isDisposing = false; // Whether disposal has started (but not finished)
      this._isDisposed = false; // Whether disposal has finished
    }

    this.applyCSSHacks();

    this.setBackgroundColor( options.backgroundColor );

    // @public {UtteranceQueue} - data structure for managing aria-live alerts the this Display instance
    const ariaLiveAnnouncer = new AriaLiveAnnouncer();
    this.descriptionUtteranceQueue = new UtteranceQueue( ariaLiveAnnouncer, {
      implementAsSkeleton: !this._accessible,
      tandem: options.tandem.createTandem( 'descriptionUtteranceQueue' )
    } );

    // @public - Manages the various types of Focus that can go through the Display, as well as Properties
    // controlling which forms of focus should be displayed in the HighlightOverlay.
    this.focusManager = new FocusManager();

    if ( this._accessible ) {

      // @private {Node}
      this._focusRootNode = new Node();

      // @private {HighlightOverlay}
      this._focusOverlay = new HighlightOverlay( this, this._focusRootNode, {
        pdomFocusHighlightsVisibleProperty: this.focusManager.pdomFocusHighlightsVisibleProperty,
        interactiveHighlightsVisibleProperty: this.focusManager.interactiveHighlightsVisibleProperty,
        readingBlockHighlightsVisibleProperty: this.focusManager.readingBlockHighlightsVisibleProperty
      } );
      this.addOverlay( this._focusOverlay );

      // @public {boolean} (scenery-internal) - During DOM operations where HTML elements are removed from and
      // reinserted into the PDOM, event callbacks related to focus should be blocked as these are internal operations
      // unrelated to application behavior user input, see https://github.com/phetsims/scenery/issues/925
      this.blockFocusCallbacks = false;

      // @public (scenery-internal) {PDOMInstance}
      this._rootPDOMInstance = PDOMInstance.createFromPool( null, this, new Trail() );
      sceneryLog && sceneryLog.PDOMInstance && sceneryLog.PDOMInstance(
        `Display root instance: ${this._rootPDOMInstance.toString()}` );
      PDOMTree.rebuildInstanceTree( this._rootPDOMInstance );

      // add the accessible DOM as a child of this DOM element
      this._domElement.appendChild( this._rootPDOMInstance.peer.primarySibling );

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

  /**
   * Returns the base DOM element that will be displayed by this Display
   * @public
   *
   * @returns {Element}
   */
  getDOMElement() {
    return this._domElement;
  }

  get domElement() { return this.getDOMElement(); }

  /**
   * Updates the display's DOM element with the current visual state of the attached root node and its descendants
   * @public
   */
  updateDisplay() {
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
      this._rootPDOMInstance.peer.updateSubtreePositioning();
    }

    // validate bounds for everywhere that could trigger bounds listeners. we want to flush out any changes, so that we can call validateBounds()
    // from code below without triggering side effects (we assume that we are not reentrant).
    this._rootNode.validateWatchedBounds();

    if ( assertSlow ) { this._accessible && this._rootPDOMInstance.auditRoot(); }

    if ( assertSlow ) { this._rootNode._picker.audit(); }

    this._baseInstance = this._baseInstance || Instance.createFromPool( this, new Trail( this._rootNode ), true, false );
    this._baseInstance.baseSyncTree();
    if ( firstRun ) {
      this.markTransformRootDirty( this._baseInstance, this._baseInstance.isTransformed ); // marks the transform root as dirty (since it is)
    }

    // update our drawable's linked lists where necessary
    while ( this._drawablesToUpdateLinks.length ) {
      this._drawablesToUpdateLinks.pop().updateLinks();
    }

    // clean change-interval information from instances, so we don't leak memory/references
    while ( this._changeIntervalsToDispose.length ) {
      this._changeIntervalsToDispose.pop().dispose();
    }

    this._rootBackbone = this._rootBackbone || this._baseInstance.groupDrawable;
    assert && assert( this._rootBackbone, 'We are guaranteed a root backbone as the groupDrawable on the base instance' );
    assert && assert( this._rootBackbone === this._baseInstance.groupDrawable, 'We don\'t want the base instance\'s groupDrawable to change' );


    if ( assertSlow ) { this._rootBackbone.audit( true, false, true ); } // allow pending blocks / dirty

    sceneryLog && sceneryLog.Display && sceneryLog.Display( 'drawable block change phase' );
    sceneryLog && sceneryLog.Display && sceneryLog.push();
    while ( this._drawablesToChangeBlock.length ) {
      const changed = this._drawablesToChangeBlock.pop().updateBlock();
      if ( sceneryLog && scenery.isLoggingPerformance() && changed ) {
        this.perfDrawableBlockChangeCount++;
      }
    }
    sceneryLog && sceneryLog.Display && sceneryLog.pop();

    if ( assertSlow ) { this._rootBackbone.audit( false, false, true ); } // allow only dirty
    if ( assertSlow ) { this._baseInstance.audit( this._frameId, false ); }

    // pre-repaint phase: update relative transform information for listeners (notification) and precomputation where desired
    this.updateDirtyTransformRoots();
    // pre-repaint phase update visibility information on instances
    this._baseInstance.updateVisibility( true, true, false );

    if ( assertSlow ) { this._baseInstance.auditVisibility( true ); }

    if ( assertSlow ) { this._baseInstance.audit( this._frameId, true ); }

    sceneryLog && sceneryLog.Display && sceneryLog.Display( 'instance root disposal phase' );
    sceneryLog && sceneryLog.Display && sceneryLog.push();
    // dispose all of our instances. disposing the root will cause all descendants to also be disposed.
    // will also dispose attached drawables (self/group/etc.)
    while ( this._instanceRootsToDispose.length ) {
      this._instanceRootsToDispose.pop().dispose();
    }
    sceneryLog && sceneryLog.Display && sceneryLog.pop();

    if ( assertSlow ) { this._rootNode.auditInstanceSubtreeForDisplay( this ); } // make sure trails are valid

    sceneryLog && sceneryLog.Display && sceneryLog.Display( 'drawable disposal phase' );
    sceneryLog && sceneryLog.Display && sceneryLog.push();
    // dispose all of our other drawables.
    while ( this._drawablesToDispose.length ) {
      this._drawablesToDispose.pop().dispose();
    }
    sceneryLog && sceneryLog.Display && sceneryLog.pop();

    if ( assertSlow ) { this._baseInstance.audit( this._frameId ); }

    if ( assert ) {
      assert( !this._isPainting, 'Display was already updating paint, may have thrown an error on the last update' );
      this._isPainting = true;
    }

    // repaint phase
    //OHTWO TODO: can anything be updated more efficiently by tracking at the Display level? Remember, we have recursive updates so things get updated in the right order!
    sceneryLog && sceneryLog.Display && sceneryLog.Display( 'repaint phase' );
    sceneryLog && sceneryLog.Display && sceneryLog.push();
    this._rootBackbone.update();
    sceneryLog && sceneryLog.Display && sceneryLog.pop();

    if ( assert ) {
      this._isPainting = false;
    }

    if ( assertSlow ) { this._rootBackbone.audit( false, false, false ); } // allow nothing
    if ( assertSlow ) { this._baseInstance.audit( this._frameId ); }

    this.updateCursor();
    this.updateBackgroundColor();

    this.updateSize();

    if ( this._overlays.length ) {
      let zIndex = this._rootBackbone.lastZIndex;
      for ( let i = 0; i < this._overlays.length; i++ ) {
        // layer the overlays properly
        const overlay = this._overlays[ i ];
        overlay.domElement.style.zIndex = zIndex++;

        overlay.update();
      }
    }

    // After our update and disposals, we want to eliminate any memory leaks from anything that wasn't updated.
    while ( this._reduceReferencesNeeded.length ) {
      this._reduceReferencesNeeded.pop().reduceReferences();
    }

    this._frameId++;

    if ( sceneryLog && scenery.isLoggingPerformance() ) {
      const syncTreeMessage = `syncTree count: ${this.perfSyncTreeCount}`;
      if ( this.perfSyncTreeCount > 500 ) {
        sceneryLog.PerfCritical && sceneryLog.PerfCritical( syncTreeMessage );
      }
      else if ( this.perfSyncTreeCount > 100 ) {
        sceneryLog.PerfMajor && sceneryLog.PerfMajor( syncTreeMessage );
      }
      else if ( this.perfSyncTreeCount > 20 ) {
        sceneryLog.PerfMinor && sceneryLog.PerfMinor( syncTreeMessage );
      }
      else if ( this.perfSyncTreeCount > 0 ) {
        sceneryLog.PerfVerbose && sceneryLog.PerfVerbose( syncTreeMessage );
      }

      const drawableBlockCountMessage = `drawable block changes: ${this.perfDrawableBlockChangeCount} for` +
                                        ` -${this.perfDrawableOldIntervalCount
                                        } +${this.perfDrawableNewIntervalCount}`;
      if ( this.perfDrawableBlockChangeCount > 200 ) {
        sceneryLog.PerfCritical && sceneryLog.PerfCritical( drawableBlockCountMessage );
      }
      else if ( this.perfDrawableBlockChangeCount > 60 ) {
        sceneryLog.PerfMajor && sceneryLog.PerfMajor( drawableBlockCountMessage );
      }
      else if ( this.perfDrawableBlockChangeCount > 10 ) {
        sceneryLog.PerfMinor && sceneryLog.PerfMinor( drawableBlockCountMessage );
      }
      else if ( this.perfDrawableBlockChangeCount > 0 ) {
        sceneryLog.PerfVerbose && sceneryLog.PerfVerbose( drawableBlockCountMessage );
      }
    }

    PDOMTree.auditPDOMDisplays( this.rootNode );

    sceneryLog && sceneryLog.Display && sceneryLog.pop();
  }

  /**
   * @private
   */
  updateSize() {
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
   * @public
   *
   * @returns {boolean} - Whether WebGL is allowed to be used in drawables for this Display
   */
  isWebGLAllowed() {
    return this._allowWebGL;
  }

  /**
   * Returns the Node at the root of the display.
   * @public
   *
   * @returns {Node}
   */
  getRootNode() {
    return this._rootNode;
  }

  get rootNode() { return this.getRootNode(); }

  /**
   * @public (scenery-internal)
   *
   * @returns {BackboneDrawable}
   */
  getRootBackbone() {
    return this._rootBackbone;
  }

  get rootBackbone() { return this.getRootBackbone(); }

  /**
   * The dimensions of the Display's DOM element
   * @public
   *
   * @returns {Dimension2}
   */
  getSize() {
    return this.sizeProperty.value;
  }

  get size() { return this.getSize(); }

  /**
   * @public
   *
   * @returns {Bounds2}
   */
  getBounds() {
    return this.size.toBounds();
  }

  get bounds() { return this.getBounds(); }

  /**
   * Changes the size that the Display's DOM element will be after the next updateDisplay()
   * @public
   *
   * @param {Dimension2} size
   */
  setSize( size ) {
    assert && assert( size instanceof Dimension2 );
    assert && assert( size.width % 1 === 0, 'Display.width should be an integer' );
    assert && assert( size.width > 0, 'Display.width should be greater than zero' );
    assert && assert( size.height % 1 === 0, 'Display.height should be an integer' );
    assert && assert( size.height > 0, 'Display.height should be greater than zero' );

    this.sizeProperty.value = size;
  }

  /**
   * Changes the size that the Display's DOM element will be after the next updateDisplay()
   * @public
   *
   * @param {number} width
   * @param {number} height
   */
  setWidthHeight( width, height ) {
    this.setSize( new Dimension2( width, height ) );
  }

  /**
   * The width of the Display's DOM element
   * @public
   *
   * @returns {number}
   */
  getWidth() {
    return this.size.width;
  }

  get width() { return this.getWidth(); }

  /**
   * Sets the width that the Display's DOM element will be after the next updateDisplay(). Should be an integral value.
   * @public
   *
   * @param {number} width
   */
  setWidth( width ) {
    assert && assert( typeof width === 'number', 'Display.width should be a number' );

    if ( this.getWidth() !== width ) {
      this.setSize( new Dimension2( width, this.getHeight() ) );
    }
  }

  set width( value ) { this.setWidth( value ); }

  /**
   * The height of the Display's DOM element
   * @public
   *
   * @returns {number}
   */
  getHeight() {
    return this.size.height;
  }

  get height() { return this.getHeight(); }

  /**
   * Sets the height that the Display's DOM element will be after the next updateDisplay(). Should be an integral value.
   * @public
   *
   * @param {number} height
   */
  setHeight( height ) {
    assert && assert( typeof height === 'number', 'Display.height should be a number' );

    if ( this.getHeight() !== height ) {
      this.setSize( new Dimension2( this.getWidth(), height ) );
    }
  }

  set height( value ) { this.setHeight( value ); }

  /**
   * Will be applied to the root DOM element on updateDisplay(), and no sooner.
   * @public
   *
   * @param {Color|string|null} color
   */
  setBackgroundColor( color ) {
    assert && assert( color === null || typeof color === 'string' || color instanceof Color );

    this._backgroundColor = color;
  }

  set backgroundColor( value ) { this.setBackgroundColor( value ); }

  /**
   * @public
   *
   * @returns {Color|string|null}
   */
  getBackgroundColor() {
    return this._backgroundColor;
  }

  get backgroundColor() { return this.getBackgroundColor(); }

  /**
   * @public
   *
   * @returns {boolean}
   */
  get interactive() { return this._interactive; }

  /**
   * @public
   *
   * @param {boolean} value
   */
  set interactive( value ) {
    if ( this._accessible && value !== this._interactive ) {
      this._rootPDOMInstance.peer.recursiveDisable( !value );
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
   * @public
   *
   * @param {*} overlay
   */
  addOverlay( overlay ) {
    this._overlays.push( overlay );
    this._domElement.appendChild( overlay.domElement );

    // ensure that the overlay is hidden from screen readers, all accessible content should be in the dom element
    // of the this._rootPDOMInstance
    overlay.domElement.setAttribute( 'aria-hidden', true );
  }

  /**
   * Removes an overlay from the display.
   * @public
   *
   * @param {*} overlay
   */
  removeOverlay( overlay ) {
    this._domElement.removeChild( overlay.domElement );
    this._overlays.splice( _.indexOf( this._overlays, overlay ), 1 );
  }

  /**
   * Get the root accessible DOM element which represents this display and provides semantics for assistive
   * technology. If this Display is not accessible, returns null.
   * @public
   *
   * @returns {HTMLElement|null}
   */
  getPDOMRootElement() {
    return this._accessible ? this._rootPDOMInstance.peer.primarySibling : null;
  }

  get pdomRootElement() { return this.getPDOMRootElement(); }

  /**
   * Has this Display enabled accessibility features like PDOM creation and support.
   * @public
   * @returns {boolean}
   */
  isAccessible() {
    return this._accessible;
  }

  /**
   * Implements a workaround that prevents DOM focus from leaving the Display in FullScreen mode. There is
   * a bug in some browsers where DOM focus can be permanently lost if tabbing out of the FullScreen element,
   * see https://github.com/phetsims/scenery/issues/883.
   * @private
   * @param {Event} domEvent
   */
  handleFullScreenNavigation( domEvent ) {
    assert && assert( this.pdomRootElement, 'There must be a PDOM to support keyboard navigation' );

    if ( FullScreen.isFullScreen() && KeyboardUtils.isKeyEvent( domEvent, KeyboardUtils.KEY_TAB ) ) {
      const rootElement = this.pdomRootElement;
      const nextElement = domEvent.shiftKey ? PDOMUtils.getPreviousFocusable( rootElement ) :
                          PDOMUtils.getNextFocusable( rootElement );
      if ( nextElement === domEvent.target ) {
        domEvent.preventDefault();
      }
    }
  }

  /*
   * Returns the bitmask union of all renderers (canvas/svg/dom/webgl) that are used for display, excluding
   * BackboneDrawables (which would be DOM).
   * @public
   *
   * @returns {number}
   */
  getUsedRenderersBitmask() {
    function renderersUnderBackbone( backbone ) {
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
    return renderersUnderBackbone( this._rootBackbone ) & Renderer.bitmaskRendererArea;
  }

  /*
   * Called from Instances that will need a transform update (for listeners and precomputation).
   * @public (scenery-internal)
   *
   * @param {Instance} instance
   * @param {boolean} passTransform - Whether we should pass the first transform root when validating transforms (should
   * be true if the instance is transformed)
   */
  markTransformRootDirty( instance, passTransform ) {
    passTransform ? this._dirtyTransformRoots.push( instance ) : this._dirtyTransformRootsWithoutPass.push( instance );
  }

  /**
   * @private
   */
  updateDirtyTransformRoots() {
    sceneryLog && sceneryLog.transformSystem && sceneryLog.transformSystem( 'updateDirtyTransformRoots' );
    sceneryLog && sceneryLog.transformSystem && sceneryLog.push();
    while ( this._dirtyTransformRoots.length ) {
      this._dirtyTransformRoots.pop().relativeTransform.updateTransformListenersAndCompute( false, false, this._frameId, true );
    }
    while ( this._dirtyTransformRootsWithoutPass.length ) {
      this._dirtyTransformRootsWithoutPass.pop().relativeTransform.updateTransformListenersAndCompute( false, false, this._frameId, false );
    }
    sceneryLog && sceneryLog.transformSystem && sceneryLog.pop();
  }

  /**
   * @public (scenery-internal)
   *
   * @param {Drawable} drawable
   */
  markDrawableChangedBlock( drawable ) {
    assert && assert( drawable instanceof Drawable );

    sceneryLog && sceneryLog.Display && sceneryLog.Display( `markDrawableChangedBlock: ${drawable.toString()}` );
    this._drawablesToChangeBlock.push( drawable );
  }

  /**
   * Marks an item for later reduceReferences() calls at the end of Display.update().
   * @public (scenery-internal)
   *
   * @param {*} item
   */
  markForReducedReferences( item ) {
    assert && assert( !!item.reduceReferences );

    this._reduceReferencesNeeded.push( item );
  }

  /**
   * @public (scenery-internal)
   *
   * @param {Instance} instance
   */
  markInstanceRootForDisposal( instance ) {
    assert && assert( instance instanceof Instance, 'How would an instance not be an instance of an instance?!?!?' );

    sceneryLog && sceneryLog.Display && sceneryLog.Display( `markInstanceRootForDisposal: ${instance.toString()}` );
    this._instanceRootsToDispose.push( instance );
  }

  /**
   * @public (scenery-internal)
   *
   * @param {Drawable} drawable
   */
  markDrawableForDisposal( drawable ) {
    assert && assert( drawable instanceof Drawable );

    sceneryLog && sceneryLog.Display && sceneryLog.Display( `markDrawableForDisposal: ${drawable.toString()}` );
    this._drawablesToDispose.push( drawable );
  }

  /**
   * @public (scenery-internal)
   *
   * @param {Drawable} drawable
   */
  markDrawableForLinksUpdate( drawable ) {
    assert && assert( drawable instanceof Drawable );

    this._drawablesToUpdateLinks.push( drawable );
  }

  /**
   * Add a {ChangeInterval} for the "remove change interval info" phase (we don't want to leak memory/references)
   * @public (scenery-internal)
   *
   * @param {ChangeInterval} changeInterval
   */
  markChangeIntervalToDispose( changeInterval ) {
    assert && assert( changeInterval instanceof ChangeInterval );

    this._changeIntervalsToDispose.push( changeInterval );
  }

  /**
   * @private
   */
  updateBackgroundColor() {
    assert && assert( this._backgroundColor === null ||
                      typeof this._backgroundColor === 'string' ||
                      this._backgroundColor instanceof Color );

    const newBackgroundCSS = this._backgroundColor === null ?
                             '' :
                             ( this._backgroundColor.toCSS ?
                               this._backgroundColor.toCSS() :
                               this._backgroundColor );
    if ( newBackgroundCSS !== this._currentBackgroundCSS ) {
      this._currentBackgroundCSS = newBackgroundCSS;

      this._domElement.style.backgroundColor = newBackgroundCSS;
    }
  }

  /*---------------------------------------------------------------------------*
   * Cursors
   *----------------------------------------------------------------------------*/

  /**
   * @private
   */
  updateCursor() {
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
   * @private
   *
   * @param {string} cursor
   */
  setElementCursor( cursor ) {
    this._domElement.style.cursor = cursor;

    // In some cases, Chrome doesn't seem to respect the cursor set on the Display's domElement. If we are using the
    // full window, we can apply the workaround of controlling the body's style.
    // See https://github.com/phetsims/scenery/issues/983
    if ( this._assumeFullWindow ) {
      document.body.style.cursor = cursor;
    }
  }

  /**
   * @private
   *
   * @param {string} cursor
   */
  setSceneCursor( cursor ) {
    if ( cursor !== this._lastCursor ) {
      this._lastCursor = cursor;
      const customCursors = Display.customCursors[ cursor ];
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

  /**
   * @private
   */
  applyCSSHacks() {
    // to use CSS3 transforms for performance, hide anything outside our bounds by default
    if ( !this._allowSceneOverflow ) {
      this._domElement.style.overflow = 'hidden';
    }

    // Prevents selection cursor issues in Safari, see https://github.com/phetsims/scenery/issues/476
    document.onselectstart = () => false;

    // forward all pointer events
    this._domElement.style.msTouchAction = 'none';

    // prevent any default zooming behavior from a trackpad on IE11 and Edge, all should be handled by scenery - must
    // be on the body, doesn't prevent behavior if on the display div
    document.body.style.msContentZooming = 'none';

    // don't allow browser to switch between font smoothing methods for text (see https://github.com/phetsims/scenery/issues/431)
    Features.setStyle( this._domElement, Features.fontSmoothing, 'antialiased' );

    if ( this._allowCSSHacks ) {
      // some css hacks (inspired from https://github.com/EightMedia/hammer.js/blob/master/hammer.js).
      // modified to only apply the proper prefixed version instead of spamming all of them, and doesn't use jQuery.
      Features.setStyle( this._domElement, Features.userDrag, 'none' );
      Features.setStyle( this._domElement, Features.userSelect, 'none' );
      Features.setStyle( this._domElement, Features.touchAction, 'none' );
      Features.setStyle( this._domElement, Features.touchCallout, 'none' );
      Features.setStyle( this._domElement, Features.tapHighlightColor, 'rgba(0,0,0,0)' );
    }
  }

  /**
   * @public
   *
   * @param {function(string)} callback
   */
  canvasDataURL( callback ) {
    this.canvasSnapshot( canvas => {
      callback( canvas.toDataURL() );
    } );
  }

  /**
   * Renders what it can into a Canvas (so far, Canvas and SVG layers work fine)
   * @public
   *
   * @param {function(HTMLCanvasElement,ImageData)} callback
   */
  canvasSnapshot( callback ) {
    const canvas = document.createElement( 'canvas' );
    canvas.width = this.size.width;
    canvas.height = this.size.height;

    const context = canvas.getContext( '2d' );

    //OHTWO TODO: allow actual background color directly, not having to check the style here!!!
    this._rootNode.renderToCanvas( canvas, context, () => {
      callback( canvas, context.getImageData( 0, 0, canvas.width, canvas.height ) );
    }, this.domElement.style.backgroundColor );
  }

  /**
   * @public
   *
   * TODO: reduce code duplication for handling overlays
   *
   * @param {boolean} visibility
   */
  setPointerDisplayVisible( visibility ) {
    assert && assert( typeof visibility === 'boolean' );

    const hasOverlay = !!this._pointerOverlay;

    if ( visibility !== hasOverlay ) {
      if ( !visibility ) {
        this.removeOverlay( this._pointerOverlay );
        this._pointerOverlay.dispose();
        this._pointerOverlay = null;
      }
      else {
        this._pointerOverlay = new PointerOverlay( this, this._rootNode );
        this.addOverlay( this._pointerOverlay );
      }
    }
  }

  /**
   * @public
   *
   * TODO: reduce code duplication for handling overlays
   *
   * @param {boolean} visibility
   */
  setPointerAreaDisplayVisible( visibility ) {
    assert && assert( typeof visibility === 'boolean' );

    const hasOverlay = !!this._pointerAreaOverlay;

    if ( visibility !== hasOverlay ) {
      if ( !visibility ) {
        this.removeOverlay( this._pointerAreaOverlay );
        this._pointerAreaOverlay.dispose();
        this._pointerAreaOverlay = null;
      }
      else {
        this._pointerAreaOverlay = new PointerAreaOverlay( this, this._rootNode );
        this.addOverlay( this._pointerAreaOverlay );
      }
    }
  }

  /**
   * @public
   *
   * TODO: reduce code duplication for handling overlays
   *
   * @param {boolean} visibility
   */
  setHitAreaDisplayVisible( visibility ) {
    assert && assert( typeof visibility === 'boolean' );

    const hasOverlay = !!this._hitAreaOverlay;

    if ( visibility !== hasOverlay ) {
      if ( !visibility ) {
        this.removeOverlay( this._hitAreaOverlay );
        this._hitAreaOverlay.dispose();
        this._hitAreaOverlay = null;
      }
      else {
        this._hitAreaOverlay = new HitAreaOverlay( this, this._rootNode );
        this.addOverlay( this._hitAreaOverlay );
      }
    }
  }

  /**
   * @public
   *
   * TODO: reduce code duplication for handling overlays
   *
   * @param {boolean} visibility
   */
  setCanvasNodeBoundsVisible( visibility ) {
    assert && assert( typeof visibility === 'boolean' );

    const hasOverlay = !!this._canvasAreaBoundsOverlay;

    if ( visibility !== hasOverlay ) {
      if ( !visibility ) {
        this.removeOverlay( this._canvasAreaBoundsOverlay );
        this._canvasAreaBoundsOverlay.dispose();
        this._canvasAreaBoundsOverlay = null;
      }
      else {
        this._canvasAreaBoundsOverlay = new CanvasNodeBoundsOverlay( this, this._rootNode );
        this.addOverlay( this._canvasAreaBoundsOverlay );
      }
    }
  }

  /**
   * @public
   *
   * TODO: reduce code duplication for handling overlays
   *
   * @param {boolean} visibility
   */
  setFittedBlockBoundsVisible( visibility ) {
    assert && assert( typeof visibility === 'boolean' );

    const hasOverlay = !!this._fittedBlockBoundsOverlay;

    if ( visibility !== hasOverlay ) {
      if ( !visibility ) {
        this.removeOverlay( this._fittedBlockBoundsOverlay );
        this._fittedBlockBoundsOverlay.dispose();
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
   * @public
   */
  resizeOnWindowResize() {
    const resizer = () => {
      this.setWidthHeight( window.innerWidth, window.innerHeight ); // eslint-disable-line bad-sim-text
    };
    window.addEventListener( 'resize', resizer );
    resizer();
  }

  /**
   * Updates on every request animation frame. If stepCallback is passed in, it is called before updateDisplay() with
   * stepCallback( timeElapsedInSeconds )
   * @public
   *
   * @param {function(dt:number)} [stepCallback]
   */
  updateOnRequestAnimationFrame( stepCallback ) {
    // keep track of how much time elapsed over the last frame
    let lastTime = 0;
    let timeElapsedInSeconds = 0;

    const self = this;
    ( function step() {
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

  /**
   * @public
   */
  cancelUpdateOnRequestAnimationFrame() {
    window.cancelAnimationFrame( this._requestAnimationFrameID );
  }

  /**
   * Initializes event handling, and connects the browser's input event handlers to notify this Display of events.
   * @public
   *
   * NOTE: This can be reversed with detachEvents().
   * @param {Object} [options] - for PhET-iO
   */
  initializeEvents( options ) {
    assert && assert( !this._input, 'Events cannot be attached twice to a display (for now)' );

    // TODO: refactor here
    const input = new Input( this, !this._listenToOnlyElement, this._batchDOMEvents, this._assumeFullWindow, this._passiveEvents, options );
    this._input = input;

    input.connectListeners();
  }

  /**
   * Detach already-attached input event handling (from initializeEvents()).
   * @public
   */
  detachEvents() {
    assert && assert( this._input, 'detachEvents() should be called only when events are attached' );

    this._input.disconnectListeners();
    this._input = null;
  }


  /**
   * Adds an input listener.
   * @public
   *
   * @param {Object} listener
   * @returns {Display} - For chaining
   */
  addInputListener( listener ) {
    assert && assert( !_.includes( this._inputListeners, listener ), 'Input listener already registered on this Display' );

    // don't allow listeners to be added multiple times
    if ( !_.includes( this._inputListeners, listener ) ) {
      this._inputListeners.push( listener );
    }
    return this;
  }

  /**
   * Removes an input listener that was previously added with addInputListener.
   * @public
   *
   * @param {Object} listener
   * @returns {Display} - For chaining
   */
  removeInputListener( listener ) {
    // ensure the listener is in our list
    assert && assert( _.includes( this._inputListeners, listener ) );

    this._inputListeners.splice( _.indexOf( this._inputListeners, listener ), 1 );

    return this;
  }

  /**
   * Returns whether this input listener is currently listening to this Display.
   * @public
   *
   * More efficient than checking display.inputListeners, as that includes a defensive copy.
   *
   * @param {Object} listener
   * @returns {boolean}
   */
  hasInputListener( listener ) {
    for ( let i = 0; i < this._inputListeners.length; i++ ) {
      if ( this._inputListeners[ i ] === listener ) {
        return true;
      }
    }
    return false;
  }

  /**
   * Returns a copy of all of our input listeners.
   * @public
   *
   * @returns {Array.<Object>}
   */
  getInputListeners() {
    return this._inputListeners.slice( 0 ); // defensive copy
  }

  get inputListeners() { return this.getInputListeners(); }

  /**
   * Interrupts all input listeners that are attached to this Display.
   * @public
   *
   * @returns {Display} - For chaining
   */
  interruptInput() {
    const listenersCopy = this.inputListeners;

    for ( let i = 0; i < listenersCopy.length; i++ ) {
      const listener = listenersCopy[ i ];

      listener.interrupt && listener.interrupt();
    }

    return this;
  }

  /**
   * @public (scenery-internal)
   */
  ensureNotPainting() {
    assert && assert( !this._isPainting,
      'This should not be run in the call tree of updateDisplay(). If you see this, it is likely that either the ' +
      'last updateDisplay() had a thrown error and it is trying to be run again (in which case, investigate that ' +
      'error), OR code was run/triggered from inside an updateDisplay() that has the potential to cause an infinite ' +
      'loop, e.g. CanvasNode paintCanvas() call manipulating another Node, or a bounds listener that Scenery missed.' );
  }

  /**
   * Triggers a loss of context for all WebGL blocks.
   * @public
   *
   * NOTE: Should generally only be used for debugging.
   */
  loseWebGLContexts() {
    ( function loseBackbone( backbone ) {
      if ( backbone.blocks ) {
        backbone.blocks.forEach( block => {
          if ( block.gl ) {
            Utils.loseContext( block.gl );
          }

          //TODO: pattern for this iteration
          for ( let drawable = block.firstDrawable; drawable !== null; drawable = drawable.nextDrawable ) {
            loseBackbone( drawable );
            if ( drawable === block.lastDrawable ) { break; }
          }
        } );
      }
    } )( this._rootBackbone );
  }

  /**
   * Makes this Display available for inspection.
   * @public
   */
  inspect() {
    localStorage.scenerySnapshot = JSON.stringify( scenery.serialize( this ) );
  }

  /**
   * Returns an HTML fragment that includes a large amount of debugging information, including a view of the
   * instance tree and drawable tree.
   * @public
   *
   * @returns {string}
   */
  getDebugHTML() {
    function str( ob ) {
      return ob ? ob.toString() : `${ob}`;
    }

    const headerStyle = 'font-weight: bold; font-size: 120%; margin-top: 5px;';

    let depth = 0;

    let result = '';

    result += `<div style="${headerStyle}">Display Summary</div>`;
    result += `${this.size.toString()} frame:${this._frameId} input:${!!this._input} cursor:${this._lastCursor}<br/>`;

    function nodeCount( node ) {
      let count = 1; // for us
      for ( let i = 0; i < node.children.length; i++ ) {
        count += nodeCount( node.children[ i ] );
      }
      return count;
    }

    result += `Nodes: ${nodeCount( this._rootNode )}<br/>`;

    function instanceCount( instance ) {
      let count = 1; // for us
      for ( let i = 0; i < instance.children.length; i++ ) {
        count += instanceCount( instance.children[ i ] );
      }
      return count;
    }

    result += this._baseInstance ? ( `Instances: ${instanceCount( this._baseInstance )}<br/>` ) : '';

    function drawableCount( drawable ) {
      let count = 1; // for us
      if ( drawable.blocks ) {
        // we're a backbone
        _.each( drawable.blocks, childDrawable => {
          count += drawableCount( childDrawable );
        } );
      }
      else if ( drawable.firstDrawable && drawable.lastDrawable ) {
        // we're a block
        for ( let childDrawable = drawable.firstDrawable; childDrawable !== drawable.lastDrawable; childDrawable = childDrawable.nextDrawable ) {
          count += drawableCount( childDrawable );
        }
        count += drawableCount( drawable.lastDrawable );
      }
      return count;
    }

    result += this._rootBackbone ? ( `Drawables: ${drawableCount( this._rootBackbone )}<br/>` ) : '';

    const drawableCountMap = {}; // {string} drawable constructor name => {number} count of seen
    // increment the count in our map
    function countRetainedDrawable( drawable ) {
      const name = drawable.constructor.name;
      if ( drawableCountMap[ name ] ) {
        drawableCountMap[ name ]++;
      }
      else {
        drawableCountMap[ name ] = 1;
      }
    }

    function retainedDrawableCount( instance ) {
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

    function blockSummary( block ) {
      // ensure we are a block
      if ( !block.firstDrawable || !block.lastDrawable ) {
        return '';
      }

      const hasBackbone = block.domDrawable && block.domDrawable.blocks;

      let div = `<div style="margin-left: ${depth * 20}px">`;

      div += block.toString();
      if ( !hasBackbone ) {
        div += ` (${block.drawableCount} drawables)`;
      }

      div += '</div>';

      depth += 1;
      if ( hasBackbone ) {
        for ( let k = 0; k < block.domDrawable.blocks.length; k++ ) {
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

    function instanceSummary( instance ) {
      let iSummary = '';

      function addQualifier( text ) {
        iSummary += ` <span style="color: #008">${text}</span>`;
      }

      const node = instance.node;

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
      if ( instance.trail.isPickable() ) {
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
        case Matrix3.Types.IDENTITY:
          transformType = '';
          break;
        case Matrix3.Types.TRANSLATION_2D:
          transformType = 'translated';
          break;
        case Matrix3.Types.SCALING:
          transformType = 'scale';
          break;
        case Matrix3.Types.AFFINE:
          transformType = 'affine';
          break;
        case Matrix3.Types.OTHER:
          transformType = 'other';
          break;
        default:
          throw new Error( `invalid matrix type: ${node.transform.getMatrix().type}` );
      }
      if ( transformType ) {
        iSummary += ` <span style="color: #88f" title="${node.transform.getMatrix().toString().replace( '\n', '&#10;' )}">${transformType}</span>`;
      }

      iSummary += ` <span style="color: #888">[Trail ${instance.trail.indices.join( '.' )}]</span>`;
      iSummary += ` <span style="color: #c88">${str( instance.state )}</span>`;
      iSummary += ` <span style="color: #8c8">${node._rendererSummary.bitmask.toString( 16 )}${node._rendererBitmask !== Renderer.bitmaskNodeDefault ? ` (${node._rendererBitmask.toString( 16 )})` : ''}</span>`;

      return iSummary;
    }

    function drawableSummary( drawable ) {
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

    function printInstanceSubtree( instance ) {
      let div = `<div style="margin-left: ${depth * 20}px">`;

      function addDrawable( name, drawable ) {
        div += ` <span style="color: #888">${name}:${drawableSummary( drawable )}</span>`;
      }

      div += instanceSummary( instance );

      instance.selfDrawable && addDrawable( 'self', instance.selfDrawable );
      instance.groupDrawable && addDrawable( 'group', instance.groupDrawable );
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

    function printDrawableSubtree( drawable ) {
      let div = `<div style="margin-left: ${depth * 20}px">`;

      div += drawableSummary( drawable );
      if ( drawable.instance ) {
        div += ` <span style="color: #0a0;">(${drawable.instance.trail.toPathString()})</span>`;
        div += `&nbsp;&nbsp;&nbsp;${instanceSummary( drawable.instance )}`;
      }
      else if ( drawable.backboneInstance ) {
        div += ` <span style="color: #a00;">(${drawable.backboneInstance.trail.toPathString()})</span>`;
        div += `&nbsp;&nbsp;&nbsp;${instanceSummary( drawable.backboneInstance )}`;
      }

      div += '</div>';
      result += div;

      if ( drawable.blocks ) {
        // we're a backbone
        depth += 1;
        _.each( drawable.blocks, childDrawable => {
          printDrawableSubtree( childDrawable );
        } );
        depth -= 1;
      }
      else if ( drawable.firstDrawable && drawable.lastDrawable ) {
        // we're a block
        depth += 1;
        for ( let childDrawable = drawable.firstDrawable; childDrawable !== drawable.lastDrawable; childDrawable = childDrawable.nextDrawable ) {
          printDrawableSubtree( childDrawable );
        }
        printDrawableSubtree( drawable.lastDrawable ); // wasn't hit in our simplified (and safer) loop
        depth -= 1;
      }
    }

    if ( this._rootBackbone ) {
      result += '<div style="font-weight: bold;">Root Drawable Tree</div>';
      printDrawableSubtree( this._rootBackbone );
    }

    //OHTWO TODO: add shared cache drawable trees

    return result;
  }

  /**
   * Returns the getDebugHTML() information, but wrapped into a full HTML page included in a data URI.
   * @public
   *
   * @returns {string}
   */
  getDebugURI() {
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
   * @public
   */
  popupDebug() {
    window.open( this.getDebugURI() );
  }

  /**
   * Attempts to open an iframe popup with the getDebugHTML() information in the same window. This is similar to
   * popupDebug(), but should work in browsers that block popups, or prevent that type of data URI being opened.
   * @public
   */
  iframeDebug() {
    const iframe = document.createElement( 'iframe' );
    iframe.width = window.innerWidth; // eslint-disable-line bad-sim-text
    iframe.height = window.innerHeight; // eslint-disable-line bad-sim-text
    iframe.style.position = 'absolute';
    iframe.style.left = '0';
    iframe.style.top = '0';
    iframe.style.zIndex = '10000';
    document.body.appendChild( iframe );

    iframe.contentWindow.document.open();
    iframe.contentWindow.document.write( this.getDebugHTML() );
    iframe.contentWindow.document.close();

    iframe.contentWindow.document.body.style.background = 'white';

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

  /**
   * @public
   *
   * @returns {string}
   */
  getPDOMDebugHTML() {
    let result = '';

    const headerStyle = 'font-weight: bold; font-size: 120%; margin-top: 5px;';
    const indent = '&nbsp;&nbsp;&nbsp;&nbsp;';

    result += `<div style="${headerStyle}">Accessible Instances</div><br>`;

    recurse( this._rootPDOMInstance, '' );

    function recurse( instance, indentation ) {
      result += `${indentation + escapeHTML( `${instance.isRootInstance ? '' : instance.node.tagName} ${instance.toString()}` )}<br>`;
      instance.children.forEach( child => {
        recurse( child, indentation + indent );
      } );
    }

    result += `<br><div style="${headerStyle}">Parallel DOM</div><br>`;

    let parallelDOM = this._rootPDOMInstance.peer.primarySibling.outerHTML;
    parallelDOM = parallelDOM.replace( /></g, '>\n<' );
    const lines = parallelDOM.split( '\n' );

    let indentation = '';
    for ( let i = 0; i < lines.length; i++ ) {
      const line = lines[ i ];
      const isEndTag = line.slice( 0, 2 ) === '</';

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
   * @public
   *
   * Only tested on recent Chrome and Firefox, not recommended for general use. Guaranteed not to work for IE <= 10.
   *
   * See https://github.com/phetsims/scenery/issues/394 for some details.
   */
  foreignObjectRasterization( callback ) {
    // Scan our drawable tree for Canvases. We'll rasterize them here (to data URLs) so we can replace them later in
    // the HTML tree (with images) before putting that in the foreignObject. That way, we can actually display
    // things rendered in Canvas in our rasterization.
    const canvasUrlMap = {};

    let unknownIds = 0;

    function addCanvas( canvas ) {
      if ( !canvas.id ) {
        canvas.id = `unknown-canvas-${unknownIds++}`;
      }
      canvasUrlMap[ canvas.id ] = canvas.toDataURL();
    }

    function scanForCanvases( drawable ) {
      if ( drawable.blocks ) {
        // we're a backbone
        _.each( drawable.blocks, childDrawable => {
          scanForCanvases( childDrawable );
        } );
      }
      else if ( drawable.firstDrawable && drawable.lastDrawable ) {
        // we're a block
        for ( let childDrawable = drawable.firstDrawable; childDrawable !== drawable.lastDrawable; childDrawable = childDrawable.nextDrawable ) {
          scanForCanvases( childDrawable );
        }
        scanForCanvases( drawable.lastDrawable ); // wasn't hit in our simplified (and safer) loop

        if ( drawable.canvas && drawable.canvas instanceof window.HTMLCanvasElement ) {
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

    scanForCanvases( this._rootBackbone );

    // Create a new document, so that we can (1) serialize it to XHTML, and (2) manipulate it independently.
    // Inspired by http://cburgmer.github.io/rasterizeHTML.js/
    const doc = document.implementation.createHTMLDocument( '' );
    doc.documentElement.innerHTML = this.domElement.outerHTML;
    doc.documentElement.setAttribute( 'xmlns', doc.documentElement.namespaceURI );

    // Hide the PDOM
    doc.documentElement.appendChild( document.createElement( 'style' ) ).innerHTML = `.${PDOMSiblingStyle.ROOT_CLASS_NAME} { display:none; } `;

    // Replace each <canvas> with an <img> that has src=canvas.toDataURL() and the same style
    let displayCanvases = doc.documentElement.getElementsByTagName( 'canvas' );
    displayCanvases = Array.prototype.slice.call( displayCanvases ); // don't use a live HTMLCollection copy!
    for ( let i = 0; i < displayCanvases.length; i++ ) {
      const displayCanvas = displayCanvases[ i ];

      const cssText = displayCanvas.style.cssText;

      const displayImg = doc.createElement( 'img' );
      const src = canvasUrlMap[ displayCanvas.id ];
      assert && assert( src, 'Must have missed a toDataURL() on a Canvas' );

      displayImg.src = src;
      displayImg.setAttribute( 'style', cssText );

      displayCanvas.parentNode.replaceChild( displayImg, displayCanvas );
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

        ( () => {
          // Closure variables need to be stored for each individual SVG image.
          const refImage = new window.Image();
          const svgImage = displaySVGImage;

          refImage.onload = () => {
            // Get a Canvas
            const refCanvas = document.createElement( 'canvas' );
            refCanvas.width = refImage.width;
            refCanvas.height = refImage.height;
            const refContext = refCanvas.getContext( '2d' );

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

  /**
   * @public
   */
  popupRasterization() {
    this.foreignObjectRasterization( window.open );
  }

  /**
   * Releases references.
   * @public
   *
   * TODO: this dispose function is not complete.
   */
  dispose() {
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
      globalKeyStateTracker.keydownEmitter.removeListener( this._boundHandleFullScreenNavigation );
      this._rootPDOMInstance.dispose();
    }

    this._focusOverlay && this._focusOverlay.dispose();

    this.sizeProperty.dispose();

    // Will immediately dispose recursively, all Instances AND their attached drawables, which will include the
    // rootBackbone.
    this._baseInstance && this._baseInstance.dispose();

    this.descriptionUtteranceQueue && this.descriptionUtteranceQueue.dispose();

    this.focusManager && this.focusManager.dispose();

    if ( assert ) {
      this._isDisposing = false;
      this._isDisposed = true;
    }
  }

  /**
   * Takes a given DOM element, and asynchronously renders it to a string that is a data URL representing an SVG
   * file.
   * @public
   *
   * @param {HTMLElement} domElement
   * @param {number} width - The width of the output SVG
   * @param {number} height - The height of the output SVG
   * @param {function} callback - Called as callback( url: {string} ), where the URL will be the encoded SVG file.
   */
  static elementToSVGDataURL( domElement, width, height, callback ) {
    const canvas = document.createElement( 'canvas' );
    const context = canvas.getContext( '2d' );
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
    const uint8array = new window.TextEncoderLite( 'utf-8' ).encode( data );
    const base64 = window.fromByteArray( uint8array );

    // turn it to base64 and wrap it in the data URL format
    img.src = `data:image/svg+xml;base64,${base64}`;
  }

  /**
   * Returns true when NO nodes in the subtree are disposed.
   * @private
   *
   * @param {Node} node
   * @returns {boolean}
   */
  static assertSubtreeDisposed( node ) {
    assert && assert( !node.isDisposed, 'Disposed nodes should not be included in a scene graph to display.' );

    if ( assert ) {
      for ( let i = 0; i < node.children.length; i++ ) {
        Display.assertSubtreeDisposed( node.children[ i ] );
      }
    }
  }

  /**
   * Adds an input listener to be fired for ANY Display
   * @public
   *
   * @param {Object} listener
   */
  static addInputListener( listener ) {
    assert && assert( !_.includes( Display.inputListeners, listener ), 'Input listener already registered' );

    // don't allow listeners to be added multiple times
    if ( !_.includes( Display.inputListeners, listener ) ) {
      Display.inputListeners.push( listener );
    }
  }

  /**
   * Removes an input listener that was previously added with Display.addInputListener.
   * @public
   *
   * @param {Object} listener
   */
  static removeInputListener( listener ) {
    // ensure the listener is in our list
    assert && assert( _.includes( Display.inputListeners, listener ) );

    Display.inputListeners.splice( _.indexOf( Display.inputListeners, listener ), 1 );
  }

  /**
   * Interrupts all input listeners that are attached to all Displays.
   * @public
   */
  static interruptInput() {
    const listenersCopy = Display.inputListeners.slice( 0 );

    for ( let i = 0; i < listenersCopy.length; i++ ) {
      const listener = listenersCopy[ i ];

      listener.interrupt && listener.interrupt();
    }
  }
}

scenery.register( 'Display', Display );

Display.customCursors = {
  'scenery-grab-pointer': [ 'grab', '-moz-grab', '-webkit-grab', 'pointer' ],
  'scenery-grabbing-pointer': [ 'grabbing', '-moz-grabbing', '-webkit-grabbing', 'pointer' ]
};

// @public {Emitter} - Fires when we detect an input event that would be considered a "user gesture" by Chrome, so
// that we can trigger browser actions that are only allowed as a result.
// See https://github.com/phetsims/scenery/issues/802 and https://github.com/phetsims/vibe/issues/32 for more
// information.
Display.userGestureEmitter = new Emitter();

// @public {Array.<Object>} - Listeners that will be called for every event on ANY Display, see
// https://github.com/phetsims/scenery/issues/1149. Do not directly modify this!
Display.inputListeners = [];

export default Display;