// Copyright 2013-2016, University of Colorado Boulder

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

define( function( require ) {
  'use strict';

  var AccessibilityTree = require( 'SCENERY/accessibility/AccessibilityTree' );
  var Dimension2 = require( 'DOT/Dimension2' );
  var Emitter = require( 'AXON/Emitter' );
  var escapeHTML = require( 'PHET_CORE/escapeHTML' );
  var Events = require( 'AXON/Events' );
  var extend = require( 'PHET_CORE/extend' );
  var inherit = require( 'PHET_CORE/inherit' );
  var KeyStateTracker = require( 'SCENERY/accessibility/KeyStateTracker' );
  var Matrix3 = require( 'DOT/Matrix3' );
  var Property = require( 'AXON/Property' );
  var PropertyIO = require( 'AXON/PropertyIO' );
  var Tandem = require( 'TANDEM/Tandem' );
  var timer = require( 'AXON/timer' );

  // TODO: Order these, and see which ones we can require?
  var Features = require( 'SCENERY/util/Features' );
  var Node = require( 'SCENERY/nodes/Node' );
  var scenery = require( 'SCENERY/scenery' );
  require( 'SCENERY/display/BackboneDrawable' );
  require( 'SCENERY/display/CanvasBlock' );
  require( 'SCENERY/display/CanvasSelfDrawable' );
  var ChangeInterval = require( 'SCENERY/display/ChangeInterval' );
  require( 'SCENERY/display/DOMSelfDrawable' );
  var Drawable = require( 'SCENERY/display/Drawable' );
  var Instance = require( 'SCENERY/display/Instance' );
  require( 'SCENERY/display/InlineCanvasCacheDrawable' );
  var Renderer = require( 'SCENERY/display/Renderer' );
  require( 'SCENERY/display/SharedCanvasCacheDrawable' );
  require( 'SCENERY/display/SVGSelfDrawable' );
  var Input = require( 'SCENERY/input/Input' );
  require( 'SCENERY/util/Trail' );
  var AccessibleInstance = require( 'SCENERY/accessibility/AccessibleInstance' );
  var CanvasNodeBoundsOverlay = require( 'SCENERY/overlays/CanvasNodeBoundsOverlay' );
  var FittedBlockBoundsOverlay = require( 'SCENERY/overlays/FittedBlockBoundsOverlay' );
  var FocusIO = require( 'SCENERY/accessibility/FocusIO' );
  var FocusOverlay = require( 'SCENERY/overlays/FocusOverlay' );
  var NullableIO = require( 'TANDEM/types/NullableIO' );
  var platform = require( 'PHET_CORE/platform' );
  var PointerAreaOverlay = require( 'SCENERY/overlays/PointerAreaOverlay' );
  var PointerOverlay = require( 'SCENERY/overlays/PointerOverlay' );
  // var SceneryStyle = require( 'SCENERY/util/SceneryStyle' );
  var Util = require( 'SCENERY/util/Util' );

  /**
   * Constructs a Display that will show the rootNode and its subtree in a visual state. Default options provided below
   *
   * @param {Node} rootNode - Displays this node and all of its descendants
   * @param {Object} options - Valid parameters in the parameter object:
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
   * @constructor
   */
  function Display( rootNode, options ) {
    assert && assert( rootNode, 'rootNode is a required parameter' );

    //OHTWO TODO: hybrid batching (option to batch until an event like 'up' that might be needed for security issues)

    // supertype call to axon.Events (should just initialize a few properties here, notably _eventListeners and _staticEventListeners)
    Events.call( this );

    options = _.extend( {
      // initial display width
      width: ( options && options.container && options.container.clientWidth ) || 640,

      // initial display height
      height: ( options && options.container && options.container.clientHeight ) || 480,

      allowCSSHacks: true,       // applies CSS styles to the root DOM element that make it amenable to interactive content
      allowSceneOverflow: false, // usually anything displayed outside of our dom element is hidden with CSS overflow
      defaultCursor: 'default',  // what cursor is used when no other cursor is specified
      backgroundColor: null,      // initial background color
      preserveDrawingBuffer: false,
      allowWebGL: true,
      accessibility: true,        // enables accessibility features
      isApplication: false,      // adds the aria-role: 'application' when accessibility is enabled
      interactive: true,         // Whether mouse/touch/keyboard inputs are enabled (if input has been added).

      // {boolean} - If true, input event listeners will be attached to the Display's DOM element instead of the window.
      // Normally, attaching listeners to the window is preferred (it will see mouse moves/ups outside of the browser
      // window, allowing correct button tracking), however there may be instances where a global listener is not
      // preferred.
      listenToOnlyElement: false,

      // TODO: doc
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
      allowBackingScaleAntialiasing: true
    }, options );

    // TODO: don't store the options, it's an anti-pattern.
    this.options = options; // @private

    // @public (scenery-internal) {boolean} - Whether accessibility is enabled for this particular display.
    this._accessible = options.accessibility;

    this._allowWebGL = options.allowWebGL;

    // The (integral, > 0) dimensions of the Display's DOM element (only updates the DOM element on updateDisplay())
    this._size = new Dimension2( this.options.width, this.options.height );
    this._currentSize = new Dimension2( -1, -1 ); // used to check against new size to see what we need to change

    this._rootNode = rootNode;
    this._rootNode.addRootedDisplay( this );
    this._rootBackbone = null; // to be filled in later
    this._domElement = ( options && options.container ) ?
                       scenery.BackboneDrawable.repurposeBackboneContainer( options.container ) :
                       scenery.BackboneDrawable.createDivBackbone();
    this._sharedCanvasInstances = {}; // map from Node ID to Instance, for fast lookup
    this._baseInstance = null; // will be filled with the root Instance

    // We have a monotonically-increasing frame ID, generally for use with a pattern where we can mark objects with this
    // to note that they are either up-to-date or need refreshing due to this particular frame (without having to clear
    // that information after use). This is incremented every frame
    this._frameId = 0; // {number}

    this._dirtyTransformRoots = [];
    this._dirtyTransformRootsWithoutPass = [];

    this._instanceRootsToDispose = [];
    this._drawablesToDispose = [];

    // Block changes are handled by changing the "pending" block/backbone on drawables. We want to change them all after
    // the main stitch process has completed, so we can guarantee that a single drawable is removed from its previous
    // block before being added to a new one. This is taken care of in an updateDisplay pass after syncTree / stitching.
    this._drawablesToChangeBlock = []; // {[Drawable]}

    // Drawables have two implicit linked-lists, "current" and "old". syncTree modifies the "current" linked-list
    // information so it is up-to-date, but needs to use the "old" information also. We move updating the
    // "current" => "old" linked-list information until after syncTree and stitching is complete, and is taken care of
    // in an updateDisplay pass.
    this._drawablesToUpdateLinks = []; // {[Drawable]}

    // We store information on {ChangeInterval}s that records change interval information, that may contain references.
    // We don't want to leave those references dangling after we don't need them, so they are recorded and cleaned in
    // one of updateDisplay's phases.
    this._changeIntervalsToDispose = []; // {[ChangeInterval]}

    this._lastCursor = null;

    this._currentBackgroundCSS = null;
    this._backgroundColor = null;

    // used for shortcut animation frame functions
    this._requestAnimationFrameID = 0;

    // will be filled in with a scenery.Input if event handling is enabled
    this._input = null; // @public (phet-io)
    this._inputListeners = []; // {Array.<Object>} - Listeners that will be called for every event.
    this._interactive = this.options.interactive; // {boolean} - Whether mouse/touch/keyboard inputs are enabled (if input has been added). Simulation will still step.
    this._listenToOnlyElement = options.listenToOnlyElement; // TODO: doc
    this._batchDOMEvents = options.batchDOMEvents; // TODO: doc
    this._assumeFullWindow = options.assumeFullWindow; // TODO: doc
    this._passiveEvents = options.passiveEvents; // @private {boolean|null}

    // @public (scenery-internal) {boolean}
    this._aggressiveContextRecreation = options.aggressiveContextRecreation;

    // @public (scenery-internal) {boolean}
    this._allowBackingScaleAntialiasing = options.allowBackingScaleAntialiasing;

    // overlays currently being displayed.
    // API expected:
    //   .domElement
    //   .update()
    this._overlays = [];
    this._pointerOverlay = null;
    this._pointerAreaOverlay = null;
    this._canvasAreaBoundsOverlay = null;
    this._fittedBlockBoundsOverlay = null;

    if ( assert ) {
      // @private @assertion-only {boolean} - Whether we are running the paint phase of updateDisplay() for this Display.
      this._isPainting = false;
    }

    this.applyCSSHacks();

    this.setBackgroundColor( this.options.backgroundColor );

    // global reference if we have a Display (useful)
    this.scenery = scenery;

    if ( this.options.accessibility ) {
      if ( this.options.isApplication ) {
        this._domElement.setAttribute( 'aria-role', 'application' );
      }

      // make the PDOM invisible in the browser - it has some width and is shifted off screen so that AT can read the
      // formatting tags, see https://github.com/phetsims/scenery/issues/730
      // SceneryStyle.addRule( '.accessibility, .accessibility * { position: relative; left: -1000px; top: 0; width: 250px; height: 0; clip: rect(0,0,0,0); pointerEvents: none }' );

      this._focusRootNode = new Node();
      this._focusOverlay = new FocusOverlay( this, this._focusRootNode );
      this.addOverlay( this._focusOverlay );

      // TODO: doc
      this._unsortedAccessibleInstances = [];

      // @public (scenery-internal) {Node|null} - When this display receives a pointer event, this is the focusable
      // node that is being interacted with. When keyboard navigation resumes after pointer interaction, this node will
      // receive focus.
      this.pointerFocus = null;

      // @public (scenery-internal) - {Node|null} - When  this display is made inactive, store the focused node
      // so that when the sim becomes interactive again this node can receive focus when we resume keyboard
      // navigation
      this.activeNode = null;

      // @public {boolean} (scenery-internal) - During DOM operations where HTML elements are removed from and
      // reinserted into the PDOM, event callbacks related to focus should be blocked as these are internal operations
      // unrelated to application behavior user input, see https://github.com/phetsims/scenery/issues/925
      this.blockFocusCallbacks = false;

      // @private - the node that currently has focus when we remove an accessible trail, tracked so that we can
      // restore focus after sorting accessible instances
      this._focusedNodeOnRemoveTrail;

      // @public (scenery-internal) {AccessibleInstance}
      this._rootAccessibleInstance = AccessibleInstance.createFromPool( null, this, new scenery.Trail() );
      sceneryLog && sceneryLog.AccessibleInstance && sceneryLog.AccessibleInstance(
        'Display root instance: ' + this._rootAccessibleInstance.toString() );
      AccessibilityTree.rebuildInstanceTree( this._rootAccessibleInstance );

      // add the accessible DOM as a child of this DOM element
      this._domElement.appendChild( this._rootAccessibleInstance.peer.primarySibling );
    }
  }

  scenery.register( 'Display', Display );

  inherit( Object, Display, extend( {
    // returns the base DOM element that will be displayed by this Display
    getDOMElement: function() {
      return this._domElement;
    },
    get domElement() { return this.getDOMElement(); },

    /**
     * Updates the display's DOM element with the current visual state of the attached root node and its descendants
     */
    updateDisplay: function() {

      //OHTWO TODO: turn off after most debugging work is done
      if ( window.sceneryDebugPause ) {
        return;
      }

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

      sceneryLog && sceneryLog.Display && sceneryLog.Display( 'updateDisplay frame ' + this._frameId );
      sceneryLog && sceneryLog.Display && sceneryLog.push();

      var firstRun = !!this._baseInstance;

      // check to see whether contents under pointers changed (and if so, send the enter/exit events) to
      // maintain consistent state
      if ( this._input ) {
        // TODO: Should this be handled elsewhere?
        this._input.validatePointers();
      }

      if ( this._accessible ) {

        // update positioning of focusable peer siblings so they are discoverable on mobile assistive devices
        this._rootAccessibleInstance.peer.updateSubtreePositioning();
      }

      // validate bounds for everywhere that could trigger bounds listeners. we want to flush out any changes, so that we can call validateBounds()
      // from code below without triggering side effects (we assume that we are not reentrant).
      this._rootNode.validateWatchedBounds();

      if ( assertSlow ) { this.options.accessibility && this._rootAccessibleInstance.auditRoot(); }

      if ( assertSlow ) { this._rootNode._picker.audit(); }

      this._baseInstance = this._baseInstance || scenery.Instance.createFromPool( this, new scenery.Trail( this._rootNode ), true, false );
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
        var changed = this._drawablesToChangeBlock.pop().updateBlock();
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
        var zIndex = this._rootBackbone.lastZIndex;
        for ( var i = 0; i < this._overlays.length; i++ ) {
          // layer the overlays properly
          var overlay = this._overlays[ i ];
          overlay.domElement.style.zIndex = zIndex++;

          overlay.update();
        }
      }

      this._frameId++;

      if ( sceneryLog && scenery.isLoggingPerformance() ) {
        var syncTreeMessage = 'syncTree count: ' + this.perfSyncTreeCount;
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

        var drawableBlockCountMessage = 'drawable block changes: ' + this.perfDrawableBlockChangeCount + ' for' +
                                        ' -' + this.perfDrawableOldIntervalCount +
                                        ' +' + this.perfDrawableNewIntervalCount;
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

      AccessibilityTree.auditAccessibleDisplays( this.rootNode );

      sceneryLog && sceneryLog.Display && sceneryLog.pop();
    },

    updateSize: function() {
      var sizeDirty = false;
      //OHTWO TODO: if we aren't clipping or setting background colors, can we get away with having a 0x0 container div and using absolutely-positioned children?
      if ( this._size.width !== this._currentSize.width ) {
        sizeDirty = true;
        this._currentSize.width = this._size.width;
        this._domElement.style.width = this._size.width + 'px';
      }
      if ( this._size.height !== this._currentSize.height ) {
        sizeDirty = true;
        this._currentSize.height = this._size.height;
        this._domElement.style.height = this._size.height + 'px';
      }
      if ( sizeDirty && !this.options.allowSceneOverflow ) {
        // to prevent overflow, we add a CSS clip
        //TODO: 0px => 0?
        this._domElement.style.clip = 'rect(0px,' + this._size.width + 'px,' + this._size.height + 'px,0px)';
      }
    },

    getRootNode: function() {
      return this._rootNode;
    },
    get rootNode() { return this.getRootNode(); },

    // The dimensions of the Display's DOM element
    getSize: function() {
      return this._size;
    },
    get size() { return this.getSize(); },

    getBounds: function() {
      return this._size.toBounds();
    },
    get bounds() { return this.getBounds(); },

    // size: dot.Dimension2. Changes the size that the Display's DOM element will be after the next updateDisplay()
    setSize: function( size ) {
      assert && assert( size instanceof Dimension2 );
      assert && assert( size.width % 1 === 0, 'Display.width should be an integer' );
      assert && assert( size.width > 0, 'Display.width should be greater than zero' );
      assert && assert( size.height % 1 === 0, 'Display.height should be an integer' );
      assert && assert( size.height > 0, 'Display.height should be greater than zero' );

      if ( !this._size.equals( size ) ) {
        this._size = size;

        this.trigger1( 'displaySize', this._size );
      }
    },

    setWidthHeight: function( width, height ) {
      // TODO: don't burn an instance here?
      this.setSize( new Dimension2( width, height ) );
    },

    // The width of the Display's DOM element
    getWidth: function() {
      return this._size.width;
    },
    get width() { return this.getWidth(); },

    // Sets the width that the Display's DOM element will be after the next updateDisplay(). Should be an integral value.
    setWidth: function( width ) {
      assert && assert( typeof width === 'number', 'Display.width should be a number' );

      if ( this.getWidth() !== width ) {
        // TODO: remove allocation here?
        this.setSize( new Dimension2( width, this.getHeight() ) );
      }
    },
    set width( value ) { this.setWidth( value ); },

    // The height of the Display's DOM element
    getHeight: function() {
      return this._size.height;
    },
    get height() { return this.getHeight(); },

    // Sets the height that the Display's DOM element will be after the next updateDisplay(). Should be an integral value.
    setHeight: function( height ) {
      assert && assert( typeof height === 'number', 'Display.height should be a number' );

      if ( this.getHeight() !== height ) {
        // TODO: remove allocation here?
        this.setSize( new Dimension2( this.getWidth(), height ) );
      }
    },
    set height( value ) { this.setHeight( value ); },

    // {string} (CSS), {Color} instance, or null (no background color).
    // Will be applied to the root DOM element on updateDisplay(), and no sooner.
    setBackgroundColor: function( color ) {
      assert && assert( color === null || typeof color === 'string' || color instanceof scenery.Color );

      this._backgroundColor = color;
    },
    set backgroundColor( value ) { this.setBackgroundColor( value ); },

    getBackgroundColor: function() {
      return this._backgroundColor;
    },
    get backgroundColor() { return this.getBackgroundColor(); },

    get interactive() { return this._interactive; },
    set interactive( value ) {
      this._interactive = value;
      if ( !this._interactive && this._input ) {
        this._input.interruptPointers();
        this._input.clearBatchedEvents();
        this._input.removeTemporaryPointers();
        this._rootNode.interruptSubtreeInput();
      }

      // when not interactive, all keyboard navigation is disabled
      // TODO: disable keyboard nav without hiding content so that it is still readable with a screen reader,
      // see https://github.com/phetsims/phet-io/issues/995
      if ( this.options.accessibility ) {
        if ( !this._interactive ) {
          this.activeNode = Display.focusedNode;

          // prevent a FF bug where hiding the element without blurring it causes focus to get stuck in the body
          this.activeNode && this.activeNode.blur();
        }

        this.accessibleDOMElement.hidden = !this._interactive;
      }
    },

    addOverlay: function( overlay ) {
      this._overlays.push( overlay );
      this._domElement.appendChild( overlay.domElement );

      // ensure that the overlay is hidden from screen readers, all accessible content should be in the dom element
      // of the this._rootAccessibleInstance
      overlay.domElement.setAttribute( 'aria-hidden', true );
    },

    removeOverlay: function( overlay ) {
      this._domElement.removeChild( overlay.domElement );
      this._overlays.splice( _.indexOf( this._overlays, overlay ), 1 );
    },

    markUnsortedAccessibleInstance: function( accessibleInstance ) {
      this._unsortedAccessibleInstances.push( accessibleInstance );
    },


    sortAccessibleInstances: function() {

      // keep reference so we can restore focus if browser blurs element while sorting
      var focusedNode = Display.focusedNode;

      while ( this._unsortedAccessibleInstances.length ) {
        this._unsortedAccessibleInstances.pop().sortChildren();
      }
      focusedNode && focusedNode.focus();
    },

    /**
     * Get the root accessible DOM element which represents this display and provides semantics for assistive
     * technology.
     * @public
     * @returns {HTMLElement}
     */
    getAccessibleDOMElement: function() {
      return this._rootAccessibleInstance.peer.primarySibling;
    },
    get accessibleDOMElement() { return this.getAccessibleDOMElement(); },

    /*
     * Returns the bitmask union of all renderers (canvas/svg/dom/webgl) that are used for display, excluding
     * BackboneDrawables (which would be DOM).
     */
    getUsedRenderersBitmask: function() {
      function renderersUnderBackbone( backbone ) {
        var bitmask = 0;
        _.each( backbone.blocks, function( block ) {
          if ( block instanceof scenery.DOMBlock && block.domDrawable instanceof scenery.BackboneDrawable ) {
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
    },

    /*
     * Called from Instances that will need a transform update (for listeners and precomputation).
     * @param passTransform {boolean} - Whether we should pass the first transform root when validating transforms (should be true if the instance is transformed)
     */
    markTransformRootDirty: function( instance, passTransform ) {
      passTransform ? this._dirtyTransformRoots.push( instance ) : this._dirtyTransformRootsWithoutPass.push( instance );
    },

    updateDirtyTransformRoots: function() {
      sceneryLog && sceneryLog.transformSystem && sceneryLog.transformSystem( 'updateDirtyTransformRoots' );
      sceneryLog && sceneryLog.transformSystem && sceneryLog.push();
      while ( this._dirtyTransformRoots.length ) {
        this._dirtyTransformRoots.pop().relativeTransform.updateTransformListenersAndCompute( false, false, this._frameId, true );
      }
      while ( this._dirtyTransformRootsWithoutPass.length ) {
        this._dirtyTransformRootsWithoutPass.pop().relativeTransform.updateTransformListenersAndCompute( false, false, this._frameId, false );
      }
      sceneryLog && sceneryLog.transformSystem && sceneryLog.pop();
    },

    markDrawableChangedBlock: function( drawable ) {
      assert && assert( drawable instanceof Drawable );

      sceneryLog && sceneryLog.Display && sceneryLog.Display( 'markDrawableChangedBlock: ' + drawable.toString() );
      this._drawablesToChangeBlock.push( drawable );
    },

    markInstanceRootForDisposal: function( instance ) {
      assert && assert( instance instanceof Instance, 'How would an instance not be an instance of an instance?!?!?' );

      sceneryLog && sceneryLog.Display && sceneryLog.Display( 'markInstanceRootForDisposal: ' + instance.toString() );
      this._instanceRootsToDispose.push( instance );
    },

    markDrawableForDisposal: function( drawable ) {
      assert && assert( drawable instanceof Drawable );

      sceneryLog && sceneryLog.Display && sceneryLog.Display( 'markDrawableForDisposal: ' + drawable.toString() );
      this._drawablesToDispose.push( drawable );
    },

    markDrawableForLinksUpdate: function( drawable ) {
      assert && assert( drawable instanceof Drawable );

      this._drawablesToUpdateLinks.push( drawable );
    },

    // Add a {ChangeInterval} for the "remove change interval info" phase (we don't want to leak memory/references)
    markChangeIntervalToDispose: function( changeInterval ) {
      assert && assert( changeInterval instanceof ChangeInterval );

      this._changeIntervalsToDispose.push( changeInterval );
    },

    updateBackgroundColor: function() {
      assert && assert( this._backgroundColor === null ||
                        typeof this._backgroundColor === 'string' ||
                        this._backgroundColor instanceof scenery.Color );

      var newBackgroundCSS = this._backgroundColor === null ?
                             '' :
                             ( this._backgroundColor.toCSS ?
                               this._backgroundColor.toCSS() :
                               this._backgroundColor );
      if ( newBackgroundCSS !== this._currentBackgroundCSS ) {
        this._currentBackgroundCSS = newBackgroundCSS;

        this._domElement.style.backgroundColor = newBackgroundCSS;
      }
    },

    /*---------------------------------------------------------------------------*
     * Cursors
     *----------------------------------------------------------------------------*/

    updateCursor: function() {
      if ( this._input && this._input.mouse && this._input.mouse.point ) {
        if ( this._input.mouse.cursor ) {
          sceneryLog && sceneryLog.Cursor && sceneryLog.Cursor( 'set on pointer: ' + this._input.mouse.cursor );
          return this.setSceneCursor( this._input.mouse.cursor );
        }

        //OHTWO TODO: For a display, just return an instance and we can avoid the garbage collection/mutation at the cost of the linked-list traversal instead of an array
        var mouseTrail = this._rootNode.trailUnderPointer( this._input.mouse );

        if ( mouseTrail ) {
          for ( var i = mouseTrail.getCursorCheckIndex(); i >= 0; i-- ) {
            var node = mouseTrail.nodes[ i ];
            var cursor = node.getCursor();

            if ( cursor ) {
              sceneryLog && sceneryLog.Cursor && sceneryLog.Cursor( cursor + ' on ' + node.constructor.name + '#' + node.id );
              return this.setSceneCursor( cursor );
            }
          }
        }

        sceneryLog && sceneryLog.Cursor && sceneryLog.Cursor( '--- for ' + ( mouseTrail ? mouseTrail.toString() : '(no hit)' ) );
      }

      // fallback case
      return this.setSceneCursor( this.options.defaultCursor );
    },

    setSceneCursor: function( cursor ) {
      if ( cursor !== this._lastCursor ) {
        this._lastCursor = cursor;
        var customCursors = Display.customCursors[ cursor ];
        if ( customCursors ) {
          // go backwards, so the most desired cursor sticks
          for ( var i = customCursors.length - 1; i >= 0; i-- ) {
            this._domElement.style.cursor = customCursors[ i ];
          }
        }
        else {
          this._domElement.style.cursor = cursor;
        }
      }
    },

    applyCSSHacks: function() {
      // to use CSS3 transforms for performance, hide anything outside our bounds by default
      if ( !this.options.allowSceneOverflow ) {
        this._domElement.style.overflow = 'hidden';
      }

      // Prevents selection cursor issues in Safari, see https://github.com/phetsims/scenery/issues/476
      document.onselectstart = function() {
        return false;
      };

      // forward all pointer events
      this._domElement.style.msTouchAction = 'none';

      // don't allow browser to switch between font smoothing methods for text (see https://github.com/phetsims/scenery/issues/431)
      Features.setStyle( this._domElement, Features.fontSmoothing, 'antialiased' );

      if ( this.options.allowCSSHacks ) {
        // some css hacks (inspired from https://github.com/EightMedia/hammer.js/blob/master/hammer.js).
        // modified to only apply the proper prefixed version instead of spamming all of them, and doesn't use jQuery.
        Features.setStyle( this._domElement, Features.userDrag, 'none' );
        Features.setStyle( this._domElement, Features.userSelect, 'none' );
        Features.setStyle( this._domElement, Features.touchAction, 'none' );
        Features.setStyle( this._domElement, Features.touchCallout, 'none' );
        Features.setStyle( this._domElement, Features.tapHighlightColor, 'rgba(0,0,0,0)' );
      }
    },

    // TODO: consider SVG data URLs
    canvasDataURL: function( callback ) {
      this.canvasSnapshot( function( canvas ) {
        callback( canvas.toDataURL() );
      } );
    },

    // renders what it can into a Canvas (so far, Canvas and SVG layers work fine)
    canvasSnapshot: function( callback ) {
      var canvas = document.createElement( 'canvas' );
      canvas.width = this._size.width;
      canvas.height = this._size.height;

      var context = canvas.getContext( '2d' );

      //OHTWO TODO: allow actual background color directly, not having to check the style here!!!
      this._rootNode.renderToCanvas( canvas, context, function() {
        callback( canvas, context.getImageData( 0, 0, canvas.width, canvas.height ) );
      }, this.domElement.style.backgroundColor );
    },

    //TODO: reduce code duplication for handling overlays
    setPointerDisplayVisible: function( visibility ) {
      assert && assert( typeof visibility === 'boolean' );

      var hasOverlay = !!this._pointerOverlay;

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
    },

    //TODO: reduce code duplication for handling overlays
    setPointerAreaDisplayVisible: function( visibility ) {
      assert && assert( typeof visibility === 'boolean' );

      var hasOverlay = !!this._pointerAreaOverlay;

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
    },

    //TODO: reduce code duplication for handling overlays
    setCanvasNodeBoundsVisible: function( visibility ) {
      assert && assert( typeof visibility === 'boolean' );

      var hasOverlay = !!this._canvasAreaBoundsOverlay;

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
    },

    //TODO: reduce code duplication for handling overlays
    setFittedBlockBoundsVisible: function( visibility ) {
      assert && assert( typeof visibility === 'boolean' );

      var hasOverlay = !!this._fittedBlockBoundsOverlay;

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
    },

    resizeOnWindowResize: function() {
      var self = this;

      var resizer = function() {
        self.setWidthHeight( window.innerWidth, window.innerHeight );
      };
      window.addEventListener( 'resize', resizer );
      resizer();
    },

    // Updates on every request animation frame. If stepCallback is passed in, it is called before updateDisplay() with
    // stepCallback( timeElapsedInSeconds )
    updateOnRequestAnimationFrame: function( stepCallback ) {
      // keep track of how much time elapsed over the last frame
      var lastTime = 0;
      var timeElapsedInSeconds = 0;

      var self = this;
      ( function step() {
        self._requestAnimationFrameID = window.requestAnimationFrame( step, self._domElement );

        // calculate how much time has elapsed since we rendered the last frame
        var timeNow = Date.now();
        if ( lastTime !== 0 ) {
          timeElapsedInSeconds = ( timeNow - lastTime ) / 1000.0;
        }
        lastTime = timeNow;

        // step the timer that drives any time dependent updates of the Display
        timer.emit( timeElapsedInSeconds );

        stepCallback && stepCallback( timeElapsedInSeconds );
        self.updateDisplay();
      } )();
    },

    cancelUpdateOnRequestAnimationFrame: function() {
      window.cancelAnimationFrame( this._requestAnimationFrameID );
    },

    /**
     * Initializes event handling, and connects the browser's input event handlers to notify this Display of events.
     * @public
     *
     * NOTE: This can be reversed with detachEvents().
     * @param {Object} [options] - for PhET-iO
     */
    initializeEvents: function( options ) {
      assert && assert( !this._input, 'Events cannot be attached twice to a display (for now)' );

      // TODO: refactor here
      var input = new Input( this, !this._listenToOnlyElement, this._batchDOMEvents, this._assumeFullWindow, this._passiveEvents, options );
      this._input = input;

      input.connectListeners();
    },

    /**
     * Detach already-attached input event handling (from initializeEvents()).
     * @public
     */
    detachEvents: function() {
      assert && assert( this._input, 'detachEvents() should be called only when events are attached' );

      this._input.disconnectListeners();
      this._input = null;
    },


    /**
     * Adds an input listener.
     * @public
     *
     * @param {Object} listener
     * @returns {Display} - For chaining
     */
    addInputListener: function( listener ) {
      assert && assert( !_.includes( this._inputListeners, listener ), 'Input listener already registered on this Node' );

      // don't allow listeners to be added multiple times
      if ( !_.includes( this._inputListeners, listener ) ) {
        this._inputListeners.push( listener );
      }
      return this;
    },

    /**
     * Removes an input listener that was previously added with addInputListener.
     * @public
     *
     * @param {Object} listener
     * @returns {Display} - For chaining
     */
    removeInputListener: function( listener ) {
      // ensure the listener is in our list
      assert && assert( _.includes( this._inputListeners, listener ) );

      this._inputListeners.splice( _.indexOf( this._inputListeners, listener ), 1 );

      return this;
    },

    /**
     * Returns whether this input listener is currently listening to this node.
     * @public
     *
     * More efficient than checking node.inputListeners, as that includes a defensive copy.
     *
     * @param {Object} listener
     * @returns {boolean}
     */
    hasInputListener: function( listener ) {
      for ( var i = 0; i < this._inputListeners.length; i++ ) {
        if ( this._inputListeners[ i ] === listener ) {
          return true;
        }
      }
      return false;
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
    get inputListeners() { return this.getInputListeners(); },

    ensureNotPainting: function() {
      assert && assert( !this._isPainting,
        'This should not be run in the call tree of updateDisplay(). If you see this, it is likely that either the ' +
        'last updateDisplay() had a thrown error and it is trying to be run again (in which case, investigate that ' +
        'error), OR code was run/triggered from inside an updateDisplay() that has the potential to cause an infinite ' +
        'loop, e.g. CanvasNode paintCanvas() call manipulating another Node, or a bounds listener that Scenery missed.' );
    },

    /**
     * Triggers a loss of context for all WebGL blocks.
     * @public
     *
     * NOTE: Should generally only be used for debugging.
     */
    loseWebGLContexts: function() {
      ( function loseBackbone( backbone ) {
        if ( backbone.blocks ) {
          backbone.blocks.forEach( function( block ) {
            if ( block.gl ) {
              Util.loseContext( block.gl );
            }

            //TODO: pattern for this iteration
            for ( var drawable = block.firstDrawable; drawable !== null; drawable = drawable.nextDrawable ) {
              loseBackbone( drawable );
              if ( drawable === block.lastDrawable ) { break; }
            }
          } );
        }
      } )( this._rootBackbone );
    },

    /**
     * Makes this Display available for inspection.
     * @public
     */
    inspect: function() {
      localStorage.scenerySnapshot = JSON.stringify( scenery.serialize( this ) );
    },

    /**
     * Returns an HTML fragment {string} that includes a large amount of debugging information, including a view of the
     * instance tree and drawable tree.
     */
    getDebugHTML: function() {
      function str( ob ) {
        return ob ? ob.toString() : ob + '';
      }

      var headerStyle = 'font-weight: bold; font-size: 120%; margin-top: 5px;';

      var depth = 0;

      var result = '';

      result += '<div style="' + headerStyle + '">Display Summary</div>';
      result += this._size.toString() + ' frame:' + this._frameId + ' input:' + !!this._input + ' cursor:' + this._lastCursor + '<br/>';

      function nodeCount( node ) {
        var count = 1; // for us
        for ( var i = 0; i < node.children.length; i++ ) {
          count += nodeCount( node.children[ i ] );
        }
        return count;
      }

      result += 'Nodes: ' + nodeCount( this._rootNode ) + '<br/>';

      function instanceCount( instance ) {
        var count = 1; // for us
        for ( var i = 0; i < instance.children.length; i++ ) {
          count += instanceCount( instance.children[ i ] );
        }
        return count;
      }

      result += this._baseInstance ? ( 'Instances: ' + instanceCount( this._baseInstance ) + '<br/>' ) : '';

      function drawableCount( drawable ) {
        var count = 1; // for us
        if ( drawable.blocks ) {
          // we're a backbone
          _.each( drawable.blocks, function( childDrawable ) {
            count += drawableCount( childDrawable );
          } );
        }
        else if ( drawable.firstDrawable && drawable.lastDrawable ) {
          // we're a block
          for ( var childDrawable = drawable.firstDrawable; childDrawable !== drawable.lastDrawable; childDrawable = childDrawable.nextDrawable ) {
            count += drawableCount( childDrawable );
          }
          count += drawableCount( drawable.lastDrawable );
        }
        return count;
      }

      result += this._rootBackbone ? ( 'Drawables: ' + drawableCount( this._rootBackbone ) + '<br/>' ) : '';

      var drawableCountMap = {}; // {string} drawable constructor name => {number} count of seen
      // increment the count in our map
      function countRetainedDrawable( drawable ) {
        var name = drawable.constructor.name;
        if ( drawableCountMap[ name ] ) {
          drawableCountMap[ name ]++;
        }
        else {
          drawableCountMap[ name ] = 1;
        }
      }

      function retainedDrawableCount( instance ) {
        var count = 0;
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
        for ( var i = 0; i < instance.children.length; i++ ) {
          count += retainedDrawableCount( instance.children[ i ] );
        }
        return count;
      }

      result += this._baseInstance ? ( 'Retained Drawables: ' + retainedDrawableCount( this._baseInstance ) + '<br/>' ) : '';
      for ( var drawableName in drawableCountMap ) {
        result += '&nbsp;&nbsp;&nbsp;&nbsp;' + drawableName + ': ' + drawableCountMap[ drawableName ] + '<br/>';
      }

      function blockSummary( block ) {
        // ensure we are a block
        if ( !block.firstDrawable || !block.lastDrawable ) {
          return;
        }

        var hasBackbone = block.domDrawable && block.domDrawable.blocks;

        var div = '<div style="margin-left: ' + ( depth * 20 ) + 'px">';

        div += block.toString();
        if ( !hasBackbone ) {
          div += ' (' + block.drawableCount + ' drawables)';
        }

        div += '</div>';

        depth += 1;
        if ( hasBackbone ) {
          for ( var k = 0; k < block.domDrawable.blocks.length; k++ ) {
            div += blockSummary( block.domDrawable.blocks[ k ] );
          }
        }
        depth -= 1;

        return div;
      }

      if ( this._rootBackbone ) {
        result += '<div style="' + headerStyle + '">Block Summary</div>';
        for ( var i = 0; i < this._rootBackbone.blocks.length; i++ ) {
          result += blockSummary( this._rootBackbone.blocks[ i ] );
        }
      }

      function instanceSummary( instance ) {
        var iSummary = '';

        function addQualifier( text ) {
          iSummary += ' <span style="color: #008">' + text + '</span>';
        }

        var node = instance.node;

        iSummary += instance.id;
        iSummary += ' ' + ( node.constructor.name ? node.constructor.name : '?' );
        iSummary += ' <span style="font-weight: ' + ( node.isPainted() ? 'bold' : 'normal' ) + '">' + node.id + '</span>';
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
          addQualifier( 'renderer:' + node.getRenderer() );
        }
        if ( node.isLayerSplit() ) {
          addQualifier( 'layerSplit' );
        }
        if ( node.opacity < 1 ) {
          addQualifier( 'opacity:' + node.opacity );
        }

        if ( node._boundsEventCount > 0 ) {
          addQualifier( '<span style="color: #800">boundsListen:' + node._boundsEventCount + ':' + node._boundsEventSelfCount + '</span>' );
        }

        var transformType = '';
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
            throw new Error( 'invalid matrix type: ' + node.transform.getMatrix().type );
        }
        if ( transformType ) {
          iSummary += ' <span style="color: #88f" title="' + node.transform.getMatrix().toString().replace( '\n', '&#10;' ) + '">' + transformType + '</span>';
        }

        iSummary += ' <span style="color: #888">[Trail ' + instance.trail.indices.join( '.' ) + ']</span>';
        iSummary += ' <span style="color: #c88">' + str( instance.state ) + '</span>';
        iSummary += ' <span style="color: #8c8">' + node._rendererSummary.bitmask.toString( 16 ) + ( node._rendererBitmask !== Renderer.bitmaskNodeDefault ? ' (' + node._rendererBitmask.toString( 16 ) + ')' : '' ) + '</span>';

        return iSummary;
      }

      function drawableSummary( drawable ) {
        var drawableString = drawable.toString();
        if ( drawable.visible ) {
          drawableString = '<strong>' + drawableString + '</strong>';
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
        var div = '<div style="margin-left: ' + ( depth * 20 ) + 'px">';

        function addDrawable( name, drawable ) {
          div += ' <span style="color: #888">' + name + ':' + drawableSummary( drawable ) + '</span>';
        }

        div += instanceSummary( instance );

        instance.selfDrawable && addDrawable( 'self', instance.selfDrawable );
        instance.groupDrawable && addDrawable( 'group', instance.groupDrawable );
        instance.sharedCacheDrawable && addDrawable( 'sharedCache', instance.sharedCacheDrawable );

        div += '</div>';
        result += div;

        depth += 1;
        _.each( instance.children, function( childInstance ) {
          printInstanceSubtree( childInstance );
        } );
        depth -= 1;
      }

      if ( this._baseInstance ) {
        result += '<div style="' + headerStyle + '">Root Instance Tree</div>';
        printInstanceSubtree( this._baseInstance );
      }

      _.each( this._sharedCanvasInstances, function( instance ) {
        result += '<div style="' + headerStyle + '">Shared Canvas Instance Tree</div>';
        printInstanceSubtree( instance );
      } );

      function printDrawableSubtree( drawable ) {
        var div = '<div style="margin-left: ' + ( depth * 20 ) + 'px">';

        div += drawableSummary( drawable );
        if ( drawable.instance ) {
          div += ' <span style="color: #0a0;">(' + drawable.instance.trail.toPathString() + ')</span>';
          div += '&nbsp;&nbsp;&nbsp;' + instanceSummary( drawable.instance );
        }
        else if ( drawable.backboneInstance ) {
          div += ' <span style="color: #a00;">(' + drawable.backboneInstance.trail.toPathString() + ')</span>';
          div += '&nbsp;&nbsp;&nbsp;' + instanceSummary( drawable.backboneInstance );
        }

        div += '</div>';
        result += div;

        if ( drawable.blocks ) {
          // we're a backbone
          depth += 1;
          _.each( drawable.blocks, function( childDrawable ) {
            printDrawableSubtree( childDrawable );
          } );
          depth -= 1;
        }
        else if ( drawable.firstDrawable && drawable.lastDrawable ) {
          // we're a block
          depth += 1;
          for ( var childDrawable = drawable.firstDrawable; childDrawable !== drawable.lastDrawable; childDrawable = childDrawable.nextDrawable ) {
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
    },

    popupDebug: function() {
      var htmlContent = '<!DOCTYPE html>' +
                        '<html lang="en">' +
                        '<head><title>Scenery Debug Snapshot</title></head>' +
                        '<body style="font-size: 12px;">' + this.getDebugHTML() + '</body>' +
                        '</html>';
      window.open( 'data:text/html;charset=utf-8,' + encodeURIComponent( htmlContent ) );
    },

    getAccessibleDebugHTML: function() {
      var result = '';

      var headerStyle = 'font-weight: bold; font-size: 120%; margin-top: 5px;';
      var indent = '&nbsp;&nbsp;&nbsp;&nbsp;';

      result += '<div style="' + headerStyle + '">Accessible Instances</div><br>';

      recurse( this._rootAccessibleInstance, '' );

      function recurse( instance, indentation ) {
        result += indentation + escapeHTML( ( instance.isRootInstance ? '' : instance.node.tagName ) + ' ' + instance.toString() ) + '<br>';
        instance.children.forEach( function( child ) {
          recurse( child, indentation + indent );
        } );
      }

      result += '<br><div style="' + headerStyle + '">Parallel DOM</div><br>';

      var parallelDOM = this._rootAccessibleInstance.peer.primarySibling.outerHTML;
      parallelDOM = parallelDOM.replace( /\>\</g, '>\n<' );
      var lines = parallelDOM.split( '\n' );

      var indentation = '';
      for ( var i = 0; i < lines.length; i++ ) {
        var line = lines[ i ];
        var isEndTag = line.slice( 0, 2 ) === '</';

        if ( isEndTag ) {
          indentation = indentation.slice( indent.length );
        }
        result += indentation + escapeHTML( line ) + '<br>';
        if ( !isEndTag ) {
          indentation += indent;
        }
      }
      return result;
    },

    /**
     * Will attempt to call callback( {string} dataURI ) with the rasterization of the entire Display's DOM structure,
     * used for internal testing. Will call-back null if there was an error
     *
     * Only tested on recent Chrome and Firefox, not recommended for general use. Guaranteed not to work for IE <= 10.
     *
     * See https://github.com/phetsims/scenery/issues/394 for some details.
     */
    foreignObjectRasterization: function( callback ) {
      // Scan our drawable tree for Canvases. We'll rasterize them here (to data URLs) so we can replace them later in
      // the HTML tree (with images) before putting that in the foreignObject. That way, we can actually display
      // things rendered in Canvas in our rasterization.
      var canvasUrlMap = {};

      var unknownIds = 0;

      function addCanvas( canvas ) {
        if ( !canvas.id ) {
          canvas.id = 'unknown-canvas-' + unknownIds++;
        }
        canvasUrlMap[ canvas.id ] = canvas.toDataURL();
      }

      function scanForCanvases( drawable ) {
        if ( drawable.blocks ) {
          // we're a backbone
          _.each( drawable.blocks, function( childDrawable ) {
            scanForCanvases( childDrawable );
          } );
        }
        else if ( drawable.firstDrawable && drawable.lastDrawable ) {
          // we're a block
          for ( var childDrawable = drawable.firstDrawable; childDrawable !== drawable.lastDrawable; childDrawable = childDrawable.nextDrawable ) {
            scanForCanvases( childDrawable );
          }
          scanForCanvases( drawable.lastDrawable ); // wasn't hit in our simplified (and safer) loop

          if ( drawable.canvas && drawable.canvas instanceof window.HTMLCanvasElement ) {
            addCanvas( drawable.canvas );
          }
        }

        if ( scenery.DOMDrawable && drawable instanceof scenery.DOMDrawable ) {
          if ( drawable.domElement instanceof window.HTMLCanvasElement ) {
            addCanvas( drawable.domElement );
          }
          Array.prototype.forEach.call( drawable.domElement.getElementsByTagName( 'canvas' ), function( canvas ) {
            addCanvas( canvas );
          } );
        }
      }

      scanForCanvases( this._rootBackbone );

      // Create a new document, so that we can (1) serialize it to XHTML, and (2) manipulate it independently.
      // Inspired by http://cburgmer.github.io/rasterizeHTML.js/
      var doc = document.implementation.createHTMLDocument( '' );
      doc.documentElement.innerHTML = this.domElement.outerHTML;
      doc.documentElement.setAttribute( 'xmlns', doc.documentElement.namespaceURI );

      // Replace each <canvas> with an <img> that has src=canvas.toDataURL() and the same style
      var displayCanvases = doc.documentElement.getElementsByTagName( 'canvas' );
      displayCanvases = Array.prototype.slice.call( displayCanvases ); // don't use a live HTMLCollection copy!
      for ( var i = 0; i < displayCanvases.length; i++ ) {
        var displayCanvas = displayCanvases[ i ];

        var cssText = displayCanvas.style.cssText;

        var displayImg = doc.createElement( 'img' );
        var src = canvasUrlMap[ displayCanvas.id ];
        assert && assert( src, 'Must have missed a toDataURL() on a Canvas' );

        displayImg.src = src;
        displayImg.setAttribute( 'style', cssText );

        displayCanvas.parentNode.replaceChild( displayImg, displayCanvas );
      }

      var displayWidth = this.width;
      var displayHeight = this.height;
      var completeFunction = function() {
        Display.elementToSVGDataURL( doc.documentElement, displayWidth, displayHeight, callback );
      };

      // Convert each <image>'s xlink:href so that it's a data URL with the relevant data, e.g.
      // <image ... xlink:href="http://localhost:8080/scenery-phet/images/battery-D-cell.png?bust=1476308407988"/>
      // gets replaced with a data URL.
      // See https://github.com/phetsims/scenery/issues/573
      var replacedImages = 0; // Count how many images get replaced. We'll decrement with each finished image.
      var hasReplacedImages = false; // Whether any images are replaced
      var displaySVGImages = Array.prototype.slice.call( doc.documentElement.getElementsByTagName( 'image' ) );
      for ( var j = 0; j < displaySVGImages.length; j++ ) {
        var displaySVGImage = displaySVGImages[ j ];
        var currentHref = displaySVGImage.getAttribute( 'xlink:href' );
        if ( currentHref.slice( 0, 5 ) !== 'data:' ) {
          replacedImages++;
          hasReplacedImages = true;

          ( function() {
            // Closure variables need to be stored for each individual SVG image.
            var refImage = new window.Image();
            var svgImage = displaySVGImage;

            refImage.onload = function() {
              // Get a Canvas
              var refCanvas = document.createElement( 'canvas' );
              refCanvas.width = refImage.width;
              refCanvas.height = refImage.height;
              var refContext = refCanvas.getContext( '2d' );

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
            refImage.onerror = function() {
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
    },

    popupRasterization: function() {
      this.foreignObjectRasterization( window.open );
    }
  }, Events.prototype ), {
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
    elementToSVGDataURL: function( domElement, width, height, callback ) {
      var canvas = document.createElement( 'canvas' );
      var context = canvas.getContext( '2d' );
      canvas.width = width;
      canvas.height = height;

      // Serialize it to XHTML that can be used in foreignObject (HTML can't be)
      var xhtml = new window.XMLSerializer().serializeToString( domElement );

      // Create an SVG container with a foreignObject.
      var data = '<svg xmlns="http://www.w3.org/2000/svg" width="' + width + '" height="' + height + '">' +
                 '<foreignObject width="100%" height="100%">' +
                 '<div xmlns="http://www.w3.org/1999/xhtml">' +
                 xhtml +
                 '</div>' +
                 '</foreignObject>' +
                 '</svg>';

      // Load an <img> with the SVG data URL, and when loaded draw it into our Canvas
      var img = new window.Image();
      img.onload = function() {
        context.drawImage( img, 0, 0 );
        callback( canvas.toDataURL() ); // Endpoint here
      };
      img.onerror = function() {
        callback( null );
      };

      // We can't btoa() arbitrary unicode, so we need another solution,
      // see https://developer.mozilla.org/en-US/docs/Web/API/WindowBase64/Base64_encoding_and_decoding#The_.22Unicode_Problem.22
      var uint8array = new window.TextEncoderLite( 'utf-8' ).encode( data );
      var base64 = window.fromByteArray( uint8array );

      // turn it to base64 and wrap it in the data URL format
      img.src = 'data:image/svg+xml;base64,' + base64;
    },

    /**
     * Set the focus for Display.  Can set to null to clear focus from Display.
     * @public
     *
     * @param  {Focus|null} value
     */
    set focus( value ) {

      // If in phet-io brand, a11y is enabled, and the focus is not null
      if ( window.phet && phet.phetio && phet.chipper.accessibility && value ) {
        var node = value.trail.lastNode();
        assert && assert( node.isPhetioInstrumented(),
          'When running phet-io mode, all focusable instances must be instrumented.' );
      }

      var previousFocus;
      if ( this.focusProperty.value ) {
        previousFocus = this.focusedNode;

        // Emit that the old focused node is no longer focused
        previousFocus.focusChangedEmitter.emit( false );
      }

      this.focusProperty.value = value;

      if ( value ) {

        // Emit that the new node is focused
        value.trail.lastNode().focusChangedEmitter.emit( true );
      }
      else {

        // if set to null, make sure that the active element is no longer focused
        if ( previousFocus ) {
          previousFocus.blur();
        }
      }
    },

    /**
     * Get the focus for Display. Null if nothing under a Display has focus.
     * @public
     *
     * @returns {Focus|null}
     */
    get focus() {
      return this.focusProperty.value;
    },

    /**
     * Get the currently focused Node, the leaf-most Node of the focusProperty value's Trail. Null if no
     * Node has focus.
     *
     * @public
     * @returns {Node|null}
     */
    getFocusedNode: function() {
      var focusedNode = null;
      var focus = this.focusProperty.get();
      if ( focus ) {
        focusedNode = focus.trail.lastNode();
      }
      return focusedNode;
    },
    get focusedNode() { return this.getFocusedNode(); }
  } );

  /**
   * Dispose function for Display.
   *
   * TODO: this dispose function is not complete.
   * TODO: Don't require overriding like this. Events prototype and non-standard inheritance forces us right now, but
   * ideally we'll stop using Events for Display and this should just work.
   * @public
   */
  Display.prototype.dispose = function() {
    if ( this._input ) {
      this.detachEvents();
    }
    this._rootNode.removeRootedDisplay( this );

    if ( this._accessible ) {
      this._rootAccessibleInstance.dispose();
    }
  };

  Display.customCursors = {
    'scenery-grab-pointer': [ 'grab', '-moz-grab', '-webkit-grab', 'pointer' ],
    'scenery-grabbing-pointer': [ 'grabbing', '-moz-grabbing', '-webkit-grabbing', 'pointer' ]
  };

  // @public (a11y, read-only, scenery-internal setable) {Property.<Focus|null>} - Display has an axon Property to
  // indicate which component is focused (or null if no scenery Node has focus). By passing the tandem and phetioValueType,
  // PhET-iO is able to interoperate (save, restore, control, observe what is currently focused. See Display.focus
  // for setting the Display's focus. Don't set the value of this Property directly.
  Display.focusProperty = new Property( null, {
    tandem: Tandem.generalTandem.createTandem( 'focusProperty' ),
    phetioDocumentation: 'Stores the current focus for the simulation, null if there is not focus. This is not updated ' +
                         'based on mouse or touch input, only keyboard and other alternative inputs.',
    phetioType: PropertyIO( NullableIO( FocusIO ) ),
    phetioState: false,
    phetioReadOnly: true
  } );

  // @public {Emitter} - Fires when we detect an input event that would be considered a "user gesture" by Chrome, so
  // that we can trigger browser actions that are only allowed as a result.
  // See https://github.com/phetsims/scenery/issues/802 and https://github.com/phetsims/vibe/issues/32 for more
  // information.
  Display.userGestureEmitter = new Emitter();

  // @public (read-only) {KeyStateTracker} - A global object that tracks the state of the keyboard for all Displays. Use this
  // to get information about which keyboard keys are pressed down and for how long.
  Display.keyStateTracker = new KeyStateTracker();

  /**
   * Returns true when NO nodes in the subtree are disposed.
   * @private
   *
   * @param {Node} node
   * @returns {boolean}
   */
  Display.assertSubtreeDisposed = function( node ) {
    assert && assert( !node.isDisposed, 'Disposed nodes should not be included in a scene graph to display.' );

    if ( assert ) {
      for ( var i = 0; i < node.children.length; i++ ) {
        Display.assertSubtreeDisposed( node.children[ i ] );
      }
    }
  };

  return Display;
} );
