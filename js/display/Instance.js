// Copyright 2013-2016, University of Colorado Boulder


/**
 * An instance that is specific to the display (not necessarily a global instance, could be in a Canvas cache, etc),
 * that is needed to tracking instance-specific display information, and signals to the display system when other
 * changes are necessary.
 *
 * Instances generally form a true tree, as opposed to the DAG of nodes. The one exception is for shared Canvas caches,
 * where multiple instances can point to one globally-stored (shared) cache instance.
 *
 * An Instance is pooled, but when constructed will not automatically create children, drawables, etc.
 * syncTree() is responsible for synchronizing the instance itself and its entire subtree.
 *
 * Instances are created as 'stateless' instances, but during syncTree the rendering state (properties to determine
 * how to construct the drawable tree for this instance and its subtree) are set.
 *
 * While Instances are considered 'stateful', they will have listeners added to their Node which records actions taken
 * in-between Display.updateDisplay().
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var arrayRemove = require( 'PHET_CORE/arrayRemove' );
  var ChangeInterval = require( 'SCENERY/display/ChangeInterval' );
  var cleanArray = require( 'PHET_CORE/cleanArray' );
  var Drawable = require( 'SCENERY/display/Drawable' );
  var Events = require( 'AXON/Events' );
  var Fittability = require( 'SCENERY/display/Fittability' );
  var inherit = require( 'PHET_CORE/inherit' );
  var Poolable = require( 'PHET_CORE/Poolable' );
  var RelativeTransform = require( 'SCENERY/display/RelativeTransform' );
  var Renderer = require( 'SCENERY/display/Renderer' );
  var scenery = require( 'SCENERY/scenery' );
  var Util = require( 'SCENERY/util/Util' );

  var globalIdCounter = 1;

  // preferences top to bottom in general
  var defaultPreferredRenderers = Renderer.createOrderBitmask(
    Renderer.bitmaskSVG, Renderer.bitmaskCanvas, Renderer.bitmaskDOM, Renderer.bitmaskWebGL );

  /**
   * @constructor
   * @mixes Poolable
   *
   * See initialize() for documentation
   *
   * @param display
   * @param trail
   * @param isDisplayRoot
   * @param isSharedCanvasCacheRoot
   */
  function Instance( display, trail, isDisplayRoot, isSharedCanvasCacheRoot ) {
    Events.call( this );

    this.active = false;

    this.initialize( display, trail, isDisplayRoot, isSharedCanvasCacheRoot );
  }

  scenery.register( 'Instance', Instance );

  inherit( Events, Instance, {
    /*
     * @param {Display} display - Instances are bound to a single display
     * @param {Trail} trail - The list of ancestors going back up to our root instance (for the display, or for a cache)
     * @param {boolean} isDisplayRoot - Whether our instance is for the root node provided to the Display.
     * @param {boolean} isSharedCanvasCacheRoot - Whether our instance is the root for a shared Canvas cache (which can
     *                                            be used multiple places in the main instance tree)
     */
    initialize: function( display, trail, isDisplayRoot, isSharedCanvasCacheRoot ) {
      assert && assert( !this.active,
        'We should never try to initialize an already active object' );

      // prevent the trail passed in from being mutated after this point (we want a consistent trail)
      trail.setImmutable();

      this.id = this.id || globalIdCounter++;

      this.isWebGLSupported = display._allowWebGL && Util.isWebGLSupported;

      // {RelativeTransform}, provides high-performance access to 'relative' transforms (from our nearest
      // transform root), and allows for listening to when our relative transform changes (called during
      // a phase of Display.updateDisplay()).
      this.relativeTransform = ( this.relativeTransform || new RelativeTransform( this ) );

      // {Fittability}, provides logic for whether our drawables (or common-fit ancestors) will support fitting for
      // FittedBlock subtypes. See https://github.com/phetsims/scenery/issues/406.
      this.fittability = ( this.fittability || new Fittability( this ) );

      // Tracking of visibility {boolean} and associated boolean flags.
      this.visible = true; // global visibility (whether this instance will end up appearing on the display)
      this.relativeVisible = true; // relative visibility (ignores the closest ancestral visibility root and below)
      this.selfVisible = true; // like relative visibility, but is always true if we are a visibility root
      this.visibilityDirty = true; // entire subtree of visibility will need to be updated
      this.childVisibilityDirty = true; // an ancestor needs its visibility updated

      // @public {Object} - Maps another instance's `instance.id` {number} => branch index {number} (first index where
      // the two trails are different). This effectively operates as a cache (since it's more expensive to compute the
      // value than it is to look up the value).
      // It is also "bidirectional", such that if we add instance A's branch index to this map, we will also add the
      // same value to instance A's map (referencing this instance). In order to clean up and prevent leaks, the
      // instance references are provided in this.branchIndexReferences (on both ends), so that when one instance is
      // disposed it can remove the references bidirectionally.
      this.branchIndexMap = {} || this.branchIndexMap;

      // @public {Array.<Instance>} - All instances where we have entries in our map. See docs for branchIndexMap.
      this.branchIndexReferences = cleanArray( this.branchIndexReferences );

      // In the range (-1,0), to help us track insertions and removals of this instance's node to its parent
      // (did we get removed but added back?).
      // If it's -1 at its parent's syncTree, we'll end up removing our reference to it.
      // We use an integer just for sanity checks (if it ever reaches -2 or 1, we've reached an invalid state)
      this.addRemoveCounter = 0;

      // If equal to the current frame ID (it is initialized as such), then it is treated during the change interval
      // waterfall as "completely changed", and an interval for the entire instance is used.
      this.stitchChangeFrame = display._frameId;

      // If equal to the current frame ID, an instance was removed from before or after this instance, so we'll want to
      // add in a proper change interval (related to siblings)
      this.stitchChangeBefore = 0;
      this.stitchChangeAfter = 0;

      // If equal to the current frame ID, child instances were added or removed from this instance.
      this.stitchChangeOnChildren = 0;

      // whether we have been included in our parent's drawables the previous frame
      this.stitchChangeIncluded = false;

      // Node listeners for tracking children. Listeners should be added only when we become stateful
      this.childInsertedListener = this.childInsertedListener || this.onChildInserted.bind( this );
      this.childRemovedListener = this.childRemovedListener || this.onChildRemoved.bind( this );
      this.childrenReorderedListener = this.childrenReorderedListener || this.onChildrenReordered.bind( this );
      this.visibilityListener = this.visibilityListener || this.onVisibilityChange.bind( this );
      this.markRenderStateDirtyListener = this.markRenderStateDirtyListener || this.markRenderStateDirty.bind( this );

      this.cleanInstance( display, trail );

      // We need to add this reference on stateless instances, so that we can find out if it was removed before our
      // syncTree was called.
      this.node.addInstance( this );

      // Outstanding external references. used for shared cache instances, where multiple instances can point to us.
      this.externalReferenceCount = 0;

      this.stateless = true; // {boolean} - Whether we have had our state initialized yet

      // Rendering state constants (will not change over the life of an instance)
      this.isDisplayRoot = isDisplayRoot; // {boolean} - Whether we are the root instance for a Display
      this.isSharedCanvasCacheRoot = isSharedCanvasCacheRoot; // {boolean} - Whether we are the root of a Canvas cache

      // 'Cascading' render state for the instance tree. These are properties that can affect the entire subtree when set
      this.preferredRenderers = 0; // {number} - Packed renderer order bitmask (what our renderer preferences are)
      this.isUnderCanvasCache = isSharedCanvasCacheRoot; // {boolean} - Whether we are beneath a Canvas cache (Canvas required)

      // Render state exports for this instance.
      this.isBackbone = false; // {boolean} - Whether we will have a BackboneDrawable group drawable
      this.isTransformed = false;  // {boolean} - Whether this instance creates a new "root" for the relative trail transforms
      this.isVisibilityApplied = false; // {boolean} - Whether this instance handles visibility with a group drawable
      this.isInstanceCanvasCache = false; // {boolean} - Whether we have a Canvas cache specific to this instance's position
      this.isSharedCanvasCachePlaceholder = false; // {boolean}
      this.isSharedCanvasCacheSelf = isSharedCanvasCacheRoot; // {boolean}
      this.selfRenderer = 0; // {number} Renderer bitmask for the 'self' drawable (if our Node is painted)
      this.groupRenderer = 0; // {number} Renderer bitmask for the 'group' drawable (if applicable)
      this.sharedCacheRenderer = 0; // {number} Renderer bitmask for the cache drawable (if applicable)

      // pruning flags (whether we need to be visited, whether updateRenderingState is required, and whether to visit children)
      this.renderStateDirtyFrame = display._frameId; // {number} - When equal to the current frame it is considered "dirty"
      this.skipPruningFrame = display._frameId; // {number} - When equal to the current frame we can't prune at this instance

      sceneryLog && sceneryLog.Instance && sceneryLog.Instance( 'initialized ' + this.toString() );

      // Whether we have been instantiated. false if we are in a pool waiting to be instantiated.
      this.active = true;

      return this;
    },

    /*
     * Called for initialization of properties (via initialize(), via constructor), and to clean the instance for
     * placement in the pool (don't leak memory).
     *
     * If the parameters are null, we remove all external references so that we don't leak memory.
     *
     * @param {Display|null} display - Instances are bound to a single display
     * @param {Trail|null} trail - The list of ancestors going back up to our root instance (for the display, or for a cache)
     */
    cleanInstance: function( display, trail ) {
      this.display = display;
      this.trail = trail;
      this.node = trail ? trail.lastNode() : null;
      this.parent = null; // will be set as needed
      this.oldParent = null; // set when removed from us, so that we can easily reattach it when necessary
      // NOTE: reliance on correct order after syncTree by at least SVGBlock/SVGGroup
      this.children = cleanArray( this.children ); // Array[Instance].
      this.sharedCacheInstance = null; // reference to a shared cache instance (different than a child)

      // initialize/clean sub-components
      this.relativeTransform.initialize( display, trail );
      this.fittability.initialize( display, trail );

      // {Instance[]} - Child instances are pushed to here when their node is removed from our node.
      // We don't immediately dispose, since it may be added back.
      this.instanceRemovalCheckList = cleanArray( this.instanceRemovalCheckList );

      // {Drawable} - references to our drawables in the drawable tree
      this.selfDrawable = null;
      this.groupDrawable = null; // e.g. backbone or non-shared cache
      this.sharedCacheDrawable = null; // our drawable if we are a shared cache

      // {Drawable} - references into the linked list of drawables (null if nothing is drawable under this)
      this.firstDrawable = null;
      this.lastDrawable = null;

      // {Drawable} - references into the linked list of drawables (excludes any group drawables handling)
      this.firstInnerDrawable = null;
      this.lastInnerDrawable = null;

      // {SVGGroup[]} - List of SVG groups associated with this display instance
      this.svgGroups = cleanArray( this.svgGroups );

      this.cleanSyncTreeResults();
    },

    /*
     * Initializes or clears properties that are all set as pseudo 'return values' of the syncTree() method. It is the
     * responsibility of the caller of syncTree() to afterwards (optionally read these results and) clear the references
     * using this method to prevent memory leaks.
     *
     * TODO: consider a pool of (or a single global) typed return object(s), since setting these values on the instance
     * generally means hitting the heap, and can slow us down.
     */
    cleanSyncTreeResults: function() {
      // Tracking bounding indices / drawables for what has changed, so we don't have to over-stitch things.

      // if (not iff) child's index <= beforeStableIndex, it hasn't been added/removed. relevant to current children.
      this.beforeStableIndex = this.children.length;

      // if (not iff) child's index >= afterStableIndex, it hasn't been added/removed. relevant to current children.
      this.afterStableIndex = -1;

      // NOTE: both of these being null indicates "there are no change intervals", otherwise it assumes it points to
      // a linked-list of change intervals. We use {ChangeInterval}s to hold this information, see ChangeInterval to see
      // the individual properties that are considered part of a change interval.

      // {ChangeInterval}, first change interval (should have nextChangeInterval linked-list to lastChangeInterval)
      this.firstChangeInterval = null;

      // {ChangeInterval}, last change interval
      this.lastChangeInterval = null;

      // {boolean} - render state change flags, all set in updateRenderingState()
      this.incompatibleStateChange = false; // {boolean} - Whether we need to recreate the instance tree
      this.groupChanged = false; // {boolean} - Whether we need to force a rebuild of the group drawable
      this.cascadingStateChange = false; // {boolean} - Whether we had a render state change that requires visiting all children
      this.anyStateChange = false; // {boolean} - Whether there was any change of rendering state with the last updateRenderingState()
    },

    /*
     * Updates the rendering state properties, and returns a {boolean} flag of whether it was successful if we were
     * already stateful.
     *
     * Rendering state properties determine how we construct the drawable tree from our instance tree (e.g. do we
     * create an SVG or Canvas rectangle, where to place CSS transforms, how to handle opacity, etc.)
     *
     * Instances start out as 'stateless' until updateRenderingState() is called the first time.
     *
     * Node changes that can cause a potential state change (using Node event listeners):
     * - hints
     * - opacity
     * - clipArea
     * - _rendererSummary
     * - _rendererBitmask
     *
     * State changes that can cause cascading state changes in descendants:
     * - isUnderCanvasCache
     * - preferredRenderers
     */
    updateRenderingState: function() {
      sceneryLog && sceneryLog.Instance && sceneryLog.Instance( 'updateRenderingState ' + this.toString() +
                                                                ( this.stateless ? ' (stateless)' : '' ) );
      sceneryLog && sceneryLog.Instance && sceneryLog.push();

      sceneryLog && sceneryLog.Instance && sceneryLog.Instance( 'old: ' + this.getStateString() );

      // old state information, so we can compare what was changed
      var wasBackbone = this.isBackbone;
      var wasTransformed = this.isTransformed;
      var wasVisibilityApplied = this.isVisibilityApplied;
      var wasInstanceCanvasCache = this.isInstanceCanvasCache;
      var wasSharedCanvasCacheSelf = this.isSharedCanvasCacheSelf;
      var wasSharedCanvasCachePlaceholder = this.isSharedCanvasCachePlaceholder;
      var wasUnderCanvasCache = this.isUnderCanvasCache;
      var oldSelfRenderer = this.selfRenderer;
      var oldGroupRenderer = this.groupRenderer;
      var oldSharedCacheRenderer = this.sharedCacheRenderer;
      var oldPreferredRenderers = this.preferredRenderers;

      // default values to set (makes the logic much simpler)
      this.isBackbone = false;
      this.isTransformed = false;
      this.isVisibilityApplied = false;
      this.isInstanceCanvasCache = false;
      this.isSharedCanvasCacheSelf = false;
      this.isSharedCanvasCachePlaceholder = false;
      this.selfRenderer = 0;
      this.groupRenderer = 0;
      this.sharedCacheRenderer = 0;

      var hints = this.node._hints;

      this.isUnderCanvasCache = this.isSharedCanvasCacheRoot ||
                                ( this.parent ? ( this.parent.isUnderCanvasCache || this.parent.isInstanceCanvasCache || this.parent.isSharedCanvasCacheSelf ) : false );

      // set up our preferred renderer list (generally based on the parent)
      this.preferredRenderers = this.parent ? this.parent.preferredRenderers : defaultPreferredRenderers;
      // allow the node to modify its preferred renderers (and those of its descendants)
      if ( hints.renderer ) {
        this.preferredRenderers = Renderer.pushOrderBitmask( this.preferredRenderers, hints.renderer );
      }

      var hasClip = this.node.hasClipArea();
      var hasTransparency = this.node.opacity !== 1 || hints.usesOpacity;
      var requiresSplit = hints.requireElement || hints.cssTransform || hints.layerSplit;
      var backboneRequired = this.isDisplayRoot || ( !this.isUnderCanvasCache && requiresSplit );

      // Support either "all Canvas" or "all SVG" opacity/clip
      var applyTransparencyWithBlock = !backboneRequired &&
                                       ( hasTransparency || hasClip ) &&
                                       ( this.node._rendererSummary.isSubtreeRenderedExclusivelySVG( this.preferredRenderers ) ||
                                         this.node._rendererSummary.isSubtreeRenderedExclusivelyCanvas( this.preferredRenderers ) );
      var useBackbone = applyTransparencyWithBlock ? false : ( backboneRequired || hasTransparency || hasClip );

      // check if we need a backbone or cache
      // if we are under a canvas cache, we will NEVER have a backbone
      // splits are accomplished just by having a backbone
      // NOTE: If changing, check RendererSummary.summaryBitmaskForNodeSelf
      //OHTWO TODO: Update this to properly identify when backbones are necessary/and-or when we forward opacity/clipping
      if ( useBackbone ) {
        this.isBackbone = true;
        this.isVisibilityApplied = true;
        this.isTransformed = this.isDisplayRoot || !!hints.cssTransform; // for now, only trigger CSS transform if we have the specific hint
        //OHTWO TODO: check whether the force acceleration hint is being used by our DOMBlock
        this.groupRenderer = Renderer.bitmaskDOM; // probably won't be used
      }
      else if ( !applyTransparencyWithBlock && ( hasTransparency || hasClip || hints.canvasCache ) ) {
        // everything underneath needs to be renderable with Canvas, otherwise we cannot cache
        assert && assert( this.node._rendererSummary.isSingleCanvasSupported(),
          'hints.canvasCache provided, but not all node contents can be rendered with Canvas under ' +
          this.node.constructor.name );

        if ( hints.singleCache ) {
          // TODO: scale options - fixed size, match highest resolution (adaptive), or mipmapped
          if ( this.isSharedCanvasCacheRoot ) {
            this.isSharedCanvasCacheSelf = true;

            this.sharedCacheRenderer = this.isWebGLSupported ? Renderer.bitmaskWebGL : Renderer.bitmaskCanvas;
          }
          else {
            // everything underneath needs to guarantee that its bounds are valid
            //OHTWO TODO: We'll probably remove this if we go with the "safe bounds" approach
            assert && assert( this.node._rendererSummary.areBoundsValid(),
              'hints.singleCache provided, but not all node contents have valid bounds under ' +
              this.node.constructor.name );

            this.isSharedCanvasCachePlaceholder = true;
          }
        }
        else {
          this.isInstanceCanvasCache = true;
          this.isUnderCanvasCache = true;
          this.groupRenderer = this.isWebGLSupported ? Renderer.bitmaskWebGL : Renderer.bitmaskCanvas;
        }
      }

      if ( this.node.isPainted() ) {
        if ( this.isUnderCanvasCache ) {
          this.selfRenderer = Renderer.bitmaskCanvas;
        }
        else {
          var supportedNodeBitmask = this.node._rendererBitmask;
          if ( !this.isWebGLSupported ) {
            var invalidBitmasks = Renderer.bitmaskWebGL;
            supportedNodeBitmask = supportedNodeBitmask ^ ( supportedNodeBitmask & invalidBitmasks );
          }

          // use the preferred rendering order if specified, otherwise use the default
          this.selfRenderer = ( supportedNodeBitmask & Renderer.bitmaskOrder( this.preferredRenderers, 0 ) ) ||
                              ( supportedNodeBitmask & Renderer.bitmaskOrder( this.preferredRenderers, 1 ) ) ||
                              ( supportedNodeBitmask & Renderer.bitmaskOrder( this.preferredRenderers, 2 ) ) ||
                              ( supportedNodeBitmask & Renderer.bitmaskOrder( this.preferredRenderers, 3 ) ) ||
                              ( supportedNodeBitmask & Renderer.bitmaskSVG ) ||
                              ( supportedNodeBitmask & Renderer.bitmaskCanvas ) ||
                              ( supportedNodeBitmask & Renderer.bitmaskDOM ) ||
                              ( supportedNodeBitmask & Renderer.bitmaskWebGL ) ||
                              0;

          assert && assert( this.selfRenderer, 'setSelfRenderer failure?' );
        }
      }

      // whether we need to force rebuilding the group drawable
      this.groupChanged = ( wasBackbone !== this.isBackbone ) ||
                          ( wasInstanceCanvasCache !== this.isInstanceCanvasCache ) ||
                          ( wasSharedCanvasCacheSelf !== this.isSharedCanvasCacheSelf );

      // whether any of our render state changes can change descendant render states
      this.cascadingStateChange = ( wasUnderCanvasCache !== this.isUnderCanvasCache ) ||
                                  ( oldPreferredRenderers !== this.preferredRenderers );

      /*
       * Whether we can just update the state on an Instance when changing from this state => otherState.
       * This is generally not possible if there is a change in whether the instance should be a transform root
       * (e.g. backbone/single-cache), so we will have to recreate the instance and its subtree if that is the case.
       *
       * Only relevant if we were previously stateful, so it can be ignored if this is our first updateRenderingState()
       */
      this.incompatibleStateChange = ( this.isTransformed !== wasTransformed ) ||
                                     ( this.isSharedCanvasCachePlaceholder !== wasSharedCanvasCachePlaceholder );

      // whether there was any render state change
      this.anyStateChange = this.groupChanged || this.cascadingStateChange || this.incompatibleStateChange ||
                            ( oldSelfRenderer !== this.selfRenderer ) ||
                            ( oldGroupRenderer !== this.groupRenderer ) ||
                            ( oldSharedCacheRenderer !== this.sharedCacheRenderer );

      // if our visibility applications changed, update the entire subtree
      if ( wasVisibilityApplied !== this.isVisibilityApplied ) {
        this.visibilityDirty = true;
        this.parent && this.parent.markChildVisibilityDirty();
      }

      // If our fittability has changed, propagate those changes. (It's generally a hint change which will trigger an
      // update of rendering state).
      this.fittability.checkSelfFittability();

      sceneryLog && sceneryLog.Instance && sceneryLog.Instance( 'new: ' + this.getStateString() );
      sceneryLog && sceneryLog.Instance && sceneryLog.pop();
    },

    /*
     * @returns A short string that contains a summary of the rendering state, for debugging/logging purposes.
     */
    getStateString: function() {
      var result = 'S[ ' +
                   ( this.isDisplayRoot ? 'displayRoot ' : '' ) +
                   ( this.isBackbone ? 'backbone ' : '' ) +
                   ( this.isInstanceCanvasCache ? 'instanceCache ' : '' ) +
                   ( this.isSharedCanvasCachePlaceholder ? 'sharedCachePlaceholder ' : '' ) +
                   ( this.isSharedCanvasCacheSelf ? 'sharedCacheSelf ' : '' ) +
                   ( this.isTransformed ? 'TR ' : '' ) +
                   ( this.isVisibilityApplied ? 'VIS ' : '' ) +
                   ( this.selfRenderer ? this.selfRenderer.toString( 16 ) : '-' ) + ',' +
                   ( this.groupRenderer ? this.groupRenderer.toString( 16 ) : '-' ) + ',' +
                   ( this.sharedCacheRenderer ? this.sharedCacheRenderer.toString( 16 ) : '-' ) + ' ';
      return result + ']';
    },

    /*
     * The main entry point for syncTree(), called on the root instance. See syncTree() for more information.
     */
    baseSyncTree: function() {
      assert && assert( this.isDisplayRoot, 'baseSyncTree() should only be called on the root instance' );

      sceneryLog && sceneryLog.Instance && sceneryLog.Instance( '-------- START baseSyncTree ' + this.toString() + ' --------' );
      this.syncTree();
      sceneryLog && sceneryLog.Instance && sceneryLog.Instance( '-------- END baseSyncTree ' + this.toString() + ' --------' );
      this.cleanSyncTreeResults();
    },

    /*
     * Updates the rendering state, synchronizes the instance sub-tree (so that our instance tree matches
     * the Node tree the client provided), and back-propagates {ChangeInterval} information for stitching backbones
     * and/or caches.
     *
     * syncTree() also sets a number of pseudo 'return values' (documented in cleanSyncTreeResults()). After calling
     * syncTree() and optionally reading those results, cleanSyncTreeResults() should be called on the same instance
     * in order to prevent memory leaks.
     *
     * @returns {boolean} - Whether the sync was possible. If it wasn't, a new instance subtree will need to be created.
     */
    syncTree: function() {
      sceneryLog && sceneryLog.Instance && sceneryLog.Instance( 'syncTree ' + this.toString() + ' ' + this.getStateString() +
                                                                ( this.stateless ? ' (stateless)' : '' ) );
      sceneryLog && sceneryLog.Instance && sceneryLog.push();

      if ( sceneryLog && scenery.isLoggingPerformance() ) {
        this.display.perfSyncTreeCount++;
      }

      // may access isTransformed up to root to determine relative trails
      assert && assert( !this.parent || !this.parent.stateless, 'We should not have a stateless parent instance' );

      var wasStateless = this.stateless;
      if ( wasStateless ||
           ( this.parent && this.parent.cascadingStateChange ) || // if our parent had cascading state changes, we need to recompute
           ( this.renderStateDirtyFrame === this.display._frameId ) ) { // if our render state is dirty
        this.updateRenderingState();
      }
      else {
        // we can check whether updating state would have made any changes when we skip it (for slow assertions)
        if ( assertSlow ) {
          this.updateRenderingState();
          assertSlow( !this.anyStateChange );
        }
      }

      if ( !wasStateless && this.incompatibleStateChange ) {
        sceneryLog && sceneryLog.Instance && sceneryLog.Instance( 'incompatible instance ' + this.toString() + ' ' + this.getStateString() + ', aborting' );
        sceneryLog && sceneryLog.Instance && sceneryLog.pop();

        // The false return will signal that a new instance needs to be used. our tree will be disposed soon.
        return false;
      }
      this.stateless = false;

      // no need to overwrite, should always be the same
      assert && assert( !wasStateless || this.children.length === 0,
        'We should not have child instances on an instance without state' );

      if ( wasStateless ) {
        // If we are a transform root, notify the display that we are dirty. We'll be validated when it's at that phase
        // at the next updateDisplay().
        if ( this.isTransformed ) {
          this.display.markTransformRootDirty( this, true );
        }

        this.attachNodeListeners();
      }

      // TODO: pruning of shared caches
      if ( this.isSharedCanvasCachePlaceholder ) {
        this.sharedSyncTree();
      }
      // pruning so that if no changes would affect a subtree it is skipped
      else if ( wasStateless || this.skipPruningFrame === this.display._frameId || this.anyStateChange ) {

        // mark fully-removed instances for disposal, and initialize child instances if we were stateless
        this.prepareChildInstances( wasStateless );

        var oldFirstDrawable = this.firstDrawable;
        var oldLastDrawable = this.lastDrawable;
        var oldFirstInnerDrawable = this.firstInnerDrawable;
        var oldLastInnerDrawable = this.lastInnerDrawable;

        var selfChanged = this.updateSelfDrawable();

        // Synchronizes our children and self, with the drawables and change intervals of both combined
        this.localSyncTree( selfChanged );

        if ( assertSlow ) {
          // before and after first/last drawables (inside any potential group drawable)
          this.auditChangeIntervals( oldFirstInnerDrawable, oldLastInnerDrawable, this.firstInnerDrawable, this.lastInnerDrawable );
        }

        // If we use a group drawable (backbone, etc.), we'll collapse our drawables and change intervals to reference
        // the group drawable (as applicable).
        this.groupSyncTree( wasStateless );

        if ( assertSlow ) {
          // before and after first/last drawables (outside of any potential group drawable)
          this.auditChangeIntervals( oldFirstDrawable, oldLastDrawable, this.firstDrawable, this.lastDrawable );
        }
      }
      else {
        // our sub-tree was not visited, since there were no relevant changes to it (that need instance synchronization
        // or drawable changes)
        sceneryLog && sceneryLog.Instance && sceneryLog.Instance( 'pruned' );
      }

      sceneryLog && sceneryLog.Instance && sceneryLog.pop();

      return true;
    },

    /*
     * Responsible for syncing children, connecting the drawable linked list as needed, and outputting change intervals
     * and first/last drawable information.
     */
    localSyncTree: function( selfChanged ) {
      var frameId = this.display._frameId;

      // local variables, since we can't overwrite our instance properties yet
      var firstDrawable = this.selfDrawable; // possibly null
      var currentDrawable = firstDrawable; // possibly null

      assert && assert( this.firstChangeInterval === null && this.lastChangeInterval === null,
        'sanity checks that cleanSyncTreeResults were called' );

      var firstChangeInterval = null;
      if ( selfChanged ) {
        sceneryLog && sceneryLog.ChangeInterval && sceneryLog.ChangeInterval( 'self' );
        sceneryLog && sceneryLog.ChangeInterval && sceneryLog.push();
        firstChangeInterval = ChangeInterval.newForDisplay( null, null, this.display );
        sceneryLog && sceneryLog.ChangeInterval && sceneryLog.pop();
      }
      var currentChangeInterval = firstChangeInterval;
      var lastUnchangedDrawable = selfChanged ? null : this.selfDrawable; // possibly null

      for ( var i = 0; i < this.children.length; i++ ) {
        var childInstance = this.children[ i ];

        var isCompatible = childInstance.syncTree();
        if ( !isCompatible ) {
          childInstance = this.updateIncompatibleChildInstance( childInstance, i );
          childInstance.syncTree();
        }

        var includeChildDrawables = childInstance.shouldIncludeInParentDrawables();

        //OHTWO TODO: only strip out invisible Canvas drawables, while leaving SVG (since we can more efficiently hide
        // SVG trees, memory-wise)
        // here we strip out invisible drawable sections out of the drawable linked list
        if ( includeChildDrawables ) {
          // if there are any drawables for that child, link them up in our linked list
          if ( childInstance.firstDrawable ) {
            if ( currentDrawable ) {
              // there is already an end of the linked list, so just append to it
              Drawable.connectDrawables( currentDrawable, childInstance.firstDrawable, this.display );
            }
            else {
              // start out the linked list
              firstDrawable = childInstance.firstDrawable;
            }
            // update the last drawable of the linked list
            currentDrawable = childInstance.lastDrawable;
          }
        }

        /*---------------------------------------------------------------------------*
         * Change intervals
         *----------------------------------------------------------------------------*/

        sceneryLog && sceneryLog.ChangeInterval && sceneryLog.ChangeInterval( 'changes for ' + childInstance.toString() +
                                                                              ' in ' + this.toString() );
        sceneryLog && sceneryLog.ChangeInterval && sceneryLog.push();

        var wasIncluded = childInstance.stitchChangeIncluded;
        var isIncluded = includeChildDrawables;
        childInstance.stitchChangeIncluded = isIncluded;

        sceneryLog && sceneryLog.ChangeInterval && sceneryLog.ChangeInterval( 'included: ' + wasIncluded + ' => ' + isIncluded );

        // check for forcing full change-interval on child
        if ( childInstance.stitchChangeFrame === frameId ) {
          sceneryLog && sceneryLog.ChangeInterval && sceneryLog.ChangeInterval( 'stitchChangeFrame full change interval' );
          sceneryLog && sceneryLog.ChangeInterval && sceneryLog.push();

          // e.g. it was added, moved, or had visibility changes. requires full change interval
          childInstance.firstChangeInterval = childInstance.lastChangeInterval = ChangeInterval.newForDisplay( null, null, this.display );

          sceneryLog && sceneryLog.ChangeInterval && sceneryLog.pop();
        }
        else {
          assert && assert( wasIncluded === isIncluded,
            'If we do not have stitchChangeFrame activated, our inclusion should not have changed' );
        }

        var firstChildChangeInterval = childInstance.firstChangeInterval;
        var isBeforeOpen = currentChangeInterval && currentChangeInterval.drawableAfter === null;
        var isAfterOpen = firstChildChangeInterval && firstChildChangeInterval.drawableBefore === null;
        var needsBridge = childInstance.stitchChangeBefore === frameId && !isBeforeOpen && !isAfterOpen;

        // We need to insert an additional change interval (bridge) when we notice a link in the drawable linked list
        // where there were nodes that needed stitch changes that aren't still children, or were moved. We create a
        // "bridge" change interval to span the gap where nodes were removed.
        if ( needsBridge ) {
          sceneryLog && sceneryLog.ChangeInterval && sceneryLog.ChangeInterval( 'bridge' );
          sceneryLog && sceneryLog.ChangeInterval && sceneryLog.push();

          var bridge = ChangeInterval.newForDisplay( lastUnchangedDrawable, null, this.display );
          if ( currentChangeInterval ) {
            currentChangeInterval.nextChangeInterval = bridge;
          }
          currentChangeInterval = bridge;
          firstChangeInterval = firstChangeInterval || currentChangeInterval; // store if it is the first
          isBeforeOpen = true;

          sceneryLog && sceneryLog.ChangeInterval && sceneryLog.pop();
        }

        // Exclude child instances that are now (and were before) not included. NOTE: We still need to include those in
        // bridge calculations, since a removed (before-included) instance could be between two still-invisible
        // instances.
        if ( wasIncluded || isIncluded ) {
          if ( isBeforeOpen ) {
            // we want to try to glue our last ChangeInterval up
            if ( firstChildChangeInterval ) {
              if ( firstChildChangeInterval.drawableBefore === null ) {
                // we want to glue from both sides

                // basically have our current change interval replace the child's first change interval
                currentChangeInterval.drawableAfter = firstChildChangeInterval.drawableAfter;
                currentChangeInterval.nextChangeInterval = firstChildChangeInterval.nextChangeInterval;

                currentChangeInterval = childInstance.lastChangeInterval === firstChildChangeInterval ?
                                        currentChangeInterval : // since we are replacing, don't give an origin reference
                                        childInstance.lastChangeInterval;
              }
              else {
                // only a desire to glue from before
                currentChangeInterval.drawableAfter = childInstance.firstDrawable; // either null or the correct drawable
                currentChangeInterval.nextChangeInterval = firstChildChangeInterval;
                currentChangeInterval = childInstance.lastChangeInterval;
              }
            }
            else {
              // no changes to the child. grabs the first drawable reference it can
              currentChangeInterval.drawableAfter = childInstance.firstDrawable; // either null or the correct drawable
            }
          }
          else if ( firstChildChangeInterval ) {
            firstChangeInterval = firstChangeInterval || firstChildChangeInterval; // store if it is the first
            if ( firstChildChangeInterval.drawableBefore === null ) {
              assert && assert( !currentChangeInterval || lastUnchangedDrawable,
                'If we have a current change interval, we should be guaranteed a non-null ' +
                'lastUnchangedDrawable' );
              firstChildChangeInterval.drawableBefore = lastUnchangedDrawable; // either null or the correct drawable
            }
            if ( currentChangeInterval ) {
              currentChangeInterval.nextChangeInterval = firstChildChangeInterval;
            }
            currentChangeInterval = childInstance.lastChangeInterval;
          }
          lastUnchangedDrawable = ( currentChangeInterval && currentChangeInterval.drawableAfter === null ) ?
                                  null :
                                  ( childInstance.lastDrawable ?
                                    childInstance.lastDrawable :
                                    lastUnchangedDrawable );
        }

        // if the last instance, check for post-bridge
        if ( i === this.children.length - 1 ) {
          if ( childInstance.stitchChangeAfter === frameId && !( currentChangeInterval && currentChangeInterval.drawableAfter === null ) ) {
            var endingBridge = ChangeInterval.newForDisplay( lastUnchangedDrawable, null, this.display );
            if ( currentChangeInterval ) {
              currentChangeInterval.nextChangeInterval = endingBridge;
            }
            currentChangeInterval = endingBridge;
            firstChangeInterval = firstChangeInterval || currentChangeInterval; // store if it is the first
          }
        }

        // clean up the metadata on our child (can't be done in the child call, since we use these values like a
        // composite return value)
        //OHTWO TODO: only do this on instances that were actually traversed
        childInstance.cleanSyncTreeResults();

        sceneryLog && sceneryLog.ChangeInterval && sceneryLog.pop();
      }

      // it's really the easiest way to compare if two things (casted to booleans) are the same?
      assert && assert( !!firstChangeInterval === !!currentChangeInterval,
        'Presence of first and current change intervals should be equal' );

      // Check to see if we are emptied and marked as changed (but without change intervals). This should imply we have
      // no children (and thus no stitchChangeBefore / stitchChangeAfter to use), so we'll want to create a change
      // interval to cover our entire range.
      if ( !firstChangeInterval && this.stitchChangeOnChildren === this.display._frameId && this.children.length === 0 ) {
        firstChangeInterval = currentChangeInterval = ChangeInterval.newForDisplay( null, null, this.display );
      }

      // store our results
      // NOTE: these may get overwritten with the group change intervals (in that case, groupSyncTree will read from these)
      this.firstChangeInterval = firstChangeInterval;
      this.lastChangeInterval = currentChangeInterval;

      // NOTE: these may get overwritten with the group drawable (in that case, groupSyncTree will read from these)
      this.firstDrawable = this.firstInnerDrawable = firstDrawable;
      this.lastDrawable = this.lastInnerDrawable = currentDrawable; // either null, or the drawable itself

      // ensure that our firstDrawable and lastDrawable are correct
      if ( assertSlow ) {
        var firstDrawableCheck = null;
        for ( var j = 0; j < this.children.length; j++ ) {
          if ( this.children[ j ].shouldIncludeInParentDrawables() && this.children[ j ].firstDrawable ) {
            firstDrawableCheck = this.children[ j ].firstDrawable;
            break;
          }
        }
        if ( this.selfDrawable ) {
          firstDrawableCheck = this.selfDrawable;
        }

        var lastDrawableCheck = this.selfDrawable;
        for ( var k = this.children.length - 1; k >= 0; k-- ) {
          if ( this.children[ k ].shouldIncludeInParentDrawables() && this.children[ k ].lastDrawable ) {
            lastDrawableCheck = this.children[ k ].lastDrawable;
            break;
          }
        }

        assertSlow( firstDrawableCheck === this.firstDrawable );
        assertSlow( lastDrawableCheck === this.lastDrawable );
      }
    },

    /*
     * If necessary, create/replace/remove our selfDrawable.
     *
     * @returns whether the selfDrawable changed
     */
    updateSelfDrawable: function() {
      if ( this.node.isPainted() ) {
        var selfRenderer = this.selfRenderer; // our new self renderer bitmask

        // bitwise trick, since only one of Canvas/SVG/DOM/WebGL/etc. flags will be chosen, and bitmaskRendererArea is
        // the mask for those flags. In English, "Is the current selfDrawable compatible with our selfRenderer (if any),
        // or do we need to create a selfDrawable?"
        //OHTWO TODO: For Canvas, we won't care about anything else for the drawable, but for DOM we care about the
        // force-acceleration flag! That's stripped out here.
        if ( !this.selfDrawable || ( ( this.selfDrawable.renderer & selfRenderer & Renderer.bitmaskRendererArea ) === 0 ) ) {
          if ( this.selfDrawable ) {
            sceneryLog && sceneryLog.Instance && sceneryLog.Instance( 'replacing old drawable ' + this.selfDrawable.toString() + ' with new renderer' );

            // scrap the previous selfDrawable, we need to create one with a different renderer.
            this.selfDrawable.markForDisposal( this.display );
          }

          this.selfDrawable = Renderer.createSelfDrawable( this, this.node, selfRenderer, this.fittability.ancestorsFittable );
          assert && assert( this.selfDrawable );

          return true;
        }
      }
      else {
        assert && assert( this.selfDrawable === null, 'Non-painted nodes should not have a selfDrawable' );
      }

      return false;
    },

    // returns the up-to-date instance
    updateIncompatibleChildInstance: function( childInstance, index ) {
      if ( sceneryLog && scenery.isLoggingPerformance() ) {
        var affectedInstanceCount = childInstance.getDescendantCount() + 1; // +1 for itself

        if ( affectedInstanceCount > 100 ) {
          sceneryLog.PerfCritical && sceneryLog.PerfCritical( 'incompatible instance rebuild at ' + this.trail.toPathString() + ': ' + affectedInstanceCount );
        }
        else if ( affectedInstanceCount > 40 ) {
          sceneryLog.PerfMajor && sceneryLog.PerfMajor( 'incompatible instance rebuild at ' + this.trail.toPathString() + ': ' + affectedInstanceCount );
        }
        else if ( affectedInstanceCount > 0 ) {
          sceneryLog.PerfMinor && sceneryLog.PerfMinor( 'incompatible instance rebuild at ' + this.trail.toPathString() + ': ' + affectedInstanceCount );
        }
      }

      // mark it for disposal
      this.display.markInstanceRootForDisposal( childInstance );

      // swap in a new instance
      var replacementInstance = Instance.createFromPool( this.display, this.trail.copy().addDescendant( childInstance.node, index ), false, false );
      this.replaceInstanceWithIndex( childInstance, replacementInstance, index );
      return replacementInstance;
    },

    groupSyncTree: function( wasStateless ) {
      var groupRenderer = this.groupRenderer;
      assert && assert( ( this.isBackbone ? 1 : 0 ) +
                        ( this.isInstanceCanvasCache ? 1 : 0 ) +
                        ( this.isSharedCanvasCacheSelf ? 1 : 0 ) === ( groupRenderer ? 1 : 0 ),
        'We should have precisely one of these flags set for us to have a groupRenderer' );

      // if we switched to/away from a group, our group type changed, or our group renderer changed
      var groupChanged = ( !!groupRenderer !== !!this.groupDrawable ) ||
                         ( !wasStateless && this.groupChanged ) ||
                         ( this.groupDrawable && this.groupDrawable.renderer !== groupRenderer );

      // if there is a change, prepare
      if ( groupChanged ) {
        if ( this.groupDrawable ) {
          sceneryLog && sceneryLog.Instance && sceneryLog.Instance( 'replacing group drawable ' + this.groupDrawable.toString() );

          this.groupDrawable.markForDisposal( this.display );
          this.groupDrawable = null;
        }

        // change everything, since we may need a full restitch
        this.firstChangeInterval = this.lastChangeInterval = ChangeInterval.newForDisplay( null, null, this.display );
      }

      if ( groupRenderer ) {
        // ensure our linked list is fully disconnected from others
        this.firstDrawable && Drawable.disconnectBefore( this.firstDrawable, this.display );
        this.lastDrawable && Drawable.disconnectAfter( this.lastDrawable, this.display );

        if ( this.isBackbone ) {
          if ( groupChanged ) {
            this.groupDrawable = scenery.BackboneDrawable.createFromPool( this.display, this, this.getTransformRootInstance(), groupRenderer, this.isDisplayRoot );

            if ( this.isTransformed ) {
              this.display.markTransformRootDirty( this, true );
            }
          }

          if ( this.firstChangeInterval ) {
            this.groupDrawable.stitch( this.firstDrawable, this.lastDrawable, this.firstChangeInterval, this.lastChangeInterval );
          }
        }
        else if ( this.isInstanceCanvasCache ) {
          if ( groupChanged ) {
            this.groupDrawable = scenery.InlineCanvasCacheDrawable.createFromPool( groupRenderer, this );
          }
          if ( this.firstChangeInterval ) {
            this.groupDrawable.stitch( this.firstDrawable, this.lastDrawable, this.firstChangeInterval, this.lastChangeInterval );
          }
        }
        else if ( this.isSharedCanvasCacheSelf ) {
          if ( groupChanged ) {
            this.groupDrawable = scenery.CanvasBlock.createFromPool( groupRenderer, this );
          }
          //OHTWO TODO: restitch here??? implement it
        }
        // Update the fittable flag
        this.groupDrawable.setFittable( this.fittability.ancestorsFittable );

        this.firstDrawable = this.lastDrawable = this.groupDrawable;
      }

      // change interval handling
      if ( groupChanged ) {
        // if our group status changed, mark EVERYTHING as potentially changed
        this.firstChangeInterval = this.lastChangeInterval = ChangeInterval.newForDisplay( null, null, this.display );
      }
      else if ( groupRenderer ) {
        // our group didn't have to change at all, so we prevent any change intervals
        this.firstChangeInterval = this.lastChangeInterval = null;
      }
    },

    sharedSyncTree: function() {
      //OHTWO TODO: we are probably missing syncTree for shared trees properly with pruning. investigate!!

      this.ensureSharedCacheInitialized();

      var sharedCacheRenderer = this.sharedCacheRenderer;

      if ( !this.sharedCacheDrawable || this.sharedCacheDrawable.renderer !== sharedCacheRenderer ) {
        //OHTWO TODO: mark everything as changed (big change interval)

        if ( this.sharedCacheDrawable ) {
          sceneryLog && sceneryLog.Instance && sceneryLog.Instance( 'replacing shared cache drawable ' + this.sharedCacheDrawable.toString() );

          this.sharedCacheDrawable.markForDisposal( this.display );
        }

        //OHTWO TODO: actually create the proper shared cache drawable depending on the specified renderer
        // (update it if necessary)
        this.sharedCacheDrawable = new scenery.SharedCanvasCacheDrawable( this.trail, sharedCacheRenderer, this, this.sharedCacheInstance );
        this.firstDrawable = this.sharedCacheDrawable;
        this.lastDrawable = this.sharedCacheDrawable;

        // basically everything changed now, and won't from now on
        this.firstChangeInterval = this.lastChangeInterval = ChangeInterval.newForDisplay( null, null, this.display );
      }
    },

    prepareChildInstances: function( wasStateless ) {
      // mark all removed instances to be disposed (along with their subtrees)
      while ( this.instanceRemovalCheckList.length ) {
        var instanceToMark = this.instanceRemovalCheckList.pop();
        if ( instanceToMark.addRemoveCounter === -1 ) {
          instanceToMark.addRemoveCounter = 0; // reset it, so we don't mark it for disposal more than once
          this.display.markInstanceRootForDisposal( instanceToMark );
        }
      }

      if ( wasStateless ) {
        // we need to create all of the child instances
        for ( var k = 0; k < this.node.children.length; k++ ) {
          // create a child instance
          var child = this.node.children[ k ];
          this.appendInstance( Instance.createFromPool( this.display, this.trail.copy().addDescendant( child, k ), false, false ) );
        }
      }
    },

    ensureSharedCacheInitialized: function() {
      // we only need to initialize this shared cache reference once
      if ( !this.sharedCacheInstance ) {
        var instanceKey = this.node.getId();
        // TODO: have this abstracted away in the Display?
        this.sharedCacheInstance = this.display._sharedCanvasInstances[ instanceKey ];

        // TODO: increment reference counting?
        if ( !this.sharedCacheInstance ) {
          this.sharedCacheInstance = Instance.createFromPool( this.display, new scenery.Trail( this.node ), false, true );
          this.sharedCacheInstance.syncTree();
          this.display._sharedCanvasInstances[ instanceKey ] = this.sharedCacheInstance;
          // TODO: reference counting?

          // TODO: this.sharedCacheInstance.isTransformed?

          //OHTWO TODO: is this necessary?
          this.display.markTransformRootDirty( this.sharedCacheInstance, true );
        }

        this.sharedCacheInstance.externalReferenceCount++;

        //OHTWO TODO: is this necessary?
        if ( this.isTransformed ) {
          this.display.markTransformRootDirty( this, true );
        }
      }
    },

    // @private, whether out drawables (from firstDrawable to lastDrawable) should be included in our parent's drawables
    shouldIncludeInParentDrawables: function() {
      return this.node.isVisible() || !this.node.isExcludeInvisible();
    },

    // @private, finds the closest drawable (not including the child instance at childIndex) using lastDrawable, or null
    findPreviousDrawable: function( childIndex ) {
      for ( var i = childIndex - 1; i >= 0; i-- ) {
        var option = this.children[ i ].lastDrawable;
        if ( option !== null ) {
          return option;
        }
      }

      return null;
    },

    // @private, finds the closest drawable (not including the child instance at childIndex) using nextDrawable, or null
    findNextDrawable: function( childIndex ) {
      var len = this.children.length;
      for ( var i = childIndex + 1; i < len; i++ ) {
        var option = this.children[ i ].firstDrawable;
        if ( option !== null ) {
          return option;
        }
      }

      return null;
    },

    /*---------------------------------------------------------------------------*
     * Children handling
     *----------------------------------------------------------------------------*/

    appendInstance: function( instance ) {
      this.insertInstance( instance, this.children.length );
    },

    // NOTE: different parameter order compared to Node
    insertInstance: function( instance, index ) {
      assert && assert( instance instanceof Instance );
      assert && assert( index >= 0 && index <= this.children.length,
        'Instance insertion bounds check for index ' + index + ' with previous children length ' +
        this.children.length );

      sceneryLog && sceneryLog.InstanceTree && sceneryLog.InstanceTree(
        'inserting ' + instance.toString() + ' into ' + this.toString() );
      sceneryLog && sceneryLog.InstanceTree && sceneryLog.push();

      // mark it as changed during this frame, so that we can properly set the change interval
      instance.stitchChangeFrame = this.display._frameId;
      this.stitchChangeOnChildren = this.display._frameId;

      this.children.splice( index, 0, instance );
      instance.parent = this;
      instance.oldParent = this;

      // maintain our stitch-change interval
      if ( index <= this.beforeStableIndex ) {
        this.beforeStableIndex = index - 1;
      }
      if ( index > this.afterStableIndex ) {
        this.afterStableIndex = index + 1;
      }
      else {
        this.afterStableIndex++;
      }

      // maintain fittable flags
      this.fittability.onInsert( instance.fittability );

      this.relativeTransform.addInstance( instance );

      this.markChildVisibilityDirty();

      sceneryLog && sceneryLog.InstanceTree && sceneryLog.pop();
    },

    removeInstance: function( instance ) {
      return this.removeInstanceWithIndex( instance, _.indexOf( this.children, instance ) );
    },

    removeInstanceWithIndex: function( instance, index ) {
      assert && assert( instance instanceof Instance );
      assert && assert( index >= 0 && index < this.children.length,
        'Instance removal bounds check for index ' + index + ' with previous children length ' +
        this.children.length );

      sceneryLog && sceneryLog.InstanceTree && sceneryLog.InstanceTree(
        'removing ' + instance.toString() + ' from ' + this.toString() );
      sceneryLog && sceneryLog.InstanceTree && sceneryLog.push();

      var frameId = this.display._frameId;

      // mark it as changed during this frame, so that we can properly set the change interval
      instance.stitchChangeFrame = frameId;
      this.stitchChangeOnChildren = frameId;

      // mark neighbors so that we can add a change interval for our removal area
      if ( index - 1 >= 0 ) {
        this.children[ index - 1 ].stitchChangeAfter = frameId;
      }
      if ( index + 1 < this.children.length ) {
        this.children[ index + 1 ].stitchChangeBefore = frameId;
      }

      this.children.splice( index, 1 ); // TODO: replace with a 'remove' function call
      instance.parent = null;
      instance.oldParent = this;

      // maintain our stitch-change interval
      if ( index <= this.beforeStableIndex ) {
        this.beforeStableIndex = index - 1;
      }
      if ( index >= this.afterStableIndex ) {
        this.afterStableIndex = index;
      }
      else {
        this.afterStableIndex--;
      }

      // maintain fittable flags
      this.fittability.onRemove( instance.fittability );

      this.relativeTransform.removeInstance( instance );

      sceneryLog && sceneryLog.InstanceTree && sceneryLog.pop();
    },

    replaceInstanceWithIndex: function( childInstance, replacementInstance, index ) {
      // TODO: optimization? hopefully it won't happen often, so we just do this for now
      this.removeInstanceWithIndex( childInstance, index );
      this.insertInstance( replacementInstance, index );
    },

    /**
     * For handling potential reordering of child instances inclusively between the min and max indices.
     * @private
     *
     * @param {number} minChangeIndex
     * @param {number} maxChangeIndex
     */
    reorderInstances: function( minChangeIndex, maxChangeIndex ) {
      assert && assert( typeof minChangeIndex === 'number' );
      assert && assert( typeof maxChangeIndex === 'number' );
      assert && assert( minChangeIndex <= maxChangeIndex );

      sceneryLog && sceneryLog.InstanceTree && sceneryLog.InstanceTree( 'Reordering ' + this.toString() );
      sceneryLog && sceneryLog.InstanceTree && sceneryLog.push();

      // NOTE: For implementation, we've basically set parameters as if we removed all of the relevant instances and
      // then added them back in. There may be more efficient ways to do this, but the stitching and change interval
      // process is a bit complicated right now.

      var frameId = this.display._frameId;

      // Remove the old ordering of instances
      this.children.splice( minChangeIndex, maxChangeIndex - minChangeIndex + 1 );

      // Add the instances back in the correct order
      for ( var i = minChangeIndex; i <= maxChangeIndex; i++ ) {
        var child = this.findChildInstanceOnNode( this.node._children[ i ] );
        this.children.splice( i, 0, child );
        child.stitchChangeFrame = frameId;

        // mark neighbors so that we can add a change interval for our change area
        if ( i > minChangeIndex ) {
          child.stitchChangeAfter = frameId;
        }
        if ( i < maxChangeIndex ) {
          child.stitchChangeBefore = frameId;
        }
      }

      this.stitchChangeOnChildren = frameId;
      this.beforeStableIndex = Math.min( this.beforeStableIndex, minChangeIndex - 1 );
      this.afterStableIndex = Math.max( this.afterStableIndex, maxChangeIndex + 1 );

      sceneryLog && sceneryLog.InstanceTree && sceneryLog.pop();
    },

    // if we have a child instance that corresponds to this node, return it (otherwise null)
    findChildInstanceOnNode: function( node ) {
      var instances = node.getInstances();
      for ( var i = 0; i < instances.length; i++ ) {
        if ( instances[ i ].oldParent === this ) {
          return instances[ i ];
        }
      }
      return null;
    },

    // event callback for Node's 'childInserted' event, used to track children
    onChildInserted: function( childNode, index ) {
      sceneryLog && sceneryLog.Instance && sceneryLog.Instance(
        'inserting child node ' + childNode.constructor.name + '#' + childNode.id + ' into ' + this.toString() );
      sceneryLog && sceneryLog.Instance && sceneryLog.push();

      assert && assert( !this.stateless, 'If we are stateless, we should not receive these notifications' );

      var instance = this.findChildInstanceOnNode( childNode );

      if ( instance ) {
        sceneryLog && sceneryLog.Instance && sceneryLog.Instance( 'instance already exists' );
        // it must have been added back. increment its counter
        instance.addRemoveCounter += 1;
        assert && assert( instance.addRemoveCounter === 0 );
      }
      else {
        sceneryLog && sceneryLog.Instance && sceneryLog.Instance( 'creating stub instance' );
        sceneryLog && sceneryLog.Instance && sceneryLog.push();
        instance = Instance.createFromPool( this.display, this.trail.copy().addDescendant( childNode, index ), false, false );
        sceneryLog && sceneryLog.Instance && sceneryLog.pop();
      }

      this.insertInstance( instance, index );

      // make sure we are visited for syncTree()
      this.markSkipPruning();

      sceneryLog && sceneryLog.Instance && sceneryLog.pop();
    },

    // event callback for Node's 'childRemoved' event, used to track children
    onChildRemoved: function( childNode, index ) {
      sceneryLog && sceneryLog.Instance && sceneryLog.Instance(
        'removing child node ' + childNode.constructor.name + '#' + childNode.id + ' from ' + this.toString() );
      sceneryLog && sceneryLog.Instance && sceneryLog.push();

      assert && assert( !this.stateless, 'If we are stateless, we should not receive these notifications' );
      assert && assert( this.children[ index ].node === childNode, 'Ensure that our instance matches up' );

      var instance = this.findChildInstanceOnNode( childNode );
      assert && assert( instance !== null, 'We should always have a reference to a removed instance' );

      instance.addRemoveCounter -= 1;
      assert && assert( instance.addRemoveCounter === -1 );

      // track the removed instance here. if it doesn't get added back, this will be the only reference we have (we'll
      // need to dispose it)
      this.instanceRemovalCheckList.push( instance );

      this.removeInstanceWithIndex( instance, index );

      // make sure we are visited for syncTree()
      this.markSkipPruning();

      sceneryLog && sceneryLog.Instance && sceneryLog.pop();
    },

    // event callback for Node's 'childrenReordered' event
    onChildrenReordered: function( minChangeIndex, maxChangeIndex ) {
      sceneryLog && sceneryLog.Instance && sceneryLog.Instance(
        'reordering children for ' + this.toString() );
      sceneryLog && sceneryLog.Instance && sceneryLog.push();

      this.reorderInstances( minChangeIndex, maxChangeIndex );

      // make sure we are visited for syncTree()
      this.markSkipPruning();

      sceneryLog && sceneryLog.Instance && sceneryLog.pop();
    },

    // event callback for Node's 'visibility' event, used to notify about stitch changes
    onVisibilityChange: function() {
      assert && assert( !this.stateless, 'If we are stateless, we should not receive these notifications' );

      // for now, just mark which frame we were changed for our change interval
      this.stitchChangeFrame = this.display._frameId;

      // make sure we aren't pruned in the next syncTree()
      this.parent && this.parent.markSkipPruning();

      // mark visibility changes
      this.visibilityDirty = true;
      this.parent && this.parent.markChildVisibilityDirty();
    },

    // event callback for Node's 'opacity' change event
    onOpacityChange: function() {
      assert && assert( !this.stateless, 'If we are stateless, we should not receive these notifications' );

      this.markRenderStateDirty();
    },

    markChildVisibilityDirty: function() {
      if ( !this.childVisibilityDirty ) {
        this.childVisibilityDirty = true;
        this.parent && this.parent.markChildVisibilityDirty();
      }
    },

    /**
     * Updates the currently fittability for all of the drawables attached to this instance.
     * @public
     *
     * @param {boolean} fittable
     */
    updateDrawableFittability: function( fittable ) {
      this.selfDrawable && this.selfDrawable.setFittable( fittable );
      this.groupDrawable && this.groupDrawable.setFittable( fittable );
      // this.sharedCacheDrawable && this.sharedCacheDrawable.setFittable( fittable );
    },

    /**
     * Updates the visible/relativeVisible flags on the Instance and its entire subtree.
     *
     * @param {boolean} parentGloballyVisible - Whether our parent (if any) is globally visible
     * @param {boolean} parentRelativelyVisible - Whether our parent (if any) is relatively visible
     * @param {boolean} updateFullSubtree - If true, we will visit the entire subtree to ensure visibility is correct.
     */
    updateVisibility: function( parentGloballyVisible, parentRelativelyVisible, updateFullSubtree ) {
      // If our visibility flag for ourself is dirty, we need to update our entire subtree
      if ( this.visibilityDirty ) {
        updateFullSubtree = true;
      }

      // calculate our visibilities
      var nodeVisible = this.node.isVisible();
      var wasVisible = this.visible;
      var wasRelativeVisible = this.relativeVisible;
      var wasSelfVisible = this.selfVisible;
      this.visible = parentGloballyVisible && nodeVisible;
      this.relativeVisible = parentRelativelyVisible && nodeVisible;
      this.selfVisible = this.isVisibilityApplied ? true : this.relativeVisible;

      var len = this.children.length;
      for ( var i = 0; i < len; i++ ) {
        var child = this.children[ i ];

        if ( updateFullSubtree || child.visibilityDirty || child.childVisibilityDirty ) {
          // if we are a visibility root (isVisibilityApplied===true), disregard ancestor visibility
          child.updateVisibility( this.visible, this.isVisibilityApplied ? true : this.relativeVisible, updateFullSubtree );
        }
      }

      this.visibilityDirty = false;
      this.childVisibilityDirty = false;

      // trigger changes after we do the full visibility update
      if ( this.visible !== wasVisible ) {
        this.trigger0( 'visibility' );
      }
      if ( this.relativeVisible !== wasRelativeVisible ) {
        this.trigger0( 'relativeVisibility' );
      }
      if ( this.selfVisible !== wasSelfVisible ) {
        this.trigger0( 'selfVisibility' );
      }
    },

    getDescendantCount: function() {
      var count = this.children.length;
      for ( var i = 0; i < this.children.length; i++ ) {
        count += this.children[ i ].getDescendantCount();
      }
      return count;
    },

    /*---------------------------------------------------------------------------*
     * Miscellaneous
     *----------------------------------------------------------------------------*/

    // add a reference for an SVG group (fastest way to track them)
    addSVGGroup: function( group ) {
      this.svgGroups.push( group );
    },

    // remove a reference for an SVG group (fastest way to track them)
    removeSVGGroup: function( group ) {
      var index = _.indexOf( this.svgGroups, group );
      assert && assert( index >= 0, 'Tried to remove an SVGGroup from an Instance when it did not exist' );

      this.svgGroups.splice( index, 1 ); // TODO: remove function
    },

    // returns null when a lookup fails (which is legitimate)
    lookupSVGGroup: function( block ) {
      var len = this.svgGroups.length;
      for ( var i = 0; i < len; i++ ) {
        var group = this.svgGroups[ i ];
        if ( group.block === block ) {
          return group;
        }
      }
      return null;
    },

    // what instance have filters (opacity/visibility/clip) been applied up to?
    getFilterRootInstance: function() {
      if ( this.isBackbone || this.isInstanceCanvasCache || !this.parent ) {
        return this;
      }
      else {
        return this.parent.getFilterRootInstance();
      }
    },

    // what instance transforms have been applied up to?
    getTransformRootInstance: function() {
      if ( this.isTransformed || !this.parent ) {
        return this;
      }
      else {
        return this.parent.getTransformRootInstance();
      }
    },

    getVisibilityRootInstance: function() {
      if ( this.isVisibilityApplied || !this.parent ) {
        return this;
      }
      else {
        return this.parent.getVisibilityRootInstance();
      }
    },

    attachNodeListeners: function() {
      // attach listeners to our node
      this.relativeTransform.attachNodeListeners();

      if ( !this.isSharedCanvasCachePlaceholder ) {
        this.node.onStatic( 'childInserted', this.childInsertedListener );
        this.node.onStatic( 'childRemoved', this.childRemovedListener );
        this.node.onStatic( 'childrenReordered', this.childrenReorderedListener );
        this.node.onStatic( 'visibility', this.visibilityListener );

        this.node.onStatic( 'opacity', this.markRenderStateDirtyListener );
        this.node.onStatic( 'hint', this.markRenderStateDirtyListener );
        this.node.onStatic( 'clip', this.markRenderStateDirtyListener );
        this.node.onStatic( 'rendererBitmask', this.markRenderStateDirtyListener );
        this.node.onStatic( 'rendererSummary', this.markRenderStateDirtyListener );
      }
    },

    detachNodeListeners: function() {
      this.relativeTransform.detachNodeListeners();

      if ( !this.isSharedCanvasCachePlaceholder ) {
        this.node.offStatic( 'childInserted', this.childInsertedListener );
        this.node.offStatic( 'childRemoved', this.childRemovedListener );
        this.node.offStatic( 'childrenReordered', this.childrenReorderedListener );
        this.node.offStatic( 'visibility', this.visibilityListener );

        this.node.offStatic( 'opacity', this.markRenderStateDirtyListener );
        this.node.offStatic( 'hint', this.markRenderStateDirtyListener );
        this.node.offStatic( 'clip', this.markRenderStateDirtyListener );
        this.node.offStatic( 'rendererBitmask', this.markRenderStateDirtyListener );
        this.node.offStatic( 'rendererSummary', this.markRenderStateDirtyListener );
      }
    },

    // ensure that the render state is updated in the next syncTree()
    markRenderStateDirty: function() {
      this.renderStateDirtyFrame = this.display._frameId;

      // ensure we aren't pruned (not set on this instance, since we may not need to visit our children)
      this.parent && this.parent.markSkipPruning();
    },

    // ensure that this instance and its children will be visited in the next syncTree()
    markSkipPruning: function() {
      this.skipPruningFrame = this.display._frameId;

      // walk it up to the root
      this.parent && this.parent.markSkipPruning();
    },

    getBranchIndexTo: function( instance ) {
      var cachedValue = this.branchIndexMap[ instance.id ];
      if ( cachedValue !== undefined ) {
        return cachedValue;
      }

      var branchIndex = this.trail.getBranchIndexTo( instance.trail );
      this.branchIndexMap[ instance.id ] = branchIndex;
      instance.branchIndexMap[ this.id ] = branchIndex;
      this.branchIndexReferences.push( instance );
      instance.branchIndexReferences.push( this );

      return branchIndex;
    },

    // clean up listeners and garbage, so that we can be recycled (or pooled)
    dispose: function() {
      sceneryLog && sceneryLog.Instance && sceneryLog.Instance( 'dispose ' + this.toString() );
      sceneryLog && sceneryLog.Instance && sceneryLog.push();

      assert && assert( this.active, 'Seems like we tried to dispose this Instance twice, it is not active' );

      this.active = false;

      // Remove the bidirectional branch index reference data from this instance and any referenced instances.
      while ( this.branchIndexReferences.length ) {
        var referenceInstance = this.branchIndexReferences.pop();
        delete this.branchIndexMap[ referenceInstance.id ];
        delete referenceInstance.branchIndexMap[ this.id ];
        arrayRemove( referenceInstance.branchIndexReferences, this );
      }

      // order is somewhat important
      this.groupDrawable && this.groupDrawable.disposeImmediately( this.display );
      this.sharedCacheDrawable && this.sharedCacheDrawable.disposeImmediately( this.display );
      this.selfDrawable && this.selfDrawable.disposeImmediately( this.display );

      // Dispose the rest of our subtree
      var numChildren = this.children.length;
      for ( var i = 0; i < numChildren; i++ ) {
        this.children[ i ].dispose();
      }
      // Check for child instances that were removed (we are still responsible for disposing them, since we didn't get
      // synctree to happen for them).
      while ( this.instanceRemovalCheckList.length ) {
        var child = this.instanceRemovalCheckList.pop();

        // they could have already been disposed, so we need a guard here
        if ( child.active ) {
          child.dispose();
        }
      }

      // we don't originally add in the listener if we are stateless
      if ( !this.stateless ) {
        this.detachNodeListeners();
      }

      this.node.removeInstance( this );

      // release our reference to a shared cache if applicable, and dispose if there are no other references
      if ( this.sharedCacheInstance ) {
        this.sharedCacheInstance.externalReferenceCount--;
        if ( this.sharedCacheInstance.externalReferenceCount === 0 ) {
          delete this.display._sharedCanvasInstances[ this.node.getId() ];
          this.sharedCacheInstance.dispose();
        }
      }

      // clean our variables out to release memory
      this.cleanInstance( null, null );

      this.removeAllEventListeners();

      this.freeToPool();

      sceneryLog && sceneryLog.Instance && sceneryLog.pop();
    },

    audit: function( frameId, allowValidationNotNeededChecks ) {
      if ( assertSlow ) {
        if ( frameId === undefined ) {
          frameId = this.display._frameId;
        }

        assertSlow( !this.stateless,
          'State is required for all display instances' );

        assertSlow( ( this.firstDrawable === null ) === ( this.lastDrawable === null ),
          'First/last drawables need to both be null or non-null' );

        assertSlow( ( !this.isBackbone && !this.isSharedCanvasCachePlaceholder ) || this.groupDrawable,
          'If we are a backbone or shared cache, we need to have a groupDrawable reference' );

        assertSlow( !this.isSharedCanvasCachePlaceholder || !this.node.isPainted() || this.selfDrawable,
          'We need to have a selfDrawable if we are painted and not a shared cache' );

        assertSlow( ( !this.isTransformed && !this.isCanvasCache ) || this.groupDrawable,
          'We need to have a groupDrawable if we are a backbone or any type of canvas cache' );

        assertSlow( !this.isSharedCanvasCachePlaceholder || this.sharedCacheDrawable,
          'We need to have a sharedCacheDrawable if we are a shared cache' );

        assertSlow( this.addRemoveCounter === 0,
          'Our addRemoveCounter should always be 0 at the end of syncTree' );

        // validate the subtree
        for ( var i = 0; i < this.children.length; i++ ) {
          var childInstance = this.children[ i ];

          childInstance.audit( frameId, allowValidationNotNeededChecks );
        }

        this.relativeTransform.audit( frameId, allowValidationNotNeededChecks );

        this.fittability.audit();
      }
    },

    // @public (scenery-internal) - Applies checks to make sure our visibility tracking is working as expected.
    auditVisibility: function( parentVisible ) {
      if ( assertSlow ) {
        var visible = parentVisible && this.node.isVisible();
        var trailVisible = this.trail.isVisible();
        assertSlow( visible === trailVisible, 'Trail visibility failure' );
        assertSlow( visible === this.visible, 'Visible flag failure' );

        // validate the subtree
        for ( var i = 0; i < this.children.length; i++ ) {
          var childInstance = this.children[ i ];

          childInstance.auditVisibility( visible );
        }
      }
    },

    auditChangeIntervals: function( oldFirstDrawable, oldLastDrawable, newFirstDrawable, newLastDrawable ) {
      if ( oldFirstDrawable ) {
        var oldOne = oldFirstDrawable;

        // should hit, or will have NPE
        while ( oldOne !== oldLastDrawable ) {
          oldOne = oldOne.oldNextDrawable;
        }
      }

      if ( newFirstDrawable ) {
        var newOne = newFirstDrawable;

        // should hit, or will have NPE
        while ( newOne !== newLastDrawable ) {
          newOne = newOne.nextDrawable;
        }
      }

      function checkBetween( a, b ) {
        // have the body of the function stripped (it's not inside the if statement due to JSHint)
        if ( assertSlow ) {
          assertSlow( a !== null );
          assertSlow( b !== null );

          while ( a !== b ) {
            assertSlow( a.nextDrawable === a.oldNextDrawable, 'Change interval mismatch' );
            a = a.nextDrawable;
          }
        }
      }

      if ( assertSlow ) {
        var firstChangeInterval = this.firstChangeInterval;
        var lastChangeInterval = this.lastChangeInterval;

        if ( !firstChangeInterval || firstChangeInterval.drawableBefore !== null ) {
          assertSlow( oldFirstDrawable === newFirstDrawable,
            'If we have no changes, or our first change interval is not open, our firsts should be the same' );
        }
        if ( !lastChangeInterval || lastChangeInterval.drawableAfter !== null ) {
          assertSlow( oldLastDrawable === newLastDrawable,
            'If we have no changes, or our last change interval is not open, our lasts should be the same' );
        }

        if ( !firstChangeInterval ) {
          assertSlow( !lastChangeInterval, 'We should not be missing only one change interval' );

          // with no changes, everything should be identical
          oldFirstDrawable && checkBetween( oldFirstDrawable, oldLastDrawable );
        }
        else {
          assertSlow( lastChangeInterval, 'We should not be missing only one change interval' );

          // endpoints
          if ( firstChangeInterval.drawableBefore !== null ) {
            // check to the start if applicable
            checkBetween( oldFirstDrawable, firstChangeInterval.drawableBefore );
          }
          if ( lastChangeInterval.drawableAfter !== null ) {
            // check to the end if applicable
            checkBetween( lastChangeInterval.drawableAfter, oldLastDrawable );
          }

          // between change intervals (should always be guaranteed to be fixed)
          var interval = firstChangeInterval;
          while ( interval && interval.nextChangeInterval ) {
            var nextInterval = interval.nextChangeInterval;

            assertSlow( interval.drawableAfter !== null );
            assertSlow( nextInterval.drawableBefore !== null );

            checkBetween( interval.drawableAfter, nextInterval.drawableBefore );

            interval = nextInterval;
          }
        }
      }
    },

    toString: function() {
      return this.id + '#' + ( this.node ? ( this.node.constructor.name ? this.node.constructor.name : '?' ) + '#' + this.node.id : '-' );
    }
  } );

  // object pooling
  Poolable.mixInto( Instance, {
    initialize: Instance.prototype.initialize
  } );

  return Instance;
} );
