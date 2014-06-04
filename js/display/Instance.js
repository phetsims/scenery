// Copyright 2002-2014, University of Colorado

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
 **********************
 * Relative transform system description:
 *
 * A "relative" transform here is the transform that a Trail would have, not necessarily rooted at the display's root.
 * Imagine we have a CSS-transformed backbone div, and nodes underneath that render to Canvas. On the Canvas, we will
 * need to set the context's transform to the matrix that will transform from the displayed instances' local coordinates
 * frames to the CSS-transformed backbone instance. Notably, transforming the backbone instance or any of its ancestors
 * does NOT affect this "relative" transform from the instance to the displayed instances, while any Node transform
 * changes between (not including) the backbone instance and (including) the displayed instance WILL affect that
 * relative transform. This is key to setting the CSS transform on backbones, DOM nodes, having the transforms necessary
 * for the fastest Canvas display, and determining fitting bounds for layers.
 *
 * Each Instance has its own "relative trail", although these aren't stored. We use implicit hierarchies in the Instance
 * tree for this purpose. If an Instance is a CSS-transformed backbone, or any other case that requires drawing beneath
 * to be done relative to its local coordinate frame, we call it a transform "root", and it has instance.isTransformed
 * set to true. This should NEVER change for an instance (any changes that would do this require reconstructing the
 * instance tree).
 *
 * There are implicit hierarchies for each root, with trails starting from that root's children (they won't apply that
 * root's transform since we assume we are working within that root's local coordinate frame). These should be
 * effectively independent (if there are no bugs), so that flags affecting one implicit hierarchy will not affect the
 * other (dirty flags, etc.), and traversals should not cross these boundaries.
 * 
 * For various purposes, we want a system that can:
 * - every frame before repainting: notify listeners on instances whether its relative transform has changed
 *                                  (add|removeRelativeTransformListener)
 * - every frame before repainting: precompute relative transforms on instances where we know this is required
 *                                  (add|removeRelativeTransformPrecompute)
 * - any time during repainting:    provide an efficient way to lazily compute relative transforms when needed
 *
 * This is done by first having one step in the pre-repaint phase that traverses the tree where necessary, notifying
 * relative transform listeners, and precomputing relative transforms when they have changed (and precomputation is
 * requested). This traversal leaves metadata on the instances so that we can (fairly) efficiently force relative
 * transform "validation" any time afterwards that makes sure the relativeMatrix property is up-to-date.
 *
 * First of all, to ensure we traverse the right parts of the tree, we need to keep metadata on what needs to be
 * traversed. This is done by tracking counts of listeners/precompution needs, both on the instance itself, and how many
 * children have these needs. We use counts instead of boolean flags so that we can update this quickly while (a) never
 * requiring full children scans to update this metadata, and (b) minimizing the need to traverse all the way up to the
 * root to update the metadata. The end result is hasDescendantListenerNeed and hasDescendantComputeNeed which compute,
 * respectively, whether we need to traverse this instance for listeners and precomputation. Additionally,
 * hasAncestorListenerNeed and hasAncestorComputeNeed compute whether our parent needs to traverse up to us.
 *
 * The other tricky bits to remember for this traversal are the flags it sets, and how later validation uses and updates
 * these flags. First of all, we have relativeSelfDirty and relativeChildDirtyFrame. When a node's transform changes,
 * we mark relativeSelfDirty on the node, and relativeChildDirtyFrame for all ancestors up to (and including) the
 * transform root. relativeChildDirtyFrame allows us to prune our traversal to only modified subtrees. Additionally, so
 * that we can retain the invariant that it is "set" parent node if it is set on a child, we store the rendering frame
 * ID (unique to traversals) instead of a boolean true/false. Our traversal may skip subtrees where
 * relativeChildDirtyFrame is "set" due to no listeners or precomputation needed for that subtree, so if we used
 * booleans this would be violated. Violating that invariant would prevent us from "bailing out" when setting the
 * relativeChildDirtyFrame flag, and on EVERY transform change we would have to traverse ALL of the way to the root
 * (instead of the efficient "stop at the ancestor where it is also set").
 *
 * relativeSelfDirty is initially set on instances whose nodes had transform changes (they mark that this relative
 * transform, and all transforms beneath, are dirty). We maintain the invariant that if a relative transform needs to be
 * recomputed, it or one of its ancestors WILL ALWAYS have this flag set. This is required so that later validation of
 * the relative transform can verify whether it has been changed in an efficient way. When we recompute the relative
 * transform for one instance, we have to set this flag on all children to maintain this invariant.
 *
 * Additionally, so that we can have fast "validation" speed, we also store into relativeFrameId the last rendering
 * frame ID (counter) where we either verified that the relative transform is up to date, or we have recomputed it. Thus
 * when "validating" a relative transform that wasn't precomputed, we only need to scan up the ancestors to the first
 * one that was verified OK this frame (boolean flags are insufficient for this, since we would have to clear them all
 * to false on every frame, requiring a full tree traversal). In the future, we may set this flag to the frame
 * proactively during traversal to speed up validation, but that is not done at the time of this writing.
 *
 * Some helpful notes for the scope of various relativeTransform bits:
 *                         (transformRoot) (regular) (regular) (transformRoot)
 * relativeChildDirtyFrame [---------------------------------]                 (int)
 * relativeSelfDirty                       [---------------------------------]
 * relativeTransform                       [---------------------------------] (transform on root applies to
 *                                                                             its parent context)
 * relativeFrameId                         [---------------------------------] (int)
 * child counts            [---------------------------------]                 (e.g. relativeChildrenListenersCount,
 *                                                                             relativeChildrenPrecomputeCount)
 * self counts                             [---------------------------------] (e.g. relativePrecomputeCount,
 *                                                                             relativeTransformListeners.length)
 **********************
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';
  
  var inherit = require( 'PHET_CORE/inherit' );
  var Poolable = require( 'PHET_CORE/Poolable' );
  var cleanArray = require( 'PHET_CORE/cleanArray' );
  var Matrix3 = require( 'DOT/Matrix3' );
  var scenery = require( 'SCENERY/scenery' );
  var Drawable = require( 'SCENERY/display/Drawable' );
  var Renderer = require( 'SCENERY/display/Renderer' );
  require( 'SCENERY/display/RenderState' );
  
  var globalIdCounter = 1;
  
  scenery.Instance = function Instance( display, trail ) {
    this.active = false;
    
    this.initialize( display, trail );
  };
  var Instance = scenery.Instance;
  
  inherit( Object, Instance, {
    initialize: function( display, trail ) {
      assert && assert( !this.active, 'We should never try to initialize an already active object' );
      
      // prevent the trail passed in from being mutated after this point (we want a consistent trail)
      trail.setImmutable();
      
      this.id = this.id || globalIdCounter++;
      
      this.cleanInstance( display, trail, trail.lastNode() );
      
      // the actual cached transform to the root
      this.relativeMatrix = this.relativeMatrix || Matrix3.identity();
      
      // whether our relativeMatrix is dirty
      this.relativeSelfDirty = true;
      
      // how many children have (or have descendants with) relativeTransformListeners
      this.relativeChildrenListenersCount = 0;
      
      // if >0, indicates this should be precomputed in the pre-repaint phase
      this.relativePrecomputeCount = 0;
      
      // how many children have (or have descendants with) >0 relativePrecomputeCount
      this.relativeChildrenPrecomputeCount = 0;
      
      // used to mark what frame the transform was updated in (to accelerate non-precomputed relative transform access)
      this.relativeFrameId = -1;
      
      // Whether children have dirty transforms (if it is the current frame) NOTE: used only for pre-repaint traversal,
      // and can be ignored if it has a value less than the current frame ID. This allows us to traverse and hit all
      // listeners for this particular traversal, without leaving an invalid subtree (a boolean flag here is
      // insufficient, since our traversal handling would validate our invariant of
      // this.relativeChildDirtyFrame => parent.relativeChildDirtyFrame). In this case, they are both effectively
      // "false" unless they are the current frame ID, in which case that invariant holds.
      this.relativeChildDirtyFrame = display._frameId;
      
      // properties relevant to the node's direct transform
      this.transformDirty = true; // whether the node's transform has changed (until the pre-repaint phase)
      this.nodeTransformListener = this.nodeTransformListener || this.markTransformDirty.bind( this );
      
      // In the range (-1,0), to help us track insertions and removals of this instance's node to its parent
      // (did we get removed but added back?).
      // If it's -1 at its parent's syncTree, we'll end up removing our reference to it.
      // We use an integer just for sanity checks (if it ever reaches -2 or 1, we've reached an invalid state)
      this.addRemoveCounter = 0;
      
      // Node listeners for tracking children. Listeners should be added only when we become stateful
      this.childInsertedListener = this.childInsertedListener || this.onChildInserted.bind( this );
      this.childRemovedListener = this.childRemovedListener || this.onChildRemoved.bind( this );
      
      // We need to add this reference on stateless instances, so that we can find out if it was removed before our
      // syncTree was called.
      this.node.addInstance( this );
      
      // Outstanding external references. used for shared cache instances, where multiple instances can point to us.
      this.externalReferenceCount = 0;
      
      // Whether we have been instantiated. false if we are in a pool waiting to be instantiated.
      this.active = true;
      
      sceneryLog && sceneryLog.Instance && sceneryLog.Instance( 'initialized ' + this.toString() );
      
      return this;
    },
    
    // called for initialization of properties (via initialize(), via constructor), or to clean the instance for
    // placement in the pool (don't leak memory)
    cleanInstance: function( display, trail, node ) {
      this.display = display;
      this.trail = trail;
      this.node = node;
      this.parent = null; // will be set as needed
      // NOTE: reliance on correct order after syncTree by at least SVGBlock/SVGGroup
      this.children = cleanArray( this.children ); // Array[Instance].
      this.sharedCacheInstance = null; // reference to a shared cache instance (different than a child)
      
      // Child instances are pushed to here when their node is removed from our node. we don't immediately dispose,
      // since it may be added back.
      this.instanceRemovalCheckList = cleanArray( this.instanceRemovalCheckList );
      
      // linked-list handling for sibling instances (null for no next/previous sibling)
      this.previousSibling = null;
      this.nextSibling = null;
      
      this.selfDrawable = null;
      this.groupDrawable = null; // e.g. backbone or non-shared cache
      this.sharedCacheDrawable = null; // our drawable if we are a shared cache
      
      // references into the linked list of drawables (null if nothing is drawable under this)
      this.firstDrawable = null;
      this.lastDrawable = null;
      
      // references that will be filled in with syncTree
      if ( this.state ) {
        // NOTE: assumes that we aren't reusing states across instances
        this.state.freeToPool();
      }
      this.state = null;
      this.isTransformed = false; // whether this instance creates a new "root" for the relative trail transforms
      
      this.svgGroups = []; // list of SVG groups associated with this display instance
      
      // will be notified in pre-repaint phase that our relative transform has changed (but not computed by default)
      // NOTE: it's part of the relative transform feature, see above for documentation
      //OHTWO TODO: should we rely on listeners removing themselves?
      this.relativeTransformListeners = cleanArray( this.relativeTransformListeners );
      
      this.cleanSyncTreeResults();
    },
    
    cleanSyncTreeResults: function() {
      // Tracking bounding indices / drawables for what has changed, so we don't have to over-stitch things.
      
      // if (not iff) child's index <= beforeStableIndex, it hasn't been added/removed. relevant to current children.
      this.beforeStableIndex = this.children.length;
      
      // if (not iff) child's index >= afterStableIndex, it hasn't been added/removed. relevant to current children.
      this.afterStableIndex = -1;
    },
    
    // @public
    baseSyncTree: function() {
      sceneryLog && sceneryLog.Instance && sceneryLog.Instance( '-------- START baseSyncTree ' + this.toString() + ' --------' );
      this.syncTree( scenery.RenderState.RegularState.createRootState( this.node ) );
      sceneryLog && sceneryLog.Instance && sceneryLog.Instance( '-------- END baseSyncTree ' + this.toString() + ' --------' );
      this.cleanSyncTreeResults();
    },
    
    // updates the internal {RenderState}, and fully synchronizes the instance subtree
    /*OHTWO TODO:
     * Pruning:
     *   - If children haven't changed, skip instance add/move/remove
     *   - If RenderState hasn't changed AND there are no render/instance/stitch changes below us, EXIT (whenever we are assured children don't need sync)
     * Return linked-list of alternating changed (add/remove) and keep sections of drawables, for processing with backbones/canvas caches
     *
     * Other notes:
     *
     *    Traversal hits every child if parent render state changed. Otherwise, only hits children who have (or has descendants who have) potential render state changes. If we haven't hit a "change" yet from ancestors, don't re-evaluate any render states (UNLESS renderer summary changed!)
     *      Need recursive flag for "render state needs reevaluation" / "child render state needs reevaluation
     *        Don't flag changes when they won't actually change the "current" render state!!!
     *        Changes in renderer summary (subtree combined) can cause changes in the render state
     *    OK for traversal to return "no change", doesn't specify any drawables changes
     *
     * Returns: Whether there was a stitching-relevant drawable change.
     */
    syncTree: function( state ) {
      sceneryLog && sceneryLog.Instance && sceneryLog.Instance( 'syncTree ' + this.toString() + ' ' + state.toString() + ( this.isStateless() ? ' (stateless)' : '' ) );
      sceneryLog && sceneryLog.Instance && sceneryLog.push();
      
      assert && assert( state && state.isSharedCanvasCachePlaceholder !== undefined, 'RenderState duck-typing instanceof' );
      assert && assert( !this.parent || !this.parent.isStateless(), 'We should not have a stateless parent instance' ); // may access isTransformed up to root to determine relative trails
      
      var oldState = this.state;
      var wasStateless = !oldState;
      
      assert && assert( wasStateless || oldState.isInstanceCompatibleWith( state ),
                        'Attempt to update to a render state that is not compatible with this instance\'s current state' );
      assert && assert( wasStateless || oldState.isTransformed === state.isTransformed ); // no need to overwrite, should always be the same
      assert && assert( !wasStateless || this.children.length === 0, 'We should not have child instances on an instance without state' );
      
      this.state = state;
      this.isTransformed = state.isTransformed;
      
      // whether we have a stitch change. initialized based on how our children were added/removed/moved, but can be set later down in this function
      var hasStitchChange = this.beforeStableIndex < this.afterStableIndex;
      
      if ( wasStateless ) {
        // If we are a transform root, notify the display that we are dirty. We'll be validated when it's at that phase at the next updateDisplay().
        //OHTWO TODO: when else do we have to call this?
        if ( this.isTransformed ) {
          this.display.markTransformRootDirty( this, true );
        }
        
        this.onStateCreation();
      }
      
      if ( state.isSharedCanvasCachePlaceholder ) {
        hasStitchChange = this.sharedSyncTree( state ) || hasStitchChange;
      } else {
        hasStitchChange = this.normalSyncTree( state, oldState ) || hasStitchChange;
      }
      
      if ( oldState && oldState !== this.state ) {
        oldState.freeToPool();
      }
      
      sceneryLog && sceneryLog.Instance && sceneryLog.pop();
      
      return hasStitchChange;
    },
    
    normalSyncTree: function( state, oldState ) {
      var hasStitchChange = false;
      
      // not a shared cache
      
      // local variables, since we can't overwrite our instance properties yet
      var firstDrawable = null;
      var lastDrawable = null;
      
      var currentDrawable = null;
      
      if ( this.node.isPainted() ) {
        var selfRenderer = state.selfRenderer; // our new self renderer bitmask
        
        // bitwise trick, since only one of Canvas/SVG/DOM/WebGL/etc. flags will be chosen, and bitmaskRendererArea is the mask for those flags
        // In English, "Is the current selfDrawable compatible with our selfRenderer (if any), or do we need to create a selfDrawable?"
        //OHTWO TODO: For Canvas, we won't care about anything else for the drawable, but for DOM we care about the force-acceleration flag! That's stripped out here
        if ( !this.selfDrawable || ( ( this.selfDrawable.renderer & selfRenderer & Renderer.bitmaskRendererArea ) === 0 ) ) {
          // we'll need stitch updates
          hasStitchChange = true;
          
          // maintain our stitch change interval
          this.beforeStableIndex = -1;
          if ( this.afterStableIndex < 0 ) {
            this.afterStableIndex = 0;
          }
          
          if ( this.selfDrawable ) {
            // scrap the previous selfDrawable, we need to create one with a different renderer.
            this.selfDrawable.markForDisposal( this.display );
          }
          
          this.selfDrawable = Renderer.createSelfDrawable( this, this.node, selfRenderer );
        }
        
        firstDrawable = currentDrawable = this.selfDrawable;
      }
      
      // mark all removed instances to be disposed (along with their subtrees)
      while ( this.instanceRemovalCheckList.length ) {
        var instanceToMark = this.instanceRemovalCheckList.pop();
        if ( instanceToMark.addRemoveCounter === -1 ) {
          instanceToMark.addRemoveCounter = 0; // reset it, so we don't mark it for disposal more than once
          this.display.markInstanceRootForDisposal( instanceToMark );
        }
      }
      
      if ( !oldState ) {
        // we need to create all of the child instances
        for ( var k = 0; k < this.node.children.length; k++ ) {
          // create a child instance
          var child = this.node.children[k];
          this.appendInstance( Instance.createFromPool( this.display, this.trail.copy().addDescendant( child, k ) ) );
        }
      }
      
      for ( var i = 0; i < this.children.length; i++ ) {
        var childInstance = this.children[i];
        var childState = state.getStateForDescendant( childInstance.node );
        
        // see if we need to rebuild the instance tree due to an incompatible render state
        if ( !childInstance.isStateless() && !childState.isInstanceCompatibleWith( childInstance.state ) ) {
          // mark it for disposal
          this.display.markInstanceRootForDisposal( childInstance );
          
          // swap in a new instance
          var replacementInstance = Instance.createFromPool( this.display, this.trail.copy().addDescendant( childInstance.node, i ) );
          this.replaceInstanceWithIndex( childInstance, replacementInstance, i );
          childInstance = replacementInstance;
        }
        
        // sync the tree
        var childStitchChanged = childInstance.syncTree( childState );
        
        // maintain our stitch-change interval
        if ( childStitchChanged ) {
          hasStitchChange = true;
          
          // if the instance is before our stitch change interval created based on child node add/remove/move, record the old drawable
          if ( i <= this.beforeStableIndex ) {
            this.beforeStableIndex = i - 1;
          }
          // if the instance is after our stitch change interval created based on child node add/remove/move, record the old drawable
          if ( i >= this.afterStableIndex ) {
            // NOTE: this may happen multiple times if multiple child instances after our interval are changed. consider iteration in the reverse order?
            // It's done here since we clean the stitch-based changes below during our first (this) iteration, and probably isn't a worry.
            this.afterStableIndex = i + 1;
          }
          
        }
        // clean up the metadata on our child (can't be done in the child call, since we use these values like a composite return value)
        //OHTWO TODO: only do this on instances that were actually traversed
        childInstance.cleanSyncTreeResults();
        
        //OHTWO TODO: only strip out invisible Canvas drawables, while leaving SVG (since we can more efficiently hide SVG trees, memory-wise)
        if ( childInstance.node.isVisible() ) {
          // figure out what the first and last drawable should be hooked into for the child
          var firstChildDrawable = null;
          var lastChildDrawable = null;
          
          //OHTWO TODO: set first/last drawables to the group?
          if ( childInstance.groupDrawable ) {
            // if there is a group (e.g. non-shared cache or backbone), use it
            firstChildDrawable = lastChildDrawable = childInstance.groupDrawable;
          } else if ( childInstance.sharedCacheDrawable ) {
            // if there is a shared cache drawable, use it
            firstChildDrawable = lastChildDrawable = childInstance.sharedCacheDrawable;
          } else if ( childInstance.firstDrawable ) {
            // otherwise, if they exist, pick the node's first/last directly
            assert && assert( childInstance.lastDrawable, 'Any display instance with firstDrawable should also have lastDrawable' );
            firstChildDrawable = childInstance.firstDrawable;
            lastChildDrawable = childInstance.lastDrawable;
          }
          
          // if there are any drawables for that child, link them up in our linked list
          if ( firstChildDrawable ) {
            if ( currentDrawable ) {
              // there is already an end of the linked list, so just append to it
              Drawable.connectDrawables( currentDrawable, firstChildDrawable, this.display );
            } else {
              // start out the linked list
              firstDrawable = firstChildDrawable;
            }
            // update the last drawable of the linked list
            currentDrawable = lastChildDrawable;
          }
        }
      }
      // finish setting up references to the linked list (now firstDrawable and lastDrawable should be set properly)
      lastDrawable = currentDrawable; // either null, or the drawable itself
      
      var groupRenderer = state.groupRenderer;
      assert && assert( ( state.isBackbone ? 1 : 0 ) + ( state.isInstanceCanvasCache ? 1 : 0 ) + ( state.isSharedCanvasCacheSelf ? 1 : 0 ) === ( groupRenderer ? 1 : 0 ),
                        'We should have precisely one of these flags set for us to have a groupRenderer' );
      
      // if we switched to/away from a group, our group type changed, or our group renderer changed
      /* jshint -W018 */
      var groupChanged = !!groupRenderer !== !!this.groupDrawable ||
                         ( oldState && ( oldState.isBackbone !== state.isBackbone ||
                                         oldState.isInstanceCanvasCache !== state.isInstanceCanvasCache ||
                                         oldState.isSharedCanvasCacheSelf !== state.isSharedCanvasCacheSelf ) ) ||
                         ( this.groupDrawable && this.groupDrawable.renderer !== groupRenderer );
      
      // if there is a change, prepare
      if ( groupChanged ) {
        hasStitchChange = true;
        
        this.markFullStitchChangeInterval();
        
        if ( this.groupDrawable ) {
          this.groupDrawable.markForDisposal( this.display );
          this.groupDrawable = null;
        }
      }
      
      if ( groupRenderer ) {
        // ensure our linked list is fully disconnected from others
        firstDrawable && Drawable.disconnectBefore( firstDrawable, this.display );
        lastDrawable && Drawable.disconnectAfter( lastDrawable, this.display );
        
        if ( state.isBackbone ) {
          if ( groupChanged ) {
            this.groupDrawable = scenery.BackboneDrawable.createFromPool( this.display, this, this.getTransformRootInstance(), groupRenderer, state.isDisplayRoot );
            
            if ( this.isTransformed ) {
              this.display.markTransformRootDirty( this, true );
            }
          }
          
          this.groupDrawable.stitch( firstDrawable, lastDrawable );
        } else if ( state.isInstanceCanvasCache ) {
          if ( groupChanged ) {
            this.groupDrawable = scenery.InlineCanvasCacheDrawable.createFromPool( groupRenderer, this );
          }
          this.groupDrawable.stitch( firstDrawable, lastDrawable );
        } else if ( state.isSharedCanvasCacheSelf ) {
          if ( groupChanged ) {
            this.groupDrawable = scenery.CanvasBlock.createFromPool( groupRenderer, this );
          }
          //OHTWO TODO: restitch here??? implement it
        }
        
        //OHTWO TODO: set first/last drawables to the group?
      }
      
      // now set the first/last drawables, since we are finished with referencing the old ones
      //OHTWO TODO: set first/last drawables to the group, if available?
      this.firstDrawable = firstDrawable;
      this.lastDrawable = lastDrawable;
      
      return hasStitchChange;
    },
    
    sharedSyncTree: function( state ) {
      var hasStitchChange = false;
      
      this.ensureSharedCacheInitialized();
      
      var sharedCacheRenderer = state.sharedCacheRenderer;
      
      if ( !this.sharedCacheDrawable || this.sharedCacheDrawable.renderer !== sharedCacheRenderer ) {
        // we'll need stitch updates
        hasStitchChange = true;
        
        this.markFullStitchChangeInterval();
        
        if ( this.sharedCacheDrawable ) {
          this.sharedCacheDrawable.markForDisposal( this.display );
        }
        
        //OHTWO TODO: actually create the proper shared cache drawable depending on the specified renderer (update it if necessary)
        this.sharedCacheDrawable = new scenery.SharedCanvasCacheDrawable( this.trail, sharedCacheRenderer, this, this.sharedCacheInstance );
        this.firstDrawable = this.sharedCacheDrawable;
        this.lastDrawable = this.sharedCacheDrawable;
      }
      
      return hasStitchChange;
    },
    
    ensureSharedCacheInitialized: function() {
      // we only need to initialize this shared cache reference once
      if ( !this.sharedCacheInstance ) {
        var instanceKey = this.node.getId();
        this.sharedCacheInstance = this.display._sharedCanvasInstances[instanceKey]; // TODO: have this abstracted away in the Display?
        
        // TODO: increment reference counting?
        if ( !this.sharedCacheInstance ) {
          this.sharedCacheInstance = Instance.createFromPool( this.display, new scenery.Trail( this.node ) );
          this.sharedCacheInstance.syncTree( scenery.RenderState.RegularState.createSharedCacheState( this.node ) );
          this.display._sharedCanvasInstances[instanceKey] = this.sharedCacheInstance;
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
    
    // whether we don't have an associated RenderState attached. If we are stateless, we won't have children, and won't have listeners attached to our node yet.
    isStateless: function() {
      return !this.state;
    },
    
    markFullStitchChangeInterval: function() {
      // full interval
      this.beforeStableIndex = -1;
      this.afterStableIndex = this.children.length;
    },
    
    // @private, finds the closest drawable (not including the child instance at childIndex) using lastDrawable, or null
    findPreviousDrawable: function( childIndex ) {
      for ( var i = childIndex - 1; i >= 0; i-- ) {
        var option = this.children[i].lastDrawable;
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
        var option = this.children[i].firstDrawable;
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
                        'Instance insertion bounds check for index ' + index + ' with previous children length ' + this.children.length );
      
      // maintain the linked-list handling for sibling instances
      var previousInstance = ( index - 1 >= 0 ) ? this.children[index-1] : null;
      var nextInstance = ( index < this.children.length ) ? this.children[index] : null;
      if ( previousInstance ) {
        previousInstance.nextSibling = instance;
        instance.previousSibling = previousInstance;
      }
      if ( nextInstance ) {
        nextInstance.previousSibling = instance;
        instance.nextSibling = nextInstance;
      }
      
      this.children.splice( index, 0, instance );
      instance.parent = this;
      
      // maintain our stitch-change interval
      if ( index <= this.beforeStableIndex ) {
        this.beforeStableIndex = index - 1;
      }
      if ( index > this.afterStableIndex ) {
        this.afterStableIndex = index + 1;
      } else {
        this.afterStableIndex++;
      }
      
      if ( instance.isStateless ) {
        assert && assert( !instance.hasAncestorListenerNeed(), 'We only track changes properly if stateless instances do not have needs' );
        assert && assert( !instance.hasAncestorComputeNeed(), 'We only track changes properly if stateless instances do not have needs' );
      } else {
        if ( !instance.isTransformed ) {
          if ( instance.hasAncestorListenerNeed() ) {
            this.incrementTransformListenerChildren();
          }
          if ( instance.hasAncestorComputeNeed() ) {
            this.incrementTransformPrecomputeChildren();
          }
        }
      }
      
      // mark the instance's transform as dirty, so that it will be reachable in the pre-repaint traversal pass
      instance.forceMarkTransformDirty();
    },
    
    removeInstance: function( instance ) {
      return this.removeInstanceWithIndex( instance, _.indexOf( this.children, instance ) );
    },
    
    removeInstanceWithIndex: function( instance, index ) {
      assert && assert( instance instanceof Instance );
      assert && assert( index >= 0 && index < this.children.length,
                        'Instance removal bounds check for index ' + index + ' with previous children length ' + this.children.length );
      
      // maintain the linked-list handling for sibling instances
      var previousInstance = ( index - 1 >= 0 ) ? this.children[index-1] : null;
      var nextInstance = ( index + 1 < this.children.length ) ? this.children[index+1] : null;
      if ( previousInstance ) {
        previousInstance.nextSibling = nextInstance;
      }
      if ( nextInstance ) {
        nextInstance.previousSibling = previousInstance;
      }
      
      this.children.splice( index, 1 ); // TODO: replace with a 'remove' function call
      instance.parent = null;
      instance.nextSibling = null;
      instance.previousSibling = null;
      
      // maintain our stitch-change interval
      if ( index <= this.beforeStableIndex ) {
        this.beforeStableIndex = index - 1;
      }
      if ( index >= this.afterStableIndex ) {
        this.afterStableIndex = index;
      } else {
        this.afterStableIndex--;
      }
      
      if ( !instance.isTransformed ) {
        if ( instance.hasAncestorListenerNeed() ) {
          this.decrementTransformListenerChildren();
        }
        if ( instance.hasAncestorComputeNeed() ) {
          this.decrementTransformPrecomputeChildren();
        }
      }
    },
    
    replaceInstanceWithIndex: function( childInstance, replacementInstance, index ) {
      // TODO: optimization? hopefully it won't happen often, so we just do this for now
      this.removeInstanceWithIndex( childInstance, index );
      this.insertInstance( replacementInstance, index );
    },
    
    // if we have a child instance that corresponds to this node, return it (otherwise null)
    findChildInstanceOnNode: function( node ) {
      var instances = node.getInstances();
      for ( var i = 0; i < instances.length; i++ ) {
        if ( instances[i].parent === this ) {
          return instances[i];
        }
      }
      return null;
    },
    
    // event callback for Node's 'childInserted' event, used to track children
    onChildInserted: function( childNode, index ) {
      assert && assert( !this.isStateless(), 'If we are stateless, we should not receive these notifications' );
      
      var instance = this.findChildInstanceOnNode( childNode );
      
      if ( instance ) {
        // it must have been added back. increment its counter
        instance.addRemoveCounter += 1;
        assert && assert( instance.addRemoveCounter === 0 );
      } else {
        instance = Instance.createFromPool( this.display, this.trail.copy().addDescendant( childNode, index ) );
      }
      
      this.insertInstance( instance, index );
    },
    
    // event callback for Node's 'childRemoved' event, used to track children
    onChildRemoved: function( childNode, index ) {
      assert && assert( !this.isStateless(), 'If we are stateless, we should not receive these notifications' );
      assert && assert( this.children[index].node === childNode, 'Ensure that our instance matches up' );
      
      var instance = this.findChildInstanceOnNode( childNode );
      assert && assert( instance !== null, 'We should always have a reference to a removed instance' );
      
      instance.addRemoveCounter -= 1;
      assert && assert( instance.addRemoveCounter === -1 );
      
      // track the removed instance here. if it doesn't get added back, this will be the only reference we have (we'll need to dispose it)
      this.instanceRemovalCheckList.push( instance );
      
      this.removeInstanceWithIndex( instance, index );
    },
    
    /*---------------------------------------------------------------------------*
    * Relative transform listener count recursive handling
    *----------------------------------------------------------------------------*/
    
    // @private: Only for descendants need, ignores 'self' need on isTransformed
    hasDescendantListenerNeed: function() {
      if ( this.isTransformed ) {
        return this.relativeChildrenListenersCount > 0;
      } else {
        return this.relativeChildrenListenersCount > 0 || this.relativeTransformListeners.length > 0;
      }
      return ;
    },
    // @private: Only for ancestors need, ignores child need on isTransformed
    hasAncestorListenerNeed: function() {
      if ( this.isTransformed ) {
        return this.relativeTransformListeners.length > 0;
      } else {
        return this.relativeChildrenListenersCount > 0 || this.relativeTransformListeners.length > 0;
      }
    },
    // @private
    hasSelfListenerNeed: function() {
      return this.relativeTransformListeners.length > 0;
    },
    // @private (called on the ancestor of the instance with the need)
    incrementTransformListenerChildren: function() {
      var before = this.hasAncestorListenerNeed();
      
      this.relativeChildrenListenersCount++;
      if ( before !== this.hasAncestorListenerNeed() ) {
        assert && assert( !this.isTransformed, 'Should not be a change in need if we have the isTransformed flag' );
        
        this.parent && this.parent.incrementTransformListenerChildren();
      }
    },
    // @private (called on the ancestor of the instance with the need)
    decrementTransformListenerChildren: function() {
      var before = this.hasAncestorListenerNeed();
      
      this.relativeChildrenListenersCount--;
      if ( before !== this.hasAncestorListenerNeed() ) {
        assert && assert( !this.isTransformed, 'Should not be a change in need if we have the isTransformed flag' );
        
        this.parent && this.parent.decrementTransformListenerChildren();
      }
    },
    
    // @public (called on the instance itself)
    addRelativeTransformListener: function( listener ) {
      var before = this.hasAncestorListenerNeed();
      
      this.relativeTransformListeners.push( listener );
      if ( before !== this.hasAncestorListenerNeed() ) {
        this.parent && this.parent.incrementTransformListenerChildren();
        
        // if we just went from "not needing to be traversed" to "needing to be traversed", mark ourselves as dirty so that we for-sure get future updates
        if ( !this.hasAncestorComputeNeed() ) {
          // TODO: can we do better than this?
          this.forceMarkTransformDirty();
        }
      }
    },
    
    // @public (called on the instance itself)
    removeRelativeTransformListener: function( listener ) {
      var before = this.hasAncestorListenerNeed();
      
      this.relativeTransformListeners.splice( _.indexOf( this.relativeTransformListeners, listener ), 1 ); // TODO: replace with a 'remove' function call
      if ( before !== this.hasAncestorListenerNeed() ) {
        this.parent && this.parent.decrementTransformListenerChildren();
      }
    },
    
    /*---------------------------------------------------------------------------*
    * Relative transform precompute flag recursive handling
    *----------------------------------------------------------------------------*/
    
    // @private: Only for descendants need, ignores 'self' need on isTransformed
    hasDescendantComputeNeed: function() {
      if ( this.isTransformed ) {
        return this.relativeChildrenPrecomputeCount > 0;
      } else {
        return this.relativeChildrenPrecomputeCount > 0 || this.relativePrecomputeCount > 0;
      }
      return ;
    },
    // @private: Only for ancestors need, ignores child need on isTransformed
    hasAncestorComputeNeed: function() {
      if ( this.isTransformed ) {
        return this.relativePrecomputeCount > 0;
      } else {
        return this.relativeChildrenPrecomputeCount > 0 || this.relativePrecomputeCount > 0;
      }
    },
    // @private
    hasSelfComputeNeed: function() {
      return this.relativePrecomputeCount > 0;
    },
    // @private (called on the ancestor of the instance with the need)
    incrementTransformPrecomputeChildren: function() {
      var before = this.hasAncestorComputeNeed();
      
      this.relativeChildrenPrecomputeCount++;
      if ( before !== this.hasAncestorComputeNeed() ) {
        assert && assert( !this.isTransformed, 'Should not be a change in need if we have the isTransformed flag' );
        
        this.parent && this.parent.incrementTransformPrecomputeChildren();
      }
    },
    // @private (called on the ancestor of the instance with the need)
    decrementTransformPrecomputeChildren: function() {
      var before = this.hasAncestorComputeNeed();
      
      this.relativeChildrenPrecomputeCount--;
      if ( before !== this.hasAncestorComputeNeed() ) {
        assert && assert( !this.isTransformed, 'Should not be a change in need if we have the isTransformed flag' );
        
        this.parent && this.parent.decrementTransformPrecomputeChildren();
      }
    },
    
    // @public (called on the instance itself)
    addRelativeTransformPrecompute: function() {
      var before = this.hasAncestorComputeNeed();
      
      this.relativePrecomputeCount++;
      if ( before !== this.hasAncestorComputeNeed() ) {
        this.parent && this.parent.incrementTransformPrecomputeChildren();
        
        // if we just went from "not needing to be traversed" to "needing to be traversed", mark ourselves as dirty so that we for-sure get future updates
        if ( !this.hasAncestorListenerNeed() ) {
          // TODO: can we do better than this?
          this.forceMarkTransformDirty();
        }
      }
    },
    
    // @public (called on the instance itself)
    removeRelativeTransformPrecompute: function() {
      var before = this.hasAncestorComputeNeed();
      
      this.relativePrecomputeCount--;
      if ( before !== this.hasAncestorComputeNeed() ) {
        this.parent && this.parent.decrementTransformPrecomputeChildren();
      }
    },
    
    /*---------------------------------------------------------------------------*
    * Relative transform handling
    *----------------------------------------------------------------------------*/
    
    // called immediately when the corresponding node has a transform change (can happen multiple times between renders)
    markTransformDirty: function() {
      if ( !this.transformDirty ) {
        this.forceMarkTransformDirty();
      }
    },
    
    forceMarkTransformDirty: function() {
      this.transformDirty = true;
      this.relativeSelfDirty = true;
      
      var frameId = this.display._frameId;
      
      // mark all ancestors with relativeChildDirtyFrame, bailing out when possible
      var instance = this.parent;
      while ( instance && instance.relativeChildDirtyFrame !== frameId ) {
        var parentInstance = instance.parent;
        var isTransformed = instance.isTransformed;
        
        // NOTE: our while loop guarantees that it wasn't frameId
        instance.relativeChildDirtyFrame = frameId;
        
        // always mark an instance without a parent (root instance!)
        if ( parentInstance === null ) {
          this.display.markTransformRootDirty( instance, isTransformed ); // passTransform depends on whether it is marked as a transform root
          break;
        } else if ( isTransformed ) {
          this.display.markTransformRootDirty( instance, true ); // passTransform true
          break;
        }
        
        instance = parentInstance;
      }
    },
    
    // updates our relativeMatrix based on any parents, and the node's current transform
    computeRelativeTransform: function() {
      var nodeMatrix = this.node.getTransform().getMatrix();
      
      if ( this.parent && !this.parent.isTransformed ) {
        // mutable form of parentMatrix * nodeMatrix
        this.relativeMatrix.set( this.parent.relativeMatrix );
        this.relativeMatrix.multiplyMatrix( nodeMatrix );
      } else {
        // we are the first in the trail transform, so we just directly copy the matrix over
        this.relativeMatrix.set( nodeMatrix );
      }
      
      // mark the frame where this transform was updated, to accelerate non-precomputed access
      this.relativeFrameId = this.display._frameId;
      this.relativeSelfDirty = false;
    },
    
    isValidationNotNeeded: function() {
      return this.hasAncestorComputeNeed() || this.relativeFrameId === this.display._frameId;
    },
    
    // Called from any place in the rendering process where we are not guaranteed to have a fresh relative transform. needs to scan up the tree, so it is
    // more expensive than precomputed transforms.
    // returns whether we had to update this transform
    validateRelativeTransform: function() {
      // if we are clean, bail out. If we have a compute "need", we will always be clean here since this is after the traversal step.
      // if we did not have a compute "need", we check whether we were already updated this frame by computeRelativeTransform
      if ( this.isValidationNotNeeded() ) {
        return;
      }
      
      // if we are not the first transform from the root, validate our parent. isTransform check prevents us from passing a transform root
      if ( this.parent && !this.parent.isTransformed ) {
        this.parent.validateRelativeTransform();
      }
      
      // validation of the parent may have changed our relativeSelfDirty flag to true, so we check now (could also have been true before)
      if ( this.relativeSelfDirty ) {
        // compute the transform, and mark us as not relative-dirty
        this.computeRelativeTransform();
        
        // mark all children now as dirty, since we had to update (marked so that other children from the one we are validating will know that they need updates)
        // if we were called from a child's validateRelativeTransform, they will now need to compute their transform
        var len = this.children.length;
        for ( var i = 0; i < len; i++ ) {
          this.children[i].relativeSelfDirty = true;
        }
      }
    },
    
    // called during the pre-repaint phase to (a) fire off all relative transform listeners that should be fired, and (b) precompute transforms were desired
    updateTransformListenersAndCompute: function( ancestorWasDirty, ancestorIsDirty, frameId, passTransform ) {
      sceneryLog && sceneryLog.transformSystem && sceneryLog.transformSystem(
        'update/compute: ' + this.toString() + ' ' + ancestorWasDirty + ' => ' + ancestorIsDirty + ( passTransform ? ' passTransform' : '' ) );
      sceneryLog && sceneryLog.transformSystem && sceneryLog.push();
      
      var len, i;
      
      if ( passTransform ) {
        // if we are passing isTransform, just apply this to the children
        len = this.children.length;
        for ( i = 0; i < len; i++ ) {
          this.children[i].updateTransformListenersAndCompute( false, false, frameId, false );
        }
      } else {
        var wasDirty = ancestorWasDirty || this.relativeSelfDirty;
        var wasSubtreeDirty = wasDirty || this.relativeChildDirtyFrame === frameId;
        var hasComputeNeed = this.hasDescendantComputeNeed();
        var hasListenerNeed = this.hasDescendantListenerNeed();
        var hasSelfComputeNeed = this.hasSelfComputeNeed();
        var hasSelfListenerNeed = this.hasSelfListenerNeed();
        
        // if our relative transform will be dirty but our parents' transform will be clean, we need to mark ourselves as dirty (so that later access can identify we are dirty).
        if ( !hasComputeNeed && wasDirty && !ancestorIsDirty ) {
          this.relativeSelfDirty = true;
        }
        
        // check if traversal isn't needed (no instances marked as having listeners or needing computation)
        // either the subtree is clean (no traversal needed for compute/listeners), or we have no compute/listener needs
        if ( !wasSubtreeDirty || ( !hasComputeNeed && !hasListenerNeed && !hasSelfComputeNeed && !hasSelfListenerNeed ) ) {
          sceneryLog && sceneryLog.transformSystem && sceneryLog.pop();
          return;
        }
        
        // if desired, compute the transform
        if ( wasDirty && ( hasComputeNeed || hasSelfComputeNeed ) ) {
          // compute this transform in the pre-repaint phase, so it is cheap when always used/
          // we update when the child-precompute count >0, since those children will need 
          this.computeRelativeTransform();
        }
        
        if ( this.transformDirty ) {
          this.transformDirty = false;
        }
        
        // no hasListenerNeed guard needed?
        this.notifyRelativeTransformListeners();
        
        // only update children if we aren't transformed (completely other context)
        if ( !this.isTransformed || passTransform ) {
          
          var isDirty = wasDirty && !( hasComputeNeed || hasSelfComputeNeed );
          
          // continue the traversal
          len = this.children.length;
          for ( i = 0; i < len; i++ ) {
            this.children[i].updateTransformListenersAndCompute( wasDirty, isDirty, frameId, false );
          }
        }
      }
      
      sceneryLog && sceneryLog.transformSystem && sceneryLog.pop();
    },
    
    notifyRelativeTransformListeners: function() {
      var len = this.relativeTransformListeners.length;
      for ( var i = 0; i < len; i++ ) {
        this.relativeTransformListeners[i]();
      }
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
        var group = this.svgGroups[i];
        if ( group.block === block ) {
          return group;
        }
      }
      return null;
    },
    
    // what instance have filters (opacity/visibility/clip) been applied up to?
    getFilterRootInstance: function() {
      if ( this.state.isBackbone || this.state.isInstanceCanvasCache || !this.parent ) {
        return this;
      } else {
        return this.parent.getFilterRootInstance();
      }
    },
    
    // what instance transforms have been applied up to?
    getTransformRootInstance: function() {
      if ( this.state.isTransformed || !this.parent ) {
        return this;
      } else {
        return this.parent.getTransformRootInstance();
      }
    },
    
    onStateCreation: function() {
      // attach listeners to our node
      this.node.onStatic( 'transform', this.nodeTransformListener );
      
      if ( !this.state.isSharedCanvasCachePlaceholder ) {
        this.node.onStatic( 'childInserted', this.childInsertedListener );
        this.node.onStatic( 'childRemoved', this.childRemovedListener );
      }
    },
    
    onStateRemoval: function() {
      this.node.offStatic( 'transform', this.nodeTransformListener );
      
      if ( !this.state.isSharedCanvasCachePlaceholder ) {
        this.node.offStatic( 'childInserted', this.childInsertedListener );
        this.node.offStatic( 'childRemoved', this.childRemovedListener );
      }
    },
    
    // clean up listeners and garbage, so that we can be recycled (or pooled)
    dispose: function() {
      sceneryLog && sceneryLog.Instance && sceneryLog.Instance( 'dispose ' + this.toString() );
      
      assert && assert( this.active, 'Seems like we tried to dispose this Instance twice, it is not active' );
      
      this.active = false;
      
      // order is somewhat important
      this.groupDrawable && this.groupDrawable.dispose();
      this.sharedCacheDrawable && this.sharedCacheDrawable.dispose();
      this.selfDrawable && this.selfDrawable.dispose();
      
      for ( var i = 0; i < this.children.length; i++ ) {
        this.children[i].dispose();
      }
      
      // we don't originally add in the listener if we are stateless
      if ( !this.isStateless() ) {
        this.onStateRemoval();
      }
      
      this.node.removeInstance( this );
      
      // release our reference to a shared cache if applicable, and dispose if there are no other references
      if ( this.sharedCacheInstance ) {
        this.sharedCacheInstance.externalReferenceCount--;
        if ( this.sharedCacheInstance.externalReferenceCount === 0 ) {
          delete this.display._sharedCanvasInstances[this.node.getId()];
          this.sharedCacheInstance.dispose();
        }
      }
      
      // clean our variables out to release memory
      this.cleanInstance( null, null, null );
      
      this.freeToPool();
    },
    
    audit: function( frameId, allowValidationNotNeededChecks ) {
      // get the relative matrix, computed to be up-to-date, and ignores any flags/counts so we can check whether our state is consistent
      function currentRelativeMatrix( instance ) {
        var resultMatrix = Matrix3.dirtyFromPool();
        var nodeMatrix = instance.node.getTransform().getMatrix();
        
        if ( instance.parent && !instance.parent.isTransformed ) {
          // mutable form of parentMatrix * nodeMatrix
          resultMatrix.set( currentRelativeMatrix( instance.parent ) );
          resultMatrix.multiplyMatrix( nodeMatrix );
        } else {
          // we are the first in the trail transform, so we just directly copy the matrix over
          resultMatrix.set( nodeMatrix );
        }
        
        return resultMatrix;
      }
      
      function hasRelativeSelfDirty( instance ) {
        // if validation isn't needed, act like nothing is dirty (matching our validate behavior)
        if ( allowValidationNotNeededChecks && instance.isValidationNotNeeded() ) {
          return false;
        }
        
        return instance.relativeSelfDirty || ( instance.parent && hasRelativeSelfDirty( instance.parent ) );
      }
      
      if ( assertSlow ) {
        if ( frameId === undefined ) {
          frameId = this.display._frameId;
        }
        
        assertSlow( this.state,
                    'State is required for all display instances' );
        
        assertSlow( ( this.firstDrawable === null ) === ( this.lastDrawable === null ),
                    'First/last drawables need to both be null or non-null' );
        
        assertSlow( ( !this.state.isBackbone && !this.state.isSharedCanvasCachePlaceholder ) || this.groupDrawable,
                    'If we are a backbone or shared cache, we need to have a groupDrawable reference' );
        
        assertSlow( !this.state.isSharedCanvasCachePlaceholder || !this.node.isPainted() || this.selfDrawable,
                    'We need to have a selfDrawable if we are painted and not a shared cache' );
        
        assertSlow( ( !this.state.isTransformed && !this.state.isCanvasCache ) || this.groupDrawable,
                    'We need to have a groupDrawable if we are a backbone or any type of canvas cache' );
        
        assertSlow( !this.state.isSharedCanvasCachePlaceholder || this.sharedCacheDrawable,
                    'We need to have a sharedCacheDrawable if we are a shared cache' );
        
        assertSlow( this.state.isTransformed === this.isTransformed,
                    'isTransformed should match' );
        
        assertSlow( !this.parent || this.isTransformed || ( this.relativeChildDirtyFrame !== frameId ) || ( this.parent.relativeChildDirtyFrame === frameId ),
                    'If we have a parent, we need to hold the invariant this.relativeChildDirtyFrame => parent.relativeChildDirtyFrame' );
        
        // count verification for invariants
        var notifyRelativeCount = 0;
        var precomputeRelativeCount = 0;
        for ( var i = 0; i < this.children.length; i++ ) {
          var childInstance = this.children[i];
          
          childInstance.audit( frameId, allowValidationNotNeededChecks );
          
          if ( childInstance.hasAncestorListenerNeed() ) {
            notifyRelativeCount++;
          }
          if ( childInstance.hasAncestorComputeNeed() ) {
            precomputeRelativeCount++;
          }
        }
        assertSlow( notifyRelativeCount === this.relativeChildrenListenersCount,
                    'Relative listener count invariant' );
        assertSlow( precomputeRelativeCount === this.relativeChildrenPrecomputeCount,
                    'Relative precompute count invariant' );
        
        if ( !hasRelativeSelfDirty( this ) ) {
          var matrix = currentRelativeMatrix( this );
          assertSlow( matrix.equals( this.relativeMatrix ), 'If there is no relativeSelfDirty flag set here or in our ancestors, our relativeMatrix should be up-to-date' );
        }
      }
    },
    
    toString: function() {
      return 'I#' + this.id + '/' + ( this.node ? this.node.id : '-' );
    }
  } );
  
  // object pooling
  /* jshint -W064 */
  Poolable( Instance, {
    constructorDuplicateFactory: function( pool ) {
      return function( display, trail ) {
        if ( pool.length ) {
          sceneryLog && sceneryLog.Instance && sceneryLog.Instance( 'new from pool' );
          return pool.pop().initialize( display, trail );
        } else {
          sceneryLog && sceneryLog.Instance && sceneryLog.Instance( 'new from constructor' );
          return new Instance( display, trail );
        }
      };
    }
  } );
  
  return Instance;
} );
