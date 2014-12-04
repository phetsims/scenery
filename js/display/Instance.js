// Copyright 2002-2014, University of Colorado Boulder


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
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var PoolableMixin = require( 'PHET_CORE/PoolableMixin' );
  var cleanArray = require( 'PHET_CORE/cleanArray' );
  var scenery = require( 'SCENERY/scenery' );
  var ChangeInterval = require( 'SCENERY/display/ChangeInterval' );
  var Drawable = require( 'SCENERY/display/Drawable' );
  var Renderer = require( 'SCENERY/display/Renderer' );
  var RelativeTransform = require( 'SCENERY/display/RelativeTransform' );
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

      this.relativeTransform = ( this.relativeTransform || new RelativeTransform() ).initialize( this, display, trail );

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
      this.visibilityListener = this.visibilityListener || this.onVisibilityChange.bind( this );

      this.cleanInstance( display, trail );

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
    cleanInstance: function( display, trail ) {
      this.display = display;
      this.trail = trail;
      this.node = trail ? trail.lastNode() : null;
      this.parent = null; // will be set as needed
      this.oldParent = null; // set when removed from us, so that we can easily reattach it when necessary
      // NOTE: reliance on correct order after syncTree by at least SVGBlock/SVGGroup
      this.children = cleanArray( this.children ); // Array[Instance].
      this.sharedCacheInstance = null; // reference to a shared cache instance (different than a child)

      // Child instances are pushed to here when their node is removed from our node. we don't immediately dispose,
      // since it may be added back.
      this.instanceRemovalCheckList = cleanArray( this.instanceRemovalCheckList );

      this.selfDrawable = null;
      this.groupDrawable = null; // e.g. backbone or non-shared cache
      this.sharedCacheDrawable = null; // our drawable if we are a shared cache

      // references into the linked list of drawables (null if nothing is drawable under this)
      this.firstDrawable = null;
      this.lastDrawable = null;

      // references into the linked list of drawables (excludes any group drawables handling)
      this.firstInnerDrawable = null;
      this.lastInnerDrawable = null;

      // references that will be filled in with syncTree
      if ( this.state ) {
        // NOTE: assumes that we aren't reusing states across instances
        this.state.freeToPool();
      }
      this.state = null;
      this.isTransformed = false; // whether this instance creates a new "root" for the relative trail transforms

      this.svgGroups = []; // list of SVG groups associated with this display instance

      this.cleanSyncTreeResults();

      this.relativeTransform.clean();
    },

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
    },

    // @public
    baseSyncTree: function() {
      sceneryLog && sceneryLog.Instance && sceneryLog.Instance( '-------- START baseSyncTree ' + this.toString() + ' --------' );
      this.syncTree( scenery.RenderState.createRootState( this.node ) );
      sceneryLog && sceneryLog.Instance && sceneryLog.Instance( '-------- END baseSyncTree ' + this.toString() + ' --------' );
      this.cleanSyncTreeResults();
    },

    // updates the internal {RenderState}, and fully synchronizes the instance subtree
    /*OHTWO TODO:
     * Pruning:
     *   - If children haven't changed, skip instance add/move/remove
     *   - If RenderState hasn't changed AND there are no render/instance/stitch changes below us, EXIT (whenever we are
     *     assured children don't need sync)
     * Return linked-list of alternating changed (add/remove) and keep sections of drawables, for processing with
     * backbones/canvas caches.
     *
     * Other notes:
     *
     *    Traversal hits every child if parent render state changed. Otherwise, only hits children who have (or has
     *                            descendants who have) potential render state changes. If we haven't hit a "change" yet
     *                            from ancestors, don't re-evaluate any render states (UNLESS renderer summary changed!)
     *      Need recursive flag for "render state needs reevaluation" / "child render state needs reevaluation
     *        Don't flag changes when they won't actually change the "current" render state!!!
     *        Changes in renderer summary (subtree combined) can cause changes in the render state
     *    OK for traversal to return "no change", doesn't specify any drawables changes
     */
    syncTree: function( state ) {
      sceneryLog && sceneryLog.Instance && sceneryLog.Instance( 'syncTree ' + this.toString() + ' ' + state.toString() +
                                                                ( this.isStateless() ? ' (stateless)' : '' ) );
      sceneryLog && sceneryLog.Instance && sceneryLog.push();

      if ( sceneryLog && scenery.isLoggingPerformance() ) {
        this.display.perfSyncTreeCount++;
      }

      assert && assert( state && state.isSharedCanvasCachePlaceholder !== undefined, 'RenderState duck-typing instanceof' );

      // may access isTransformed up to root to determine relative trails
      assert && assert( !this.parent || !this.parent.isStateless(), 'We should not have a stateless parent instance' );

      var oldState = this.state;
      var wasStateless = !oldState;

      assert && assert( wasStateless || oldState.isInstanceCompatibleWith( state ),
        'Attempt to update to a render state that is not compatible with this instance\'s current state' );
      // no need to overwrite, should always be the same
      assert && assert( wasStateless || oldState.isTransformed === state.isTransformed );
      assert && assert( !wasStateless || this.children.length === 0, 'We should not have child instances on an instance without state' );

      this.state = state;
      this.isTransformed = state.isTransformed;

      if ( wasStateless ) {
        // If we are a transform root, notify the display that we are dirty. We'll be validated when it's at that phase
        // at the next updateDisplay().
        //OHTWO TODO: when else do we have to call this?
        if ( this.isTransformed ) {
          this.display.markTransformRootDirty( this, true );
        }

        this.onStateCreation();
      }

      if ( state.isSharedCanvasCachePlaceholder ) {
        this.sharedSyncTree( state );
      }
      else {
        // mark fully-removed instances for disposal, and initialize child instances if we were stateless
        this.prepareChildInstances( state, oldState );

        var oldFirstDrawable = this.firstDrawable;
        var oldLastDrawable = this.lastDrawable;
        var oldFirstInnerDrawable = this.firstInnerDrawable;
        var oldLastInnerDrawable = this.lastInnerDrawable;

        // properly handle our self and children
        this.localSyncTree( state, oldState );

        if ( assertSlow ) {
          // before and after first/last drawables (inside any potential group drawable)
          this.auditChangeIntervals( oldFirstInnerDrawable, oldLastInnerDrawable, this.firstInnerDrawable, this.lastInnerDrawable );
        }

        // apply any group changes necessary
        this.groupSyncTree( state, oldState );

        if ( assertSlow ) {
          // before and after first/last drawables (outside of any potential group drawable)
          this.auditChangeIntervals( oldFirstDrawable, oldLastDrawable, this.firstDrawable, this.lastDrawable );
        }
      }

      if ( oldState && oldState !== this.state ) {
        oldState.freeToPool();
      }

      sceneryLog && sceneryLog.Instance && sceneryLog.pop();
    },

    localSyncTree: function( state, oldState ) {
      var frameId = this.display._frameId;

      var selfChanged = this.updateSelfDrawable( state, oldState );

      // local variables, since we can't overwrite our instance properties yet
      var firstDrawable = this.selfDrawable; // possibly null
      var currentDrawable = firstDrawable; // possibly null

      assert && assert( this.firstChangeInterval === null &&
                        this.lastChangeInterval === null,
        'sanity checks that cleanSyncTreeResults were called' );

      var firstChangeInterval = null;
      if ( selfChanged ) {
        firstChangeInterval = ChangeInterval.newForDisplay( null, null, this.display );
      }
      var currentChangeInterval = firstChangeInterval;
      var lastUnchangedDrawable = selfChanged ? null : this.selfDrawable;

      for ( var i = 0; i < this.children.length; i++ ) {
        var childInstance = this.children[i];
        var childState = state.getStateForDescendant( childInstance.node );
        childInstance = this.updateChildInstanceIfIncompatible( childInstance, childState, i );

        // grab the first/last drawables before our syncTree
        // var oldChildFirstDrawable = childInstance.firstDrawable;
        // var oldChildLastDrawable = childInstance.lastDrawable;

        // sync the tree
        childInstance.syncTree( childState );

        //OHTWO TODO: only strip out invisible Canvas drawables, while leaving SVG (since we can more efficiently hide
        // SVG trees, memory-wise)
        if ( childInstance.node.isVisible() ) {
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

        var wasIncluded = childInstance.stitchChangeIncluded;
        var isIncluded = childInstance.node.isVisible();
        childInstance.stitchChangeIncluded = isIncluded;

        // check for forcing full change-interval on child
        if ( childInstance.stitchChangeFrame === frameId ) {
          // e.g. it was added, moved, or had visibility changes. requires full change interval
          childInstance.firstChangeInterval = childInstance.lastChangeInterval = ChangeInterval.newForDisplay( null, null, this.display );
        }
        else {
          assert && assert( wasIncluded === isIncluded, 'If we do not have stitchChangeFrame activated, our inclusion should not have changed' );
        }

        var firstChildChangeInterval = childInstance.firstChangeInterval;
        var isBeforeOpen = currentChangeInterval && currentChangeInterval.drawableAfter === null;
        var isAfterOpen = firstChildChangeInterval && firstChildChangeInterval.drawableBefore === null;
        var needsBridge = childInstance.stitchChangeBefore === frameId && !isBeforeOpen && !isAfterOpen;

        if ( needsBridge ) {
          var bridge = ChangeInterval.newForDisplay( lastUnchangedDrawable, null, this.display );
          if ( currentChangeInterval ) {
            currentChangeInterval.nextChangeInterval = bridge;
          }
          currentChangeInterval = bridge;
          firstChangeInterval = firstChangeInterval || currentChangeInterval; // store if it is the first
          isBeforeOpen = true;
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
      }

      /* jshint -W018 */ // it's really the easiest way to compare if two things (casted to booleans) are the same?
      assert && assert( !!firstChangeInterval === !!currentChangeInterval,
        'Presence of first and current change intervals should be equal' );

      // Check to see if we are emptied and marked as changed (but without change intervals). This should imply we have
      // no children (and thus no stitchChangeBefore / stitchChangeAfter to use), so we'll want to create a change
      // interval to cover our entire range.
      if ( !firstChangeInterval && this.stitchChangeOnChildren === this.display._frameId ) {
        assert && assert( this.children.length === 0 );
        firstChangeInterval = currentChangeInterval = ChangeInterval.newForDisplay( null, null, this.display );
      }

      // store our results
      // NOTE: these may get overwritten with the group change intervals (in that case, groupSyncTree will read from these)
      this.firstChangeInterval = firstChangeInterval;
      this.lastChangeInterval = currentChangeInterval;

      // NOTE: these may get overwritten with the group drawable (in that case, groupSyncTree will read from these)
      this.firstDrawable = this.firstInnerDrawable = firstDrawable;
      this.lastDrawable = this.lastInnerDrawable = currentDrawable; // either null, or the drawable itself

      // drawable range checks
      if ( assertSlow ) {
        var firstDrawableCheck = null;
        for ( var j = 0; j < this.children.length; j++ ) {
          if ( this.children[j].node.isVisible() && this.children[j].firstDrawable ) {
            firstDrawableCheck = this.children[j].firstDrawable;
            break;
          }
        }
        if ( this.selfDrawable ) {
          firstDrawableCheck = this.selfDrawable;
        }

        var lastDrawableCheck = this.selfDrawable;
        for ( var k = this.children.length - 1; k >= 0; k-- ) {
          if ( this.children[k].node.isVisible() && this.children[k].lastDrawable ) {
            lastDrawableCheck = this.children[k].lastDrawable;
            break;
          }
        }

        assertSlow( firstDrawableCheck === this.firstDrawable );
        assertSlow( lastDrawableCheck === this.lastDrawable );
      }
    },

    // returns whether the selfDrawable changed
    updateSelfDrawable: function( state, oldState ) {
      if ( this.node.isPainted() ) {
        var selfRenderer = state.selfRenderer; // our new self renderer bitmask

        // bitwise trick, since only one of Canvas/SVG/DOM/WebGL/etc. flags will be chosen, and bitmaskRendererArea is
        // the mask for those flags. In English, "Is the current selfDrawable compatible with our selfRenderer (if any),
        // or do we need to create a selfDrawable?"
        //OHTWO TODO: For Canvas, we won't care about anything else for the drawable, but for DOM we care about the
        // force-acceleration flag! That's stripped out here.
        if ( !this.selfDrawable || ( ( this.selfDrawable.renderer & selfRenderer & Renderer.bitmaskRendererArea ) === 0 ) ) {
          if ( this.selfDrawable ) {
            // scrap the previous selfDrawable, we need to create one with a different renderer.
            this.selfDrawable.markForDisposal( this.display );
          }

          this.selfDrawable = Renderer.createSelfDrawable( this, this.node, selfRenderer );

          return true;
        }
      }
      else {
        assert && assert( this.selfDrawable === null, 'Non-painted nodes should not have a selfDrawable' );
      }

      return false;
    },

    // returns the up-to-date instance
    updateChildInstanceIfIncompatible: function( childInstance, childState, index ) {
      // see if we need to rebuild the instance tree due to an incompatible render state
      if ( !childInstance.isStateless() && !childState.isInstanceCompatibleWith( childInstance.state ) ) {
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
        var replacementInstance = Instance.createFromPool( this.display, this.trail.copy().addDescendant( childInstance.node, index ) );
        this.replaceInstanceWithIndex( childInstance, replacementInstance, index );
        return replacementInstance;
      }
      else {
        return childInstance;
      }
    },

    groupSyncTree: function( state, oldState ) {
      var groupRenderer = state.groupRenderer;
      assert && assert( ( state.isBackbone ? 1 : 0 ) +
                        ( state.isInstanceCanvasCache ? 1 : 0 ) +
                        ( state.isSharedCanvasCacheSelf ? 1 : 0 ) === ( groupRenderer ? 1 : 0 ),
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
        if ( this.groupDrawable ) {
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

        if ( state.isBackbone ) {
          if ( groupChanged ) {
            this.groupDrawable = scenery.BackboneDrawable.createFromPool( this.display, this, this.getTransformRootInstance(), groupRenderer, state.isDisplayRoot );

            if ( this.isTransformed ) {
              this.display.markTransformRootDirty( this, true );
            }
          }

          if ( this.firstChangeInterval ) {
            this.groupDrawable.stitch( this.firstDrawable, this.lastDrawable, this.firstChangeInterval, this.lastChangeInterval );
          }
        }
        else if ( state.isInstanceCanvasCache ) {
          if ( groupChanged ) {
            this.groupDrawable = scenery.InlineCanvasCacheDrawable.createFromPool( groupRenderer, this );
          }
          if ( this.firstChangeInterval ) {
            this.groupDrawable.stitch( this.firstDrawable, this.lastDrawable, this.firstChangeInterval, this.lastChangeInterval );
          }
        }
        else if ( state.isSharedCanvasCacheSelf ) {
          if ( groupChanged ) {
            this.groupDrawable = scenery.CanvasBlock.createFromPool( groupRenderer, this );
          }
          //OHTWO TODO: restitch here??? implement it
        }

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

    sharedSyncTree: function( state ) {
      this.ensureSharedCacheInitialized();

      var sharedCacheRenderer = state.sharedCacheRenderer;

      if ( !this.sharedCacheDrawable || this.sharedCacheDrawable.renderer !== sharedCacheRenderer ) {
        //OHTWO TODO: mark everything as changed (big change interval)

        if ( this.sharedCacheDrawable ) {
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

    prepareChildInstances: function( state, oldState ) {
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
    },

    ensureSharedCacheInitialized: function() {
      // we only need to initialize this shared cache reference once
      if ( !this.sharedCacheInstance ) {
        var instanceKey = this.node.getId();
        // TODO: have this abstracted away in the Display?
        this.sharedCacheInstance = this.display._sharedCanvasInstances[instanceKey];

        // TODO: increment reference counting?
        if ( !this.sharedCacheInstance ) {
          this.sharedCacheInstance = Instance.createFromPool( this.display, new scenery.Trail( this.node ) );
          this.sharedCacheInstance.syncTree( scenery.RenderState.createSharedCacheState( this.node ) );
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

    // whether we don't have an associated RenderState attached. If we are stateless, we won't have children, and won't
    // have listeners attached to our node yet.
    isStateless: function() {
      return !this.state;
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
          'Instance insertion bounds check for index ' + index + ' with previous children length ' +
          this.children.length );

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

      this.relativeTransform.insertInstance( instance, index );
    },

    removeInstance: function( instance ) {
      return this.removeInstanceWithIndex( instance, _.indexOf( this.children, instance ) );
    },

    removeInstanceWithIndex: function( instance, index ) {
      assert && assert( instance instanceof Instance );
      assert && assert( index >= 0 && index < this.children.length,
          'Instance removal bounds check for index ' + index + ' with previous children length ' +
          this.children.length );

      var frameId = this.display._frameId;

      // mark it as changed during this frame, so that we can properly set the change interval
      instance.stitchChangeFrame = frameId;
      this.stitchChangeOnChildren = frameId;

      // mark neighbors so that we can add a change interval for our removal area
      if ( index - 1 >= 0 ) {
        this.children[index - 1].stitchChangeAfter = frameId;
      }
      if ( index + 1 < this.children.length ) {
        this.children[index + 1].stitchChangeBefore = frameId;
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

      this.relativeTransform.removeInstanceWithIndex( instance, index );
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
        if ( instances[i].oldParent === this ) {
          return instances[i];
        }
      }
      return null;
    },

    // event callback for Node's 'childInserted' event, used to track children
    onChildInserted: function( childNode, index ) {
      sceneryLog && sceneryLog.Instance && sceneryLog.Instance(
          'inserting child node ' + childNode.constructor.name + '#' + childNode.id + ' into ' + this.toString() );
      sceneryLog && sceneryLog.Instance && sceneryLog.push();

      assert && assert( !this.isStateless(), 'If we are stateless, we should not receive these notifications' );

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
        instance = Instance.createFromPool( this.display, this.trail.copy().addDescendant( childNode, index ) );
        sceneryLog && sceneryLog.Instance && sceneryLog.pop();
      }

      this.insertInstance( instance, index );

      sceneryLog && sceneryLog.Instance && sceneryLog.pop();
    },

    // event callback for Node's 'childRemoved' event, used to track children
    onChildRemoved: function( childNode, index ) {
      sceneryLog && sceneryLog.Instance && sceneryLog.Instance(
          'removing child node ' + childNode.constructor.name + '#' + childNode.id + ' from ' + this.toString() );
      sceneryLog && sceneryLog.Instance && sceneryLog.push();

      assert && assert( !this.isStateless(), 'If we are stateless, we should not receive these notifications' );
      assert && assert( this.children[index].node === childNode, 'Ensure that our instance matches up' );

      var instance = this.findChildInstanceOnNode( childNode );
      assert && assert( instance !== null, 'We should always have a reference to a removed instance' );

      instance.addRemoveCounter -= 1;
      assert && assert( instance.addRemoveCounter === -1 );

      // track the removed instance here. if it doesn't get added back, this will be the only reference we have (we'll
      // need to dispose it)
      this.instanceRemovalCheckList.push( instance );

      this.removeInstanceWithIndex( instance, index );

      sceneryLog && sceneryLog.Instance && sceneryLog.pop();
    },

    // event callback for Node's 'visibility' event, used to notify about stitch changes
    onVisibilityChange: function() {
      assert && assert( !this.isStateless(), 'If we are stateless, we should not receive these notifications' );

      // for now, just mark which frame we were changed for our change interval
      this.stitchChangeFrame = this.display._frameId;

      //OHTWO TODO: mark as needing to not be pruned for syncTree
    },

    getDescendantCount: function() {
      var count = this.children.length;
      for ( var i = 0; i < this.children.length; i++ ) {
        count += this.children[i].getDescendantCount();
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
      }
      else {
        return this.parent.getFilterRootInstance();
      }
    },

    // what instance transforms have been applied up to?
    getTransformRootInstance: function() {
      if ( this.state.isTransformed || !this.parent ) {
        return this;
      }
      else {
        return this.parent.getTransformRootInstance();
      }
    },

    onStateCreation: function() {
      // attach listeners to our node
      this.relativeTransform.onStateCreation();

      if ( !this.state.isSharedCanvasCachePlaceholder ) {
        this.node.onStatic( 'childInserted', this.childInsertedListener );
        this.node.onStatic( 'childRemoved', this.childRemovedListener );
        this.node.onStatic( 'visibility', this.visibilityListener );
      }
    },

    onStateRemoval: function() {
      this.relativeTransform.onStateRemoval();

      if ( !this.state.isSharedCanvasCachePlaceholder ) {
        this.node.offStatic( 'childInserted', this.childInsertedListener );
        this.node.offStatic( 'childRemoved', this.childRemovedListener );
        this.node.offStatic( 'visibility', this.visibilityListener );
      }
    },

    // clean up listeners and garbage, so that we can be recycled (or pooled)
    dispose: function() {
      sceneryLog && sceneryLog.Instance && sceneryLog.Instance( 'dispose ' + this.toString() );

      assert && assert( this.active, 'Seems like we tried to dispose this Instance twice, it is not active' );

      this.active = false;

      // order is somewhat important
      this.groupDrawable && this.groupDrawable.disposeImmediately( this.display );
      this.sharedCacheDrawable && this.sharedCacheDrawable.disposeImmediately( this.display );
      this.selfDrawable && this.selfDrawable.disposeImmediately( this.display );

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
      this.cleanInstance( null, null );

      this.freeToPool();
    },

    audit: function( frameId, allowValidationNotNeededChecks ) {
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

        // validate the subtree
        for ( var i = 0; i < this.children.length; i++ ) {
          var childInstance = this.children[i];

          childInstance.audit( frameId, allowValidationNotNeededChecks );
        }

        this.relativeTransform.audit( frameId, allowValidationNotNeededChecks );
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
  /* jshint -W064 */
  PoolableMixin( Instance, {
    constructorDuplicateFactory: function( pool ) {
      return function( display, trail ) {
        if ( pool.length ) {
          sceneryLog && sceneryLog.Instance && sceneryLog.Instance( 'new from pool' );
          return pool.pop().initialize( display, trail );
        }
        else {
          sceneryLog && sceneryLog.Instance && sceneryLog.Instance( 'new from constructor' );
          return new Instance( display, trail );
        }
      };
    }
  } );

  return Instance;
} );
