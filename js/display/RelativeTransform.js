// Copyright 2014-2016, University of Colorado Boulder

/**
 * RelativeTransform is a component of an Instance. It is responsible for tracking changes to "relative" transforms, and
 * computing them in an efficient manner.
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
 *                                  (add|removeListener)
 * - every frame before repainting: precompute relative transforms on instances where we know this is required
 *                                  (add|removePrecompute)
 * - any time during repainting:    provide an efficient way to lazily compute relative transforms when needed
 *
 * This is done by first having one step in the pre-repaint phase that traverses the tree where necessary, notifying
 * relative transform listeners, and precomputing relative transforms when they have changed (and precomputation is
 * requested). This traversal leaves metadata on the instances so that we can (fairly) efficiently force relative
 * transform "validation" any time afterwards that makes sure the matrix property is up-to-date.
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
 * matrix                                  [---------------------------------] (transform on root applies to
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

  var cleanArray = require( 'PHET_CORE/cleanArray' );
  var inherit = require( 'PHET_CORE/inherit' );
  var Matrix3 = require( 'DOT/Matrix3' );
  var scenery = require( 'SCENERY/scenery' );

  function RelativeTransform( instance ) {
    this.instance = instance;
  }

  scenery.register( 'RelativeTransform', RelativeTransform );

  inherit( Object, RelativeTransform, {
    /**
     * Responsible for initialization and cleaning of this. If the parameters are both null, we'll want to clean our
     * external references (like Instance does).
     *
     * @param {Display|null} display
     * @param {Trail|null} trail
     * @returns {RelativeTransform} - Returns this, to allow chaining.
     */
    initialize: function( display, trail ) {
      this.display = display;
      this.trail = trail;
      this.node = trail && trail.lastNode();

      // properties relevant to the node's direct transform
      this.transformDirty = true; // whether the node's transform has changed (until the pre-repaint phase)
      this.nodeTransformListener = this.nodeTransformListener || this.onNodeTransformDirty.bind( this );

      // the actual cached transform to the root
      this.matrix = this.matrix || Matrix3.identity();

      // whether our matrix is dirty
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
      this.relativeChildDirtyFrame = display ? display._frameId : 0;

      // will be notified in pre-repaint phase that our relative transform has changed (but not computed by default)
      //OHTWO TODO: should we rely on listeners removing themselves?
      this.relativeTransformListeners = cleanArray( this.relativeTransformListeners );

      return this; // allow chaining
    },

    get parent() {
      return this.instance.parent ? this.instance.parent.relativeTransform : null;
    },

    addInstance: function( instance ) {
      if ( instance.stateless ) {
        assert && assert( !instance.relativeTransform.hasAncestorListenerNeed(),
          'We only track changes properly if stateless instances do not have needs' );
        assert && assert( !instance.relativeTransform.hasAncestorComputeNeed(),
          'We only track changes properly if stateless instances do not have needs' );
      }
      else {
        if ( instance.relativeTransform.hasAncestorListenerNeed() ) {
          this.incrementTransformListenerChildren();
        }
        if ( instance.relativeTransform.hasAncestorComputeNeed() ) {
          this.incrementTransformPrecomputeChildren();
        }
      }

      // mark the instance's transform as dirty, so that it will be reachable in the pre-repaint traversal pass
      instance.relativeTransform.forceMarkTransformDirty();
    },

    removeInstance: function( instance ) {
      if ( instance.relativeTransform.hasAncestorListenerNeed() ) {
        this.decrementTransformListenerChildren();
      }
      if ( instance.relativeTransform.hasAncestorComputeNeed() ) {
        this.decrementTransformPrecomputeChildren();
      }
    },

    attachNodeListeners: function() {
      this.node.onStatic( 'transform', this.nodeTransformListener );
    },

    detachNodeListeners: function() {
      this.node.offStatic( 'transform', this.nodeTransformListener );
    },

    /*---------------------------------------------------------------------------*
     * Relative transform listener count recursive handling
     *----------------------------------------------------------------------------*/

    // @private: Only for descendants need, ignores 'self' need on isTransformed
    hasDescendantListenerNeed: function() {
      if ( this.instance.isTransformed ) {
        return this.relativeChildrenListenersCount > 0;
      }
      else {
        return this.relativeChildrenListenersCount > 0 || this.relativeTransformListeners.length > 0;
      }
    },
    // @private: Only for ancestors need, ignores child need on isTransformed
    hasAncestorListenerNeed: function() {
      if ( this.instance.isTransformed ) {
        return this.relativeTransformListeners.length > 0;
      }
      else {
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
        assert && assert( !this.instance.isTransformed, 'Should not be a change in need if we have the isTransformed flag' );

        this.parent && this.parent.incrementTransformListenerChildren();
      }
    },
    // @private (called on the ancestor of the instance with the need)
    decrementTransformListenerChildren: function() {
      var before = this.hasAncestorListenerNeed();

      this.relativeChildrenListenersCount--;
      if ( before !== this.hasAncestorListenerNeed() ) {
        assert && assert( !this.instance.isTransformed, 'Should not be a change in need if we have the isTransformed flag' );

        this.parent && this.parent.decrementTransformListenerChildren();
      }
    },

    // @public (called on the instance itself)
    addListener: function( listener ) {
      var before = this.hasAncestorListenerNeed();

      this.relativeTransformListeners.push( listener );
      if ( before !== this.hasAncestorListenerNeed() ) {
        this.parent && this.parent.incrementTransformListenerChildren();

        // if we just went from "not needing to be traversed" to "needing to be traversed", mark ourselves as dirty so
        // that we for-sure get future updates
        if ( !this.hasAncestorComputeNeed() ) {
          // TODO: can we do better than this?
          this.forceMarkTransformDirty();
        }
      }
    },

    // @public (called on the instance itself)
    removeListener: function( listener ) {
      var before = this.hasAncestorListenerNeed();

      // TODO: replace with a 'remove' function call
      this.relativeTransformListeners.splice( _.indexOf( this.relativeTransformListeners, listener ), 1 );
      if ( before !== this.hasAncestorListenerNeed() ) {
        this.parent && this.parent.decrementTransformListenerChildren();
      }
    },

    /*---------------------------------------------------------------------------*
     * Relative transform precompute flag recursive handling
     *----------------------------------------------------------------------------*/

    // @private: Only for descendants need, ignores 'self' need on isTransformed
    hasDescendantComputeNeed: function() {
      if ( this.instance.isTransformed ) {
        return this.relativeChildrenPrecomputeCount > 0;
      }
      else {
        return this.relativeChildrenPrecomputeCount > 0 || this.relativePrecomputeCount > 0;
      }
    },
    // @private: Only for ancestors need, ignores child need on isTransformed
    hasAncestorComputeNeed: function() {
      if ( this.instance.isTransformed ) {
        return this.relativePrecomputeCount > 0;
      }
      else {
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
        assert && assert( !this.instance.isTransformed, 'Should not be a change in need if we have the isTransformed flag' );

        this.parent && this.parent.incrementTransformPrecomputeChildren();
      }
    },
    // @private (called on the ancestor of the instance with the need)
    decrementTransformPrecomputeChildren: function() {
      var before = this.hasAncestorComputeNeed();

      this.relativeChildrenPrecomputeCount--;
      if ( before !== this.hasAncestorComputeNeed() ) {
        assert && assert( !this.instance.isTransformed, 'Should not be a change in need if we have the isTransformed flag' );

        this.parent && this.parent.decrementTransformPrecomputeChildren();
      }
    },

    // @public (called on the instance itself)
    addPrecompute: function() {
      var before = this.hasAncestorComputeNeed();

      this.relativePrecomputeCount++;
      if ( before !== this.hasAncestorComputeNeed() ) {
        this.parent && this.parent.incrementTransformPrecomputeChildren();

        // if we just went from "not needing to be traversed" to "needing to be traversed", mark ourselves as dirty so
        // that we for-sure get future updates
        if ( !this.hasAncestorListenerNeed() ) {
          // TODO: can we do better than this?
          this.forceMarkTransformDirty();
        }
      }
    },

    // @public (called on the instance itself)
    removePrecompute: function() {
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
    onNodeTransformDirty: function() {
      if ( !this.transformDirty ) {
        this.forceMarkTransformDirty();
      }
    },

    forceMarkTransformDirty: function() {
      this.transformDirty = true;
      this.relativeSelfDirty = true;

      var frameId = this.display._frameId;

      // mark all ancestors with relativeChildDirtyFrame, bailing out when possible
      var instance = this.instance.parent;
      while ( instance && instance.relativeTransform.relativeChildDirtyFrame !== frameId ) {
        var parentInstance = instance.parent;
        var isTransformed = instance.isTransformed;

        // NOTE: our while loop guarantees that it wasn't frameId
        instance.relativeTransform.relativeChildDirtyFrame = frameId;

        // always mark an instance without a parent (root instance!)
        if ( parentInstance === null ) {
          // passTransform depends on whether it is marked as a transform root
          this.display.markTransformRootDirty( instance, isTransformed );
          break;
        }
        else if ( isTransformed ) {
          this.display.markTransformRootDirty( instance, true ); // passTransform true
          break;
        }

        instance = parentInstance;
      }
    },

    // @private, updates our matrix based on any parents, and the node's current transform
    computeRelativeTransform: function() {
      var nodeMatrix = this.node.getMatrix();

      if ( this.instance.parent && !this.instance.parent.isTransformed ) {
        // mutable form of parentMatrix * nodeMatrix
        this.matrix.set( this.parent.matrix );
        this.matrix.multiplyMatrix( nodeMatrix );
      }
      else {
        // we are the first in the trail transform, so we just directly copy the matrix over
        this.matrix.set( nodeMatrix );
      }

      // mark the frame where this transform was updated, to accelerate non-precomputed access
      this.relativeFrameId = this.display._frameId;
      this.relativeSelfDirty = false;
    },

    // @public
    isValidationNotNeeded: function() {
      return this.hasAncestorComputeNeed() || this.relativeFrameId === this.display._frameId;
    },

    // Called from any place in the rendering process where we are not guaranteed to have a fresh relative transform.
    // needs to scan up the tree, so it is more expensive than precomputed transforms.
    // @returns Whether we had to update this transform
    validate: function() {
      // if we are clean, bail out. If we have a compute "need", we will always be clean here since this is after the
      // traversal step. If we did not have a compute "need", we check whether we were already updated this frame by
      // computeRelativeTransform.
      if ( this.isValidationNotNeeded() ) {
        return;
      }

      // if we are not the first transform from the root, validate our parent. isTransform check prevents us from
      // passing a transform root.
      if ( this.instance.parent && !this.instance.parent.isTransformed ) {
        this.parent.validate();
      }

      // validation of the parent may have changed our relativeSelfDirty flag to true, so we check now (could also have
      // been true before)
      if ( this.relativeSelfDirty ) {
        // compute the transform, and mark us as not relative-dirty
        this.computeRelativeTransform();

        // mark all children now as dirty, since we had to update (marked so that other children from the one we are
        // validating will know that they need updates)
        // if we were called from a child's validate(), they will now need to compute their transform
        var len = this.instance.children.length;
        for ( var i = 0; i < len; i++ ) {
          this.instance.children[ i ].relativeTransform.relativeSelfDirty = true;
        }
      }
    },

    // called during the pre-repaint phase to (a) fire off all relative transform listeners that should be fired, and
    // (b) precompute transforms were desired
    updateTransformListenersAndCompute: function( ancestorWasDirty, ancestorIsDirty, frameId, passTransform ) {
      sceneryLog && sceneryLog.RelativeTransform && sceneryLog.RelativeTransform(
        'update/compute: ' + this.toString() + ' ' + ancestorWasDirty + ' => ' + ancestorIsDirty +
        ( passTransform ? ' passTransform' : '' ) );
      sceneryLog && sceneryLog.RelativeTransform && sceneryLog.push();

      var len;
      var i;

      if ( passTransform ) {
        // if we are passing isTransform, just apply this to the children
        len = this.instance.children.length;
        for ( i = 0; i < len; i++ ) {
          this.instance.children[ i ].relativeTransform.updateTransformListenersAndCompute( false, false, frameId, false );
        }
      }
      else {
        var wasDirty = ancestorWasDirty || this.relativeSelfDirty;
        var wasSubtreeDirty = wasDirty || this.relativeChildDirtyFrame === frameId;
        var hasComputeNeed = this.hasDescendantComputeNeed();
        var hasListenerNeed = this.hasDescendantListenerNeed();
        var hasSelfComputeNeed = this.hasSelfComputeNeed();
        var hasSelfListenerNeed = this.hasSelfListenerNeed();

        // if our relative transform will be dirty but our parents' transform will be clean, we need to mark ourselves
        // as dirty (so that later access can identify we are dirty).
        if ( !hasComputeNeed && wasDirty && !ancestorIsDirty ) {
          this.relativeSelfDirty = true;
        }

        // check if traversal isn't needed (no instances marked as having listeners or needing computation)
        // either the subtree is clean (no traversal needed for compute/listeners), or we have no compute/listener needs
        if ( !wasSubtreeDirty || ( !hasComputeNeed && !hasListenerNeed && !hasSelfComputeNeed && !hasSelfListenerNeed ) ) {
          sceneryLog && sceneryLog.RelativeTransform && sceneryLog.pop();
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
        if ( !this.instance.isTransformed || passTransform ) {

          var isDirty = wasDirty && !( hasComputeNeed || hasSelfComputeNeed );

          // continue the traversal
          len = this.instance.children.length;
          for ( i = 0; i < len; i++ ) {
            this.instance.children[ i ].relativeTransform.updateTransformListenersAndCompute( wasDirty, isDirty, frameId, false );
          }
        }
      }

      sceneryLog && sceneryLog.RelativeTransform && sceneryLog.pop();
    },

    // @private
    notifyRelativeTransformListeners: function() {
      var len = this.relativeTransformListeners.length;
      for ( var i = 0; i < len; i++ ) {
        this.relativeTransformListeners[ i ]();
      }
    },

    audit: function( frameId, allowValidationNotNeededChecks ) {
      // get the relative matrix, computed to be up-to-date, and ignores any flags/counts so we can check whether our
      // state is consistent
      function currentRelativeMatrix( instance ) {
        var resultMatrix = Matrix3.dirtyFromPool();
        var nodeMatrix = instance.node.getMatrix();

        if ( !instance.parent ) {
          // if our instance has no parent, ignore its transform
          resultMatrix.set( Matrix3.IDENTITY );
        }
        else if ( !instance.parent.isTransformed ) {
          // mutable form of parentMatrix * nodeMatrix
          resultMatrix.set( currentRelativeMatrix( instance.parent ) );
          resultMatrix.multiplyMatrix( nodeMatrix );
        }
        else {
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
        // count verification for invariants
        var notifyRelativeCount = 0;
        var precomputeRelativeCount = 0;
        for ( var i = 0; i < this.instance.children.length; i++ ) {
          var childInstance = this.instance.children[ i ];

          if ( childInstance.relativeTransform.hasAncestorListenerNeed() ) {
            notifyRelativeCount++;
          }
          if ( childInstance.relativeTransform.hasAncestorComputeNeed() ) {
            precomputeRelativeCount++;
          }
        }
        assertSlow( notifyRelativeCount === this.relativeChildrenListenersCount,
          'Relative listener count invariant' );
        assertSlow( precomputeRelativeCount === this.relativeChildrenPrecomputeCount,
          'Relative precompute count invariant' );

        assertSlow( !this.parent || this.instance.isTransformed || ( this.relativeChildDirtyFrame !== frameId ) ||
                    ( this.parent.relativeChildDirtyFrame === frameId ),
          'If we have a parent, we need to hold the invariant ' +
          'this.relativeChildDirtyFrame => parent.relativeChildDirtyFrame' );

        // Since we check to see if something is not dirty, we need to handle this when we are actually reporting
        // what is dirty. See https://github.com/phetsims/scenery/issues/512
        if ( !allowValidationNotNeededChecks && !hasRelativeSelfDirty( this ) ) {
          var matrix = currentRelativeMatrix( this );
          assertSlow( matrix.equals( this.matrix ), 'If there is no relativeSelfDirty flag set here or in our' +
                                                    ' ancestors, our matrix should be up-to-date' );
        }
      }
    }
  } );

  return RelativeTransform;
} );
