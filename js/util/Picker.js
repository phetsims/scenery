// Copyright 2016, University of Colorado Boulder

/**
 * Sub-component of a Node that handles pickability and hit testing.
 *
 * A "listener equivalent" is either the existence of at least one input listener, or pickable:true. Nodes with
 * listener equivalents will basically try to hit-test ALL descendants that aren't invisible or pickable:false.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var Bounds2 = require( 'DOT/Bounds2' );
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  var Vector2 = require( 'DOT/Vector2' );

  /**
   * @constructor
   *
   * @param {Node} node - Our node.
   */
  function Picker( node ) {
    // @private {Node} - Our node reference (does not change)
    this.node = node;

    // @private {boolean} - Whether our last-known state would have us be pruned by hit-test searches.
    //                      Should be equal to node.pickable === false || node.isVisible() === false.
    //                      Updated synchronously.
    this.selfPruned = false;

    // @private {boolean} - Whether our last-known state would have us not prune descendant subtrees for the lack of
    //                      listener equivalents (whether we have a listener equivalent).
    //                      Should be equal to node.pickable === true || node._inputListeners.length > 0.
    //                      Updated synchronously.
    this.selfInclusive = false;

    // @private {boolean} - Whether our subtree can be pruned IF no ancestor (or us) has selfInclusive as true.
    //                      Equivalent to:
    //                        node.pickable === false ||
    //                        !node.isVisible() ||
    //                        ( node.pickable !== true && subtreePickableCount === 0 )
    this.subtreePrunable = true;

    // @private {number} - Count designed to be non-zero when there is a listener equivalent in this node's subtree.
    //                     Effectively the sum of #inputListeners + (1?isPickable:true) + #childrenWithNonZeroCount.
    //                     Notably, it ignores children who are guaranteed to be pruned (selfPruned:true).
    this.subtreePickableCount = 0;

    // NOTE: We need "inclusive" and "exclusive" bounds to ideally be separate, so that they can be cached
    // independently. It's possible for one trail to have an ancestor with pickable:true (inclusive) while another
    // trail has no ancestors that make the search inclusive. This would introduce "thrashing" in the older version,
    // where it would continuously compute one or the other. Here, both versions can be stored.

    // @private {Bounds2} - Bounds to be used for pruning mouse hit tests when an ancestor has a listener equivalent.
    //                      Updated lazily, while the dirty flag is updated synchronously.
    this.mouseInclusiveBounds = Bounds2.NOTHING.copy();

    // @private {Bounds2} - Bounds to be used for pruning mouse hit tests when ancestors have NO listener equivalent.
    //                      Updated lazily, while the dirty flag is updated synchronously.
    this.mouseExclusiveBounds = Bounds2.NOTHING.copy();

    // @private {Bounds2} - Bounds to be used for pruning touch hit tests when an ancestor has a listener equivalent.
    //                      Updated lazily, while the dirty flag is updated synchronously.
    this.touchInclusiveBounds = Bounds2.NOTHING.copy();

    // @private {Bounds2} - Bounds to be used for pruning touch hit tests when ancestors have NO listener equivalent.
    //                      Updated lazily, while the dirty flag is updated synchronously.
    this.touchExclusiveBounds = Bounds2.NOTHING.copy();

    // @private {boolean} - Dirty flags, one for each Bounds.
    this.mouseInclusiveDirty = true;
    this.mouseExclusiveDirty = true;
    this.touchInclusiveDirty = true;
    this.touchExclusiveDirty = true;

    // @private {Vector2} - Used to minimize garbage created in the hit-testing process
    this.scratchVector = new Vector2( 0, 0 );
  }

  scenery.register( 'Picker', Picker );

  inherit( Object, Picker, {

    /*
     * Return a trail to the top node (if any, otherwise null) whose self-rendered area contains the
     * point (in parent coordinates).
     * @public
     *
     * @param {Vector2} point
     * @param {boolean} useMouse - Whether mouse-specific customizations (and acceleration) applies
     * @param {boolean} useTouch - Whether touch-specific customizations (and acceleration) applies
     * @returns {Trail|null}
     */
    hitTest: function( point, useMouse, useTouch ) {
      assert && assert( point, 'trailUnderPointer requires a point' );

      sceneryLog && sceneryLog.hitTest && sceneryLog.hitTest( '-------------- ' + this.node.constructor.name + '#' + this.node.id );

      var isBaseInclusive = this.selfInclusive;

      // Validate the bounds that we will be using for hit acceleration. This should validate all bounds that could be
      // hit by recursiveHitTest.
      if ( useMouse ) {
        if ( isBaseInclusive ) {
          this.validateMouseInclusive();
        }
        else {
          this.validateMouseExclusive();
        }
      }
      else if ( useTouch ) {
        if ( isBaseInclusive ) {
          this.validateTouchInclusive();
        }
        else {
          this.validateTouchExclusive();
        }
      }
      else {
        this.node.validateBounds();
      }

      // Kick off recursive handling, with isInclusive:false
      return this.recursiveHitTest( point, useMouse, useTouch, false );
    },

    recursiveHitTest: function( point, useMouse, useTouch, isInclusive ) {
      isInclusive = isInclusive || this.selfInclusive;

      // If we are selfPruned, ignore this node and its subtree (invisible or pickable:false).
      // If the search is NOT inclusive (no listener equivalent), also ignore this subtree if subtreePrunable is true.
      if ( this.selfPruned || ( !isInclusive && this.subtreePrunable ) ) {
        sceneryLog && sceneryLog.hitTest && sceneryLog.hitTest( this.node.constructor.name + '#' + this.node.id +
          ' pruned ' + ( this.selfPruned ? '(self)' : '(subtree)' ) );
        return null;
      }

      // Validation should have already been done in hitTest(), we just need to grab the accelerated bounds.
      var pruningBounds;
      if ( useMouse ) {
        pruningBounds = isInclusive ? this.mouseInclusiveBounds : this.mouseExclusiveBounds;
        assert && assert( isInclusive ? !this.mouseInclusiveDirty : !this.mouseExclusiveDirty );
      }
      else if ( useTouch ) {
        pruningBounds = isInclusive ? this.touchInclusiveBounds : this.touchExclusiveBounds;
        assert && assert( isInclusive ? !this.touchInclusiveDirty : !this.touchExclusiveDirty );
      }
      else {
        pruningBounds = this.node._bounds;
        assert && assert( !this.node._boundsDirty );
      }

      // Bail quickly if our point is not inside the bounds for the subtree.
      if ( !pruningBounds.containsPoint( point ) ) {
        sceneryLog && sceneryLog.hitTest && sceneryLog.hitTest( this.node.constructor.name + '#' + this.node.id + ' pruned: ' + ( useMouse ? 'mouse' : ( useTouch ? 'touch' : 'regular' ) ) );
        return null; // not in our bounds, so this point can't possibly be contained
      }

      // Transform the point in the local coordinate frame, so we can test it with the clipArea/children
      var localPoint = this.node._transform.getInverse().multiplyVector2( this.scratchVector.set( point ) );

      // If our point is outside of the local-coordinate clipping area, there should be no hit.
      if ( this.node.hasClipArea() && !this.node._clipArea.containsPoint( localPoint ) ) {
        sceneryLog && sceneryLog.hitTest && sceneryLog.hitTest( this.node.constructor.name + '#' + this.node.id + ' out of clip area' );
        return null;
      }

      sceneryLog && sceneryLog.hitTest && sceneryLog.hitTest( this.node.constructor.name + '#' + this.node.id );

      // Check children before our "self", since the children are rendered on top.
      // Manual iteration here so we can return directly, and so we can iterate backwards (last node is in front).
      for ( var i = this.node._children.length - 1; i >= 0; i-- ) {
        var child = this.node._children[ i ];

        sceneryLog && sceneryLog.hitTest && sceneryLog.push();
        var childHit = child._picker.recursiveHitTest( localPoint, useMouse, useTouch, isInclusive );
        sceneryLog && sceneryLog.hitTest && sceneryLog.pop();

        // If there was a hit, immediately add our node to the start of the Trail (will recursively build the Trail).
        if ( childHit ) {
          return childHit.addAncestor( this.node, i );
        }
      }

      // Tests for mouse and touch hit areas before testing containsPointSelf
      if ( useMouse && this.node._mouseArea ) {
        sceneryLog && sceneryLog.hitTest && sceneryLog.hitTest( this.node.constructor.name + '#' + this.node.id + ' mouse area hit' );
        // NOTE: both Bounds2 and Shape have containsPoint! We use both here!
        return this.node._mouseArea.containsPoint( localPoint ) ? new scenery.Trail( this.node ) : null;
      }
      if ( useTouch && this.node._touchArea ) {
        sceneryLog && sceneryLog.hitTest && sceneryLog.hitTest( this.node.constructor.name + '#' + this.node.id + ' touch area hit' );
        // NOTE: both Bounds2 and Shape have containsPoint! We use both here!
        return this.node._touchArea.containsPoint( localPoint ) ? new scenery.Trail( this.node ) : null;
      }

      // Didn't hit our children, so check ourself as a last resort. Check our selfBounds first, so we can potentially
      // avoid hit-testing the actual object (which may be more expensive).
      if ( this.node.selfBounds.containsPoint( localPoint ) ) {
        if ( this.node.containsPointSelf( localPoint ) ) {
          sceneryLog && sceneryLog.hitTest && sceneryLog.hitTest( this.node.constructor.name + '#' + this.node.id + ' self hit' );
          return new scenery.Trail( this.node );
        }
      }

      // No hit
      return null;
    },

    /**
     * Recursively sets dirty flags to true. If the andExclusive parameter is false, only the "inclusive" flags
     * are set to dirty.
     * @private
     *
     * @param {boolean} andExclusive
     * @param {boolean} [ignoreSelfDirty] - If true, will invalidate parents even if we were dirty.
     */
    invalidate: function( andExclusive, ignoreSelfDirty ) {
      assert && assert( typeof andExclusive === 'boolean' );

      // Track whether a 'dirty' flag was changed from false=>true (or if ignoreSelfDirty is passed).
      var wasNotDirty = !!ignoreSelfDirty || !this.mouseInclusiveDirty || !this.touchInclusiveDirty;

      this.mouseInclusiveDirty = true;
      this.touchInclusiveDirty = true;
      if ( andExclusive ) {
        wasNotDirty = wasNotDirty || !this.mouseExclusiveDirty || !this.touchExclusiveDirty;
        this.mouseExclusiveDirty = true;
        this.touchExclusiveDirty = true;
      }

      // If we are selfPruned (or if we were already fully dirty), there should be no reason to call this on our
      // parents. If we are selfPruned, we are guaranteed to not be visited in a search by our parents, so changes
      // that make this picker dirty should NOT affect our parents' pickers values.
      if ( !this.selfPruned && wasNotDirty ) {
        var parents = this.node._parents;
        for ( var i = 0; i < parents.length; i++ ) {
          parents[ i ]._picker.invalidate( andExclusive || this.selfInclusive, false );
        }
      }
    },

    /**
     * Computes the mouseInclusiveBounds for this picker (if dirty), and recursively validates it for all non-pruned
     * children.
     * @private
     *
     * NOTE: For the future, consider sharing more code with related functions. I tried this, and it made things look
     * more complicated (and probably slower), so I've kept some duplication. If a change is made to this function,
     * please check the other validate* methods to see if they also need a change.
     */
    validateMouseInclusive: function() {
      if ( !this.mouseInclusiveDirty ) {
        return;
      }

      this.mouseInclusiveBounds.set( this.node.selfBounds );

      var children = this.node._children;
      for ( var i = 0; i < children.length; i++ ) {
        var childPicker = children[ i ]._picker;

        // Since we are "inclusive", we don't care about subtreePrunable (we won't prune for that). Only check
        // if pruning is force (selfPruned).
        if ( !childPicker.selfPruned ) {
          childPicker.validateMouseInclusive();
          this.mouseInclusiveBounds.includeBounds( childPicker.mouseInclusiveBounds );
        }
      }

      // Include mouseArea (if applicable), exclude outside clipArea (if applicable), and transform to the parent
      // coordinate frame.
      this.applyAreasAndTransform( this.mouseInclusiveBounds, this.node._mouseArea );

      this.mouseInclusiveDirty = false;
    },

    /**
     * Computes the mouseExclusiveBounds for this picker (if dirty), and recursively validates the mouse-related bounds
     * for all non-pruned children.
     * @private
     *
     * Notably, if a picker is selfInclusive, we will switch to validating mouseInclusiveBounds for its subtree, as this
     * is what the hit-testing will use.
     *
     * NOTE: For the future, consider sharing more code with related functions. I tried this, and it made things look
     * more complicated (and probably slower), so I've kept some duplication. If a change is made to this function,
     * please check the other validate* methods to see if they also need a change.
     */
    validateMouseExclusive: function() {
      if ( !this.mouseExclusiveDirty ) {
        return;
      }

      this.mouseExclusiveBounds.set( this.node.selfBounds );

      var children = this.node._children;
      for ( var i = 0; i < children.length; i++ ) {
        var childPicker = children[ i ]._picker;

        // Since we are not "inclusive", we will prune the search if subtreePrunable is true.
        if ( !childPicker.subtreePrunable ) {
          // If our child is selfInclusive, we need to switch to the "inclusive" validation.
          if ( childPicker.selfInclusive ) {
            childPicker.validateMouseInclusive();
            this.mouseExclusiveBounds.includeBounds( childPicker.mouseInclusiveBounds );
          }
          // Otherwise, keep with the exclusive validation.
          else {
            childPicker.validateMouseExclusive();
            this.mouseExclusiveBounds.includeBounds( childPicker.mouseExclusiveBounds );
          }
        }
      }

      // Include mouseArea (if applicable), exclude outside clipArea (if applicable), and transform to the parent
      // coordinate frame.
      this.applyAreasAndTransform( this.mouseExclusiveBounds, this.node._mouseArea );

      this.mouseExclusiveDirty = false;
    },

    /**
     * Computes the touchInclusiveBounds for this picker (if dirty), and recursively validates it for all non-pruned
     * children.
     * @private
     *
     * NOTE: For the future, consider sharing more code with related functions. I tried this, and it made things look
     * more complicated (and probably slower), so I've kept some duplication. If a change is made to this function,
     * please check the other validate* methods to see if they also need a change.
     */
    validateTouchInclusive: function() {
      if ( !this.touchInclusiveDirty ) {
        return;
      }

      this.touchInclusiveBounds.set( this.node.selfBounds );

      var children = this.node._children;
      for ( var i = 0; i < children.length; i++ ) {
        var childPicker = children[ i ]._picker;

        // Since we are "inclusive", we don't care about subtreePrunable (we won't prune for that). Only check
        // if pruning is force (selfPruned).
        if ( !childPicker.selfPruned ) {
          childPicker.validateTouchInclusive();
          this.touchInclusiveBounds.includeBounds( childPicker.touchInclusiveBounds );
        }
      }

      // Include touchArea (if applicable), exclude outside clipArea (if applicable), and transform to the parent
      // coordinate frame.
      this.applyAreasAndTransform( this.touchInclusiveBounds, this.node._touchArea );

      this.touchInclusiveDirty = false;
    },

    /**
     * Computes the touchExclusiveBounds for this picker (if dirty), and recursively validates the touch-related bounds
     * for all non-pruned children.
     * @private
     *
     * Notably, if a picker is selfInclusive, we will switch to validating touchInclusiveBounds for its subtree, as this
     * is what the hit-testing will use.
     *
     * NOTE: For the future, consider sharing more code with related functions. I tried this, and it made things look
     * more complicated (and probably slower), so I've kept some duplication. If a change is made to this function,
     * please check the other validate* methods to see if they also need a change.
     */
    validateTouchExclusive: function() {
      if ( !this.touchExclusiveDirty ) {
        return;
      }

      this.touchExclusiveBounds.set( this.node.selfBounds );

      var children = this.node._children;
      for ( var i = 0; i < children.length; i++ ) {
        var childPicker = children[ i ]._picker;

        // Since we are not "inclusive", we will prune the search if subtreePrunable is true.
        if ( !childPicker.subtreePrunable ) {
          // If our child is selfInclusive, we need to switch to the "inclusive" validation.
          if ( childPicker.selfInclusive ) {
            childPicker.validateTouchInclusive();
            this.touchExclusiveBounds.includeBounds( childPicker.touchInclusiveBounds );
          }
          // Otherwise, keep with the exclusive validation.
          else {
            childPicker.validateTouchExclusive();
            this.touchExclusiveBounds.includeBounds( childPicker.touchExclusiveBounds );
          }
        }
      }

      // Include touchArea (if applicable), exclude outside clipArea (if applicable), and transform to the parent
      // coordinate frame.
      this.applyAreasAndTransform( this.touchExclusiveBounds, this.node._touchArea );

      this.touchExclusiveDirty = false;
    },

    /**
     * Include pointer areas (if applicable), exclude bounds outside the clip area (if applicable), and transform
     * into the parent coordinate frame. Mutates the bounds provided.
     * @private
     *
     * Meant to be called by the validation methods, as this part is the same for every validation that is done.
     *
     * @param {Bounds2} mutableBounds - The bounds to be mutated (e.g. mouseExclusiveBounds).
     * @param {Bounds2|Shape|null} pointerArea - A mouseArea/touchArea that should be included in the search.
     */
    applyAreasAndTransform: function( mutableBounds, pointerArea ) {
      // do this before the transformation to the parent coordinate frame (the mouseArea is in the local coordinate frame)
      if ( pointerArea ) {
        // we accept either Bounds2, or a Shape (in which case, we take the Shape's bounds)
        mutableBounds.includeBounds( pointerArea.isBounds ? pointerArea : pointerArea.bounds );
      }

      if ( this.node.hasClipArea() ) {
        var clipBounds = this.node._clipArea.bounds;
        // exclude areas outside of the clipping area's bounds (for efficiency)
        // Uses Bounds2.constrainBounds, but inlined to prevent https://github.com/phetsims/projectile-motion/issues/155
        mutableBounds.minX = Math.max( mutableBounds.minX, clipBounds.minX );
        mutableBounds.minY = Math.max( mutableBounds.minY, clipBounds.minY );
        mutableBounds.maxX = Math.min( mutableBounds.maxX, clipBounds.maxX );
        mutableBounds.maxY = Math.min( mutableBounds.maxY, clipBounds.maxY );
      }

      // transform it to the parent coordinate frame
      this.node.transformBoundsFromLocalToParent( mutableBounds );
    },

    /**
     * Called from Node when a child is inserted.
     * @public (scenery-internal)
     *
     * NOTE: The child may not be fully added when this is called. Don't audit, or assume that calls to the Node would
     * indicate the parent-child relationship.
     *
     * @param {Node} childNode - Our picker node's new child node.
     */
    onInsertChild: function( childNode ) {
      // If the child is selfPruned, we don't have to update any metadata.
      if ( !childNode._picker.selfPruned ) {
        var hasPickable = childNode._picker.subtreePickableCount > 0;

        // If it has a non-zero subtreePickableCount, we'll need to increment our own count by 1.
        if ( hasPickable ) {
          this.changePickableCount( 1 );
        }

        // If it has a subtreePickableCount of zero, it would be pruned by "exclusive" searches, so we only need to
        // invalidate the "inclusive" bounds.
        this.invalidate( hasPickable, true );
      }
    },

    /**
     * Called from Node when a child is removed.
     * @public (scenery-internal)
     *
     * NOTE: The child may not be fully removed when this is called. Don't audit, or assume that calls to the Node would
     * indicate the parent-child relationship.
     *
     * @param {Node} childNode - Our picker node's child that will be removed.
     */
    onRemoveChild: function( childNode ) {
      // If the child is selfPruned, we don't have to update any metadata.
      if ( !childNode._picker.selfPruned ) {
        var hasPickable = childNode._picker.subtreePickableCount > 0;

        // If it has a non-zero subtreePickableCount, we'll need to decrement our own count by 1.
        if ( hasPickable ) {
          this.changePickableCount( -1 );
        }

        // If it has a subtreePickableCount of zero, it would be pruned by "exclusive" searches, so we only need to
        // invalidate the "inclusive" bounds.
        this.invalidate( hasPickable, true );
      }
    },

    /**
     * Called from Node when an input listener is added to our node.
     * @public (scenery-internal)
     */
    onAddInputListener: function() {
      // Update flags that depend on listener count
      this.checkSelfInclusive();
      this.checkSubtreePrunable();

      // Update our pickable count, since it includes a count of how many input listeners we have.
      this.changePickableCount( 1 ); // NOTE: this should also trigger invalidation of mouse/touch bounds

      if ( assertSlow ) { this.audit(); }
    },

    /**
     * Called from Node when an input listener is removed from our node.
     * @public (scenery-internal)
     */
    onRemoveInputListener: function() {
      // Update flags that depend on listener count
      this.checkSelfInclusive();
      this.checkSubtreePrunable();

      // Update our pickable count, since it includes a count of how many input listeners we have.
      this.changePickableCount( -1 ); // NOTE: this should also trigger invalidation of mouse/touch bounds

      if ( assertSlow ) { this.audit(); }
    },

    /**
     * Called when the 'pickable' value of our Node is changed.
     * @public (scenery-internal)
     *
     * @param {boolean|null} oldPickable - Old value
     * @param {boolean|null} pickable - New value
     */
    onPickableChange: function( oldPickable, pickable ) {
      // Update flags that depend on our pickable setting.
      this.checkSelfPruned();
      this.checkSelfInclusive();
      this.checkSubtreePrunable();

      // Compute our pickable count change (pickable:true counts for 1)
      var change = ( oldPickable === true ? -1 : 0 ) + ( pickable === true ? 1 : 0 );

      if ( change ) {
        this.changePickableCount( change );
      }

      if ( assertSlow ) { this.audit(); }
    },

    /**
     * Called when the visibility of our Node is changed.
     * @public (scenery-internal)
     */
    onVisibilityChange: function() {
      // Update flags that depend on our visibility.
      this.checkSelfPruned();
      this.checkSubtreePrunable();
    },

    /**
     * Called when the mouseArea of the Node is changed.
     * @public (scenery-internal)
     */
    onMouseAreaChange: function() {
      // Bounds can depend on the mouseArea, so we'll invalidate those.
      // TODO: Consider bounds invalidation that only does the 'mouse' flags, since we don't need to invalidate touches.
      this.invalidate( true );
    },

    /**
     * Called when the mouseArea of the Node is changed.
     * @public (scenery-internal)
     */
    onTouchAreaChange: function() {
      // Bounds can depend on the touchArea, so we'll invalidate those.
      // TODO: Consider bounds invalidation that only does the 'touch' flags, since we don't need to invalidate mice.
      this.invalidate( true );
    },

    /**
     * Called when the transform of the Node is changed.
     * @public (scenery-internal)
     */
    onTransformChange: function() {
      // Can affect our bounds
      this.invalidate( true );
    },

    /**
     * Called when the transform of the Node is changed.
     * @public (scenery-internal)
     */
    onSelfBoundsDirty: function() {
      // Can affect our bounds
      this.invalidate( true );
    },

    /**
     * Called when the transform of the Node is changed.
     * @public (scenery-internal)
     */
    onClipAreaChange: function() {
      // Can affect our bounds.
      this.invalidate( true );
    },

    /**
     * Check to see if we are 'selfPruned', and update the value. If it changed, we'll need to notify our parents.
     * @private
     *
     * Note that the prunability "pickable:false" or "invisible" won't affect our computed bounds, so we don't
     * invalidate ourself.
     */
    checkSelfPruned: function() {
      var selfPruned = this.node._pickable === false || !this.node.isVisible();
      if ( this.selfPruned !== selfPruned ) {
        this.selfPruned = selfPruned;

        // Notify parents
        var parents = this.node._parents;
        for ( var i = 0; i < parents.length; i++ ) {
          var picker = parents[ i ]._picker;

          // If we have an input listener/pickable:true in our subtree, we'll need to invalidate exclusive bounds also,
          // and we'll want to update the pickable count of our parent.
          if ( this.subtreePickableCount > 0 ) {
            picker.invalidate( true, true );
            picker.changePickableCount( this.selfPruned ? -1 : 1 );
          }
          // If we have nothing in our subtree that would force a visit, we only need to invalidate the "inclusive"
          // bounds.
          else {
            picker.invalidate( false, true );
          }
        }
      }
    },

    /**
     * Check to see if we are 'selfInclusive', and update the value. If it changed, we'll need to invalidate ourself.
     * @private
     */
    checkSelfInclusive: function() {
      var selfInclusive = this.node._pickable === true || this.node._inputListeners.length > 0;
      if ( this.selfInclusive !== selfInclusive ) {
        this.selfInclusive = selfInclusive;

        // Our dirty flag handling for both inclusive and exclusive depend on this value.
        this.invalidate( true, true );
      }
    },

    /**
     * Update our 'subtreePrunable' flag.
     * @private
     */
    checkSubtreePrunable: function() {
      var subtreePrunable = this.node._pickable === false ||
                            !this.node.isVisible() ||
                            ( this.node._pickable !== true && this.subtreePickableCount === 0 );

      if ( this.subtreePrunable !== subtreePrunable ) {
        this.subtreePrunable = subtreePrunable;

        // Our dirty flag handling for both inclusive and exclusive depend on this value.
        this.invalidate( true, true );
      }
    },

    /**
     * Propagate the pickable count change down to our ancestors.
     * @private
     *
     * @param {number} n - The delta of how many pickable counts have been added/removed
     */
    changePickableCount: function( n ) {
      if ( n === 0 ) {
        return;
      }

      // Switching between 0 and 1 matters, since we then need to update the counts of our parents.
      var wasZero = this.subtreePickableCount === 0;
      this.subtreePickableCount += n;
      var isZero = this.subtreePickableCount === 0;

      // Our subtreePrunable value depends on our pickable count, make sure it gets updated.
      this.checkSubtreePrunable();

      assert && assert( this.subtreePickableCount >= 0, 'subtree pickable count should be guaranteed to be >= 0' );

      if ( !this.selfPruned && wasZero !== isZero ) {
        // Update our parents if our count changed (AND if it matters, i.e. we aren't selfPruned).
        var len = this.node._parents.length;
        for ( var i = 0; i < len; i++ ) {
          this.node._parents[ i ]._picker.changePickableCount( wasZero ? 1 : -1 );
        }
      }
    },

    /**
     * Runs a number of consistency tests when assertSlow is enabled. Verifies most conditions, and helps to catch
     * bugs earlier when they are initially triggered.
     * @public (scenery-internal)
     */
    audit: function() {
      if ( assertSlow ) {
        var self = this;

        _.each( this.node._children, function( node ) {
          node._picker.audit();
        } );

        var expectedSelfPruned = this.node.pickable === false || !this.node.isVisible();
        var expectedSelfInclusive = this.node.pickable === true || this.node._inputListeners.length > 0;
        var expectedSubtreePrunable = this.node.pickable === false ||
                                      !this.node.isVisible() ||
                                      ( this.node.pickable !== true && this.subtreePickableCount === 0 );
        var expectedSubtreePickableCount = this.node._inputListeners.length +
                                           ( this.node._pickable === true ? 1 : 0 ) +
                                           _.filter( this.node._children, function( child ) {
                                             return !child._picker.selfPruned && child._picker.subtreePickableCount > 0;
                                           } ).length;

        assertSlow( this.selfPruned === expectedSelfPruned, 'selfPruned mismatch' );
        assertSlow( this.selfInclusive === expectedSelfInclusive, 'selfInclusive mismatch' );
        assertSlow( this.subtreePrunable === expectedSubtreePrunable, 'subtreePrunable mismatch' );
        assertSlow( this.subtreePickableCount === expectedSubtreePickableCount, 'subtreePickableCount mismatch' );

        _.each( this.node._parents, function( parent ) {
          var parentPicker = parent._picker;
          var childPicker = self;

          if ( !parentPicker.mouseInclusiveDirty ) {
            assertSlow( childPicker.selfPruned || !childPicker.mouseInclusiveDirty );
          }

          if ( !parentPicker.mouseExclusiveDirty ) {
            if ( childPicker.selfInclusive ) {
              assertSlow( childPicker.selfPruned || !childPicker.mouseInclusiveDirty );
            }
            else {
              assertSlow( childPicker.selfPruned || childPicker.subtreePrunable || !childPicker.mouseExclusiveDirty );
            }
          }

          if ( !parentPicker.touchInclusiveDirty ) {
            assertSlow( childPicker.selfPruned || !childPicker.touchInclusiveDirty );
          }

          if ( !parentPicker.touchExclusiveDirty ) {
            if ( childPicker.selfInclusive ) {
              assertSlow( childPicker.selfPruned || !childPicker.touchInclusiveDirty );
            }
            else {
              assertSlow( childPicker.selfPruned || childPicker.subtreePrunable || !childPicker.touchExclusiveDirty );
            }
          }
        } );
      }
    }
  } );

  return Picker;
} );
