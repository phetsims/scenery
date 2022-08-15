// Copyright 2016-2022, University of Colorado Boulder

/**
 * Sub-component of a Node that handles pickability and hit testing.
 *
 * A "listener equivalent" is either the existence of at least one input listener, or pickable:true. Nodes with
 * listener equivalents will basically try to hit-test ALL descendants that aren't invisible or pickable:false.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Bounds2 from '../../../dot/js/Bounds2.js';
import Vector2 from '../../../dot/js/Vector2.js';
import { Node, scenery, Trail } from '../imports.js';
import { Shape } from '../../../kite/js/imports.js';

export default class Picker {

  // Our node
  private readonly node: Node;

  // Whether our last-known state would have us be pruned by hit-test searches. Should be equal to
  // node.pickable === false || node.isVisible() === false. Updated synchronously.
  private selfPruned: boolean;

  // Whether our last-known state would have us not prune descendant subtrees for the lack of listener equivalents
  // (whether we have a listener equivalent). Should be equal to
  // node.pickable === true || node._inputListeners.length > 0. Updated synchronously.
  private selfInclusive: boolean;

  // Whether our subtree can be pruned IF no ancestor (or us) has selfInclusive as true. Equivalent to:
  // node.pickable === false || !node.isVisible() || ( node.pickable !== true && subtreePickableCount === 0 )
  private subtreePrunable: boolean;

  // Count designed to be non-zero when there is a listener equivalent in this node's subtree. Effectively the sum of
  // #inputListeners + (1?isPickable:true) + #childrenWithNonZeroCount. Notably, it ignores children who are guaranteed
  // to be pruned (selfPruned:true).
  private subtreePickableCount: number;

  // NOTE: We need "inclusive" and "exclusive" bounds to ideally be separate, so that they can be cached
  // independently. It's possible for one trail to have an ancestor with pickable:true (inclusive) while another
  // trail has no ancestors that make the search inclusive. This would introduce "thrashing" in the older version,
  // where it would continuously compute one or the other. Here, both versions can be stored.

  // Bounds to be used for pruning mouse hit tests when an ancestor has a listener equivalent. Updated lazily, while
  // the dirty flag is updated synchronously.
  private mouseInclusiveBounds: Bounds2;

  // Bounds to be used for pruning mouse hit tests when ancestors have NO listener equivalent. Updated lazily, while
  // the dirty flag is updated synchronously.
  private mouseExclusiveBounds: Bounds2;

  // Bounds to be used for pruning touch hit tests when an ancestor has a listener equivalent. Updated lazily, while
  // the dirty flag is updated synchronously.
  private touchInclusiveBounds: Bounds2;

  // Bounds to be used for pruning touch hit tests when ancestors have NO listener equivalent. Updated lazily, while
  // the dirty flag is updated synchronously.
  private touchExclusiveBounds: Bounds2;

  // Dirty flags, one for each Bounds.
  private mouseInclusiveDirty: boolean;
  private mouseExclusiveDirty: boolean;
  private touchInclusiveDirty: boolean;
  private touchExclusiveDirty: boolean;

  // Used to minimize garbage created in the hit-testing process
  private scratchVector: Vector2;

  public constructor( node: Node ) {
    this.node = node;
    this.selfPruned = false;
    this.selfInclusive = false;
    this.subtreePrunable = true;
    this.subtreePickableCount = 0;
    this.mouseInclusiveBounds = Bounds2.NOTHING.copy();
    this.mouseExclusiveBounds = Bounds2.NOTHING.copy();
    this.touchInclusiveBounds = Bounds2.NOTHING.copy();
    this.touchExclusiveBounds = Bounds2.NOTHING.copy();
    this.mouseInclusiveDirty = true;
    this.mouseExclusiveDirty = true;
    this.touchInclusiveDirty = true;
    this.touchExclusiveDirty = true;
    this.scratchVector = new Vector2( 0, 0 );
  }

  /*
   * Return a trail to the top node (if any, otherwise null) whose self-rendered area contains the
   * point (in parent coordinates).
   *
   * @param point
   * @param useMouse - Whether mouse-specific customizations (and acceleration) applies
   * @param useTouch - Whether touch-specific customizations (and acceleration) applies
   */
  public hitTest( point: Vector2, useMouse: boolean, useTouch: boolean ): Trail | null {
    assert && assert( point, 'trailUnderPointer requires a point' );

    sceneryLog && sceneryLog.hitTest && sceneryLog.hitTest( `-------------- ${this.node.constructor.name}#${this.node.id}` );

    const isBaseInclusive = this.selfInclusive;

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
  }

  /**
   * @param point
   * @param useMouse
   * @param useTouch
   * @param isInclusive - Essentially true if there is an ancestor or self with an input listener
   */
  private recursiveHitTest( point: Vector2, useMouse: boolean, useTouch: boolean, isInclusive: boolean ): Trail | null {
    isInclusive = isInclusive || this.selfInclusive;

    // If we are selfPruned, ignore this node and its subtree (invisible or pickable:false).
    // If the search is NOT inclusive (no listener equivalent), also ignore this subtree if subtreePrunable is true.
    if ( this.selfPruned || ( !isInclusive && this.subtreePrunable ) ) {
      sceneryLog && sceneryLog.hitTest && sceneryLog.hitTest( `${this.node.constructor.name}#${this.node.id
      } pruned ${this.selfPruned ? '(self)' : '(subtree)'}` );
      return null;
    }

    // Validation should have already been done in hitTest(), we just need to grab the accelerated bounds.
    let pruningBounds;
    if ( useMouse ) {
      pruningBounds = isInclusive ? this.mouseInclusiveBounds : this.mouseExclusiveBounds;
      assert && assert( isInclusive ? !this.mouseInclusiveDirty : !this.mouseExclusiveDirty );
    }
    else if ( useTouch ) {
      pruningBounds = isInclusive ? this.touchInclusiveBounds : this.touchExclusiveBounds;
      assert && assert( isInclusive ? !this.touchInclusiveDirty : !this.touchExclusiveDirty );
    }
    else {
      pruningBounds = this.node.bounds;
      assert && assert( !this.node._boundsDirty );
    }

    // Bail quickly if our point is not inside the bounds for the subtree.
    if ( !pruningBounds.containsPoint( point ) ) {
      sceneryLog && sceneryLog.hitTest && sceneryLog.hitTest( `${this.node.constructor.name}#${this.node.id} pruned: ${useMouse ? 'mouse' : ( useTouch ? 'touch' : 'regular' )}` );
      return null; // not in our bounds, so this point can't possibly be contained
    }

    // Transform the point in the local coordinate frame, so we can test it with the clipArea/children
    const localPoint = this.node._transform.getInverse().multiplyVector2( this.scratchVector.set( point ) );

    // If our point is outside of the local-coordinate clipping area, there should be no hit.
    const clipArea = this.node.clipArea;
    if ( clipArea !== null && !clipArea.containsPoint( localPoint ) ) {
      sceneryLog && sceneryLog.hitTest && sceneryLog.hitTest( `${this.node.constructor.name}#${this.node.id} out of clip area` );
      return null;
    }

    sceneryLog && sceneryLog.hitTest && sceneryLog.hitTest( `${this.node.constructor.name}#${this.node.id}` );

    // Check children before our "self", since the children are rendered on top.
    // Manual iteration here so we can return directly, and so we can iterate backwards (last node is in front).
    for ( let i = this.node._children.length - 1; i >= 0; i-- ) {
      const child = this.node._children[ i ];

      sceneryLog && sceneryLog.hitTest && sceneryLog.push();
      const childHit = child._picker.recursiveHitTest( localPoint, useMouse, useTouch, isInclusive );
      sceneryLog && sceneryLog.hitTest && sceneryLog.pop();

      // If there was a hit, immediately add our node to the start of the Trail (will recursively build the Trail).
      if ( childHit ) {
        return childHit.addAncestor( this.node, i );
      }
    }

    // Tests for mouse and touch hit areas before testing containsPointSelf
    if ( useMouse && this.node._mouseArea ) {
      sceneryLog && sceneryLog.hitTest && sceneryLog.hitTest( `${this.node.constructor.name}#${this.node.id} mouse area hit` );
      // NOTE: both Bounds2 and Shape have containsPoint! We use both here!
      return this.node._mouseArea.containsPoint( localPoint ) ? new Trail( this.node ) : null;
    }
    if ( useTouch && this.node._touchArea ) {
      sceneryLog && sceneryLog.hitTest && sceneryLog.hitTest( `${this.node.constructor.name}#${this.node.id} touch area hit` );
      // NOTE: both Bounds2 and Shape have containsPoint! We use both here!
      return this.node._touchArea.containsPoint( localPoint ) ? new Trail( this.node ) : null;
    }

    // Didn't hit our children, so check ourself as a last resort. Check our selfBounds first, so we can potentially
    // avoid hit-testing the actual object (which may be more expensive).
    if ( this.node.selfBounds.containsPoint( localPoint ) ) {
      if ( this.node.containsPointSelf( localPoint ) ) {
        sceneryLog && sceneryLog.hitTest && sceneryLog.hitTest( `${this.node.constructor.name}#${this.node.id} self hit` );
        return new Trail( this.node );
      }
    }

    // No hit
    return null;
  }

  /**
   * Recursively sets dirty flags to true. If the andExclusive parameter is false, only the "inclusive" flags
   * are set to dirty.
   *
   * @param andExclusive
   * @param [ignoreSelfDirty] - If true, will invalidate parents even if we were dirty.
   */
  private invalidate( andExclusive: boolean, ignoreSelfDirty?: boolean ): void {

    // Track whether a 'dirty' flag was changed from false=>true (or if ignoreSelfDirty is passed).
    let wasNotDirty = !!ignoreSelfDirty || !this.mouseInclusiveDirty || !this.touchInclusiveDirty;

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
      const parents = this.node._parents;
      for ( let i = 0; i < parents.length; i++ ) {
        parents[ i ]._picker.invalidate( andExclusive || this.selfInclusive, false );
      }
    }
  }

  /**
   * Computes the mouseInclusiveBounds for this picker (if dirty), and recursively validates it for all non-pruned
   * children.
   *
   * NOTE: For the future, consider sharing more code with related functions. I tried this, and it made things look
   * more complicated (and probably slower), so I've kept some duplication. If a change is made to this function,
   * please check the other validate* methods to see if they also need a change.
   */
  private validateMouseInclusive(): void {
    if ( !this.mouseInclusiveDirty ) {
      return;
    }

    this.mouseInclusiveBounds.set( this.node.selfBounds );

    const children = this.node._children;
    for ( let i = 0; i < children.length; i++ ) {
      const childPicker = children[ i ]._picker;

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
  }

  /**
   * Computes the mouseExclusiveBounds for this picker (if dirty), and recursively validates the mouse-related bounds
   * for all non-pruned children.
   *
   * Notably, if a picker is selfInclusive, we will switch to validating mouseInclusiveBounds for its subtree, as this
   * is what the hit-testing will use.
   *
   * NOTE: For the future, consider sharing more code with related functions. I tried this, and it made things look
   * more complicated (and probably slower), so I've kept some duplication. If a change is made to this function,
   * please check the other validate* methods to see if they also need a change.
   */
  private validateMouseExclusive(): void {
    if ( !this.mouseExclusiveDirty ) {
      return;
    }

    this.mouseExclusiveBounds.set( this.node.selfBounds );

    const children = this.node._children;
    for ( let i = 0; i < children.length; i++ ) {
      const childPicker = children[ i ]._picker;

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
  }

  /**
   * Computes the touchInclusiveBounds for this picker (if dirty), and recursively validates it for all non-pruned
   * children.
   *
   * NOTE: For the future, consider sharing more code with related functions. I tried this, and it made things look
   * more complicated (and probably slower), so I've kept some duplication. If a change is made to this function,
   * please check the other validate* methods to see if they also need a change.
   */
  private validateTouchInclusive(): void {
    if ( !this.touchInclusiveDirty ) {
      return;
    }

    this.touchInclusiveBounds.set( this.node.selfBounds );

    const children = this.node._children;
    for ( let i = 0; i < children.length; i++ ) {
      const childPicker = children[ i ]._picker;

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
  }

  /**
   * Computes the touchExclusiveBounds for this picker (if dirty), and recursively validates the touch-related bounds
   * for all non-pruned children.
   *
   * Notably, if a picker is selfInclusive, we will switch to validating touchInclusiveBounds for its subtree, as this
   * is what the hit-testing will use.
   *
   * NOTE: For the future, consider sharing more code with related functions. I tried this, and it made things look
   * more complicated (and probably slower), so I've kept some duplication. If a change is made to this function,
   * please check the other validate* methods to see if they also need a change.
   */
  private validateTouchExclusive(): void {
    if ( !this.touchExclusiveDirty ) {
      return;
    }

    this.touchExclusiveBounds.set( this.node.selfBounds );

    const children = this.node._children;
    for ( let i = 0; i < children.length; i++ ) {
      const childPicker = children[ i ]._picker;

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
  }

  /**
   * Include pointer areas (if applicable), exclude bounds outside the clip area (if applicable), and transform
   * into the parent coordinate frame. Mutates the bounds provided.
   *
   * Meant to be called by the validation methods, as this part is the same for every validation that is done.
   *
   * @param mutableBounds - The bounds to be mutated (e.g. mouseExclusiveBounds).
   * @param pointerArea - A mouseArea/touchArea that should be included in the search.
   */
  private applyAreasAndTransform( mutableBounds: Bounds2, pointerArea: Shape | Bounds2 | null ): void {
    // do this before the transformation to the parent coordinate frame (the mouseArea is in the local coordinate frame)
    if ( pointerArea ) {
      // we accept either Bounds2, or a Shape (in which case, we take the Shape's bounds)
      mutableBounds.includeBounds( pointerArea instanceof Bounds2 ? ( pointerArea ) : ( pointerArea as unknown as Shape ).bounds );
    }

    const clipArea = this.node.clipArea;
    if ( clipArea ) {
      const clipBounds = clipArea.bounds;
      // exclude areas outside of the clipping area's bounds (for efficiency)
      // Uses Bounds2.constrainBounds, but inlined to prevent https://github.com/phetsims/projectile-motion/issues/155
      mutableBounds.minX = Math.max( mutableBounds.minX, clipBounds.minX );
      mutableBounds.minY = Math.max( mutableBounds.minY, clipBounds.minY );
      mutableBounds.maxX = Math.min( mutableBounds.maxX, clipBounds.maxX );
      mutableBounds.maxY = Math.min( mutableBounds.maxY, clipBounds.maxY );
    }

    // transform it to the parent coordinate frame
    this.node.transformBoundsFromLocalToParent( mutableBounds );
  }

  /**
   * Called from Node when a child is inserted. (scenery-internal)
   *
   * NOTE: The child may not be fully added when this is called. Don't audit, or assume that calls to the Node would
   * indicate the parent-child relationship.
   *
   * @param childNode - Our picker node's new child node.
   */
  public onInsertChild( childNode: Node ): void {
    // If the child is selfPruned, we don't have to update any metadata.
    if ( !childNode._picker.selfPruned ) {
      const hasPickable = childNode._picker.subtreePickableCount > 0;

      // If it has a non-zero subtreePickableCount, we'll need to increment our own count by 1.
      if ( hasPickable ) {
        this.changePickableCount( 1 );
      }

      // If it has a subtreePickableCount of zero, it would be pruned by "exclusive" searches, so we only need to
      // invalidate the "inclusive" bounds.
      this.invalidate( hasPickable, true );
    }
  }

  /**
   * Called from Node when a child is removed. (scenery-internal)
   *
   * NOTE: The child may not be fully removed when this is called. Don't audit, or assume that calls to the Node would
   * indicate the parent-child relationship.
   *
   * @param childNode - Our picker node's child that will be removed.
   */
  public onRemoveChild( childNode: Node ): void {
    // If the child is selfPruned, we don't have to update any metadata.
    if ( !childNode._picker.selfPruned ) {
      const hasPickable = childNode._picker.subtreePickableCount > 0;

      // If it has a non-zero subtreePickableCount, we'll need to decrement our own count by 1.
      if ( hasPickable ) {
        this.changePickableCount( -1 );
      }

      // If it has a subtreePickableCount of zero, it would be pruned by "exclusive" searches, so we only need to
      // invalidate the "inclusive" bounds.
      this.invalidate( hasPickable, true );
    }
  }

  /**
   * Called from Node when an input listener is added to our node. (scenery-internal)
   */
  public onAddInputListener(): void {
    // Update flags that depend on listener count
    this.checkSelfInclusive();
    this.checkSubtreePrunable();

    // Update our pickable count, since it includes a count of how many input listeners we have.
    this.changePickableCount( 1 ); // NOTE: this should also trigger invalidation of mouse/touch bounds

    if ( assertSlow ) { this.audit(); }
  }

  /**
   * Called from Node when an input listener is removed from our node. (scenery-internal)
   */
  public onRemoveInputListener(): void {
    // Update flags that depend on listener count
    this.checkSelfInclusive();
    this.checkSubtreePrunable();

    // Update our pickable count, since it includes a count of how many input listeners we have.
    this.changePickableCount( -1 ); // NOTE: this should also trigger invalidation of mouse/touch bounds

    if ( assertSlow ) { this.audit(); }
  }

  /**
   * Called when the 'pickable' value of our Node is changed. (scenery-internal)
   */
  public onPickableChange( oldPickable: boolean | null, pickable: boolean | null ): void {
    // Update flags that depend on our pickable setting.
    this.checkSelfPruned();
    this.checkSelfInclusive();
    this.checkSubtreePrunable();

    // Compute our pickable count change (pickable:true counts for 1)
    const change = ( oldPickable === true ? -1 : 0 ) + ( pickable === true ? 1 : 0 );

    if ( change ) {
      this.changePickableCount( change );
    }

    if ( assertSlow ) { this.audit(); }
  }

  /**
   * Called when the visibility of our Node is changed. (scenery-internal)
   */
  public onVisibilityChange(): void {
    // Update flags that depend on our visibility.
    this.checkSelfPruned();
    this.checkSubtreePrunable();
  }

  /**
   * Called when the mouseArea of the Node is changed. (scenery-internal)
   */
  public onMouseAreaChange(): void {
    // Bounds can depend on the mouseArea, so we'll invalidate those.
    // TODO: Consider bounds invalidation that only does the 'mouse' flags, since we don't need to invalidate touches.
    this.invalidate( true );
  }

  /**
   * Called when the mouseArea of the Node is changed. (scenery-internal)
   */
  public onTouchAreaChange(): void {
    // Bounds can depend on the touchArea, so we'll invalidate those.
    // TODO: Consider bounds invalidation that only does the 'touch' flags, since we don't need to invalidate mice.
    this.invalidate( true );
  }

  /**
   * Called when the transform of the Node is changed. (scenery-internal)
   */
  public onTransformChange(): void {
    // Can affect our bounds
    this.invalidate( true );
  }

  /**
   * Called when the transform of the Node is changed. (scenery-internal)
   */
  public onSelfBoundsDirty(): void {
    // Can affect our bounds
    this.invalidate( true );
  }

  /**
   * Called when the transform of the Node is changed. (scenery-internal)
   */
  public onClipAreaChange(): void {
    // Can affect our bounds.
    this.invalidate( true );
  }

  /**
   * Check to see if we are 'selfPruned', and update the value. If it changed, we'll need to notify our parents.
   *
   * Note that the prunability "pickable:false" or "invisible" won't affect our computed bounds, so we don't
   * invalidate ourself.
   */
  private checkSelfPruned(): void {
    const selfPruned = this.node.pickableProperty.value === false || !this.node.isVisible();
    if ( this.selfPruned !== selfPruned ) {
      this.selfPruned = selfPruned;

      // Notify parents
      const parents = this.node._parents;
      for ( let i = 0; i < parents.length; i++ ) {
        const picker = parents[ i ]._picker;

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
  }

  /**
   * Check to see if we are 'selfInclusive', and update the value. If it changed, we'll need to invalidate ourself.
   */
  private checkSelfInclusive(): void {
    const selfInclusive = this.node.pickableProperty.value === true || this.node._inputListeners.length > 0;
    if ( this.selfInclusive !== selfInclusive ) {
      this.selfInclusive = selfInclusive;

      // Our dirty flag handling for both inclusive and exclusive depend on this value.
      this.invalidate( true, true );
    }
  }

  /**
   * Update our 'subtreePrunable' flag.
   */
  private checkSubtreePrunable(): void {
    const subtreePrunable = this.node.pickableProperty.value === false ||
                            !this.node.isVisible() ||
                            ( this.node.pickableProperty.value !== true && this.subtreePickableCount === 0 );

    if ( this.subtreePrunable !== subtreePrunable ) {
      this.subtreePrunable = subtreePrunable;

      // Our dirty flag handling for both inclusive and exclusive depend on this value.
      this.invalidate( true, true );
    }
  }

  /**
   * Propagate the pickable count change down to our ancestors.
   *
   * @param n - The delta of how many pickable counts have been added/removed
   */
  private changePickableCount( n: number ): void {
    if ( n === 0 ) {
      return;
    }

    // Switching between 0 and 1 matters, since we then need to update the counts of our parents.
    const wasZero = this.subtreePickableCount === 0;
    this.subtreePickableCount += n;
    const isZero = this.subtreePickableCount === 0;

    // Our subtreePrunable value depends on our pickable count, make sure it gets updated.
    this.checkSubtreePrunable();

    assert && assert( this.subtreePickableCount >= 0, 'subtree pickable count should be guaranteed to be >= 0' );

    if ( !this.selfPruned && wasZero !== isZero ) {
      // Update our parents if our count changed (AND if it matters, i.e. we aren't selfPruned).
      const len = this.node._parents.length;
      for ( let i = 0; i < len; i++ ) {
        this.node._parents[ i ]._picker.changePickableCount( wasZero ? 1 : -1 );
      }
    }
  }

  /**
   * Runs a number of consistency tests when assertSlow is enabled. Verifies most conditions, and helps to catch
   * bugs earlier when they are initially triggered. (scenery-internal)
   */
  public audit(): void {
    if ( assertSlow ) {
      this.node._children.forEach( node => {
        node._picker.audit();
      } );

      const localAssertSlow = assertSlow;

      const expectedSelfPruned = this.node.pickable === false || !this.node.isVisible();
      const expectedSelfInclusive = this.node.pickable === true || this.node._inputListeners.length > 0;
      const expectedSubtreePrunable = this.node.pickable === false ||
                                      !this.node.isVisible() ||
                                      ( this.node.pickable !== true && this.subtreePickableCount === 0 );
      const expectedSubtreePickableCount = this.node._inputListeners.length +
                                           ( this.node.pickableProperty.value === true ? 1 : 0 ) +
                                           _.filter( this.node._children, child => !child._picker.selfPruned && child._picker.subtreePickableCount > 0 ).length;

      assertSlow( this.selfPruned === expectedSelfPruned, 'selfPruned mismatch' );
      assertSlow( this.selfInclusive === expectedSelfInclusive, 'selfInclusive mismatch' );
      assertSlow( this.subtreePrunable === expectedSubtreePrunable, 'subtreePrunable mismatch' );
      assertSlow( this.subtreePickableCount === expectedSubtreePickableCount, 'subtreePickableCount mismatch' );

      this.node._parents.forEach( parent => {
        const parentPicker = parent._picker;

        // eslint-disable-next-line @typescript-eslint/no-this-alias
        const childPicker = this; // eslint-disable-line consistent-this

        if ( !parentPicker.mouseInclusiveDirty ) {
          localAssertSlow( childPicker.selfPruned || !childPicker.mouseInclusiveDirty );
        }

        if ( !parentPicker.mouseExclusiveDirty ) {
          if ( childPicker.selfInclusive ) {
            localAssertSlow( childPicker.selfPruned || !childPicker.mouseInclusiveDirty );
          }
          else {
            localAssertSlow( childPicker.selfPruned || childPicker.subtreePrunable || !childPicker.mouseExclusiveDirty );
          }
        }

        if ( !parentPicker.touchInclusiveDirty ) {
          localAssertSlow( childPicker.selfPruned || !childPicker.touchInclusiveDirty );
        }

        if ( !parentPicker.touchExclusiveDirty ) {
          if ( childPicker.selfInclusive ) {
            localAssertSlow( childPicker.selfPruned || !childPicker.touchInclusiveDirty );
          }
          else {
            localAssertSlow( childPicker.selfPruned || childPicker.subtreePrunable || !childPicker.touchExclusiveDirty );
          }
        }
      } );
    }
  }
}

scenery.register( 'Picker', Picker );
