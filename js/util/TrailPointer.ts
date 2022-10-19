// Copyright 2013-2022, University of Colorado Boulder

/**
 * Points to a specific node (with a trail), and whether it is conceptually before or after the node.
 *
 * There are two orderings:
 * - rendering order: the order that node selves would be rendered, matching the Trail implicit order
 * - nesting order:   the order in depth first with entering a node being "before" and exiting a node being "after"
 *
 * TODO: more seamless handling of the orders. or just exclusively use the nesting order
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import WithoutNull from '../../../phet-core/js/types/WithoutNull.js';
import { Node, scenery, Trail } from '../imports.js';
import { TrailCallback } from './Trail.js';

export type ActiveTrailPointer = WithoutNull<TrailPointer, 'trail'>;

type ActiveTrailPointerCallback = ( ( trailPointer: ActiveTrailPointer ) => boolean ) | ( ( trailPointer: ActiveTrailPointer ) => void );

export default class TrailPointer {

  public trail: Trail | null;
  public isBefore!: boolean;
  public isAfter!: boolean;

  /**
   * @param trail
   * @param isBefore - whether this points to before the node (and its children) have been rendered, or after
   */
  public constructor( trail: Trail, isBefore: boolean ) {
    this.trail = trail;
    this.setBefore( isBefore );
  }

  public isActive(): this is ActiveTrailPointer {
    return !!this.trail;
  }

  public copy(): TrailPointer {
    assert && assert( this.isActive() );
    return new TrailPointer( ( this as ActiveTrailPointer ).trail.copy(), this.isBefore );
  }

  public setBefore( isBefore: boolean ): void {
    this.isBefore = isBefore;
    this.isAfter = !isBefore;
  }

  /**
   * Return the equivalent pointer that swaps before and after (may return null if it doesn't exist)
   */
  public getRenderSwappedPointer(): TrailPointer | null {
    assert && assert( this.isActive() );
    const activeSelf = this as ActiveTrailPointer;

    const newTrail = this.isBefore ? activeSelf.trail.previous() : activeSelf.trail.next();

    if ( newTrail === null ) {
      return null;
    }
    else {
      return new TrailPointer( newTrail, !this.isBefore );
    }
  }

  public getRenderBeforePointer(): TrailPointer | null {
    return this.isBefore ? this : this.getRenderSwappedPointer();
  }

  public getRenderAfterPointer(): TrailPointer | null {
    return this.isAfter ? this : this.getRenderSwappedPointer();
  }

  /**
   * In the render order, will return 0 if the pointers are equivalent, -1 if this pointer is before the
   * other pointer, and 1 if this pointer is after the other pointer.
   */
  public compareRender( other: TrailPointer ): number {
    assert && assert( other !== null );

    const a = this.getRenderBeforePointer();
    const b = other.getRenderBeforePointer();

    if ( a !== null && b !== null ) {
      assert && assert( a.isActive() && b.isActive() );

      // normal (non-degenerate) case
      return ( a as ActiveTrailPointer ).trail.compare( ( b as ActiveTrailPointer ).trail );
    }
    else {
      // null "before" point is equivalent to the "after" pointer on the last rendered node.
      if ( a === b ) {
        return 0; // uniqueness guarantees they were the same
      }
      else {
        return a === null ? 1 : -1;
      }
    }
  }

  /**
   * Like compareRender, but for the nested (depth-first) order
   *
   * TODO: optimization?
   */
  public compareNested( other: TrailPointer ): number {
    assert && assert( other );

    assert && assert( this.isActive() && other.isActive() );
    const activeSelf = this as ActiveTrailPointer;
    const activeOther = other as ActiveTrailPointer;

    const comparison = activeSelf.trail.compare( activeOther.trail );

    if ( comparison === 0 ) {
      // if trails are equal, just compare before/after
      if ( this.isBefore === other.isBefore ) {
        return 0;
      }
      else {
        return this.isBefore ? -1 : 1;
      }
    }
    else {
      // if one is an extension of the other, the shorter isBefore flag determines the order completely
      if ( activeSelf.trail.isExtensionOf( activeOther.trail ) ) {
        return other.isBefore ? 1 : -1;
      }
      else if ( activeOther.trail.isExtensionOf( activeSelf.trail ) ) {
        return this.isBefore ? -1 : 1;
      }
      else {
        // neither is a subtrail of the other, so a straight trail comparison should give the answer
        return comparison;
      }
    }
  }

  public equalsRender( other: TrailPointer ): boolean {
    return this.compareRender( other ) === 0;
  }

  public equalsNested( other: TrailPointer ): boolean {
    return this.compareNested( other ) === 0;
  }

  /**
   * Will return false if this pointer has gone off of the beginning or end of the tree (will be marked with isAfter or
   * isBefore though)
   */
  public hasTrail(): boolean {
    return !!this.trail;
  }

  /**
   * Moves this pointer forwards one step in the nested order
   *
   * TODO: refactor with "Side"-like handling
   */
  public nestedForwards(): this | null {
    assert && assert( this.isActive() );
    const activeSelf = this as ActiveTrailPointer;

    if ( this.isBefore ) {
      const children = activeSelf.trail.lastNode()._children;
      if ( children.length > 0 ) {
        // stay as before, just walk to the first child
        activeSelf.trail.addDescendant( children[ 0 ], 0 );
      }
      else {
        // stay on the same node, but switch to after
        this.setBefore( false );
      }
    }
    else {
      if ( activeSelf.trail.indices.length === 0 ) {
        // nothing else to jump to below, so indicate the lack of existence
        this.trail = null;
        // stays isAfter
        return null;
      }
      else {
        const index = activeSelf.trail.indices[ activeSelf.trail.indices.length - 1 ];
        activeSelf.trail.removeDescendant();

        const children = activeSelf.trail.lastNode()._children;
        if ( children.length > index + 1 ) {
          // more siblings, switch to the beginning of the next one
          activeSelf.trail.addDescendant( children[ index + 1 ], index + 1 );
          this.setBefore( true );
        }
        else {
          // no more siblings. exit on parent. nothing else needed since we're already isAfter
        }
      }
    }
    return this;
  }

  /**
   * Moves this pointer backwards one step in the nested order
   */
  public nestedBackwards(): this | null {
    assert && assert( this.isActive() );
    const activeSelf = this as ActiveTrailPointer;

    if ( this.isBefore ) {
      if ( activeSelf.trail.indices.length === 0 ) {
        // jumping off the front
        this.trail = null;
        // stays isBefore
        return null;
      }
      else {
        const index = activeSelf.trail.indices[ activeSelf.trail.indices.length - 1 ];
        activeSelf.trail.removeDescendant();

        if ( index - 1 >= 0 ) {
          // more siblings, switch to the beginning of the previous one and switch to isAfter
          activeSelf.trail.addDescendant( activeSelf.trail.lastNode()._children[ index - 1 ], index - 1 );
          this.setBefore( false );
        }
        else {
          // no more siblings. enter on parent. nothing else needed since we're already isBefore
        }
      }
    }
    else {
      if ( activeSelf.trail.lastNode()._children.length > 0 ) {
        // stay isAfter, but walk to the last child
        const children = activeSelf.trail.lastNode()._children;
        activeSelf.trail.addDescendant( children[ children.length - 1 ], children.length - 1 );
      }
      else {
        // switch to isBefore, since this is a leaf node
        this.setBefore( true );
      }
    }
    return this;
  }

  /**
   * Treats the pointer as render-ordered (includes the start pointer 'before' if applicable, excludes the end pointer
   * 'before' if applicable
   */
  public eachNodeBetween( other: TrailPointer, callback: ( node: Node ) => void ): void {
    this.eachTrailBetween( other, ( trail: Trail ) => callback( trail.lastNode() ) );
  }

  /**
   * Treats the pointer as render-ordered (includes the start pointer 'before' if applicable, excludes the end pointer
   * 'before' if applicable
   */
  public eachTrailBetween( other: TrailPointer, callback: TrailCallback ): void {
    // this should trigger on all pointers that have the 'before' flag, except a pointer equal to 'other'.

    // since we exclude endpoints in the depthFirstUntil call, we need to fire this off first
    if ( this.isBefore ) {
      assert && assert( this.isActive() );
      callback( ( this as ActiveTrailPointer ).trail );
    }

    this.depthFirstUntil( other, ( pointer: ActiveTrailPointer ) => {
      if ( pointer.isBefore ) {
        return callback( pointer.trail );
      }
      return false;
    }, true ); // exclude the endpoints so we can ignore the ending 'before' case
  }

  /**
   * Recursively (depth-first) iterates over all pointers between this pointer and 'other', calling
   * callback( pointer ) for each pointer. If excludeEndpoints is truthy, the callback will not be
   * called if pointer is equivalent to this pointer or 'other'.
   *
   * If the callback returns a truthy value, the subtree for the current pointer will be skipped
   * (applies only to before-pointers)
   */
  public depthFirstUntil( other: TrailPointer, callback: ActiveTrailPointerCallback, excludeEndpoints: boolean ): void {
    assert && assert( this.isActive() && other.isActive() );
    const activeSelf = this as ActiveTrailPointer;
    const activeOther = other as ActiveTrailPointer;

    // make sure this pointer is before the other, but allow start === end if we are not excluding endpoints
    assert && assert( this.compareNested( other ) <= ( excludeEndpoints ? -1 : 0 ), 'TrailPointer.depthFirstUntil pointers out of order, possibly in both meanings of the phrase!' );
    assert && assert( activeSelf.trail.rootNode() === activeOther.trail.rootNode(), 'TrailPointer.depthFirstUntil takes pointers with the same root' );

    // sanity check TODO: remove later
    activeSelf.trail.reindex();
    activeOther.trail.reindex();

    const pointer = this.copy() as ActiveTrailPointer;
    pointer.trail.setMutable(); // this trail will be modified in the iteration, so references to it may be modified

    let first = true;

    while ( !pointer.equalsNested( other ) ) {
      assert && assert( pointer.compareNested( other ) !== 1, 'skipped in depthFirstUntil' );
      let skipSubtree: boolean | void = false; // eslint-disable-line @typescript-eslint/no-invalid-void-type

      if ( first ) {
        // start point
        if ( !excludeEndpoints ) {
          skipSubtree = callback( pointer );
        }
        first = false;
      }
      else {
        // between point
        skipSubtree = callback( pointer );
      }

      if ( skipSubtree && pointer.isBefore ) {
        // to skip the subtree, we just change to isAfter
        pointer.setBefore( false );

        // if we skip a subtree, make sure we don't run past the ending pointer
        if ( pointer.compareNested( other ) === 1 ) {
          break;
        }
      }
      else {
        pointer.nestedForwards();
      }
    }

    // end point
    if ( !excludeEndpoints ) {
      callback( pointer );
    }
  }

  /**
   * Returns a string form of this object
   */
  public toString(): string {
    assert && assert( this.isActive() );

    return `[${this.isBefore ? 'before' : 'after'} ${( this as ActiveTrailPointer ).trail.toString().slice( 1 )}`;
  }

  /**
   * Same as new TrailPointer( trailA, isBeforeA ).compareNested( new TrailPointer( trailB, isBeforeB ) )
   */
  public static compareNested( trailA: Trail, isBeforeA: boolean, trailB: Trail, isBeforeB: boolean ): number {
    const comparison = trailA.compare( trailB );

    if ( comparison === 0 ) {
      // if trails are equal, just compare before/after
      if ( isBeforeA === isBeforeB ) {
        return 0;
      }
      else {
        return isBeforeA ? -1 : 1;
      }
    }
    else {
      // if one is an extension of the other, the shorter isBefore flag determines the order completely
      if ( trailA.isExtensionOf( trailB ) ) {
        return isBeforeB ? 1 : -1;
      }
      else if ( trailB.isExtensionOf( trailA ) ) {
        return isBeforeA ? -1 : 1;
      }
      else {
        // neither is a subtrail of the other, so a straight trail comparison should give the answer
        return comparison;
      }
    }
  }
}

scenery.register( 'TrailPointer', TrailPointer );
