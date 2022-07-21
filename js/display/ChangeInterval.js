// Copyright 2014-2022, University of Colorado Boulder

/**
 * An interval (implicit consecutive sequence of drawables) that has a recorded change in-between the two ends.
 * We store the closest drawables to the interval that aren't changed, or null itself to indicate "to the end".
 *
 * isEmpty() should be used before checking the endpoints, since it could have a null-to-null state but be empty,
 * since we arrived at that state from constriction.
 *
 * For documentation purposes, an 'internal' drawable is one that is in-between (but not including) our un-changed ends
 * (drawableBefore and drawableAfter), and 'external' drawables are outside (or including) the un-changed ends.
 *
 * For stitching purposes, a ChangeInterval effectively represents two linked lists: the "old" one that was displayed
 * in the previous frame (using oldNextDrawable for iteration across the drawable linked-list), or the "new" one that
 * will be displayed in the next frame (using nextDrawable for iteration).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Poolable from '../../../phet-core/js/Poolable.js';
import { Drawable, scenery } from '../imports.js';

class ChangeInterval {
  /**
   * @mixes Poolable
   *
   * @param {Drawable} drawableBefore
   * @param {Drawable} drawableAfter
   */
  constructor( drawableBefore, drawableAfter ) {
    this.initialize( drawableBefore, drawableAfter );
  }

  /**
   * @public
   *
   * @param {Drawable} drawableBefore
   * @param {Drawable} drawableAfter
   */
  initialize( drawableBefore, drawableAfter ) {
    assert && assert( drawableBefore === null || ( drawableBefore instanceof Drawable ),
      'drawableBefore can either be null to indicate that there is no un-changed drawable before our changes, ' +
      'or it can reference an un-changed drawable' );
    assert && assert( drawableAfter === null || ( drawableAfter instanceof Drawable ),
      'drawableAfter can either be null to indicate that there is no un-changed drawable after our changes, ' +
      'or it can reference an un-changed drawable' );

    /*---------------------------------------------------------------------------*
     * All @public properties
     *----------------------------------------------------------------------------*/

    // @public {ChangeInterval|null}, singly-linked list
    this.nextChangeInterval = null;

    // @public {Drawable|null}, the drawable before our ChangeInterval that is not modified. null indicates that we
    // don't yet have a "before" boundary, and should be connected to the closest drawable that is unchanged.
    this.drawableBefore = drawableBefore;

    // @public {Drawable|null}, the drawable after our ChangeInterval that is not modified. null indicates that we
    // don't yet have a "after" boundary, and should be connected to the closest drawable that is unchanged.
    this.drawableAfter = drawableAfter;

    // @public {boolean} If a null-to-X interval gets collapsed all the way, we want to have a flag that indicates that.
    // Otherwise, it would be interpreted as a null-to-null change interval ("change everything"), instead of the
    // correct "change nothing".
    this.collapsedEmpty = false;
  }

  /**
   * Releases references
   * @public
   */
  dispose() {
    // release our references
    this.nextChangeInterval = null;
    this.drawableBefore = null;
    this.drawableAfter = null;

    this.freeToPool();
  }

  /**
   * Make our interval as tight as possible (we may have over-estimated it before)
   * @public
   *
   * @returns {boolean} - Whether it was changed
   */
  constrict() {
    let changed = false;

    if ( this.isEmpty() ) { return true; }

    // Notes: We don't constrict null boundaries, and we should never constrict a non-null boundary to a null
    // boundary (this the this.drawableX.Xdrawable truthy check), since going from a null-to-X interval to
    // null-to-null has a completely different meaning. This should be checked by a client of this API.

    while ( this.drawableBefore && this.drawableBefore.nextDrawable === this.drawableBefore.oldNextDrawable ) {
      this.drawableBefore = this.drawableBefore.nextDrawable;
      changed = true;

      // check for a totally-collapsed state
      if ( !this.drawableBefore ) {
        assert && assert( !this.drawableAfter );
        this.collapsedEmpty = true;
      }

      // if we are empty, bail out before continuing
      if ( this.isEmpty() ) { return true; }
    }

    while ( this.drawableAfter && this.drawableAfter.previousDrawable === this.drawableAfter.oldPreviousDrawable ) {
      this.drawableAfter = this.drawableAfter.previousDrawable;
      changed = true;

      // check for a totally-collapsed state
      if ( !this.drawableAfter ) {
        assert && assert( !this.drawableBefore );
        this.collapsedEmpty = true;
      }

      // if we are empty, bail out before continuing
      if ( this.isEmpty() ) { return true; }
    }

    return changed;
  }

  /**
   * @public
   *
   * @returns {boolean}
   */
  isEmpty() {
    return this.collapsedEmpty || ( this.drawableBefore !== null && this.drawableBefore === this.drawableAfter );
  }

  /**
   * The quantity of "old" internal drawables. Requires the old first/last drawables for the backbone, since
   * we need that information for null-before/after boundaries.
   * @public
   *
   * @param {Drawable} oldStitchFirstDrawable
   * @param {Drawable} oldStitchLastDrawable
   * @returns {number}
   */
  getOldInternalDrawableCount( oldStitchFirstDrawable, oldStitchLastDrawable ) {
    const firstInclude = this.drawableBefore ? this.drawableBefore.oldNextDrawable : oldStitchFirstDrawable;
    const lastExclude = this.drawableAfter; // null is OK here

    let count = 0;
    for ( let drawable = firstInclude; drawable !== lastExclude; drawable = drawable.oldNextDrawable ) {
      count++;
    }

    return count;
  }

  /**
   * The quantity of "new" internal drawables. Requires the old first/last drawables for the backbone, since
   * we need that information for null-before/after boundaries.
   * @public
   *
   * @param {Drawable} newStitchFirstDrawable
   * @param {Drawable} newStitchLastDrawable
   *
   * @returns {number}
   */
  getNewInternalDrawableCount( newStitchFirstDrawable, newStitchLastDrawable ) {
    const firstInclude = this.drawableBefore ? this.drawableBefore.nextDrawable : newStitchFirstDrawable;
    const lastExclude = this.drawableAfter; // null is OK here

    let count = 0;
    for ( let drawable = firstInclude; drawable !== lastExclude; drawable = drawable.nextDrawable ) {
      count++;
    }

    return count;
  }

  /**
   * Creates a ChangeInterval that will be disposed after syncTree is complete (see Display phases).
   * @public
   *
   * @param {Drawable} drawableBefore
   * @param {Drawable} drawableAfter
   * @param {Display} display
   *
   * @returns {ChangeInterval}
   */
  static newForDisplay( drawableBefore, drawableAfter, display ) {
    const changeInterval = ChangeInterval.createFromPool( drawableBefore, drawableAfter );
    display.markChangeIntervalToDispose( changeInterval );
    return changeInterval;
  }
}

scenery.register( 'ChangeInterval', ChangeInterval );

Poolable.mixInto( ChangeInterval );

export default ChangeInterval;