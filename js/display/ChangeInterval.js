// Copyright 2014-2016, University of Colorado Boulder


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

define( function( require ) {
  'use strict';

  var Drawable = require( 'SCENERY/display/Drawable' );
  var inherit = require( 'PHET_CORE/inherit' );
  var Poolable = require( 'PHET_CORE/Poolable' );
  var scenery = require( 'SCENERY/scenery' );

  /**
   * @constructor
   * @mixes Poolable
   *
   * @param drawableBefore
   * @param drawableAfter
   */
  function ChangeInterval( drawableBefore, drawableAfter ) {
    this.initialize( drawableBefore, drawableAfter );
  }

  scenery.register( 'ChangeInterval', ChangeInterval );

  inherit( Object, ChangeInterval, {
    initialize: function( drawableBefore, drawableAfter ) {
      assert && assert( drawableBefore === null || ( drawableBefore instanceof Drawable ),
        'drawableBefore can either be null to indicate that there is no un-changed drawable before our changes, ' +
        'or it can reference an un-changed drawable' );
      assert && assert( drawableAfter === null || ( drawableAfter instanceof Drawable ),
        'drawableAfter can either be null to indicate that there is no un-changed drawable after our changes, ' +
        'or it can reference an un-changed drawable' );

      /*---------------------------------------------------------------------------*
       * All @public properties
       *----------------------------------------------------------------------------*/

      // {ChangeInterval|null}, singly-linked list
      this.nextChangeInterval = null;

      // {Drawable|null}, the drawable before our ChangeInterval that is not modified. null indicates that we don't yet
      // have a "before" boundary, and should be connected to the closest drawable that is unchanged.
      this.drawableBefore = drawableBefore;

      // {Drawable|null}, the drawable after our ChangeInterval that is not modified. null indicates that we don't yet
      // have a "after" boundary, and should be connected to the closest drawable that is unchanged.
      this.drawableAfter = drawableAfter;

      // {boolean} If a null-to-X interval gets collapsed all the way, we want to have a flag that indicates that.
      // Otherwise, it would be interpreted as a null-to-null change interval ("change everything"), instead of the
      // correct "change nothing".
      this.collapsedEmpty = false;

      // chaining for PoolableMixin
      return this;
    },

    dispose: function() {
      // release our references
      this.nextChangeInterval = null;
      this.drawableBefore = null;
      this.drawableAfter = null;

      this.freeToPool();
    },

    // Make our interval as tight as possible (we may have over-estimated it before)
    constrict: function() {
      var changed = false;

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
    },

    isEmpty: function() {
      return this.collapsedEmpty || ( this.drawableBefore !== null && this.drawableBefore === this.drawableAfter );
    },

    // {number} The quantity of "old" internal drawables. Requires the old first/last drawables for the backbone, since
    // we need that information for null-before/after boundaries.
    getOldInternalDrawableCount: function( oldStitchFirstDrawable, oldStitchLastDrawable ) {
      var firstInclude = this.drawableBefore ? this.drawableBefore.oldNextDrawable : oldStitchFirstDrawable;
      var lastExclude = this.drawableAfter; // null is OK here

      var count = 0;
      for ( var drawable = firstInclude; drawable !== lastExclude; drawable = drawable.oldNextDrawable ) {
        count++;
      }

      return count;
    },

    // {number} The quantity of "new" internal drawables. Requires the old first/last drawables for the backbone, since
    // we need that information for null-before/after boundaries.
    getNewInternalDrawableCount: function( newStitchFirstDrawable, newStitchLastDrawable ) {
      var firstInclude = this.drawableBefore ? this.drawableBefore.nextDrawable : newStitchFirstDrawable;
      var lastExclude = this.drawableAfter; // null is OK here

      var count = 0;
      for ( var drawable = firstInclude; drawable !== lastExclude; drawable = drawable.nextDrawable ) {
        count++;
      }

      return count;
    }
  } );

  Poolable.mixInto( ChangeInterval, {
    initialize: ChangeInterval.prototype.initialize
  } );

  // creates a ChangeInterval that will be disposed after syncTree is complete (see Display phases)
  ChangeInterval.newForDisplay = function( drawableBefore, drawableAfter, display ) {
    var changeInterval = ChangeInterval.createFromPool( drawableBefore, drawableAfter );
    display.markChangeIntervalToDispose( changeInterval );
    return changeInterval;
  };

  return ChangeInterval;
} );
