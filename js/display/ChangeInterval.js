// Copyright 2002-2014, University of Colorado

/**
 * An interval (implicit consecutive sequence of drawables) that has a recorded change in-between the two ends.
 * We store the closest drawables to the interval that aren't changed, or null itself to indicate "to the end".
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';
  
  var inherit = require( 'PHET_CORE/inherit' );
  var Poolable = require( 'PHET_CORE/Poolable' );
  var scenery = require( 'SCENERY/scenery' );
  var Drawable = require( 'SCENERY/display/Drawable' );
  
  scenery.ChangeInterval = function ChangeInterval( drawableBefore, drawableAfter ) {
    this.initialize( drawableBefore, drawableAfter );
  };
  var ChangeInterval = scenery.ChangeInterval;
  
  inherit( Object, ChangeInterval, {
    initialize: function( drawableBefore, drawableAfter ) {
      assert && assert( drawableBefore === null || ( drawableBefore instanceof Drawable ) );
      assert && assert( drawableAfter === null || ( drawableAfter instanceof Drawable ) );
      
      // all @public, for modification
      this.nextChangeInterval = null;       // {ChangeInterval|null}, singly-linked list
      this.drawableBefore = drawableBefore; // {Drawable|null}, the drawable before our ChangeInterval that is not
                                            // modified. null indicates that we don't yet have a "before" boundary,
                                            // and should be connected to the closest drawable that is unchanged.
      this.drawableAfter = drawableAfter;   // {Drawable|null}, the drawable after our ChangeInterval that is not
                                            // modified. null indicates that we don't yet have a "after" boundary,
                                            // and should be connected to the closest drawable that is unchanged.
      this.collapsedEmpty = false;          // If a null-to-X interval gets collapsed all the way, we want to signal
                                            // that (null-to-null is now the state of it).
      return this;
    },
    
    dispose: function() {
      this.nextChangeInterval = null;
      this.drawableBefore = null;
      this.drawableAfter = null;
      
      this.freeToPool();
    },
    
    // Make our interval as tight as possible (we may have over-estimated it before)
    constrict: function() {
      if ( this.isEmpty() ) { return; }
      
      // Notes: We don't constrict null boundaries, and we should never constrict a non-null boundary to a null
      // boundary (this the this.drawableX.Xdrawable truthy check), since going from a null-to-X interval to
      // null-to-null has a completely different meaning. This should be checked by a client of this API.
      
      while ( this.drawableBefore && this.drawableBefore.nextDrawable === this.drawableBefore.oldNextDrawable ) {
        this.drawableBefore = this.drawableBefore.nextDrawable;
        
        // check for a totally-collapsed state
        if ( !this.drawableBefore ) {
          assert && assert( !this.drawableAfter );
          this.collapsedEmpty = true;
        }
        
        // if we are empty, bail out before continuing
        if ( this.isEmpty() ) { return; }
      }
      
      while ( this.drawableAfter && this.drawableAfter.previousDrawable === this.drawableAfter.oldPreviousDrawable ) {
        this.drawableAfter = this.drawableAfter.previousDrawable;
        
        // check for a totally-collapsed state
        if ( !this.drawableAfter ) {
          assert && assert( !this.drawableBefore );
          this.collapsedEmpty = true;
        }
        
        // if we are empty, bail out before continuing
        if ( this.isEmpty() ) { return; }
      }
    },
    
    isEmpty: function() {
      return this.collapsedEmpty || this.drawableBefore === this.drawableAfter;
    }
  } );
  
  /* jshint -W064 */
  Poolable( ChangeInterval, {
    constructorDuplicateFactory: function( pool ) {
      return function( drawableBefore, drawableAfter ) {
        if ( pool.length ) {
          sceneryLog && sceneryLog.ChangeInterval && sceneryLog.ChangeInterval( 'new from pool' );
          return pool.pop().initialize( drawableBefore, drawableAfter );
        } else {
          sceneryLog && sceneryLog.ChangeInterval && sceneryLog.ChangeInterval( 'new from constructor' );
          return new ChangeInterval( drawableBefore, drawableAfter );
        }
      };
    }
  } );
  
  // creates a ChangeInterval that will be disposed after syncTree is complete (see Display phases)
  ChangeInterval.newForDisplay = function( drawableBefore, drawableAfter, display ) {
    var changeInterval = ChangeInterval.createFromPool( drawableBefore, drawableAfter );
    display.markChangeIntervalToDispose( changeInterval );
    return changeInterval;
  };
  
  return ChangeInterval;
} );
