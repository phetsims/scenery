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
      return this;
    },
    
    dispose: function() {
      this.nextChangeInterval = null;
      this.drawableBefore = null;
      this.drawableAfter = null;
      
      this.freeToPool();
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
