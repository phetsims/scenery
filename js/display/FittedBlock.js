// Copyright 2002-2014, University of Colorado Boulder


/**
 * A Block that needs to be fitted to either the screen bounds or other local bounds.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var Bounds2 = require( 'DOT/Bounds2' );
  var Vector2 = require( 'DOT/Vector2' );
  var scenery = require( 'SCENERY/scenery' );
  var Block = require( 'SCENERY/display/Block' );
  var Renderer = require( 'SCENERY/display/Renderer' );

  scenery.FittedBlock = function FittedBlock( display, renderer, transformRootInstance ) {
    this.initialize( display, renderer, transformRootInstance );
  };
  var FittedBlock = scenery.FittedBlock;

  inherit( Block, FittedBlock, {
    initializeFittedBlock: function( display, renderer, transformRootInstance ) {
      this.initializeBlock( display, renderer );

      this.transformRootInstance = transformRootInstance;

      assert && assert( typeof transformRootInstance.isDisplayRoot === 'boolean' );
      // var canBeFullDisplay = transformRootInstance.isDisplayRoot;

      //OHTWO TODO: change fit based on renderer flags or extra parameters
      // this.fit = canBeFullDisplay ? FittedBlock.FULL_DISPLAY : FittedBlock.COMMON_ANCESTOR;
      this.fit = FittedBlock.COMMON_ANCESTOR;

      this.dirtyFit = true;
      this.dirtyFitListener = this.dirtyFitListener || this.markDirtyFit.bind( this );
      this.commonFitInstance = null; // filled in if COMMON_ANCESTOR
      this.fitBounds = Bounds2.NOTHING.copy(); // tracks the "tight" bounds for fitting, not the actually-displayed bounds
      this.oldFitBounds = Bounds2.NOTHING.copy(); // copy for storage
      this.fitOffset = new Vector2();

      // TODO: I can't find documentation about forceAcceleration anywhere.  How is this used?  What is it for?  How does it work?
      this.forceAcceleration = ( renderer & Renderer.bitmaskForceAcceleration ) !== 0;

      if ( this.fit === FittedBlock.FULL_DISPLAY ) {
        this.display.onStatic( 'displaySize', this.dirtyFitListener );
      }

      // TODO: add count of boundsless objects?
      return this;
    },

    markDirtyFit: function() {
      sceneryLog && sceneryLog.dirty && sceneryLog.dirty( 'markDirtyFit on FittedBlock#' + this.id );
      this.dirtyFit = true;
      this.markDirty();
    },

    // should be called from update() whenever this block is dirty
    updateFit: function() {
      assert && assert( this.fit === FittedBlock.FULL_DISPLAY || this.fit === FittedBlock.COMMON_ANCESTOR,
        'Unsupported fit' );

      // check to see if we don't need to re-fit
      if ( !this.dirtyFit && this.fit === FittedBlock.FULL_DISPLAY ) {
        return;
      }

      sceneryLog && sceneryLog.FittedBlock && sceneryLog.FittedBlock( 'updateFit #' + this.id );

      this.dirtyFit = false;

      if ( this.fit === FittedBlock.FULL_DISPLAY ) {
        this.setSizeFullDisplay();
      }
      else if ( this.fit === FittedBlock.COMMON_ANCESTOR ) {
        assert && assert( this.commonFitInstance.trail.length >= this.transformRootInstance.trail.length );

        // will trigger bounds validation (for now) until we have a better way of handling this
        this.fitBounds.set( this.commonFitInstance.node.getLocalBounds() );

        //OHTWO TODO: bail out here when possible (should store an old "local" one to compare with?)

        // walk it up, transforming so it is relative to our transform root
        var instance = this.commonFitInstance;
        while ( instance !== this.transformRootInstance ) {
          // shouldn't infinite loop, we'll null-pointer beforehand unless something is seriously wrong
          this.fitBounds.transform( instance.node.getMatrix() );
          instance = instance.parent;
        }

        //OHTWO TODO: change only when necessary
        if ( !this.fitBounds.equals( this.oldFitBounds ) ) {
          // store our copy for future checks (and do it before we modify this.fitBounds)
          this.oldFitBounds.set( this.fitBounds );

          this.fitBounds.roundOut();
          this.fitBounds.dilate( 4 ); // for safety, modify in the future

          this.setSizeFitBounds();
        }
      }
      else {
        throw new Error( 'unknown fit' );
      }
    },

    setSizeFullDisplay: function() {
      // override in subtypes, use this.display.getSize()
    },

    setSizeFitBounds: function() {
      // override in subtypes, use this.fitBounds
    },

    dispose: function() {
      sceneryLog && sceneryLog.FittedBlock && sceneryLog.FittedBlock( 'dispose #' + this.id );

      if ( this.fit === FittedBlock.FULL_DISPLAY ) {
        this.display.offStatic( 'displaySize', this.dirtyFitListener );
      }

      // clear references
      this.transformRootInstance = null;
      this.commonFitInstance = null;

      Block.prototype.dispose.call( this );
    },

    onIntervalChange: function( firstDrawable, lastDrawable ) {
      sceneryLog && sceneryLog.FittedBlock && sceneryLog.FittedBlock( '#' + this.id + '.onIntervalChange ' + firstDrawable.toString() + ' to ' + lastDrawable.toString() );

      Block.prototype.onIntervalChange.call( this, firstDrawable, lastDrawable );

      // if we use a common ancestor fit, find the common ancestor instance
      if ( this.fit === FittedBlock.COMMON_ANCESTOR ) {
        assert && assert( firstDrawable.instance && lastDrawable.instance,
          'For common-ancestor SVG fits, we need the first and last drawables to have direct instance references' );

        var firstInstance = firstDrawable.instance;
        var lastInstance = lastDrawable.instance;

        // walk down the longest one until they are a common length
        var minLength = Math.min( firstInstance.trail.length, lastInstance.trail.length );
        while ( firstInstance.trail.length > minLength ) {
          firstInstance = firstInstance.parent;
        }
        while ( lastInstance.trail.length > minLength ) {
          lastInstance = lastInstance.parent;
        }

        // step down until they match
        while ( firstInstance !== lastInstance ) {
          firstInstance = firstInstance.parent;
          lastInstance = lastInstance.parent;
        }

        this.commonFitInstance = firstInstance;
        sceneryLog && sceneryLog.FittedBlock && sceneryLog.FittedBlock( '   common fit instance: ' + this.commonFitInstance.toString() );

        assert && assert( this.commonFitInstance.trail.length >= this.transformRootInstance.trail.length );

        this.markDirtyFit();
      }
    }
  } );

  FittedBlock.FULL_DISPLAY = 1;
  FittedBlock.COMMON_ANCESTOR = 2;

  FittedBlock.fitString = {
    1: 'fullDisplay',
    2: 'commonAncestor'
  };

  return FittedBlock;
} );
