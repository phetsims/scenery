// Copyright 2014-2015, University of Colorado Boulder

/**
 * A Block that needs to be fitted to either the screen bounds or other local bounds. This potentially reduces memory
 * usage and can make graphical operations in the browser faster, yet if the fit is rapidly changing could cause
 * performance degradation.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var Block = require( 'SCENERY/display/Block' );
  var Bounds2 = require( 'DOT/Bounds2' );
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  var Vector2 = require( 'DOT/Vector2' );

  var scratchBounds2 = Bounds2.NOTHING.copy();

  function FittedBlock( display, renderer, transformRootInstance ) {
    this.initialize( display, renderer, transformRootInstance );
  }

  scenery.register( 'FittedBlock', FittedBlock );

  inherit( Block, FittedBlock, {
    initializeFittedBlock: function( display, renderer, transformRootInstance, preferredFit ) {
      this.initializeBlock( display, renderer );

      this.transformRootInstance = transformRootInstance;

      assert && assert( typeof transformRootInstance.isDisplayRoot === 'boolean' );
      this.canBeFullDisplay = transformRootInstance.isDisplayRoot;

      assert && assert( preferredFit === FittedBlock.FULL_DISPLAY || preferredFit === FittedBlock.COMMON_ANCESTOR );

      // @private {FittedBlock.Fit} - Our preferred fit IF we can be fitted. Our fit can fall back if something's unfittable.
      this.preferredFit = preferredFit;

      // @protected {FittedBlock.Fit} - Our current fitting method.
      this.fit = preferredFit;

      this.dirtyFit = true;
      this.dirtyFitListener = this.dirtyFitListener || this.markDirtyFit.bind( this );
      this.commonFitInstance = null; // filled in if COMMON_ANCESTOR
      this.fitBounds = Bounds2.NOTHING.copy(); // tracks the "tight" bounds for fitting, not the actually-displayed bounds
      this.oldFitBounds = Bounds2.NOTHING.copy(); // copy for storage
      this.fitOffset = new Vector2( 0, 0 );

      // {number} - Number of child drawables that are marked as unfittable.
      this.unfittableDrawableCount = 0;

      this.fittableListener = this.onFittabilityChange.bind( this );

      // TODO: improve how we handle graphical acceleration with transforms
      this.forceAcceleration = false;

      // now we always add a listener to the display size to invalidate our fit
      this.display.onStatic( 'displaySize', this.dirtyFitListener );

      // TODO: add count of boundsless objects?
      return this;
    },

    /**
     * Changes the current fit, if it's currently different from the argument.
     * @private
     *
     * @param {FittedBlock.Fit} fit
     */
    setFit: function( fit ) {
      // We can't allow full-display fits sometimes
      if ( !this.canBeFullDisplay && fit === FittedBlock.FULL_DISPLAY ) {
        fit = FittedBlock.COMMON_ANCESTOR;
      }

      if ( this.fit !== fit ) {
        this.fit = fit;

        // updateFit() needs to be called in the repaint phase
        this.markDirtyFit();

        // Reset the oldFitBounds so that any updates that check bounds changes will update it.
        // TODO: remove duplication here with updateFit()
        this.oldFitBounds.set( Bounds2.NOTHING );

        // If we switched to the common-ancestor fit, we need to compute the common-ancestor instance.
        if ( fit === FittedBlock.COMMON_ANCESTOR ) {
          this.removeCommonFitInstance();
        }
      }
    },

    markDirtyFit: function() {
      sceneryLog && sceneryLog.dirty && sceneryLog.dirty( 'markDirtyFit on FittedBlock#' + this.id );
      this.dirtyFit = true;

      // Make sure we are visited in the repaint phase
      this.markDirty();
    },

    /*
     * Should be called from update() whenever this block is dirty
     */
    updateFit: function() {
      assert && assert( this.fit === FittedBlock.FULL_DISPLAY || this.fit === FittedBlock.COMMON_ANCESTOR,
        'Unsupported fit' );

      // check to see if we don't need to re-fit
      if ( !this.dirtyFit && this.fit === FittedBlock.FULL_DISPLAY ) {
        return;
      }

      sceneryLog && sceneryLog.FittedBlock && sceneryLog.FittedBlock( 'updateFit #' + this.id );

      this.dirtyFit = false;

      if ( this.fit === FittedBlock.COMMON_ANCESTOR && this.commonFitInstance === null ) {
        this.addCommonFitInstance( this.computeCommonAncestorInstance() );
      }

      // If our fit WAS common-ancestor and our common fit instance's subtree as something unfittable, switch to
      // full-display fit.
      if ( this.fit === FittedBlock.COMMON_ANCESTOR &&
           this.commonFitInstance.fittability.subtreeUnfittableCount > 0 &&
           this.canBeFullDisplay ) {
        // Reset the oldFitBounds so that any updates that check bounds changes will update it.
        this.oldFitBounds.set( Bounds2.NOTHING );

        this.fit = FittedBlock.FULL_DISPLAY;
      }

      if ( this.fit === FittedBlock.FULL_DISPLAY ) {
        this.fitBounds.set( Bounds2.NOTHING );

        this.setSizeFullDisplay();
      }
      else if ( this.fit === FittedBlock.COMMON_ANCESTOR ) {
        assert && assert( this.commonFitInstance.trail.length >= this.transformRootInstance.trail.length );

        // will trigger bounds validation (for now) until we have a better way of handling this
        this.fitBounds.set( this.commonFitInstance.node.getLocalBounds() );

        // walk it up, transforming so it is relative to our transform root
        var instance = this.commonFitInstance;
        while ( instance !== this.transformRootInstance ) {
          // shouldn't infinite loop, we'll null-pointer beforehand unless something is seriously wrong
          this.fitBounds.transform( instance.node.getMatrix() );
          instance = instance.parent;
        }

        this.fitBounds.roundOut();
        this.fitBounds.dilate( 4 ); // for safety, modify in the future

        // ensure that our fitted bounds don't go outside of our display's bounds (see https://github.com/phetsims/scenery/issues/390)
        if ( this.transformRootInstance.isDisplayRoot ) {
          // Only apply this effect if our transform root is the display root. Otherwise we might be transformed, and
          // this could cause buggy situations. See https://github.com/phetsims/scenery/issues/454
          scratchBounds2.setMinMax( 0, 0, this.display.width, this.display.height );
          this.fitBounds.constrainBounds( scratchBounds2 );
        }

        if ( !this.fitBounds.isValid() ) {
          this.fitBounds.setMinMax( 0, 0, 0, 0 );
        }

        if ( !this.fitBounds.equals( this.oldFitBounds ) ) {
          // store our copy for future checks (and do it before we modify this.fitBounds)
          this.oldFitBounds.set( this.fitBounds );

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

    addCommonFitInstance: function( instance ) {
      assert && assert( this.commonFitInstance === null );

      if ( instance ) {
        this.commonFitInstance = instance;
        this.commonFitInstance.fittability.subtreeFittabilityChange.addListener( this.dirtyFitListener );
      }
    },

    removeCommonFitInstance: function() {
      if ( this.commonFitInstance ) {
        this.commonFitInstance.fittability.subtreeFittabilityChange.removeListener( this.dirtyFitListener );
        this.commonFitInstance = null;
      }
    },

    dispose: function() {
      sceneryLog && sceneryLog.FittedBlock && sceneryLog.FittedBlock( 'dispose #' + this.id );

      this.display.offStatic( 'displaySize', this.dirtyFitListener );

      this.removeCommonFitInstance();

      // clear references
      this.transformRootInstance = null;

      Block.prototype.dispose.call( this );
    },

    /**
     * @override
     * Track the fittability of the added drawable.
     *
     * @param {Drawable} drawable
     */
    addDrawable: function( drawable ) {
      Block.prototype.addDrawable.call( this, drawable );

      drawable.onStatic( 'fittability', this.fittableListener );

      if ( !drawable.fittable ) {
        this.incrementUnfittable();
      }
    },

    /**
     * @override
     * Stop tracking the fittability of the removed drawable.
     *
     * @param {Drawable} drawable
     */
    removeDrawable: function( drawable ) {
      Block.prototype.removeDrawable.call( this, drawable );

      drawable.offStatic( 'fittability', this.fittableListener );

      if ( !drawable.fittable ) {
        this.decrementUnfittable();
      }
    },

    /**
     * Called from the fittability listener attached to child drawables when their fittability changes.
     * @private
     *
     * @param {Drawable} drawable
     */
    onFittabilityChange: function( drawable ) {
      assert && assert( drawable.parentDrawable === this );

      if ( drawable.isFittable() ) {
        this.decrementUnfittable();
      }
      else {
        this.incrementUnfittable();
      }
    },

    /**
     * The number of unfittable child drawables was increased by 1.
     * @private
     */
    incrementUnfittable: function() {
      this.unfittableDrawableCount++;

      if ( this.unfittableDrawableCount === 1 ) {
        this.checkFitConstraints();
      }
    },

    /**
     * The number of unfittable child drawables was decreased by 1.
     * @private
     */
    decrementUnfittable: function() {
      this.unfittableDrawableCount--;

      if ( this.unfittableDrawableCount === 0 ) {
        this.checkFitConstraints();
      }
    },

    /**
     * Check to make sure we are using the correct current fit.
     * @private
     */
    checkFitConstraints: function() {
      // If we have ANY unfittable drawables, take up the full display.
      if ( this.unfittableDrawableCount > 0 && this.canBeFullDisplay ) {
        this.setFit( FittedBlock.FULL_DISPLAY );
      }
      // Otherwise fall back to our "default"
      else {
        this.setFit( this.preferredFit );
      }
    },

    computeCommonAncestorInstance: function() {
      assert && assert( this.firstDrawable.instance && this.lastDrawable.instance,
        'For common-ancestor fits, we need the first and last drawables to have direct instance references' );

      var firstInstance = this.firstDrawable.instance;
      var lastInstance = this.lastDrawable.instance;

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

      var commonFitInstance = firstInstance;

      assert && assert( commonFitInstance.trail.length >= this.transformRootInstance.trail.length );

      return commonFitInstance;
    },

    onIntervalChange: function( firstDrawable, lastDrawable ) {
      sceneryLog && sceneryLog.FittedBlock && sceneryLog.FittedBlock( '#' + this.id + '.onIntervalChange ' + firstDrawable.toString() + ' to ' + lastDrawable.toString() );

      Block.prototype.onIntervalChange.call( this, firstDrawable, lastDrawable );

      // if we use a common ancestor fit, find the common ancestor instance
      if ( this.fit === FittedBlock.COMMON_ANCESTOR ) {
        this.removeCommonFitInstance();
        this.markDirtyFit();
      }
    }
  } );

  // Defines the FittedBlock.Fit enumeration type.
  FittedBlock.FULL_DISPLAY = 1;
  FittedBlock.COMMON_ANCESTOR = 2;

  FittedBlock.fitString = {
    1: 'fullDisplay',
    2: 'commonAncestor'
  };

  return FittedBlock;
} );
