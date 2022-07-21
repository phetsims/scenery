// Copyright 2014-2022, University of Colorado Boulder

/**
 * A Block that needs to be fitted to either the screen bounds or other local bounds. This potentially reduces memory
 * usage and can make graphical operations in the browser faster, yet if the fit is rapidly changing could cause
 * performance degradation.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Bounds2 from '../../../dot/js/Bounds2.js';
import { Block, scenery } from '../imports.js';

const scratchBounds2 = Bounds2.NOTHING.copy();

class FittedBlock extends Block {
  /**
   * @public
   *
   * @param {Display} display
   * @param {number} renderer
   * @param {Instance} transformRootInstance
   * @param {FittedBlock.Fit} preferredFit
   * @returns {FittedBlock} - For chaining
   */
  initialize( display, renderer, transformRootInstance, preferredFit ) {
    super.initialize( display, renderer );

    // @private {Instance}
    this.transformRootInstance = transformRootInstance;

    assert && assert( typeof transformRootInstance.isDisplayRoot === 'boolean' );

    // @private {boolean}
    this.canBeFullDisplay = transformRootInstance.isDisplayRoot;

    assert && assert( preferredFit === FittedBlock.FULL_DISPLAY || preferredFit === FittedBlock.COMMON_ANCESTOR );

    // @private {FittedBlock.Fit} - Our preferred fit IF we can be fitted. Our fit can fall back if something's unfittable.
    this.preferredFit = preferredFit;

    // @protected {FittedBlock.Fit} - Our current fitting method.
    this.fit = preferredFit;

    // @protected {boolean}
    this.dirtyFit = true;

    // @private {Instance|null} - filled in if COMMON_ANCESTOR
    this.commonFitInstance = null;

    // @public {Bounds2} - tracks the "tight" bounds for fitting, not the actually-displayed bounds
    this.fitBounds = Bounds2.NOTHING.copy();

    // @private {Bounds2} - copy for storage
    this.oldFitBounds = Bounds2.NOTHING.copy();

    // {number} - Number of child drawables that are marked as unfittable.
    this.unfittableDrawableCount = 0;

    // @private {function}
    this.dirtyFitListener = this.dirtyFitListener || this.markDirtyFit.bind( this );
    this.fittableListener = this.fittableListener || this.onFittabilityChange.bind( this );

    // now we always add a listener to the display size to invalidate our fit
    this.display.sizeProperty.lazyLink( this.dirtyFitListener );

    // TODO: add count of boundsless objects?
    return this;
  }

  /**
   * Changes the current fit, if it's currently different from the argument.
   * @private
   *
   * @param {FittedBlock.Fit} fit
   */
  setFit( fit ) {
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
  }

  /**
   * @public
   */
  markDirtyFit() {
    sceneryLog && sceneryLog.dirty && sceneryLog.dirty( `markDirtyFit on FittedBlock#${this.id}` );
    this.dirtyFit = true;

    // Make sure we are visited in the repaint phase
    this.markDirty();
  }

  /**
   * Should be called from update() whenever this block is dirty
   * @protected
   */
  updateFit() {
    assert && assert( this.fit === FittedBlock.FULL_DISPLAY || this.fit === FittedBlock.COMMON_ANCESTOR,
      'Unsupported fit' );

    // check to see if we don't need to re-fit
    if ( !this.dirtyFit && this.fit === FittedBlock.FULL_DISPLAY ) {
      return;
    }

    sceneryLog && sceneryLog.FittedBlock && sceneryLog.FittedBlock( `updateFit #${this.id}` );

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
      let instance = this.commonFitInstance;
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
  }

  /**
   * @public
   */
  setSizeFullDisplay() {
    // override in subtypes, use this.display.getSize()
  }

  /**
   * @public
   */
  setSizeFitBounds() {
    // override in subtypes, use this.fitBounds
  }

  /**
   * @public
   *
   * @param {Instance|null} instance
   */
  addCommonFitInstance( instance ) {
    assert && assert( this.commonFitInstance === null );

    if ( instance ) {
      this.commonFitInstance = instance;
      this.commonFitInstance.fittability.subtreeFittabilityChangeEmitter.addListener( this.dirtyFitListener );
    }
  }

  /**
   * @public
   */
  removeCommonFitInstance() {
    if ( this.commonFitInstance ) {
      this.commonFitInstance.fittability.subtreeFittabilityChangeEmitter.removeListener( this.dirtyFitListener );
      this.commonFitInstance = null;
    }
  }

  /**
   * Releases references
   * @public
   * @override
   */
  dispose() {
    sceneryLog && sceneryLog.FittedBlock && sceneryLog.FittedBlock( `dispose #${this.id}` );

    this.display.sizeProperty.unlink( this.dirtyFitListener );

    this.removeCommonFitInstance();

    // clear references
    this.transformRootInstance = null;

    super.dispose();
  }

  /**
   * Track the fittability of the added drawable.
   * @public
   * @override
   *
   * @param {Drawable} drawable
   */
  addDrawable( drawable ) {
    super.addDrawable( drawable );

    drawable.fittableProperty.lazyLink( this.fittableListener );

    if ( !drawable.fittable ) {
      this.incrementUnfittable();
    }
  }

  /**
   * Stop tracking the fittability of the removed drawable.
   * @public
   * @override
   *
   * @param {Drawable} drawable
   */
  removeDrawable( drawable ) {
    super.removeDrawable( drawable );

    drawable.fittableProperty.unlink( this.fittableListener );

    if ( !drawable.fittable ) {
      this.decrementUnfittable();
    }
  }

  /**
   * Called from the fittability listener attached to child drawables when their fittability changes.
   * @private
   *
   * @param {boolean} fittable - Whether the particular child drawable is fittable
   */
  onFittabilityChange( fittable ) {
    if ( fittable ) {
      this.decrementUnfittable();
    }
    else {
      this.incrementUnfittable();
    }
  }

  /**
   * The number of unfittable child drawables was increased by 1.
   * @private
   */
  incrementUnfittable() {
    this.unfittableDrawableCount++;

    if ( this.unfittableDrawableCount === 1 ) {
      this.checkFitConstraints();
    }
  }

  /**
   * The number of unfittable child drawables was decreased by 1.
   * @private
   */
  decrementUnfittable() {
    this.unfittableDrawableCount--;

    if ( this.unfittableDrawableCount === 0 ) {
      this.checkFitConstraints();
    }
  }

  /**
   * Check to make sure we are using the correct current fit.
   * @private
   */
  checkFitConstraints() {
    // If we have ANY unfittable drawables, take up the full display.
    if ( this.unfittableDrawableCount > 0 && this.canBeFullDisplay ) {
      this.setFit( FittedBlock.FULL_DISPLAY );
    }
    // Otherwise fall back to our "default"
    else {
      this.setFit( this.preferredFit );
    }
  }

  /**
   * @private
   *
   * @returns {Instance}
   */
  computeCommonAncestorInstance() {
    assert && assert( this.firstDrawable.instance && this.lastDrawable.instance,
      'For common-ancestor fits, we need the first and last drawables to have direct instance references' );

    let firstInstance = this.firstDrawable.instance;
    let lastInstance = this.lastDrawable.instance;

    // walk down the longest one until they are a common length
    const minLength = Math.min( firstInstance.trail.length, lastInstance.trail.length );
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

    const commonFitInstance = firstInstance;

    assert && assert( commonFitInstance.trail.length >= this.transformRootInstance.trail.length );

    return commonFitInstance;
  }

  /**
   * @public
   * @override
   *
   * @param {Drawable} firstDrawable
   * @param {Drawable} lastDrawable
   */
  onIntervalChange( firstDrawable, lastDrawable ) {
    sceneryLog && sceneryLog.FittedBlock && sceneryLog.FittedBlock( `#${this.id}.onIntervalChange ${firstDrawable.toString()} to ${lastDrawable.toString()}` );

    super.onIntervalChange( firstDrawable, lastDrawable );

    // if we use a common ancestor fit, find the common ancestor instance
    if ( this.fit === FittedBlock.COMMON_ANCESTOR ) {
      this.removeCommonFitInstance();
      this.markDirtyFit();
    }
  }
}

scenery.register( 'FittedBlock', FittedBlock );

// Defines the FittedBlock.Fit enumeration type.
FittedBlock.FULL_DISPLAY = 1;
FittedBlock.COMMON_ANCESTOR = 2;

// TODO: enumeration these?
FittedBlock.fitString = {
  1: 'fullDisplay',
  2: 'commonAncestor'
};

export default FittedBlock;