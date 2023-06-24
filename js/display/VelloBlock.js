// Copyright 2023, University of Colorado Boulder

/**
 * Renders a visual layer of WebGL drawables.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 * @author Sam Reid (PhET Interactive Simulations)
 * @author Sharfudeen Ashraf (For Ghent University)
 */

import cleanArray from '../../../phet-core/js/cleanArray.js';
import Poolable from '../../../phet-core/js/Poolable.js';
import { FittedBlock, scenery, Utils } from '../imports.js';
import { Affine } from './vello/Affine.js';
import DeviceContext from './vello/DeviceContext.js';
import PhetEncoding from './vello/PhetEncoding.js';
import render from './vello/render.js';

// TODO: don't let this hackiness in
PhetEncoding.load();

class VelloBlock extends FittedBlock {
  /**
   * @mixes Poolable
   *
   * @param {Display} display
   * @param {number} renderer
   * @param {Instance} transformRootInstance
   * @param {Instance} filterRootInstance
   */
  constructor( display, renderer, transformRootInstance, filterRootInstance ) {
    super();

    this.initialize( display, renderer, transformRootInstance, filterRootInstance );
  }

  /**
   * @public
   *
   * @param {Display} display
   * @param {number} renderer
   * @param {Instance} transformRootInstance
   * @param {Instance} filterRootInstance
   * @returns {VelloBlock} - For chaining
   */
  initialize( display, renderer, transformRootInstance, filterRootInstance ) {
    sceneryLog && sceneryLog.VelloBlock && sceneryLog.VelloBlock( `initialize #${this.id}` );
    sceneryLog && sceneryLog.VelloBlock && sceneryLog.push();

    // VelloBlocks are hard-coded to take the full display size (as opposed to svg and canvas)
    // Since we saw some jitter on iPad, see #318 and generally expect WebGPU layers to span the entire display
    // In the future, it would be good to understand what was causing the problem and make webgl consistent
    // with svg and canvas again.
    // TODO: Don't have it be a "Fitted" block then
    super.initialize( display, renderer, transformRootInstance, FittedBlock.FULL_DISPLAY );

    // TODO: Uhh, is this not used?
    this.filterRootInstance = filterRootInstance;

    // // {boolean} - Whether we pass this flag to the WebGL Context. It will store the contents displayed on the screen,
    // // so that canvas.toDataURL() will work. It also requires clearing the context manually ever frame. Both incur
    // // performance costs, so it should be false by default.
    // // TODO: This block can be shared across displays, so we need to handle preserveDrawingBuffer separately?
    // this.preserveDrawingBuffer = display._preserveDrawingBuffer;

    // list of {Drawable}s that need to be updated before we update
    this.dirtyDrawables = cleanArray( this.dirtyDrawables );

    if ( !this.domElement ) {
      // @public (scenery-internal) {HTMLCanvasElement} - Div wrapper used so we can switch out Canvases if necessary.
      this.canvas = document.createElement( 'canvas' );
      this.canvas.style.position = 'absolute';
      this.canvas.style.left = '0';
      this.canvas.style.top = '0';
      this.canvas.style.pointerEvents = 'none';

      // @private {number} - unique ID so that we can support rasterization with Display.foreignObjectRasterization
      this.canvasId = this.canvas.id = `scenery-vello${this.id}`;

      this.domElement = this.canvas;

      // TODO: handle context restoration/loss
      this.deviceContext = DeviceContext.getSync();
      this.canvasContext = this.deviceContext.getCanvasContext( this.canvas );
    }

    // TODO: perhaps these might reduce performance? kill them?
    // reset any fit transforms that were applied
    Utils.prepareForTransform( this.canvas ); // Apply CSS needed for future CSS transforms to work properly.
    Utils.unsetTransform( this.canvas ); // clear out any transforms that could have been previously applied

    sceneryLog && sceneryLog.VelloBlock && sceneryLog.pop();

    return this;
  }

  /**
   * @public
   * @override
   */
  setSizeFullDisplay() {
    const size = this.display.getSize();
    this.canvas.width = Math.ceil( size.width * window.devicePixelRatio );
    this.canvas.height = Math.ceil( size.height * window.devicePixelRatio );
    this.canvas.style.width = `${size.width}px`;
    this.canvas.style.height = `${size.height}px`;
  }

  /**
   * @public
   * @override
   */
  setSizeFitBounds() {
    throw new Error( 'setSizeFitBounds unimplemented for VelloBlock' );
  }

  /**
   * Updates the DOM appearance of this drawable (whether by preparing/calling draw calls, DOM element updates, etc.)
   * @public
   * @override
   *
   * @returns {boolean} - Whether the update should continue (if false, further updates in supertype steps should not
   *                      be done).
   */
  update() {
    // See if we need to actually update things (will bail out if we are not dirty, or if we've been disposed)
    if ( !super.update() ) {
      return false;
    }

    sceneryLog && sceneryLog.VelloBlock && sceneryLog.VelloBlock( `update #${this.id}` );
    sceneryLog && sceneryLog.VelloBlock && sceneryLog.push();

    // update drawables, so that they have vertex arrays up to date, etc.
    while ( this.dirtyDrawables.length ) {
      this.dirtyDrawables.pop().update();
    }

    // update the fit BEFORE drawing, since it may change our offset
    // TODO: make not fittable
    this.updateFit();

    const sceneEncoding = new phet.scenery.PhetEncoding();
    sceneEncoding.reset( false );

    const encoding = new phet.scenery.PhetEncoding();

    // Iterate through all of our drawables (linked list)
    //OHTWO TODO: PERFORMANCE: create an array for faster drawable iteration (this is probably a hellish memory access pattern)
    for ( let drawable = this.firstDrawable; drawable !== null; drawable = drawable.nextDrawable ) {
      // ignore invisible drawables
      if ( drawable.visible ) {
        encoding.append( drawable.encoding );
      }

      // exit loop end case
      if ( drawable === this.lastDrawable ) { break; }
    }

    // TODO: really get rid of Affine!!!
    sceneEncoding.append( encoding, new Affine( window.devicePixelRatio, 0, 0, window.devicePixelRatio, 0, 0 ) );
    sceneEncoding.finalize_scene();

    const outTexture = this.canvasContext.getCurrentTexture();
    console.log( outTexture.width );
    const renderInfo = sceneEncoding.resolve( this.deviceContext );
    renderInfo.prepareRender( outTexture.width, outTexture.height, 0 );

    render( renderInfo, this.deviceContext, outTexture );

    sceneryLog && sceneryLog.VelloBlock && sceneryLog.pop();

    return true;
  }

  /**
   * Releases references
   * @public
   */
  dispose() {
    sceneryLog && sceneryLog.VelloBlock && sceneryLog.VelloBlock( `dispose #${this.id}` );

    // TODO: many things to dispose!?

    // clear references
    cleanArray( this.dirtyDrawables );

    super.dispose();
  }

  /**
   * @public
   *
   * @param {Drawable} drawable
   */
  markDirtyDrawable( drawable ) {
    sceneryLog && sceneryLog.dirty && sceneryLog.dirty( `markDirtyDrawable on VelloBlock#${this.id} with ${drawable.toString()}` );

    assert && assert( drawable );
    assert && assert( !drawable.isDisposed );

    // TODO: instance check to see if it is a canvas cache (usually we don't need to call update on our drawables)
    this.dirtyDrawables.push( drawable );
    this.markDirty();
  }

  /**
   * @public
   * @override
   *
   * @param {Drawable} drawable
   */
  addDrawable( drawable ) {
    sceneryLog && sceneryLog.VelloBlock && sceneryLog.VelloBlock( `#${this.id}.addDrawable ${drawable.toString()}` );

    super.addDrawable( drawable );
  }

  /**
   * @public
   * @override
   *
   * @param {Drawable} drawable
   */
  removeDrawable( drawable ) {
    sceneryLog && sceneryLog.VelloBlock && sceneryLog.VelloBlock( `#${this.id}.removeDrawable ${drawable.toString()}` );

    // Ensure a removed drawable is not present in the dirtyDrawables array afterwards. Don't want to update it.
    // See https://github.com/phetsims/scenery/issues/635
    let index = 0;
    while ( ( index = this.dirtyDrawables.indexOf( drawable, index ) ) >= 0 ) {
      this.dirtyDrawables.splice( index, 1 );
    }

    super.removeDrawable( drawable );
  }

  /**
   * @public
   * @override
   *
   * @param {Drawable} firstDrawable
   * @param {Drawable} lastDrawable
   */
  onIntervalChange( firstDrawable, lastDrawable ) {
    sceneryLog && sceneryLog.VelloBlock && sceneryLog.VelloBlock( `#${this.id}.onIntervalChange ${firstDrawable.toString()} to ${lastDrawable.toString()}` );

    super.onIntervalChange( firstDrawable, lastDrawable );

    this.markDirty();
  }

  /**
   * @public
   *
   * @param {Drawable} drawable
   */
  onPotentiallyMovedDrawable( drawable ) {
    sceneryLog && sceneryLog.VelloBlock && sceneryLog.VelloBlock( `#${this.id}.onPotentiallyMovedDrawable ${drawable.toString()}` );
    sceneryLog && sceneryLog.VelloBlock && sceneryLog.push();

    assert && assert( drawable.parentDrawable === this );

    this.markDirty();

    sceneryLog && sceneryLog.VelloBlock && sceneryLog.pop();
  }

  /**
   * Returns a string form of this object
   * @public
   *
   * @returns {string}
   */
  toString() {
    return `VelloBlock#${this.id}-${FittedBlock.fitString[ this.fit ]}`;
  }
}

scenery.register( 'VelloBlock', VelloBlock );

Poolable.mixInto( VelloBlock );

export default VelloBlock;