// Copyright 2017-2020, University of Colorado Boulder

/**
 * Base type for controllers that create and keep an SVG gradient element up-to-date with a Scenery gradient.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import cleanArray from '../../../phet-core/js/cleanArray.js';
import scenery from '../scenery.js';
import SVGGradientStop from './SVGGradientStop.js';

class SVGGradient {
  /**
   * @param {SVGBlock} svgBlock
   * @param {Gradient} gradient
   */
  constructor( svgBlock, gradient ) {
    this.initialize( svgBlock, gradient );
  }

  /**
   * Poolable initializer.
   * @private
   *
   * @param {SVGBlock} svgBlock
   * @param {Gradient} gradient
   */
  initialize( svgBlock, gradient ) {
    sceneryLog && sceneryLog.Paints && sceneryLog.Paints( `[SVGGradient] initialize ${gradient.id}` );
    sceneryLog && sceneryLog.Paints && sceneryLog.push();

    // @private {SVGBlock} - transient
    this.svgBlock = svgBlock;

    // @private {Gradient} - transient
    this.gradient = gradient;

    const hasPreviousDefinition = this.definition !== undefined;

    // @public {SVGGradientElement} - persistent
    this.definition = this.definition || this.createDefinition();

    if ( !hasPreviousDefinition ) {
      // so we don't depend on the bounds of the object being drawn with the gradient
      this.definition.setAttribute( 'gradientUnits', 'userSpaceOnUse' );
    }

    if ( gradient.transformMatrix ) {
      this.definition.setAttribute( 'gradientTransform', gradient.transformMatrix.getSVGTransform() );
    }
    else {
      this.definition.removeAttribute( 'gradientTransform' );
    }

    // We need to make a function call, as stops need to be rescaled/reversed in some radial gradient cases.
    const gradientStops = gradient.getSVGStops();

    // @private {Array.<SVGGradientStop>} - transient
    this.stops = cleanArray( this.stops );
    for ( let i = 0; i < gradientStops.length; i++ ) {
      const stop = new SVGGradientStop( this, gradientStops[ i ].ratio, gradientStops[ i ].color );
      this.stops.push( stop );
      this.definition.appendChild( stop.svgElement );
    }

    // @private {boolean}
    this.dirty = false;

    sceneryLog && sceneryLog.Paints && sceneryLog.pop();
  }

  /**
   * Creates the gradient-type-specific definition.
   * @protected
   * @abstract
   *
   * @returns {SVGGradientElement}
   */
  createDefinition() {
    throw new Error( 'abstract method' );
  }

  /**
   * Called from SVGGradientStop when a stop needs to change the actual color.
   * @public
   */
  markDirty() {
    if ( !this.dirty ) {
      sceneryLog && sceneryLog.Paints && sceneryLog.Paints( `[SVGGradient] switched to dirty: ${this.gradient.id}` );
      sceneryLog && sceneryLog.Paints && sceneryLog.push();

      this.dirty = true;

      this.svgBlock.markDirtyGradient( this );

      sceneryLog && sceneryLog.Paints && sceneryLog.pop();
    }
  }

  /**
   * Called from SVGBlock when we need to update our color stops.
   * @public
   */
  update() {
    if ( !this.dirty ) {
      return;
    }
    this.dirty = false;

    sceneryLog && sceneryLog.Paints && sceneryLog.Paints( `[SVGGradient] update: ${this.gradient.id}` );
    sceneryLog && sceneryLog.Paints && sceneryLog.push();

    for ( let i = 0; i < this.stops.length; i++ ) {
      this.stops[ i ].update();
    }

    sceneryLog && sceneryLog.Paints && sceneryLog.pop();
  }

  /**
   * Disposes, so that it can be reused from the pool.
   * @public
   */
  dispose() {
    sceneryLog && sceneryLog.Paints && sceneryLog.Paints( `[SVGGradient] dispose ${this.gradient.id}` );
    sceneryLog && sceneryLog.Paints && sceneryLog.push();

    // Dispose and clean up stops
    for ( let i = 0; i < this.stops.length; i++ ) {
      const stop = this.stops[ i ]; // SVGGradientStop
      this.definition.removeChild( stop.svgElement );
      stop.dispose();
    }
    cleanArray( this.stops );

    this.svgBlock = null;
    this.gradient = null;

    this.freeToPool();

    sceneryLog && sceneryLog.Paints && sceneryLog.pop();
  }
}

scenery.register( 'SVGGradient', SVGGradient );

export default SVGGradient;