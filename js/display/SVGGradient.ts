// Copyright 2017-2022, University of Colorado Boulder

/**
 * Base type for controllers that create and keep an SVG gradient element up-to-date with a Scenery gradient.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import cleanArray from '../../../phet-core/js/cleanArray.js';
import WithoutNull from '../../../phet-core/js/types/WithoutNull.js';
import { Gradient, scenery, SVGBlock, SVGGradientStop } from '../imports.js';

export type ActiveSVGGradient = WithoutNull<SVGGradient, 'svgBlock' | 'gradient'>;

export default abstract class SVGGradient {

  // transient (scenery-internal)
  public svgBlock!: SVGBlock | null;
  public gradient!: Gradient | null;
  public stops!: SVGGradientStop[];

  // persistent
  public definition!: SVGGradientElement;

  private dirty!: boolean;

  public constructor( svgBlock: SVGBlock, gradient: Gradient ) {
    this.initialize( svgBlock, gradient );
  }

  public isActiveSVGGradient(): this is ActiveSVGGradient { return !!this.svgBlock; }

  public initialize( svgBlock: SVGBlock, gradient: Gradient ): void {
    sceneryLog && sceneryLog.Paints && sceneryLog.Paints( `[SVGGradient] initialize ${gradient.id}` );
    sceneryLog && sceneryLog.Paints && sceneryLog.push();

    this.svgBlock = svgBlock;
    this.gradient = gradient;

    const hasPreviousDefinition = this.definition !== undefined;

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

    this.stops = cleanArray( this.stops );
    for ( let i = 0; i < gradientStops.length; i++ ) {
      const stop = new SVGGradientStop( this as ActiveSVGGradient, gradientStops[ i ].ratio, gradientStops[ i ].color );
      this.stops.push( stop );
      this.definition.appendChild( stop.svgElement );
    }

    this.dirty = false;

    sceneryLog && sceneryLog.Paints && sceneryLog.pop();
  }

  /**
   * Creates the gradient-type-specific definition.
   */
  protected abstract createDefinition(): SVGGradientElement;

  /**
   * Called from SVGGradientStop when a stop needs to change the actual color.
   */
  public markDirty(): void {
    if ( !this.dirty ) {
      assert && assert( this.isActiveSVGGradient() );
      const activeGradient = this as ActiveSVGGradient;

      sceneryLog && sceneryLog.Paints && sceneryLog.Paints( `[SVGGradient] switched to dirty: ${this.gradient!.id}` );
      sceneryLog && sceneryLog.Paints && sceneryLog.push();

      this.dirty = true;

      activeGradient.svgBlock.markDirtyGradient( this );

      sceneryLog && sceneryLog.Paints && sceneryLog.pop();
    }
  }

  /**
   * Called from SVGBlock when we need to update our color stops.
   */
  public update(): void {
    if ( !this.dirty ) {
      return;
    }
    this.dirty = false;

    sceneryLog && sceneryLog.Paints && sceneryLog.Paints( `[SVGGradient] update: ${this.gradient!.id}` );
    sceneryLog && sceneryLog.Paints && sceneryLog.push();

    for ( let i = 0; i < this.stops.length; i++ ) {
      this.stops[ i ].update();
    }

    sceneryLog && sceneryLog.Paints && sceneryLog.pop();
  }

  /**
   * Disposes, so that it can be reused from the pool.
   */
  public dispose(): void {
    sceneryLog && sceneryLog.Paints && sceneryLog.Paints( `[SVGGradient] dispose ${this.gradient!.id}` );
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

  public abstract freeToPool(): void;
}

scenery.register( 'SVGGradient', SVGGradient );
