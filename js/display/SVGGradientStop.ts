// Copyright 2017-2022, University of Colorado Boulder

/**
 * Handles creation of an SVG stop element, and handles keeping it updated based on property/color changes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import ReadOnlyProperty from '../../../axon/js/ReadOnlyProperty.js';
import Pool, { TPoolable } from '../../../phet-core/js/Pool.js';
import WithoutNull from '../../../phet-core/js/types/WithoutNull.js';
import { ActiveSVGGradient, Color, TColor, scenery, svgns } from '../imports.js';

const scratchColor = new Color( 'transparent' );

export type ActiveSVGGradientStop = WithoutNull<SVGGradientStop, 'svgGradient'>;

class SVGGradientStop implements TPoolable {

  // persistent
  public svgElement!: SVGStopElement;

  // transient
  public svgGradient!: ActiveSVGGradient | null;
  public color!: TColor;

  public ratio!: number;
  private dirty!: boolean;
  private propertyListener!: () => void;
  private colorListener!: () => void;

  public constructor( svgGradient: ActiveSVGGradient, ratio: number, color: TColor ) {
    this.initialize( svgGradient, ratio, color );
  }

  public isActiveSVGGradientStop(): this is ActiveSVGGradientStop { return !!this.svgGradient; }

  public initialize( svgGradient: ActiveSVGGradient, ratio: number, color: TColor ): this {
    sceneryLog && sceneryLog.Paints && sceneryLog.Paints( `[SVGGradientStop] initialize: ${svgGradient.gradient.id} : ${ratio}` );
    sceneryLog && sceneryLog.Paints && sceneryLog.push();

    this.svgGradient = svgGradient;
    this.color = color;
    this.ratio = ratio;

    this.svgElement = this.svgElement || document.createElementNS( svgns, 'stop' );

    this.svgElement.setAttribute( 'offset', '' + ratio );

    this.dirty = true; // true here so our update() actually properly initializes

    this.update();

    this.propertyListener = this.propertyListener || this.onPropertyChange.bind( this );
    this.colorListener = this.colorListener || this.markDirty.bind( this );

    if ( color instanceof ReadOnlyProperty ) {
      sceneryLog && sceneryLog.Paints && sceneryLog.Paints( `[SVGGradientStop] adding Property listener: ${this.svgGradient.gradient.id} : ${this.ratio}` );
      color.lazyLink( this.propertyListener );
      if ( color.value instanceof Color ) {
        sceneryLog && sceneryLog.Paints && sceneryLog.Paints( `[SVGGradientStop] adding Color listener: ${this.svgGradient.gradient.id} : ${this.ratio}` );
        color.value.changeEmitter.addListener( this.colorListener );
      }
    }
    else if ( color instanceof Color ) {
      sceneryLog && sceneryLog.Paints && sceneryLog.Paints( `[SVGGradientStop] adding Color listener: ${this.svgGradient.gradient.id} : ${this.ratio}` );
      color.changeEmitter.addListener( this.colorListener );
    }

    sceneryLog && sceneryLog.Paints && sceneryLog.pop();

    return this;
  }

  /**
   * Called when our color is a Property and it changes.
   */
  private onPropertyChange( newValue: Color | string | null, oldValue: Color | string | null ): void {
    assert && assert( this.isActiveSVGGradientStop() );
    const activeSelf = this as ActiveSVGGradientStop;

    if ( oldValue instanceof Color ) {
      sceneryLog && sceneryLog.Paints && sceneryLog.Paints( `[SVGGradientStop] removing Color listener: ${activeSelf.svgGradient.gradient.id} : ${this.ratio}` );
      oldValue.changeEmitter.removeListener( this.colorListener );
    }
    if ( newValue instanceof Color ) {
      sceneryLog && sceneryLog.Paints && sceneryLog.Paints( `[SVGGradientStop] adding Color listener: ${activeSelf.svgGradient.gradient.id} : ${this.ratio}` );
      newValue.changeEmitter.addListener( this.colorListener );
    }

    this.markDirty();
  }

  /**
   * Should be called when the color stop's value may have changed.
   */
  private markDirty(): void {
    assert && assert( this.isActiveSVGGradientStop() );

    this.dirty = true;
    ( this as ActiveSVGGradientStop ).svgGradient.markDirty();
  }

  /**
   * Updates the color stop to whatever the current color should be.
   */
  public update(): void {
    if ( !this.dirty ) {
      return;
    }
    this.dirty = false;

    assert && assert( this.isActiveSVGGradientStop() );
    const activeSelf = this as ActiveSVGGradientStop;

    sceneryLog && sceneryLog.Paints && sceneryLog.Paints( `[SVGGradientStop] update: ${activeSelf.svgGradient.gradient.id} : ${this.ratio}` );
    sceneryLog && sceneryLog.Paints && sceneryLog.push();

    // {Color|string|Property.<Color|string|null>|null}
    let color = this.color;

    // to {Color|string|null}
    if ( color instanceof ReadOnlyProperty ) {
      color = color.value;
    }

    // to {Color|string}
    if ( color === null ) {
      color = 'transparent';
    }

    // to {Color}, in our scratchColor
    if ( typeof color === 'string' ) {
      scratchColor.setCSS( color );
    }
    else {
      scratchColor.set( color as Color );
    }

    // Since SVG doesn't support parsing scientific notation (e.g. 7e5), we need to output fixed decimal-point strings.
    // Since this needs to be done quickly, and we don't particularly care about slight rounding differences (it's
    // being used for display purposes only, and is never shown to the user), we use the built-in JS toFixed instead of
    // Dot's version of toFixed. See https://github.com/phetsims/kite/issues/50
    const stopOpacityRule = `stop-opacity: ${scratchColor.a.toFixed( 20 )};`; // eslint-disable-line bad-sim-text

    // For GC, mutate the color so it is just RGB and output that CSS also
    scratchColor.alpha = 1;
    const stopColorRule = `stop-color: ${scratchColor.toCSS()};`;

    this.svgElement.setAttribute( 'style', `${stopColorRule} ${stopOpacityRule}` );

    sceneryLog && sceneryLog.Paints && sceneryLog.pop();
  }

  /**
   * Disposes, so that it can be reused from the pool.
   */
  public dispose(): void {
    assert && assert( this.isActiveSVGGradientStop() );
    const activeSelf = this as ActiveSVGGradientStop;

    sceneryLog && sceneryLog.Paints && sceneryLog.Paints( `[SVGGradientStop] dispose: ${activeSelf.svgGradient.gradient.id} : ${this.ratio}` );
    sceneryLog && sceneryLog.Paints && sceneryLog.push();

    const color = this.color;

    if ( color instanceof ReadOnlyProperty ) {
      sceneryLog && sceneryLog.Paints && sceneryLog.Paints( `[SVGGradientStop] removing Property listener: ${activeSelf.svgGradient.gradient.id} : ${this.ratio}` );
      if ( color.hasListener( this.propertyListener ) ) {
        color.unlink( this.propertyListener );
      }
      if ( color.value instanceof Color ) {
        sceneryLog && sceneryLog.Paints && sceneryLog.Paints( `[SVGGradientStop] removing Color listener: ${activeSelf.svgGradient.gradient.id} : ${this.ratio}` );
        color.value.changeEmitter.removeListener( this.colorListener );
      }
    }
    else if ( color instanceof Color ) {
      sceneryLog && sceneryLog.Paints && sceneryLog.Paints( `[SVGGradientStop] removing Color listener: ${activeSelf.svgGradient.gradient.id} : ${this.ratio}` );
      color.changeEmitter.removeListener( this.colorListener );
    }

    this.color = null; // clear the reference
    this.svgGradient = null; // clear the reference

    this.freeToPool();

    sceneryLog && sceneryLog.Paints && sceneryLog.pop();
  }

  public freeToPool(): void {
    SVGGradientStop.pool.freeToPool( this );
  }

  public static readonly pool = new Pool( SVGGradientStop );
}

scenery.register( 'SVGGradientStop', SVGGradientStop );

export default SVGGradientStop;