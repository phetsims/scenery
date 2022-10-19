// Copyright 2013-2022, University of Colorado Boulder

/**
 * Abstract base type for LinearGradient and RadialGradient.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import ReadOnlyProperty from '../../../axon/js/ReadOnlyProperty.js';
import cleanArray from '../../../phet-core/js/cleanArray.js';
import { Color, TColor, Paint, scenery } from '../imports.js';

export type GradientStop = {
  ratio: number;
  color: TColor;
};

export default abstract class Gradient extends Paint {

  public stops: GradientStop[]; // (scenery-internal)
  private lastStopRatio: number;
  private canvasGradient: CanvasGradient | null; // lazily created

  // Whether we should force a check of whether stops have changed
  private colorStopsDirty: boolean;

  // Used to check to see if colors have changed since last time
  private lastColorStopValues: string[];

  /**
   * TODO: add the ability to specify the color-stops inline. possibly [ [0,color1], [0.5,color2], [1,color3] ]
   */
  public constructor() {
    super();

    assert && assert( this.constructor.name !== 'Gradient',
      'Please create a LinearGradient or RadialGradient. Do not directly use the supertype Gradient.' );

    this.stops = [];
    this.lastStopRatio = 0;
    this.canvasGradient = null;
    this.colorStopsDirty = false;
    this.lastColorStopValues = [];
  }


  /**
   * Adds a color stop to the gradient.
   *
   * Color stops should be added in order (monotonically increasing ratio values).
   *
   * NOTE: Color stops should only be added before using the gradient as a fill/stroke. Adding stops afterwards
   *       will result in undefined behavior.
   * TODO: Catch attempts to do the above.
   *
   * @param ratio - Monotonically increasing value in the range of 0 to 1
   * @param color
   * @returns - for chaining
   */
  public addColorStop( ratio: number, color: TColor ): this {
    assert && assert( ratio >= 0 && ratio <= 1, 'Ratio needs to be between 0,1 inclusively' );
    assert && assert( color === null ||
                      typeof color === 'string' ||
                      color instanceof Color ||
                      ( color instanceof ReadOnlyProperty && ( color.value === null ||
                                                               typeof color.value === 'string' ||
                                                               color.value instanceof Color ) ),
      'Color should match the addColorStop type specification' );

    if ( this.lastStopRatio > ratio ) {
      // fail out, since browser quirks go crazy for this case
      throw new Error( 'Color stops not specified in the order of increasing ratios' );
    }
    else {
      this.lastStopRatio = ratio;
    }

    this.stops.push( {
      ratio: ratio,
      color: color
    } );

    // Easiest to just push a value here, so that it is always the same length as the stops array.
    this.lastColorStopValues.push( '' );

    return this;
  }

  /**
   * Subtypes should return a fresh CanvasGradient type.
   */
  public abstract createCanvasGradient(): CanvasGradient;

  /**
   * Returns stops suitable for direct SVG use.
   */
  public getSVGStops(): GradientStop[] {
    return this.stops;
  }

  /**
   * Forces a re-check of whether colors have changed, so that the Canvas gradient can be regenerated if
   * necessary.
   */
  public invalidateCanvasGradient(): void {
    sceneryLog && sceneryLog.Paints && sceneryLog.Paints( `Invalidated Canvas Gradient for #${this.id}` );
    this.colorStopsDirty = true;
  }

  /**
   * Compares the current color values with the last-recorded values for the current Canvas gradient.
   *
   * This is needed since the values of color properties (or the color itself) may change.
   */
  private haveCanvasColorStopsChanged(): boolean {
    if ( this.lastColorStopValues === null ) {
      return true;
    }

    for ( let i = 0; i < this.stops.length; i++ ) {
      if ( Gradient.colorToString( this.stops[ i ].color ) !== this.lastColorStopValues[ i ] ) {
        return true;
      }
    }

    return false;
  }

  /**
   * Returns an object that can be passed to a Canvas context's fillStyle or strokeStyle.
   */
  public getCanvasStyle(): CanvasGradient {
    // Check if we need to regenerate the Canvas gradient
    if ( !this.canvasGradient || ( this.colorStopsDirty && this.haveCanvasColorStopsChanged() ) ) {
      sceneryLog && sceneryLog.Paints && sceneryLog.Paints( `Regenerating Canvas Gradient for #${this.id}` );
      sceneryLog && sceneryLog.Paints && sceneryLog.push();

      this.colorStopsDirty = false;

      cleanArray( this.lastColorStopValues );
      this.canvasGradient = this.createCanvasGradient();

      for ( let i = 0; i < this.stops.length; i++ ) {
        const stop = this.stops[ i ];

        const colorString = Gradient.colorToString( stop.color );
        this.canvasGradient.addColorStop( stop.ratio, colorString );

        // Save it so we can compare next time whether our generated gradient would have changed
        this.lastColorStopValues.push( colorString );
      }

      sceneryLog && sceneryLog.Paints && sceneryLog.pop();
    }

    return this.canvasGradient;
  }

  /**
   * Returns the current value of the generally-allowed color types for Gradient, as a string.
   */
  public static colorToString( color: TColor ): string {
    // to {Color|string|null}
    if ( color instanceof ReadOnlyProperty ) {
      color = color.value;
    }

    // to {Color|string}
    if ( color === null ) {
      color = 'transparent';
    }

    // to {string}
    if ( color instanceof Color ) {
      color = color.toCSS();
    }

    return color as string;
  }

  public isGradient!: boolean;
}

Gradient.prototype.isGradient = true;

scenery.register( 'Gradient', Gradient );
