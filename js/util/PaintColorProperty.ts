// Copyright 2018-2022, University of Colorado Boulder

/**
 * A Property that will always hold a `Color` object representing the current value of a given paint (and can be set to
 * different paints).
 *
 * This is valuable, since:
 * ```
 *   const color = new scenery.Color( 'red' );
 *   const fill = new axon.Property( color );
 *   const paintColorProperty = new scenery.PaintColorProperty( fill );
 *
 *   // value is converted to a {Color}
 *   paintColorProperty.value; // r: 255, g: 0, b: 0, a: 1
 *
 *   // watches direct Color mutation
 *   color.red = 128;
 *   paintColorProperty.value; // r: 128, g: 0, b: 0, a: 1
 *
 *   // watches the Property mutation
 *   fill.value = 'green';
 *   paintColorProperty.value; // r: 0, g: 128, b: 0, a: 1
 *
 *   // can switch to a different paint
 *   paintColorProperty.paint = 'blue';
 *   paintColorProperty.value; // r: 0, g: 0, b: 255, a: 1
 * ```
 *
 * Basically, you don't have to add your own listeners to both (optionally) any Properties in a paint and (optionally)
 * any Color objects (since it's all handled).
 *
 * This is particularly helpful to create paints that are either lighter or darker than an original paint (where it
 * will update its color value when the original is updated).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Property, { PropertyOptions } from '../../../axon/js/Property.js';
import optionize from '../../../phet-core/js/optionize.js';
import { scenery, PaintDef, PaintObserver, Color, IPaint } from '../imports.js';

type SelfOptions = {
  // 0 applies no change. Positive numbers brighten the color up to 1 (white). Negative numbers darken
  // the color up to -1 (black). See setLuminanceFactor() for more information.
  luminanceFactor?: number;
};

export type PaintColorPropertyOptions = SelfOptions & PropertyOptions<Color>;

class PaintColorProperty extends Property<Color> {

  private _paint: IPaint;

  // See setLuminanceFactor() for more information.
  private _luminanceFactor: number;

  // Our "paint changed" listener, will update the value of this Property.
  private _changeListener: () => void;

  private _paintObserver: PaintObserver;

  constructor( paint: IPaint, providedOptions?: PaintColorPropertyOptions ) {
    const initialColor = PaintDef.toColor( paint );

    const options = optionize<PaintColorPropertyOptions, SelfOptions, PropertyOptions<Color>>( {
      luminanceFactor: 0,

      // Property options
      useDeepEquality: true // We don't need to renotify for equivalent colors
    }, providedOptions );

    super( initialColor, options );

    this._paint = null;
    this._luminanceFactor = options.luminanceFactor;
    this._changeListener = this.invalidatePaint.bind( this );
    this._paintObserver = new PaintObserver( this._changeListener );

    this.setPaint( paint );
  }

  /**
   * Sets the current paint of the PaintColorProperty.
   */
  setPaint( paint: IPaint ) {
    assert && assert( PaintDef.isPaintDef( paint ) );

    this._paint = paint;
    this._paintObserver.setPrimary( paint );
  }

  set paint( value ) { this.setPaint( value ); }

  /**
   * Returns the current paint.
   */
  getPaint(): IPaint {
    return this._paint;
  }

  get paint(): IPaint { return this.getPaint(); }

  /**
   * Sets the current value used for adjusting the brightness or darkness (luminance) of the color.
   *
   * If this factor is a non-zero value, the value of this Property will be either a brightened or darkened version of
   * the paint (depending on the value of the factor). 0 applies no change. Positive numbers brighten the color up to
   * 1 (white). Negative numbers darken the color up to -1 (black).
   *
   * For example, if the given paint is blue, the below factors will result in:
   *
   *   -1: black
   * -0.5: dark blue
   *    0: blue
   *  0.5: light blue
   *    1: white
   *
   * With intermediate values basically "interpolated". This uses the `Color` colorUtilsBrightness method to adjust
   * the paint.
   */
  setLuminanceFactor( luminanceFactor: number ) {
    assert && assert( typeof luminanceFactor === 'number' && luminanceFactor >= -1 && luminanceFactor <= 1 );

    if ( this.luminanceFactor !== luminanceFactor ) {
      this._luminanceFactor = luminanceFactor;

      this.invalidatePaint();
    }
  }

  set luminanceFactor( value: number ) { this.setLuminanceFactor( value ); }

  /**
   * Returns the current value used for adjusting the brightness or darkness (luminance) of the color.
   *
   * See setLuminanceFactor() for more information.
   */
  getLuminanceFactor(): number {
    return this._luminanceFactor;
  }

  get luminanceFactor(): number { return this.getLuminanceFactor(); }

  /**
   * Updates the value of this Property.
   */
  private invalidatePaint() {
    this.value = PaintDef.toColor( this._paint ).colorUtilsBrightness( this._luminanceFactor );
  }

  /**
   * Releases references.
   */
  dispose() {
    this.paint = null;

    super.dispose();
  }
}

scenery.register( 'PaintColorProperty', PaintColorProperty );
export default PaintColorProperty;