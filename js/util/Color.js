// Copyright 2012-2020, University of Colorado Boulder

/**
 * A color with RGBA values, assuming the sRGB color space is used.
 *
 * See http://www.w3.org/TR/css3-color/
 *
 * TODO: make a getHue, getSaturation, getLightness. we can then expose them via ES5!
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Property from '../../../axon/js/Property.js';
import TinyEmitter from '../../../axon/js/TinyEmitter.js';
import Utils from '../../../dot/js/Utils.js';
import IOType from '../../../tandem/js/types/IOType.js';
import scenery from '../scenery.js';

// constants
const clamp = Utils.clamp;
const linear = Utils.linear;

class Color {
  /**
   * Creates a Color with an initial value. Multiple different types of parameters are supported:
   * - new Color( color ) is a copy constructor, for a {Color}
   * - new Color( string ) will parse the string assuming it's a CSS-compatible color, e.g. set( 'red' )
   * - new Color( r, g, b ) is equivalent to setRGBA( r, g, b, 1 ), e.g. set( 255, 0, 128 )
   * - new Color( r, g, b, a ) is equivalent to setRGBA( r, g, b, a ), e.g. set( 255, 0, 128, 0.5 )
   * - new Color( hex ) will set RGB with alpha=1, e.g. set( 0xFF0000 )
   * - new Color( hex, a ) will set RGBA, e.g. set( 0xFF0000, 1 )
   *
   * The 'r', 'g', and 'b' values stand for red, green and blue respectively, and will be clamped to integers in 0-255.
   * The 'a' value stands for alpha, and will be clamped to 0-1 (floating point)
   * 'hex' indicates a 6-decimal-digit format hex number, for example 0xFFAA00 is equivalent to r=255, g=170, b=0.
   *
   * @param {number|Color|string} r - See above for the possible overloaded values
   * @param {number} [g] - If provided, should be the green value (or the alpha value if a hex color is given)
   * @param {number} [b] - If provided, should be the blue value
   * @param {number} [a] - If provided, should be the alpha value
   */
  constructor( r, g, b, a ) {

    // @public {Emitter}
    this.changeEmitter = new TinyEmitter();

    this.set( r, g, b, a );
  }

  /**
   * Returns a copy of this color.
   * @public
   *
   * @returns {Color}
   */
  copy() {
    return new Color( this.r, this.g, this.b, this.a );
  }

  /**
   * Sets the values of this Color. Supported styles:
   * @public
   *
   * - set( color ) is a copy constructor
   * - set( string ) will parse the string assuming it's a CSS-compatible color, e.g. set( 'red' )
   * - set( r, g, b ) is equivalent to setRGBA( r, g, b, 1 ), e.g. set( 255, 0, 128 )
   * - set( r, g, b, a ) is equivalent to setRGBA( r, g, b, a ), e.g. set( 255, 0, 128, 0.5 )
   * - set( hex ) will set RGB with alpha=1, e.g. set( 0xFF0000 )
   * - set( hex, alpha ) will set RGBA, e.g. set( 0xFF0000, 1 )
   *
   * @param {number|Color|string} r - See above for the possible overloaded values
   * @param {number} [g] - If provided, should be the green value (or the alpha value if a hex color is given)
   * @param {number} [b] - If provided, should be the blue value
   * @param {number} [a] - If provided, should be the alpha value
   * @returns {Color} - for chaining
   */
  set( r, g, b, a ) {
    assert && assert( r !== undefined, 'Can\'t call Color.set( undefined )' );

    // support for set( string )
    if ( typeof r === 'string' ) {
      this.setCSS( r );
    }
    // support for set( color )
    else if ( r instanceof Color ) {
      this.setRGBA( r.r, r.g, r.b, r.a );
    }
    // support for set( hex ) and set( hex, alpha )
    else if ( b === undefined ) {
      assert && assert( typeof r === 'number' );
      assert && assert( g === undefined || typeof g === 'number' );

      const red = ( r >> 16 ) & 0xFF;
      const green = ( r >> 8 ) & 0xFF;
      const blue = ( r >> 0 ) & 0xFF;
      const alpha = ( g === undefined ) ? 1 : g;
      this.setRGBA( red, green, blue, alpha );
    }
    // support for set( r, g, b ) and set( r, g, b, a )
    else {
      assert && assert( typeof r === 'number' );
      assert && assert( typeof g === 'number' );
      assert && assert( typeof b === 'number' );
      assert && assert( a === undefined || typeof a === 'number' );
      this.setRGBA( r, g, b, ( a === undefined ) ? 1 : a );
    }

    return this; // support chaining
  }

  /**
   * Returns the red value as an integer between 0 and 255
   * @public
   *
   * @returns {number}
   */
  getRed() {
    return this.r;
  }

  get red() { return this.getRed(); }

  /**
   * Sets the red value.
   * @public
   *
   * @param {number} value - Will be clamped to an integer between 0 and 255
   * @returns {Color} - This color, for chaining
   */
  setRed( value ) {
    return this.setRGBA( value, this.g, this.b, this.a );
  }

  set red( value ) { this.setRed( value ); }

  /**
   * Returns the green value as an integer between 0 and 255
   * @public
   *
   * @returns {number}
   */
  getGreen() {
    return this.g;
  }

  get green() { return this.getGreen(); }

  /**
   * Sets the green value.
   * @public
   *
   * @param {number} value - Will be clamped to an integer between 0 and 255
   * @returns {Color} - This color, for chaining
   */
  setGreen( value ) {
    return this.setRGBA( this.r, value, this.b, this.a );
  }

  set green( value ) { this.setGreen( value ); }

  /**
   * Returns the blue value as an integer between 0 and 255
   * @public
   *
   * @returns {number}
   */
  getBlue() {
    return this.b;
  }

  get blue() { return this.getBlue(); }

  /**
   * Sets the blue value.
   * @public
   *
   * @param {number} value - Will be clamped to an integer between 0 and 255
   * @returns {Color} - This color, for chaining
   */
  setBlue( value ) {
    return this.setRGBA( this.r, this.g, value, this.a );
  }

  set blue( value ) { this.setBlue( value ); }

  /**
   * Returns the alpha value as a floating-point value between 0 and 1
   * @public
   *
   * @returns {number}
   */
  getAlpha() {
    return this.a;
  }

  get alpha() { return this.getAlpha(); }

  /**
   * Sets the alpha value.
   * @public
   *
   * @param {number} value - Will be clamped between 0 and 1
   * @returns {Color} - This color, for chaining
   */
  setAlpha( value ) {
    return this.setRGBA( this.r, this.g, this.b, value );
  }

  set alpha( value ) { this.setAlpha( value ); }

  /**
   * Sets the value of this Color using RGB integral between 0-255, alpha (float) between 0-1.
   * @public
   *
   * @param {number} red
   * @param {number} green
   * @param {number} blue
   * @param {number} alpha
   * @returns {Color} - This color, for chaining
   */
  setRGBA( red, green, blue, alpha ) {
    this.r = Utils.roundSymmetric( clamp( red, 0, 255 ) );
    this.g = Utils.roundSymmetric( clamp( green, 0, 255 ) );
    this.b = Utils.roundSymmetric( clamp( blue, 0, 255 ) );
    this.a = clamp( alpha, 0, 1 );

    this.updateColor(); // update the cached value

    return this; // allow chaining
  }

  /**
   * A linear (gamma-corrected) interpolation between this color (ratio=0) and another color (ratio=1).
   * @public
   *
   * @param {Color} otherColor
   * @param {number} ratio - Not necessarily constrained in [0, 1]
   * @returns {Color}
   */
  blend( otherColor, ratio ) {
    assert && assert( otherColor instanceof Color );

    const gamma = 2.4;
    const linearRedA = Math.pow( this.r, gamma );
    const linearRedB = Math.pow( otherColor.r, gamma );
    const linearGreenA = Math.pow( this.g, gamma );
    const linearGreenB = Math.pow( otherColor.g, gamma );
    const linearBlueA = Math.pow( this.b, gamma );
    const linearBlueB = Math.pow( otherColor.b, gamma );

    const r = Math.pow( linearRedA + ( linearRedB - linearRedA ) * ratio, 1 / gamma );
    const g = Math.pow( linearGreenA + ( linearGreenB - linearGreenA ) * ratio, 1 / gamma );
    const b = Math.pow( linearBlueA + ( linearBlueB - linearBlueA ) * ratio, 1 / gamma );
    const a = this.a + ( otherColor.a - this.a ) * ratio;

    return new Color( r, g, b, a );
  }

  /**
   * Used internally to compute the CSS string for this color. Use toCSS()
   * @private
   *
   * @returns {string}
   */
  computeCSS() {
    if ( this.a === 1 ) {
      return 'rgb(' + this.r + ',' + this.g + ',' + this.b + ')';
    }
    else {
      // Since SVG doesn't support parsing scientific notation (e.g. 7e5), we need to output fixed decimal-point strings.
      // Since this needs to be done quickly, and we don't particularly care about slight rounding differences (it's
      // being used for display purposes only, and is never shown to the user), we use the built-in JS toFixed instead of
      // Dot's version of toFixed. See https://github.com/phetsims/kite/issues/50
      let alpha = this.a.toFixed( 20 );
      while ( alpha.length >= 2 && alpha[ alpha.length - 1 ] === '0' && alpha[ alpha.length - 2 ] !== '.' ) {
        alpha = alpha.slice( 0, alpha.length - 1 );
      }

      const alphaString = this.a === 0 || this.a === 1 ? this.a : alpha;
      return 'rgba(' + this.r + ',' + this.g + ',' + this.b + ',' + alphaString + ')';
    }
  }

  /**
   * Returns the value of this Color as a CSS string.
   * @public
   *
   * @returns {string}
   */
  toCSS() {
    // verify that the cached value is correct (in debugging builds only, defeats the point of caching otherwise)
    assert && assert( this._css === this.computeCSS(), 'CSS cached value is ' + this._css + ', but the computed value appears to be ' + this.computeCSS() );

    return this._css;
  }

  /**
   * Sets this color for a CSS color string.
   * @public
   *
   * @param {string} cssString
   *
   * @returns {Color} - This color, for chaining
   */
  setCSS( cssString ) {
    let str = cssString.replace( / /g, '' ).toLowerCase();
    let success = false;

    // replace colors based on keywords
    const keywordMatch = Color.colorKeywords[ str ];
    if ( keywordMatch ) {
      str = '#' + keywordMatch;
    }

    // run through the available text formats
    for ( let i = 0; i < Color.formatParsers.length; i++ ) {
      const parser = Color.formatParsers[ i ];

      const matches = parser.regexp.exec( str );
      if ( matches ) {
        parser.apply( this, matches );
        success = true;
        break;
      }
    }

    if ( !success ) {
      throw new Error( 'Color unable to parse color string: ' + cssString );
    }

    this.updateColor(); // update the cached value

    return this;
  }

  /**
   * Returns this color's RGB information in the hexadecimal number equivalent, e.g. 0xFF00FF
   * @public
   *
   * @returns {number}
   */
  toNumber() {
    return ( this.r << 16 ) + ( this.g << 8 ) + this.b;
  }

  /**
   * Called to update the internally cached CSS value
   * @private
   */
  updateColor() {
    assert && assert( !this.immutable,
      'Cannot modify an immutable color. Likely caused by trying to mutate a color after it was used for a node fill/stroke' );

    assert && assert( typeof this.red === 'number' &&
    typeof this.green === 'number' &&
    typeof this.blue === 'number' &&
    typeof this.alpha === 'number',
      'Ensure color components are numeric: ' + this.toString() );

    assert && assert( isFinite( this.red ) && isFinite( this.green ) && isFinite( this.blue ) && isFinite( this.alpha ),
      'Ensure color components are finite and not NaN' );

    assert && assert( this.red >= 0 && this.red <= 255 &&
    this.green >= 0 && this.green <= 255 &&
    this.red >= 0 && this.red <= 255 &&
    this.alpha >= 0 && this.alpha <= 1,
      'Ensure color components are in the proper ranges: ' + this.toString() );

    const oldCSS = this._css;
    this._css = this.computeCSS();

    // notify listeners if it changed
    if ( oldCSS !== this._css ) {
      this.changeEmitter.emit();
    }
  }

  /**
   * Allow setting this Color to be immutable when assertions are disabled. any change will throw an error
   * @public
   *
   * @returns {Color} - This color, for chaining
   */
  setImmutable() {
    if ( assert ) {
      this.immutable = true;
    }

    return this; // allow chaining
  }

  /**
   * Returns an object that can be passed to a Canvas context's fillStyle or strokeStyle.
   * @public
   *
   * @returns {string}
   */
  getCanvasStyle() {
    return this.toCSS(); // should be inlined, leave like this for future maintainability
  }

  /**
   * Sets this color using HSLA values.
   * @public
   *
   * TODO: make a getHue, getSaturation, getLightness. we can then expose them via ES5!
   *
   * @param {number} hue - integer modulo 360
   * @param {number} saturation - percentage
   * @param {number} lightness - percentage
   * @param {number} alpha
   * @returns {Color} - This color, for chaining
   */
  setHSLA( hue, saturation, lightness, alpha ) {
    hue = ( hue % 360 ) / 360;
    saturation = clamp( saturation / 100, 0, 1 );
    lightness = clamp( lightness / 100, 0, 1 );

    // see http://www.w3.org/TR/css3-color/
    let m2;
    if ( lightness < 0.5 ) {
      m2 = lightness * ( saturation + 1 );
    }
    else {
      m2 = lightness + saturation - lightness * saturation;
    }
    const m1 = lightness * 2 - m2;

    this.r = Utils.roundSymmetric( Color.hueToRGB( m1, m2, hue + 1 / 3 ) * 255 );
    this.g = Utils.roundSymmetric( Color.hueToRGB( m1, m2, hue ) * 255 );
    this.b = Utils.roundSymmetric( Color.hueToRGB( m1, m2, hue - 1 / 3 ) * 255 );
    this.a = clamp( alpha, 0, 1 );

    this.updateColor(); // update the cached value

    return this; // allow chaining
  }

  /**
   * @public
   *
   * @param {Color} color
   * @returns {boolean}
   */
  equals( color ) {
    return this.r === color.r && this.g === color.g && this.b === color.b && this.a === color.a;
  }

  /**
   * Returns a copy of this color with a different alpha value.
   * @public
   *
   * @param {number} alpha
   * @returns {Color}
   */
  withAlpha( alpha ) {
    return new Color( this.r, this.g, this.b, alpha );
  }

  /**
   * @private
   *
   * @param {number} [factor]
   * @returns {number}
   */
  checkFactor( factor ) {
    assert && assert( factor === undefined || ( factor >= 0 && factor <= 1 ), `factor must be between 0 and 1: ${factor}` );

    return ( factor === undefined ) ? 0.7 : factor;
  }

  /**
   * Matches Java's Color.brighter()
   * @public
   *
   * @param {number} [factor]
   * @returns {Color}
   */
  brighterColor( factor ) {
    factor = this.checkFactor( factor );
    const red = Math.min( 255, Math.floor( this.r / factor ) );
    const green = Math.min( 255, Math.floor( this.g / factor ) );
    const blue = Math.min( 255, Math.floor( this.b / factor ) );
    return new Color( red, green, blue, this.a );
  }

  /**
   * Brightens a color in RGB space. Useful when creating gradients from a single base color.
   * @public
   *
   * @param {number} [factor] - 0 (no change) to 1 (white)
   * @returns {Color} - (closer to white) version of the original color.
   */
  colorUtilsBrighter( factor ) {
    factor = this.checkFactor( factor );
    const red = Math.min( 255, this.getRed() + Math.floor( factor * ( 255 - this.getRed() ) ) );
    const green = Math.min( 255, this.getGreen() + Math.floor( factor * ( 255 - this.getGreen() ) ) );
    const blue = Math.min( 255, this.getBlue() + Math.floor( factor * ( 255 - this.getBlue() ) ) );
    return new Color( red, green, blue, this.getAlpha() );
  }

  /**
   * Matches Java's Color.darker()
   * @public
   *
   * @param {number} [factor]
   * @returns {Color}
   */
  darkerColor( factor ) {
    factor = this.checkFactor( factor );
    const red = Math.max( 0, Math.floor( factor * this.r ) );
    const green = Math.max( 0, Math.floor( factor * this.g ) );
    const blue = Math.max( 0, Math.floor( factor * this.b ) );
    return new Color( red, green, blue, this.a );
  }

  /**
   * Darken a color in RGB space. Useful when creating gradients from a single
   * base color.
   * @public
   *
   * @param {number} [factor] - 0 (no change) to 1 (black)
   * @returns {Color} - darker (closer to black) version of the original color.
   */
  colorUtilsDarker( factor ) {
    factor = this.checkFactor( factor );
    const red = Math.max( 0, this.getRed() - Math.floor( factor * this.getRed() ) );
    const green = Math.max( 0, this.getGreen() - Math.floor( factor * this.getGreen() ) );
    const blue = Math.max( 0, this.getBlue() - Math.floor( factor * this.getBlue() ) );
    return new Color( red, green, blue, this.getAlpha() );
  }

  /*
   * Like colorUtilsBrighter/Darker, however factor should be in the range -1 to 1, and it will call:
   *   colorUtilsBrighter( factor )   for factor >  0
   *   this                           for factor == 0
   *   colorUtilsDarker( -factor )    for factor <  0
   * @public
   *
   * @param {number} factor from -1 (black), to 0 (no change), to 1 (white)
   * @returns {Color}
   */
  colorUtilsBrightness( factor ) {
    if ( factor === 0 ) {
      return this;
    }
    else if ( factor > 0 ) {
      return this.colorUtilsBrighter( factor );
    }
    else {
      return this.colorUtilsDarker( -factor );
    }
  }

  /**
   * Returns a string form of this object
   * @public
   *
   * @returns {string}
   */
  toString() {
    return this.constructor.name + '[r:' + this.r + ' g:' + this.g + ' b:' + this.b + ' a:' + this.a + ']';
  }

  /**
   * Convert to a hex string, that starts with "#".
   * @public
   * @returns {string}
   */
  toHexString() {
    let hexString = this.toNumber().toString( 16 );
    while ( hexString.length < 6 ) {
      hexString = `0${hexString}`;
    }
    return `#${hexString}`;
  }

  /**
   * @public
   *
   * @returns {Object}
   */
  toStateObject() {
    return {
      r: this.r,
      g: this.g,
      b: this.b,
      a: this.a
    };
  }

  /**
   * Utility function, see http://www.w3.org/TR/css3-color/
   * @public
   *
   * @param {number} m1
   * @param {number} m2
   * @param {number} h
   * @returns {number}
   */
  static hueToRGB( m1, m2, h ) {
    if ( h < 0 ) {
      h = h + 1;
    }
    if ( h > 1 ) {
      h = h - 1;
    }
    if ( h * 6 < 1 ) {
      return m1 + ( m2 - m1 ) * h * 6;
    }
    if ( h * 2 < 1 ) {
      return m2;
    }
    if ( h * 3 < 2 ) {
      return m1 + ( m2 - m1 ) * ( 2 / 3 - h ) * 6;
    }
    return m1;
  }

  /**
   * Convenience function that converts a color spec to a color object if necessary, or simply returns the color object
   * if not.
   * @public
   *
   * @param {String|Color} colorSpec
   * @returns {Color}
   */
  static toColor( colorSpec ) {
    return colorSpec instanceof Color ? colorSpec : new Color( colorSpec );
  }

  /**
   * Interpolates between 2 colors in RGBA space. When distance is 0, color1 is returned. When distance is 1, color2 is
   * returned. Other values of distance return a color somewhere between color1 and color2. Each color component is
   * interpolated separately.
   * @public
   *
   * @param {Color} color1
   * @param {Color} color2
   * @param {number} distance distance between color1 and color2, 0 <= distance <= 1
   * @returns {Color}
   */
  static interpolateRGBA( color1, color2, distance ) {
    if ( distance < 0 || distance > 1 ) {
      throw new Error( 'distance must be between 0 and 1: ' + distance );
    }
    const r = Math.floor( linear( 0, 1, color1.r, color2.r, distance ) );
    const g = Math.floor( linear( 0, 1, color1.g, color2.g, distance ) );
    const b = Math.floor( linear( 0, 1, color1.b, color2.b, distance ) );
    const a = linear( 0, 1, color1.a, color2.a, distance );
    return new Color( r, g, b, a );
  }

  /**
   * Returns a blended color as a mix between the given colors.
   * @public
   *
   * @param {Array.<Color>} colors
   * @returns {Color}
   */
  static supersampleBlend( colors ) {
    // hard-coded gamma (assuming the exponential part of the sRGB curve as a simplification)
    const GAMMA = 2.2;

    // maps to [0,1] linear colorspace
    const reds = colors.map( color => Math.pow( color.r / 255, GAMMA ) );
    const greens = colors.map( color => Math.pow( color.g / 255, GAMMA ) );
    const blues = colors.map( color => Math.pow( color.b / 255, GAMMA ) );
    const alphas = colors.map( color => Math.pow( color.a, GAMMA ) );

    const alphaSum = _.sum( alphas );

    if ( alphaSum === 0 ) {
      return new Color( 0, 0, 0, 0 );
    }

    // blending of pixels, weighted by alphas
    const red = _.sum( _.range( 0, colors.length ).map( i => reds[ i ] * alphas[ i ] ) ) / alphaSum;
    const green = _.sum( _.range( 0, colors.length ).map( i => greens[ i ] * alphas[ i ] ) ) / alphaSum;
    const blue = _.sum( _.range( 0, colors.length ).map( i => blues[ i ] * alphas[ i ] ) ) / alphaSum;
    const alpha = alphaSum / colors.length; // average of alphas

    return new Color(
      Math.floor( Math.pow( red, 1 / GAMMA ) * 255 ),
      Math.floor( Math.pow( green, 1 / GAMMA ) * 255 ),
      Math.floor( Math.pow( blue, 1 / GAMMA ) * 255 ),
      Math.pow( alpha, 1 / GAMMA )
    );
  }

  /**
   * @public
   *
   * @param {Object} stateObject
   * @returns {Color}
   */
  static fromStateObject( stateObject ) {
    return new Color( stateObject.r, stateObject.g, stateObject.b, stateObject.a );
  }

  /**
   * @public
   *
   * @param {number} hue
   * @param {number} saturation
   * @param {number} lightness
   * @param {number} alpha
   * @returns {Color}
   */
  static hsla( hue, saturation, lightness, alpha ) {
    return new Color( 0, 0, 0, 1 ).setHSLA( hue, saturation, lightness, alpha );
  }

  /**
   * @public
   *
   * @param {string} cssString
   */
  static checkPaintString( cssString ) {
    if ( assert ) {
      try {
        scratchColor.setCSS( cssString );
      }
      catch( e ) {
        assert( false, 'The CSS string is an invalid color: ' + cssString );
      }
    }
  }

  /**
   * A Paint of the type that Paintable accepts as fills or strokes
   * @public
   *
   * @param {PaintDef} paint
   */
  static checkPaint( paint ) {
    if ( typeof paint === 'string' ) {
      Color.checkPaintString( paint );
    }
    else if ( ( paint instanceof Property ) && ( typeof paint.value === 'string' ) ) {
      Color.checkPaintString( paint.value );
    }
  }

  /**
   * Gets the luminance of a color, per ITU-R recommendation BT.709, https://en.wikipedia.org/wiki/Rec._709.
   * Green contributes the most to the intensity perceived by humans, and blue the least.
   * This algorithm works correctly with a grayscale color because the RGB coefficients sum to 1.
   * @public
   *
   * @param {Color|string} color
   * @returns {number} - a value in the range [0,255]
   */
  static getLuminance( color ) {
    const sceneryColor = Color.toColor( color );
    const luminance = ( sceneryColor.red * 0.2126 + sceneryColor.green * 0.7152 + sceneryColor.blue * 0.0722 );
    assert && assert( luminance >= 0 && luminance <= 255, `unexpected luminance: ${luminance}` );
    return luminance;
  }

  /**
   * Converts a color to grayscale.
   * @public
   *
   * @param {Color|string} color
   * @returns {Color}
   */
  static toGrayscale( color ) {
    const luminance = Color.getLuminance( color );
    return new Color( luminance, luminance, luminance );
  }

  /**
   * Determines whether a color is 'dark'.
   * @public
   *
   * @param {Color|string} color
   * @param {number} [luminanceThreshold] - colors with luminance < this value are dark, range [0,255], default 186
   * @returns {boolean}
   */
  static isDarkColor( color, luminanceThreshold = 186 ) {
    assert && assert( typeof luminanceThreshold === 'number' && luminanceThreshold >= 0 && luminanceThreshold <= 255,
      'invalid luminanceThreshold' );
    return ( Color.getLuminance( color ) < luminanceThreshold );
  }

  /**
   * Determines whether a color is 'light'.
   * @public
   *
   * @param {Color|string} color
   * @param {number} [luminanceThreshold] - colors with luminance >= this value are light, range [0,255], default 186
   * @returns {boolean}
   */
  static isLightColor( color, luminanceThreshold ) {
    return !Color.isDarkColor( color, luminanceThreshold );
  }

  /**
   * Creates a Color that is a shade of gray.
   * @param {number} rgb - used for red, blue, and green components
   * @param {number} [a] - defaults to 1
   * @returns {Color}
   * @public
   */
  static grayColor( rgb, a ) {
    return new Color( rgb, rgb, rgb, a );
  }
}

scenery.register( 'Color', Color );

// regex utilities
const rgbNumber = '(-?\\d{1,3}%?)'; // syntax allows negative integers and percentages
const aNumber = '(\\d+|\\d*\\.\\d+)'; // decimal point number. technically we allow for '255', even though this will be clamped to 1.
const rawNumber = '(\\d{1,3})'; // a 1-3 digit number

// handles negative and percentage values
function parseRGBNumber( str ) {
  let multiplier = 1;

  // if it's a percentage, strip it off and handle it that way
  if ( str.charAt( str.length - 1 ) === '%' ) {
    multiplier = 2.55;
    str = str.slice( 0, str.length - 1 );
  }

  return Utils.roundSymmetric( parseInt( str, 10 ) * multiplier );
}

Color.formatParsers = [
  {
    // 'transparent'
    regexp: /^transparent$/,
    apply: ( color, matches ) => {
      color.setRGBA( 0, 0, 0, 0 );
    }
  },
  {
    // short hex form, a la '#fff'
    regexp: /^#(\w{1})(\w{1})(\w{1})$/,
    apply: ( color, matches ) => {
      color.setRGBA(
        parseInt( matches[ 1 ] + matches[ 1 ], 16 ),
        parseInt( matches[ 2 ] + matches[ 2 ], 16 ),
        parseInt( matches[ 3 ] + matches[ 3 ], 16 ),
        1 );
    }
  },
  {
    // long hex form, a la '#ffffff'
    regexp: /^#(\w{2})(\w{2})(\w{2})$/,
    apply: ( color, matches ) => {
      color.setRGBA(
        parseInt( matches[ 1 ], 16 ),
        parseInt( matches[ 2 ], 16 ),
        parseInt( matches[ 3 ], 16 ),
        1 );
    }
  },
  {
    // rgb(...)
    regexp: new RegExp( '^rgb\\(' + rgbNumber + ',' + rgbNumber + ',' + rgbNumber + '\\)$' ),
    apply: ( color, matches ) => {
      color.setRGBA(
        parseRGBNumber( matches[ 1 ] ),
        parseRGBNumber( matches[ 2 ] ),
        parseRGBNumber( matches[ 3 ] ),
        1 );
    }
  },
  {
    // rgba(...)
    regexp: new RegExp( '^rgba\\(' + rgbNumber + ',' + rgbNumber + ',' + rgbNumber + ',' + aNumber + '\\)$' ),
    apply: ( color, matches ) => {
      color.setRGBA(
        parseRGBNumber( matches[ 1 ] ),
        parseRGBNumber( matches[ 2 ] ),
        parseRGBNumber( matches[ 3 ] ),
        parseFloat( matches[ 4 ] ) );
    }
  },
  {
    // hsl(...)
    regexp: new RegExp( '^hsl\\(' + rawNumber + ',' + rawNumber + '%,' + rawNumber + '%\\)$' ),
    apply: ( color, matches ) => {
      color.setHSLA(
        parseInt( matches[ 1 ], 10 ),
        parseInt( matches[ 2 ], 10 ),
        parseInt( matches[ 3 ], 10 ),
        1 );
    }
  },
  {
    // hsla(...)
    regexp: new RegExp( '^hsla\\(' + rawNumber + ',' + rawNumber + '%,' + rawNumber + '%,' + aNumber + '\\)$' ),
    apply: ( color, matches ) => {
      color.setHSLA(
        parseInt( matches[ 1 ], 10 ),
        parseInt( matches[ 2 ], 10 ),
        parseInt( matches[ 3 ], 10 ),
        parseFloat( matches[ 4 ] ) );
    }
  }
];

Color.basicColorKeywords = {
  aqua: '00ffff',
  black: '000000',
  blue: '0000ff',
  fuchsia: 'ff00ff',
  gray: '808080',
  green: '008000',
  lime: '00ff00',
  maroon: '800000',
  navy: '000080',
  olive: '808000',
  purple: '800080',
  red: 'ff0000',
  silver: 'c0c0c0',
  teal: '008080',
  white: 'ffffff',
  yellow: 'ffff00'
};

Color.colorKeywords = {
  aliceblue: 'f0f8ff',
  antiquewhite: 'faebd7',
  aqua: '00ffff',
  aquamarine: '7fffd4',
  azure: 'f0ffff',
  beige: 'f5f5dc',
  bisque: 'ffe4c4',
  black: '000000',
  blanchedalmond: 'ffebcd',
  blue: '0000ff',
  blueviolet: '8a2be2',
  brown: 'a52a2a',
  burlywood: 'deb887',
  cadetblue: '5f9ea0',
  chartreuse: '7fff00',
  chocolate: 'd2691e',
  coral: 'ff7f50',
  cornflowerblue: '6495ed',
  cornsilk: 'fff8dc',
  crimson: 'dc143c',
  cyan: '00ffff',
  darkblue: '00008b',
  darkcyan: '008b8b',
  darkgoldenrod: 'b8860b',
  darkgray: 'a9a9a9',
  darkgreen: '006400',
  darkgrey: 'a9a9a9',
  darkkhaki: 'bdb76b',
  darkmagenta: '8b008b',
  darkolivegreen: '556b2f',
  darkorange: 'ff8c00',
  darkorchid: '9932cc',
  darkred: '8b0000',
  darksalmon: 'e9967a',
  darkseagreen: '8fbc8f',
  darkslateblue: '483d8b',
  darkslategray: '2f4f4f',
  darkslategrey: '2f4f4f',
  darkturquoise: '00ced1',
  darkviolet: '9400d3',
  deeppink: 'ff1493',
  deepskyblue: '00bfff',
  dimgray: '696969',
  dimgrey: '696969',
  dodgerblue: '1e90ff',
  firebrick: 'b22222',
  floralwhite: 'fffaf0',
  forestgreen: '228b22',
  fuchsia: 'ff00ff',
  gainsboro: 'dcdcdc',
  ghostwhite: 'f8f8ff',
  gold: 'ffd700',
  goldenrod: 'daa520',
  gray: '808080',
  green: '008000',
  greenyellow: 'adff2f',
  grey: '808080',
  honeydew: 'f0fff0',
  hotpink: 'ff69b4',
  indianred: 'cd5c5c',
  indigo: '4b0082',
  ivory: 'fffff0',
  khaki: 'f0e68c',
  lavender: 'e6e6fa',
  lavenderblush: 'fff0f5',
  lawngreen: '7cfc00',
  lemonchiffon: 'fffacd',
  lightblue: 'add8e6',
  lightcoral: 'f08080',
  lightcyan: 'e0ffff',
  lightgoldenrodyellow: 'fafad2',
  lightgray: 'd3d3d3',
  lightgreen: '90ee90',
  lightgrey: 'd3d3d3',
  lightpink: 'ffb6c1',
  lightsalmon: 'ffa07a',
  lightseagreen: '20b2aa',
  lightskyblue: '87cefa',
  lightslategray: '778899',
  lightslategrey: '778899',
  lightsteelblue: 'b0c4de',
  lightyellow: 'ffffe0',
  lime: '00ff00',
  limegreen: '32cd32',
  linen: 'faf0e6',
  magenta: 'ff00ff',
  maroon: '800000',
  mediumaquamarine: '66cdaa',
  mediumblue: '0000cd',
  mediumorchid: 'ba55d3',
  mediumpurple: '9370db',
  mediumseagreen: '3cb371',
  mediumslateblue: '7b68ee',
  mediumspringgreen: '00fa9a',
  mediumturquoise: '48d1cc',
  mediumvioletred: 'c71585',
  midnightblue: '191970',
  mintcream: 'f5fffa',
  mistyrose: 'ffe4e1',
  moccasin: 'ffe4b5',
  navajowhite: 'ffdead',
  navy: '000080',
  oldlace: 'fdf5e6',
  olive: '808000',
  olivedrab: '6b8e23',
  orange: 'ffa500',
  orangered: 'ff4500',
  orchid: 'da70d6',
  palegoldenrod: 'eee8aa',
  palegreen: '98fb98',
  paleturquoise: 'afeeee',
  palevioletred: 'db7093',
  papayawhip: 'ffefd5',
  peachpuff: 'ffdab9',
  peru: 'cd853f',
  pink: 'ffc0cb',
  plum: 'dda0dd',
  powderblue: 'b0e0e6',
  purple: '800080',
  red: 'ff0000',
  rosybrown: 'bc8f8f',
  royalblue: '4169e1',
  saddlebrown: '8b4513',
  salmon: 'fa8072',
  sandybrown: 'f4a460',
  seagreen: '2e8b57',
  seashell: 'fff5ee',
  sienna: 'a0522d',
  silver: 'c0c0c0',
  skyblue: '87ceeb',
  slateblue: '6a5acd',
  slategray: '708090',
  slategrey: '708090',
  snow: 'fffafa',
  springgreen: '00ff7f',
  steelblue: '4682b4',
  tan: 'd2b48c',
  teal: '008080',
  thistle: 'd8bfd8',
  tomato: 'ff6347',
  turquoise: '40e0d0',
  violet: 'ee82ee',
  wheat: 'f5deb3',
  white: 'ffffff',
  whitesmoke: 'f5f5f5',
  yellow: 'ffff00',
  yellowgreen: '9acd32'
};

// Java compatibility
Color.BLACK = Color.black = new Color( 0, 0, 0 ).setImmutable();
Color.BLUE = Color.blue = new Color( 0, 0, 255 ).setImmutable();
Color.CYAN = Color.cyan = new Color( 0, 255, 255 ).setImmutable();
Color.DARK_GRAY = Color.darkGray = new Color( 64, 64, 64 ).setImmutable();
Color.GRAY = Color.gray = new Color( 128, 128, 128 ).setImmutable();
Color.GREEN = Color.green = new Color( 0, 255, 0 ).setImmutable();
Color.LIGHT_GRAY = Color.lightGray = new Color( 192, 192, 192 ).setImmutable();
Color.MAGENTA = Color.magenta = new Color( 255, 0, 255 ).setImmutable();
Color.ORANGE = Color.orange = new Color( 255, 200, 0 ).setImmutable();
Color.PINK = Color.pink = new Color( 255, 175, 175 ).setImmutable();
Color.RED = Color.red = new Color( 255, 0, 0 ).setImmutable();
Color.WHITE = Color.white = new Color( 255, 255, 255 ).setImmutable();
Color.YELLOW = Color.yellow = new Color( 255, 255, 0 ).setImmutable();

// Helper for transparent colors
Color.TRANSPARENT = Color.transparent = new Color( 0, 0, 0, 0 ).setImmutable();

const scratchColor = new Color( 'blue' );

Color.ColorIO = new IOType( 'ColorIO', {
  valueType: Color,
  documentation: 'A color, with rgba',
  toStateObject: color => color.toStateObject(),
  fromStateObject: stateObject => new Color( stateObject.r, stateObject.g, stateObject.b, stateObject.a )
} );

export default Color;