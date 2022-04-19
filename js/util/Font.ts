// Copyright 2013-2022, University of Colorado Boulder

/**
 * Immutable font object.
 *
 * Examples:
 * new scenery.Font().font                      // "10px sans-serif" (the default)
 * new scenery.Font( { family: 'serif' } ).font // "10px serif"
 * new scenery.Font( { weight: 'bold' } ).font  // "bold 10px sans-serif"
 * new scenery.Font( { size: 16 } ).font        // "16px sans-serif"
 * var font = new scenery.Font( {
 *   family: '"Times New Roman", serif',
 *   style: 'italic',
 *   lineHeight: 10
 * } );
 * font.font;                                   // "italic 10px/10 'Times New Roman', serif"
 * font.family;                                 // "'Times New Roman', serif"
 * font.weight;                                 // 400 (the default)
 *
 * Useful specs:
 * http://www.w3.org/TR/css3-fonts/
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import merge from '../../../phet-core/js/merge.js';
import optionize from '../../../phet-core/js/optionize.js';
import PhetioObject, { PhetioObjectOptions } from '../../../tandem/js/PhetioObject.js';
import Tandem from '../../../tandem/js/Tandem.js';
import IOType from '../../../tandem/js/types/IOType.js';
import { scenery } from '../imports.js';

// @private {Array.<string>} - Valid values for the 'style' property of Font
const VALID_STYLES = [ 'normal', 'italic', 'oblique' ];

// @private {Array.<string>} - Valid values for the 'variant' property of Font
const VALID_VARIANTS = [ 'normal', 'small-caps' ];

// @private {Array.<string>} - Valid values for the 'weight' property of Font
const VALID_WEIGHTS = [ 'normal', 'bold', 'bolder', 'lighter',
  '100', '200', '300', '400', '500', '600', '700', '800', '900' ];

// @private {Array.<string>} - Valid values for the 'stretch' property of Font
const VALID_STRETCHES = [ 'normal', 'ultra-condensed', 'extra-condensed', 'condensed', 'semi-condensed',
  'semi-expanded', 'expanded', 'extra-expanded', 'ultra-expanded' ];

export type FontStyle = 'normal' | 'italic' | 'oblique';
export type FontVariant = 'normal' | 'small-caps';
export type FontWeight =
  'normal'
  | 'bold'
  | 'bolder'
  | 'lighter'
  | '100'
  | '200'
  | '300'
  | '400'
  | '500'
  | '600'
  | '700'
  | '800'
  | '900';
export type FontStretch =
  'normal'
  | 'ultra-condensed'
  | 'extra-condensed'
  | 'condensed'
  | 'semi-condensed'
  | 'semi-expanded'
  | 'expanded'
  | 'extra-expanded'
  | 'ultra-expanded';

type SelfOptions = {
  style?: FontStyle;
  variant?: FontVariant;
  weight?: number | FontWeight;
  stretch?: FontStretch;
  size?: number | string;
  lineHeight?: string;
  family?: string;
};
export type FontOptions = SelfOptions & PhetioObjectOptions;

export default class Font extends PhetioObject {

  // See https://www.w3.org/TR/css-fonts-3/#propdef-font-style
  _style: FontStyle;

  // See https://www.w3.org/TR/css-fonts-3/#font-variant-css21-values
  _variant: FontVariant;

  // See https://www.w3.org/TR/css-fonts-3/#propdef-font-weight
  _weight: FontWeight;

  // See https://www.w3.org/TR/css-fonts-3/#propdef-font-stretch
  _stretch: FontStretch;

  // See https://www.w3.org/TR/css-fonts-3/#propdef-font-size
  _size: string;

  // See https://www.w3.org/TR/CSS2/visudet.html#propdef-line-height
  _lineHeight: string;

  // See https://www.w3.org/TR/css-fonts-3/#propdef-font-family
  _family: string;

  // Shorthand font property
  _font: string;

  constructor( options?: FontOptions ) {
    assert && assert( options === undefined || ( typeof options === 'object' && Object.getPrototypeOf( options ) === Object.prototype ),
      'options, if provided, should be a raw object' );

    const definedOptions = optionize<FontOptions, SelfOptions, PhetioObjectOptions>()( {
      // {string} - 'normal', 'italic' or 'oblique'
      style: 'normal',

      // {string} - 'normal' or 'small-caps'
      variant: 'normal',

      // {number|string} - 'normal', 'bold', 'bolder', 'lighter', '100', '200', '300', '400', '500', '600', '700',
      // '800', '900', or a number that when cast to a string will be one of the strings above.
      weight: 'normal',

      // {string} - 'normal', 'ultra-condensed', 'extra-condensed', 'condensed', 'semi-condensed', 'semi-expanded',
      // 'expanded', 'extra-expanded' or 'ultra-expanded'
      stretch: 'normal',

      // {number|string} - A valid CSS font-size string, or a number representing a quantity of 'px'.
      size: '10px',

      // {string} - A valid CSS line-height, typically 'normal', a number, a CSS length (e.g. '15px'), or a percentage
      // of the normal height.
      lineHeight: 'normal',

      // {string} - A comma-separated list of families, which can include generic families (preferably at the end) such
      // as 'serif', 'sans-serif', 'cursive', 'fantasy' and 'monospace'. If there is any question about escaping (such
      // as spaces in a font name), the family should be surrounded by double quotes.
      family: 'sans-serif',

      phetioType: Font.FontIO,
      tandem: Tandem.OPTIONAL
    }, options );

    assert && assert( typeof definedOptions.weight === 'string' || typeof definedOptions.weight === 'number', 'Font weight should be specified as a string or number' );
    assert && assert( typeof definedOptions.size === 'string' || typeof definedOptions.size === 'number', 'Font size should be specified as a string or number' );

    super( definedOptions );

    this._style = definedOptions.style;
    this._variant = definedOptions.variant;
    this._weight = `${definedOptions.weight}` as FontWeight; // cast to string, we'll double check it later
    this._stretch = definedOptions.stretch;
    this._size = Font.castSize( definedOptions.size );
    this._lineHeight = definedOptions.lineHeight;
    this._family = definedOptions.family;

    // sanity checks to prevent errors in interpretation or in the font shorthand usage
    assert && assert( typeof this._style === 'string' && _.includes( VALID_STYLES, this._style ),
      'Font style must be one of "normal", "italic", or "oblique"' );
    assert && assert( typeof this._variant === 'string' && _.includes( VALID_VARIANTS, this._variant ),
      'Font variant must be "normal" or "small-caps"' );
    assert && assert( typeof this._weight === 'string' && _.includes( VALID_WEIGHTS, this._weight ),
      'Font weight must be one of "normal", "bold", "bolder", "lighter", "100", "200", "300", "400", "500", "600", "700", "800", or "900"' );
    assert && assert( typeof this._stretch === 'string' && _.includes( VALID_STRETCHES, this._stretch ),
      'Font stretch must be one of "normal", "ultra-condensed", "extra-condensed", "condensed", "semi-condensed", "semi-expanded", "expanded", "extra-expanded", or "ultra-expanded"' );
    assert && assert( typeof this._size === 'string' && !_.includes( [ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' ], this._size[ this._size.length - 1 ] ),
      'Font size must be either passed as a number (not a string, interpreted as px), or must contain a suffix for percentage, absolute or relative units, or an explicit size constant' );
    assert && assert( typeof this._lineHeight === 'string' );
    assert && assert( typeof this._family === 'string' );

    // Initialize the shorthand font property
    this._font = this.computeShorthand();
  }

  /**
   * Returns this font's CSS shorthand, which includes all of the font's information reduced into a single string.
   *
   * This can be used for CSS as the 'font' attribute, or is needed to set Canvas fonts.
   *
   * https://www.w3.org/TR/css-fonts-3/#propdef-font contains detailed information on how this is formatted.
   */
  getFont(): string {
    return this._font;
  }

  get font(): string { return this.getFont(); }

  /**
   * Returns this font's style. See the constructor for more details on valid values.
   */
  getStyle(): FontStyle {
    return this._style;
  }

  get style(): FontStyle { return this.getStyle(); }

  /**
   * Returns this font's variant. See the constructor for more details on valid values.
   */
  getVariant(): FontVariant {
    return this._variant;
  }

  get variant(): FontVariant { return this.getVariant(); }

  /**
   * Returns this font's weight. See the constructor for more details on valid values.
   *
   * NOTE: If a numeric weight was passed in, it has been cast to a string, and a string will be returned here.
   */
  getWeight(): FontWeight {
    return this._weight;
  }

  get weight(): FontWeight { return this.getWeight(); }

  /**
   * Returns this font's stretch. See the constructor for more details on valid values.
   */
  getStretch(): FontStretch {
    return this._stretch;
  }

  get stretch(): FontStretch { return this.getStretch(); }

  /**
   * Returns this font's size. See the constructor for more details on valid values.
   *
   * NOTE: If a numeric size was passed in, it has been cast to a string, and a string will be returned here.
   */
  getSize(): string {
    return this._size;
  }

  get size(): string { return this.getSize(); }

  /**
   * Returns an approximate value of this font's size in px.
   */
  getNumericSize(): number {
    const pxMatch = this._size.match( /^(\d+)px$/ );
    if ( pxMatch ) {
      return parseInt( pxMatch[ 1 ], 10 );
    }

    const ptMatch = this._size.match( /^(\d+)pt$/ );
    if ( ptMatch ) {
      return 0.75 * parseInt( ptMatch[ 1 ], 10 );
    }

    const emMatch = this._size.match( /^(\d+)em$/ );
    if ( emMatch ) {
      return parseInt( emMatch[ 1 ], 10 ) / 16;
    }

    return 12; // a guess?
  }

  get numericSize(): number { return this.getNumericSize(); }

  /**
   * Returns this font's line-height. See the constructor for more details on valid values.
   */
  getLineHeight(): string {
    return this._lineHeight;
  }

  get lineHeight(): string { return this.getLineHeight(); }

  /**
   * Returns this font's family. See the constructor for more details on valid values.
   */
  getFamily(): string {
    return this._family;
  }

  get family(): string { return this.getFamily(); }

  /**
   * Returns a new Font object, which is a copy of this object. If options are provided, they override the current
   * values in this object.
   */
  copy( options?: FontOptions ): Font {
    // TODO: get merge working in typescript
    return new Font( merge( {
      style: this._style,
      variant: this._variant,
      weight: this._weight,
      stretch: this._stretch,
      size: this._size,
      lineHeight: this._lineHeight,
      family: this._family
    }, options ) as FontOptions );
  }

  /**
   * Computes the combined CSS shorthand font string.
   *
   * https://www.w3.org/TR/css-fonts-3/#propdef-font contains details about the format.
   */
  private computeShorthand(): string {
    let ret = '';
    if ( this._style !== 'normal' ) { ret += `${this._style} `; }
    if ( this._variant !== 'normal' ) { ret += `${this._variant} `; }
    if ( this._weight !== 'normal' ) { ret += `${this._weight} `; }
    if ( this._stretch !== 'normal' ) { ret += `${this._stretch} `; }
    ret += this._size;
    if ( this._lineHeight !== 'normal' ) { ret += `/${this._lineHeight}`; }
    ret += ` ${this._family}`;
    return ret;
  }

  /**
   * Returns this font's CSS shorthand, which includes all of the font's information reduced into a single string.
   *
   * NOTE: This is an alias of getFont().
   *
   * This can be used for CSS as the 'font' attribute, or is needed to set Canvas fonts.
   *
   * https://www.w3.org/TR/css-fonts-3/#propdef-font contains detailed information on how this is formatted.
   */
  toCSS(): string {
    return this.getFont();
  }

  /**
   * Converts a generic size to a specific CSS pixel string, assuming 'px' for numbers.
   *
   * @param size - If it's a number, 'px' will be appended
   */
  static castSize( size: string | number ): string {
    if ( typeof size === 'number' ) {
      return `${size}px`; // add the pixels suffix by default for numbers
    }
    else {
      return size; // assume that it's a valid to-spec string
    }
  }

  static isFontStyle( style: string ): style is FontStyle {
    return VALID_STYLES.includes( style );
  }

  static isFontVariant( variant: string ): variant is FontVariant {
    return VALID_VARIANTS.includes( variant );
  }

  static isFontWeight( weight: string ): weight is FontWeight {
    return VALID_WEIGHTS.includes( weight );
  }

  static isFontStretch( stretch: string ): stretch is FontStretch {
    return VALID_STRETCHES.includes( stretch );
  }

  /**
   * Parses a CSS-compliant "font" shorthand string into a Font object.
   *
   * Font strings should be a valid CSS3 font declaration value (see http://www.w3.org/TR/css3-fonts/) which consists
   * of the following pattern:
   *   [ [ <‘font-style’> || <font-variant-css21> || <‘font-weight’> || <‘font-stretch’> ]? <‘font-size’>
   *   [ / <‘line-height’> ]? <‘font-family’> ]
   */
  static fromCSS( cssString: string ): Font {
    // parse a somewhat proper CSS3 form (not guaranteed to handle it precisely the same as browsers yet)

    const options: FontOptions = {};

    // split based on whitespace allowed by CSS spec (more restrictive than regular regexp whitespace)
    const tokens = _.filter( cssString.split( /[\x09\x0A\x0C\x0D\x20]/ ), token => token.length > 0 ); // eslint-disable-line no-control-regex

    // pull tokens out until we reach something that doesn't match. that must be the font size (according to spec)
    for ( let i = 0; i < tokens.length; i++ ) {
      const token = tokens[ i ];
      if ( token === 'normal' ) {
        // nothing has to be done, everything already normal as default
      }
      else if ( Font.isFontStyle( token ) ) {
        assert && assert( options.style === undefined, `Style cannot be applied twice. Already set to "${options.style}", attempt to replace with "${token}"` );
        options.style = token;
      }
      else if ( Font.isFontVariant( token ) ) {
        assert && assert( options.variant === undefined, `Variant cannot be applied twice. Already set to "${options.variant}", attempt to replace with "${token}"` );
        options.variant = token;
      }
      else if ( Font.isFontWeight( token ) ) {
        assert && assert( options.weight === undefined, `Weight cannot be applied twice. Already set to "${options.weight}", attempt to replace with "${token}"` );
        options.weight = token;
      }
      else if ( Font.isFontStretch( token ) ) {
        assert && assert( options.stretch === undefined, `Stretch cannot be applied twice. Already set to "${options.stretch}", attempt to replace with "${token}"` );
        options.stretch = token;
      }
      else {
        // not a style/variant/weight/stretch, must be a font size, possibly with an included line-height
        const subtokens = token.split( /\// ); // extract font size from any line-height
        options.size = subtokens[ 0 ];
        if ( subtokens[ 1 ] ) {
          options.lineHeight = subtokens[ 1 ];
        }
        // all future tokens are guaranteed to be part of the font-family if it is given according to spec
        options.family = tokens.slice( i + 1 ).join( ' ' );
        break;
      }
    }

    return new Font( options );
  }

  static FontIO: IOType;
  static DEFAULT: Font;
}

scenery.register( 'Font', Font );

Font.FontIO = new IOType( 'FontIO', {
  valueType: Font,
  documentation: 'Font handling for text drawing. Options:' +
                 '<ul>' +
                 '<li><strong>style:</strong> normal      &mdash; normal | italic | oblique </li>' +
                 '<li><strong>variant:</strong> normal    &mdash; normal | small-caps </li>' +
                 '<li><strong>weight:</strong> normal     &mdash; normal | bold | bolder | lighter | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 </li>' +
                 '<li><strong>stretch:</strong> normal    &mdash; normal | ultra-condensed | extra-condensed | condensed | semi-condensed | semi-expanded | expanded | extra-expanded | ultra-expanded </li>' +
                 '<li><strong>size:</strong> 10px         &mdash; absolute-size | relative-size | length | percentage -- unitless number interpreted as px. absolute suffixes: cm, mm, in, pt, pc, px. relative suffixes: em, ex, ch, rem, vw, vh, vmin, vmax. </li>' +
                 '<li><strong>lineHeight:</strong> normal &mdash; normal | number | length | percentage -- NOTE: Canvas spec forces line-height to normal </li>' +
                 '<li><strong>family:</strong> sans-serif &mdash; comma-separated list of families, including generic families (serif, sans-serif, cursive, fantasy, monospace). ideally escape with double-quotes</li>' +
                 '</ul>',
  toStateObject: ( font: Font ): Required<SelfOptions> => ( {
    style: font.getStyle(),
    variant: font.getVariant(),
    weight: font.getWeight(),
    stretch: font.getStretch(),
    size: font.getSize(),
    lineHeight: font.getLineHeight(),
    family: font.getFamily()
  } ),

  fromStateObject( stateObject: Required<SelfOptions> ) {
    return new Font( stateObject );
  }
} );

// @public {Font} - Default Font object (since they are immutable).
Font.DEFAULT = new Font();
