// Copyright 2013-2023, University of Colorado Boulder

/**
 * Immutable font object.
 *
 * Examples:
 * new phet.scenery.Font().font                      // "10px sans-serif" (the default)
 * new phet.scenery.Font( { family: 'serif' } ).font // "10px serif"
 * new phet.scenery.Font( { weight: 'bold' } ).font  // "bold 10px sans-serif"
 * new phet.scenery.Font( { size: 16 } ).font        // "16px sans-serif"
 * var font = new phet.scenery.Font( {
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

import optionize, { combineOptions } from '../../../phet-core/js/optionize.js';
import PhetioObject, { PhetioObjectOptions } from '../../../tandem/js/PhetioObject.js';
import Tandem from '../../../tandem/js/Tandem.js';
import IOType from '../../../tandem/js/types/IOType.js';
import StringIO from '../../../tandem/js/types/StringIO.js';
import { scenery } from '../imports.js';

// Valid values for the 'style' property of Font
const VALID_STYLES = [ 'normal', 'italic', 'oblique' ];

// Valid values for the 'variant' property of Font
const VALID_VARIANTS = [ 'normal', 'small-caps' ];

// Valid values for the 'weight' property of Font
const VALID_WEIGHTS = [ 'normal', 'bold', 'bolder', 'lighter',
  '100', '200', '300', '400', '500', '600', '700', '800', '900' ];

// Valid values for the 'stretch' property of Font
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
  private readonly _style: FontStyle;

  // See https://www.w3.org/TR/css-fonts-3/#font-variant-css21-values
  private readonly _variant: FontVariant;

  // See https://www.w3.org/TR/css-fonts-3/#propdef-font-weight
  private readonly _weight: FontWeight;

  // See https://www.w3.org/TR/css-fonts-3/#propdef-font-stretch
  private readonly _stretch: FontStretch;

  // See https://www.w3.org/TR/css-fonts-3/#propdef-font-size
  private readonly _size: string;

  // See https://www.w3.org/TR/CSS2/visudet.html#propdef-line-height
  private readonly _lineHeight: string;

  // See https://www.w3.org/TR/css-fonts-3/#propdef-font-family
  private readonly _family: string;

  // Shorthand font property
  private readonly _font: string;

  public constructor( providedOptions?: FontOptions ) {
    assert && assert( providedOptions === undefined || ( typeof providedOptions === 'object' && Object.getPrototypeOf( providedOptions ) === Object.prototype ),
      'options, if provided, should be a raw object' );

    const options = optionize<FontOptions, SelfOptions, PhetioObjectOptions>()( {
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
    }, providedOptions );

    assert && assert( typeof options.weight === 'string' || typeof options.weight === 'number', 'Font weight should be specified as a string or number' );
    assert && assert( typeof options.size === 'string' || typeof options.size === 'number', 'Font size should be specified as a string or number' );

    super( options );

    this._style = options.style;
    this._variant = options.variant;
    this._weight = `${options.weight}` as FontWeight; // cast to string, we'll double check it later
    this._stretch = options.stretch;
    this._size = Font.castSize( options.size );
    this._lineHeight = options.lineHeight;
    this._family = options.family;

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
  public getFont(): string {
    return this._font;
  }

  public get font(): string { return this.getFont(); }

  /**
   * Returns this font's style. See the constructor for more details on valid values.
   */
  public getStyle(): FontStyle {
    return this._style;
  }

  public get style(): FontStyle { return this.getStyle(); }

  /**
   * Returns this font's variant. See the constructor for more details on valid values.
   */
  public getVariant(): FontVariant {
    return this._variant;
  }

  public get variant(): FontVariant { return this.getVariant(); }

  /**
   * Returns this font's weight. See the constructor for more details on valid values.
   *
   * NOTE: If a numeric weight was passed in, it has been cast to a string, and a string will be returned here.
   */
  public getWeight(): FontWeight {
    return this._weight;
  }

  public get weight(): FontWeight { return this.getWeight(); }

  /**
   * Returns this font's stretch. See the constructor for more details on valid values.
   */
  public getStretch(): FontStretch {
    return this._stretch;
  }

  public get stretch(): FontStretch { return this.getStretch(); }

  /**
   * Returns this font's size. See the constructor for more details on valid values.
   *
   * NOTE: If a numeric size was passed in, it has been cast to a string, and a string will be returned here.
   */
  public getSize(): string {
    return this._size;
  }

  public get size(): string { return this.getSize(); }

  /**
   * Returns an approximate value of this font's size in px.
   */
  public getNumericSize(): number {
    const pxMatch = this._size.match( /^(\d+)px$/ );
    if ( pxMatch ) {
      return Number( pxMatch[ 1 ] );
    }

    const ptMatch = this._size.match( /^(\d+)pt$/ );
    if ( ptMatch ) {
      return 0.75 * Number( ptMatch[ 1 ] );
    }

    const emMatch = this._size.match( /^(\d+)em$/ );
    if ( emMatch ) {
      return Number( emMatch[ 1 ] ) / 16;
    }

    return 12; // a guess?
  }

  public get numericSize(): number { return this.getNumericSize(); }

  /**
   * Returns this font's line-height. See the constructor for more details on valid values.
   */
  public getLineHeight(): string {
    return this._lineHeight;
  }

  public get lineHeight(): string { return this.getLineHeight(); }

  /**
   * Returns this font's family. See the constructor for more details on valid values.
   */
  public getFamily(): string {
    return this._family;
  }

  public get family(): string { return this.getFamily(); }

  /**
   * Returns a new Font object, which is a copy of this object. If options are provided, they override the current
   * values in this object.
   */
  public copy( options?: FontOptions ): Font {
    // TODO: get merge working in typescript
    return new Font( combineOptions<FontOptions>( {
      style: this._style,
      variant: this._variant,
      weight: this._weight,
      stretch: this._stretch,
      size: this._size,
      lineHeight: this._lineHeight,
      family: this._family
    }, options ) );
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
  public toCSS(): string {
    return this.getFont();
  }

  /**
   * Converts a generic size to a specific CSS pixel string, assuming 'px' for numbers.
   *
   * @param size - If it's a number, 'px' will be appended
   */
  public static castSize( size: string | number ): string {
    if ( typeof size === 'number' ) {
      return `${size}px`; // add the pixels suffix by default for numbers
    }
    else {
      return size; // assume that it's a valid to-spec string
    }
  }

  public static isFontStyle( style: string ): style is FontStyle {
    return VALID_STYLES.includes( style );
  }

  public static isFontVariant( variant: string ): variant is FontVariant {
    return VALID_VARIANTS.includes( variant );
  }

  public static isFontWeight( weight: string ): weight is FontWeight {
    return VALID_WEIGHTS.includes( weight );
  }

  public static isFontStretch( stretch: string ): stretch is FontStretch {
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
  public static fromCSS( cssString: string ): Font {
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

  public static FontIO: IOType<Font, FontState>;

  // {Font} - Default Font object (since they are immutable).
  public static readonly DEFAULT = new Font();
}

type FontState = Required<SelfOptions>;

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
  toStateObject: ( font: Font ): FontState => ( {
    style: font.getStyle(),
    variant: font.getVariant(),
    weight: font.getWeight(),
    stretch: font.getStretch(),
    size: font.getSize(),
    lineHeight: font.getLineHeight(),
    family: font.getFamily()
  } ),

  fromStateObject( stateObject: FontState ) {
    return new Font( stateObject );
  },

  stateSchema: {
    style: StringIO,
    variant: StringIO,
    weight: StringIO,
    stretch: StringIO,
    size: StringIO,
    lineHeight: StringIO,
    family: StringIO
  }
} );
