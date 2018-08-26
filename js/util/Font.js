// Copyright 2013-2016, University of Colorado Boulder

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

define( function( require ) {
  'use strict';

  var FontIO = require( 'SCENERY/util/FontIO' );
  var inherit = require( 'PHET_CORE/inherit' );
  var PhetioObject = require( 'TANDEM/PhetioObject' );
  var scenery = require( 'SCENERY/scenery' );
  var Tandem = require( 'TANDEM/Tandem' );

  // @private {Array.<string>} - Valid values for the 'style' property of Font
  var VALID_STYLES = [ 'normal', 'italic', 'oblique' ];

  // @private {Array.<string>} - Valid values for the 'variant' property of Font
  var VALID_VARIANTS = [ 'normal', 'small-caps' ];

  // @private {Array.<string>} - Valid values for the 'weight' property of Font
  var VALID_WEIGHTS = [ 'normal', 'bold', 'bolder', 'lighter',
    '100', '200', '300', '400', '500', '600', '700', '800', '900' ];

  // @private {Array.<string>} - Valid values for the 'stretch' property of Font
  var VALID_STRETCHES = [ 'normal', 'ultra-condensed', 'extra-condensed', 'condensed', 'semi-condensed',
    'semi-expanded', 'expanded', 'extra-expanded', 'ultra-expanded' ];

  /**
   * @public
   * @constructor
   *
   * @param {Object} [options] - See below (top of the constructor) for valid options.
   */
  function Font( options ) {
    assert && assert( options === undefined || ( typeof options === 'object' && Object.getPrototypeOf( options ) === Object.prototype ),
      'options, if provided, should be a raw object' );

    options = _.extend( {
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

      phetioType: FontIO,
      tandem: Tandem.optional
    }, options );

    assert && assert( typeof options.weight === 'string' || typeof options.weight === 'number', 'Font weight should be specified as a string or number' );
    assert && assert( typeof options.size === 'string' || typeof options.size === 'number', 'Font size should be specified as a string or number' );

    // @private {string} - See https://www.w3.org/TR/css-fonts-3/#propdef-font-style
    this._style = options.style;

    // @private {string} - See https://www.w3.org/TR/css-fonts-3/#font-variant-css21-values
    this._variant = options.variant;

    // @private {string} - See https://www.w3.org/TR/css-fonts-3/#propdef-font-weight
    this._weight = '' + options.weight; // cast to string

    // @private {string} - See https://www.w3.org/TR/css-fonts-3/#propdef-font-stretch
    this._stretch = options.stretch;

    // @private {string} - See https://www.w3.org/TR/css-fonts-3/#propdef-font-size
    this._size = Font.castSize( options.size );

    // @private {string} - See https://www.w3.org/TR/CSS2/visudet.html#propdef-line-height
    this._lineHeight = options.lineHeight;

    // @private {string} - See https://www.w3.org/TR/css-fonts-3/#propdef-font-family
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

    // @private {string} - Initialize the shorthand font property (stored as _font)
    this._font = this.computeShorthand();

    PhetioObject.call( this, options );
  }

  scenery.register( 'Font', Font );

  inherit( PhetioObject, Font, {
    /**
     * Returns this font's CSS shorthand, which includes all of the font's information reduced into a single string.
     * @public
     *
     * This can be used for CSS as the 'font' attribute, or is needed to set Canvas fonts.
     *
     * https://www.w3.org/TR/css-fonts-3/#propdef-font contains detailed information on how this is formatted.
     *
     * @returns {string}
     */
    getFont: function() {
      return this._font;
    },
    get font() { return this.getFont(); },

    /**
     * Returns this font's style. See the constructor for more details on valid values.
     * @public
     *
     * @returns {string}
     */
    getStyle: function() {
      return this._style;
    },
    get style() { return this.getStyle(); },

    /**
     * Returns this font's variant. See the constructor for more details on valid values.
     * @public
     *
     * @returns {string}
     */
    getVariant: function() {
      return this._variant;
    },
    get variant() { return this.getVariant(); },

    /**
     * Returns this font's weight. See the constructor for more details on valid values.
     * @public
     *
     * NOTE: If a numeric weight was passed in, it has been cast to a string, and a string will be returned here.
     *
     * @returns {string}
     */
    getWeight: function() {
      return this._weight;
    },
    get weight() { return this.getWeight(); },

    /**
     * Returns this font's stretch. See the constructor for more details on valid values.
     * @public
     *
     * @returns {string}
     */
    getStretch: function() {
      return this._stretch;
    },
    get stretch() { return this.getStretch(); },

    /**
     * Returns this font's size. See the constructor for more details on valid values.
     * @public
     *
     * NOTE: If a numeric size was passed in, it has been cast to a string, and a string will be returned here.
     *
     * @returns {string}
     */
    getSize: function() {
      return this._size;
    },
    get size() { return this.getSize(); },

    /**
     * Returns this font's line-height. See the constructor for more details on valid values.
     * @public
     *
     * @returns {string}
     */
    getLineHeight: function() {
      return this._lineHeight;
    },
    get lineHeight() { return this.getLineHeight(); },

    /**
     * Returns this font's family. See the constructor for more details on valid values.
     * @public
     *
     * @returns {string}
     */
    getFamily: function() {
      return this._family;
    },
    get family() { return this.getFamily(); },

    /**
     * Returns a new Font object, which is a copy of this object. If options are provided, they override the current
     * values in this object.
     * @public
     *
     * @param {Object} [options] - See the constructor for the object format
     * @returns {Font}
     */
    copy: function( options ) {
      return new Font( _.extend( {
        style: this._style,
        variant: this._variant,
        weight: this._weight,
        stretch: this._stretch,
        size: this._size,
        lineHeight: this._lineHeight,
        family: this._family
      }, options ) );
    },

    /**
     * Computes the combined CSS shorthand font string.
     * @private
     *
     * https://www.w3.org/TR/css-fonts-3/#propdef-font contains details about the format.
     *
     * @returns {string}
     */
    computeShorthand: function() {
      var ret = '';
      if ( this._style !== 'normal' ) { ret += this._style + ' '; }
      if ( this._variant !== 'normal' ) { ret += this._variant + ' '; }
      if ( this._weight !== 'normal' ) { ret += this._weight + ' '; }
      if ( this._stretch !== 'normal' ) { ret += this._stretch + ' '; }
      ret += this._size;
      if ( this._lineHeight !== 'normal' ) { ret += '/' + this._lineHeight; }
      ret += ' ' + this._family;
      return ret;
    },

    /**
     * Returns this font's CSS shorthand, which includes all of the font's information reduced into a single string.
     * @public
     *
     * NOTE: This is an alias of getFont().
     *
     * This can be used for CSS as the 'font' attribute, or is needed to set Canvas fonts.
     *
     * https://www.w3.org/TR/css-fonts-3/#propdef-font contains detailed information on how this is formatted.
     *
     * @returns {string}
     */
    toCSS: function() {
      return this.getFont();
    }
  }, {
    /**
     * Converts a generic size to a specific CSS pixel string, assuming 'px' for numbers.
     * @public
     *
     * @param {string|number} size - If it's a number, 'px' will be appended
     * @returns {string}
     */
    castSize: function( size ) {
      if ( typeof size === 'number' ) {
        return size + 'px'; // add the pixels suffix by default for numbers
      }
      else {
        return size; // assume that it's a valid to-spec string
      }
    },

    /**
     * Parses a CSS-compliant "font" shorthand string into a Font object.
     * @public
     *
     * Font strings should be a valid CSS3 font declaration value (see http://www.w3.org/TR/css3-fonts/) which consists
     * of the following pattern:
     *   [ [ <‘font-style’> || <font-variant-css21> || <‘font-weight’> || <‘font-stretch’> ]? <‘font-size’>
     *   [ / <‘line-height’> ]? <‘font-family’> ]
     *
     * @param {string} cssString
     * @returns {Font}
     */
    fromCSS: function( cssString ) {
      // parse a somewhat proper CSS3 form (not guaranteed to handle it precisely the same as browsers yet)

      var options = {};

      // split based on whitespace allowed by CSS spec (more restrictive than regular regexp whitespace)
      var tokens = _.filter( cssString.split( /[\x09\x0A\x0C\x0D\x20]/ ), function( token ) { return token.length > 0; } ); // eslint-disable-line no-control-regex

      // pull tokens out until we reach something that doesn't match. that must be the font size (according to spec)
      for ( var i = 0; i < tokens.length; i++ ) {
        var token = tokens[ i ];
        if ( token === 'normal' ) {
          // nothing has to be done, everything already normal as default
        }
        else if ( _.includes( VALID_STYLES, token ) ) {
          assert && assert( options.style === undefined, 'Style cannot be applied twice. Already set to "' + options.style + '", attempt to replace with "' + token + '"' );
          options.style = token;
        }
        else if ( _.includes( VALID_VARIANTS, token ) ) {
          assert && assert( options.variant === undefined, 'Variant cannot be applied twice. Already set to "' + options.variant + '", attempt to replace with "' + token + '"' );
          options.variant = token;
        }
        else if ( _.includes( VALID_WEIGHTS, token ) ) {
          assert && assert( options.weight === undefined, 'Weight cannot be applied twice. Already set to "' + options.weight + '", attempt to replace with "' + token + '"' );
          options.weight = token;
        }
        else if ( _.includes( VALID_STRETCHES, token ) ) {
          assert && assert( options.stretch === undefined, 'Stretch cannot be applied twice. Already set to "' + options.stretch + '", attempt to replace with "' + token + '"' );
          options.stretch = token;
        }
        else {
          // not a style/variant/weight/stretch, must be a font size, possibly with an included line-height
          var subtokens = token.split( /\// ); // extract font size from any line-height
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
  } );

  // @public {Font} - Default Font object (since they are immutable).
  Font.DEFAULT = new Font();

  return Font;
} );
