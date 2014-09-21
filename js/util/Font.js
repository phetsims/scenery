// Copyright 2002-2013, University of Colorado

/**
 * Font handling for text drawing
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
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  'use strict';

  var scenery = require( 'SCENERY/scenery' );

  // constants used for detection (since styles/variants/weights/stretches can be mixed in the preamble of the shorthand string)
  var styles = [ 'normal', 'italic', 'oblique' ];
  var variants = [ 'normal', 'small-caps' ];
  var weights = [ 'normal', 'bold', 'bolder', 'lighter', '100', '200', '300', '400', '500', '600', '700', '800', '900' ];
  var stretches = [ 'normal', 'ultra-condensed', 'extra-condensed', 'condensed', 'semi-condensed', 'semi-expanded', 'expanded', 'extra-expanded', 'ultra-expanded' ];

  function castSize( size ) {
    if ( typeof size === 'number' ) {
      return size + 'px'; // add the pixels suffix by default for numbers
    }
    else {
      return size; // assume that it's a valid to-spec string
    }
  }

  scenery.Font = function Font( options ) {
    // options from http://www.w3.org/TR/css3-fonts/
    this._style = 'normal';      // normal | italic | oblique
    this._variant = 'normal';    // normal | small-caps
    this._weight = 'normal';     // normal | bold | bolder | lighter | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900
    this._stretch = 'normal';    // normal | ultra-condensed | extra-condensed | condensed | semi-condensed | semi-expanded | expanded | extra-expanded | ultra-expanded
    this._size = '10px';         // <absolute-size> | <relative-size> | <length> | <percentage> -- unitless number interpreted as px. absolute suffixes: cm, mm, in, pt, pc, px. relative suffixes: em, ex, ch, rem, vw, vh, vmin, vmax.
    this._lineHeight = 'normal'; // normal | <number> | <length> | <percentage> -- NOTE: Canvas spec forces line-height to normal
    this._family = 'sans-serif'; // comma-separated list of families, including generic families (serif, sans-serif, cursive, fantasy, monospace). ideally escape with double-quotes

    // font  [ [ <‘font-style’> || <font-variant-css21> || <‘font-weight’> || <‘font-stretch’> ]? <‘font-size’> [ / <‘line-height’> ]? <‘font-family’> ] | caption | icon | menu | message-box | small-caption | status-bar
    // <font-variant-css21> = [normal | small-caps]

    var type = typeof options;
    if ( type === 'string' ) {
      // parse a somewhat proper CSS3 form (not guaranteed to handle it precisely the same as browsers yet)

      // split based on whitespace allowed by CSS spec (more restrictive than regular regexp whitespace)
      var tokens = _.filter( options.split( /[\x09\x0A\x0C\x0D\x20]/ ), function( token ) { return token.length > 0; } );

      // pull tokens out until we reach something that doesn't match. that must be the font size (according to spec)
      for ( var i = 0; i < tokens.length; i++ ) {
        var token = tokens[i];
        if ( token === 'normal' ) {
          // nothing has to be done, everything already normal as default
        }
        else if ( _.contains( styles, token ) ) {
          assert && assert( this._style === 'normal', 'Style cannot be applied twice. Already set to "' + this._style + '", attempt to replace with "' + token + '"' );
          this._style = token;
        }
        else if ( _.contains( variants, token ) ) {
          assert && assert( this._variant === 'normal', 'Variant cannot be applied twice. Already set to "' + this._variant + '", attempt to replace with "' + token + '"' );
          this._variant = token;
        }
        else if ( _.contains( weights, token ) ) {
          assert && assert( this._weight === 'normal', 'Weight cannot be applied twice. Already set to "' + this._weight + '", attempt to replace with "' + token + '"' );
          this._weight = token;
        }
        else if ( _.contains( stretches, token ) ) {
          assert && assert( this._stretch === 'normal', 'Stretch cannot be applied twice. Already set to "' + this._stretch + '", attempt to replace with "' + token + '"' );
          this._stretch = token;
        }
        else {
          // not a style/variant/weight/stretch, must be a font size, possibly with an included line-height
          var subtokens = token.split( /\// ); // extract font size from any line-height
          this._size = subtokens[0];
          if ( subtokens[1] ) {
            this._lineHeight = subtokens[1];
          }
          // all future tokens are guaranteed to be part of the font-family if it is given according to spec
          this._family = tokens.slice( i + 1 ).join( ' ' );
          break;
        }
      }
    }
    else if ( type === 'object' ) {
      if ( options.style !== undefined ) {
        this._style = options.style;
      }
      if ( options.variant !== undefined ) {
        this._variant = options.variant;
      }
      if ( options.weight !== undefined ) {
        this._weight = '' + options.weight; // cast it to a string explicitly
      }
      if ( options.stretch !== undefined ) {
        this._stretch = options.stretch;
      }
      if ( options.size !== undefined ) {
        this._size = castSize( options.size );
      }
      if ( options.lineHeight !== undefined ) {
        this._lineHeight = options.lineHeight;
      }
      if ( options.family !== undefined ) {
        this._family = options.family;
      }
    }

    // sanity checks to prevent errors in interpretation or in the font shorthand usage
    assert && assert( typeof this._style === 'string' &&
                      _.contains( styles, this._style ),
      'Font style must be one of "normal", "italic", or "oblique"' );
    assert && assert( typeof this._variant === 'string' &&
                      _.contains( variants, this._variant ),
      'Font variant must be "normal" or "small-caps"' );
    assert && assert( typeof this._weight === 'string' &&
                      _.contains( weights, this._weight ),
      'Font weight must be one of "normal", "bold", "bolder", "lighter", "100", "200", "300", "400", "500", "600", "700", "800", or "900"' );
    assert && assert( typeof this._stretch === 'string' &&
                      _.contains( stretches, this._stretch ),
      'Font stretch must be one of "normal", "ultra-condensed", "extra-condensed", "condensed", "semi-condensed", "semi-expanded", "expanded", "extra-expanded", or "ultra-expanded"' );
    assert && assert( typeof this._size === 'string' && !_.contains( [ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' ], this._size[this._size.length - 1] ),
      'Font size must be either passed as a number (not a string, interpreted as px), or must contain a suffix for percentage, absolute or relative units, or an explicit size constant' );
    assert && assert( typeof this._lineHeight === 'string' );
    assert && assert( typeof this._family === 'string' );

    // initialize the shorthand font property (stored as _font)
    this._font = this.computeShorthand();

    phetAllocation && phetAllocation( 'Font' );
  };
  var Font = scenery.Font;

  Font.prototype = {
    constructor: Font,

    getFont: function() { return this._font; },
    getStyle: function() { return this._style; },
    getVariant: function() { return this._variant; },
    getWeight: function() { return this._weight; },
    getStretch: function() { return this._stretch; },
    getSize: function() { return this._size; },
    getLineHeight: function() { return this._lineHeight; },
    getFamily: function() { return this._family; },

    get font() { return this.getFont(); },
    get style() { return this.getStyle(); },
    get variant() { return this.getVariant(); },
    get weight() { return this.getWeight(); },
    get stretch() { return this.getStretch(); },
    get size() { return this.getSize(); },
    get lineHeight() { return this.getLineHeight(); },
    get family() { return this.getFamily(); },

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

    toCSS: function() {
      return this.getFont();
    }
  };

  Font.DEFAULT = new Font();

  return Font;
} );
