// Copyright 2002-2012, University of Colorado

/**
 * Encapsulates common color information and transformations.
 *
 * Consider it immutable!
 *
 * See http://www.w3.org/TR/css3-color/
 *
 * TODO: consider using https://github.com/One-com/one-color internally
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  'use strict';
  
  var scenery = require( 'SCENERY/scenery' );
  
  var clamp = require( 'DOT/Util' ).clamp;
  
  // r,g,b integers 0-255, 'a' float 0-1
  scenery.Color = function Color( r, g, b, a ) {
    
    if ( typeof r === 'string' ) {
      var str = r.replace( / /g, '' ).toLowerCase();
      var success = false;
      
      // replace colors based on keywords
      var keywordMatch = Color.colorKeywords[str];
      if ( keywordMatch ) {
        str = '#' + keywordMatch;
      }
      
      // run through the available text formats
      for ( var i = 0; i < Color.formatParsers.length; i++ ) {
        var parser = Color.formatParsers[i];
        
        var matches = parser.regexp.exec( str );
        if ( matches ) {
          parser.apply( this, matches );
          success = true;
          break;
        }
      }
      
      if ( !success ) {
        throw new Error( 'scenery.Color unable to parse color string: ' + r );
      }
    } else {
      // alpha
      this.a = a === undefined ? 1 : a;

      // bitwise handling if 3 elements aren't defined
      if ( g === undefined || b === undefined ) {
        this.r = ( r >> 16 ) && 0xFF;
        this.g = ( r >> 8 ) && 0xFF;
        this.b = ( r >> 0 ) && 0xFF;
      }
      else {
        // otherwise, copy them over
        this.r = r;
        this.g = g;
        this.b = b;
      }
    }
  };
  var Color = scenery.Color;
  
  // regex utilities
  var rgbNumber = '(-?\\d{1,3}%?)'; // syntax allows negative integers and percentages
  var aNumber = '(\\d+|\\d*\\.\\d+)'; // decimal point number. technically we allow for '255', even though this will be clamped to 1.
  var rawNumber = '(\\d{1,3})'; // a 1-3 digit number
  
  // handles negative and percentage values
  function parseRGBNumber( str ) {
    var multiplier = 1;
    
    // if it's a percentage, strip it off and handle it that way
    if ( str.charAt( str.length - 1 ) === '%' ) {
      multiplier = 2.55;
      str = str.slice( 0, str.length - 1 );
    }
    
    return Math.round( parseInt( str, 10 ) * multiplier );
  }
  
  Color.formatParsers = [
    {
      // 'transparent'
      regexp: /^transparent$/,
      apply: function( color, matches ) {
        color.setRGBA( 0, 0, 0, 0 );
      }
    },{
      // short hex form, a la '#fff'
      regexp: /^#(\w{1})(\w{1})(\w{1})$/,
      apply: function( color, matches ) {
        color.setRGBA( parseInt( matches[1] + matches[1], 16 ),
                       parseInt( matches[2] + matches[2], 16 ),
                       parseInt( matches[3] + matches[3], 16 ),
                       1 );
      }
    },{
      // long hex form, a la '#ffffff'
      regexp: /^#(\w{2})(\w{2})(\w{2})$/,
      apply: function( color, matches ) {
        color.setRGBA( parseInt( matches[1], 16 ),
                       parseInt( matches[2], 16 ),
                       parseInt( matches[3], 16 ),
                       1 );
      }
    },{
      // rgb(...)
      regexp: new RegExp( '^rgb\\(' + rgbNumber + ',' + rgbNumber + ',' + rgbNumber + '\\)$' ),
      apply: function( color, matches ) {
        color.setRGBA( parseRGBNumber( matches[1] ),
                       parseRGBNumber( matches[2] ),
                       parseRGBNumber( matches[3] ),
                       1 );
      }
    },{
      // rgba(...)
      regexp: new RegExp( '^rgba\\(' + rgbNumber + ',' + rgbNumber + ',' + rgbNumber + ',' + aNumber + '\\)$' ),
      apply: function( color, matches ) {
        color.setRGBA( parseRGBNumber( matches[1] ),
                       parseRGBNumber( matches[2] ),
                       parseRGBNumber( matches[3] ),
                       parseFloat( matches[4] ) );
      }
    },{
      // hsl(...)
      regexp: new RegExp( '^hsl\\(' + rawNumber + ',' + rawNumber + '%,' + rawNumber + '%\\)$' ),
      apply: function( color, matches ) {
        color.setHSLA( parseInt( matches[1], 10 ),
                       parseInt( matches[2], 10 ),
                       parseInt( matches[3], 10 ),
                       1 );
      }
    },{
      // hsla(...)
      regexp: new RegExp( '^hsla\\(' + rawNumber + ',' + rawNumber + '%,' + rawNumber + '%,' + aNumber + '\\)$' ),
      apply: function( color, matches ) {
        color.setHSLA( parseInt( matches[1], 10 ),
                       parseInt( matches[2], 10 ),
                       parseInt( matches[3], 10 ),
                       parseFloat( matches[4] ) );
      }
    }
  ];
  
  // see http://www.w3.org/TR/css3-color/
  Color.hueToRGB = function( m1, m2, h ) {
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
  };
  
  Color.prototype = {
    constructor: Color,
    
    // RGB integral between 0-255, alpha (float) between 0-1
    setRGBA: function( red, green, blue, alpha ) {
      this.r = Math.round( clamp( red, 0, 255 ) );
      this.g = Math.round( clamp( green, 0, 255 ) );
      this.b = Math.round( clamp( blue, 0, 255 ) );
      this.a = clamp( alpha, 0, 1 );
    },
    
    // TODO: on modification, cache this.
    toCSS: function() {
      if ( this.a === 1 ) {
        return 'rgb(' + this.r + ',' + this.g + ',' + this.b + ')';
      } else {
        var alphaString = this.a === 0 || this.a === 1 ? this.a : this.a.toFixed( 20 ); // toFixed prevents scientific notation
        return 'rgba(' + this.r + ',' + this.g + ',' + this.b + ',' + alphaString + ')';
      }
    },
    
    setHSLA: function( hue, saturation, lightness, alpha ) {
      hue = ( hue % 360 ) / 360;                    // integer modulo 360
      saturation = clamp( saturation / 100, 0, 1 ); // percentage
      lightness = clamp( lightness / 100, 0, 1 );   // percentage
      
      // see http://www.w3.org/TR/css3-color/
      var m1, m2;
      if ( lightness < 0.5 ) {
        m2 = lightness * ( saturation + 1 );
      } else {
        m2 = lightness + saturation - lightness * saturation;
      }
      m1 = lightness * 2 - m2;
      
      this.r = Math.round( Color.hueToRGB( m1, m2, hue + 1/3 ) * 255 );
      this.g = Math.round( Color.hueToRGB( m1, m2, hue ) * 255 );
      this.b = Math.round( Color.hueToRGB( m1, m2, hue - 1/3 ) * 255 );
      this.a = clamp( alpha, 0, 1 );
    },
    
    equals: function( color ) {
      return this.r === color.r && this.g === color.g && this.b === color.b && this.a === color.a;
    },
    
    withAlpha: function( alpha ) {
      return new Color( this.r, this.g, this.b, alpha );
    },
    
    brighterColor: function( factor ) {
      if ( factor < 0 || factor > 1 ) {
        throw new Error( "factor must be between 0 and 1: " + factor );
      }
      factor = ( factor === undefined ) ? 0.7 : factor;
      var red = Math.min( 255, Math.floor( this.r / factor ) );
      var green = Math.min( 255, Math.floor( this.g / factor ) );
      var blue = Math.min( 255, Math.floor( this.b / factor ) );
      return new Color( red, green, blue, this.a );
    },
    
    darkerColor: function( factor ) {
      if ( factor < 0 || factor > 1 ) {
        throw new Error( "factor must be between 0 and 1: " + factor );
      }
      factor = ( factor === undefined ) ? 0.7 : factor;
      var red = Math.max( 0, Math.floor( factor * this.r ) );
      var green = Math.max( 0, Math.floor( factor * this.g ) );
      var blue = Math.max( 0, Math.floor( factor * this.b ) );
      return new Color( red, green, blue, this.a );
    }
  };
  
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
    darkturquoise: '00ced1',
    darkviolet: '9400d3',
    deeppink: 'ff1493',
    deepskyblue: '00bfff',
    dimgray: '696969',
    dodgerblue: '1e90ff',
    feldspar: 'd19275',
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
    honeydew: 'f0fff0',
    hotpink: 'ff69b4',
    indianred : 'cd5c5c',
    indigo : '4b0082',
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
    lightgrey: 'd3d3d3',
    lightgreen: '90ee90',
    lightpink: 'ffb6c1',
    lightsalmon: 'ffa07a',
    lightseagreen: '20b2aa',
    lightskyblue: '87cefa',
    lightslateblue: '8470ff',
    lightslategray: '778899',
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
    mediumpurple: '9370d8',
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
    palevioletred: 'd87093',
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
    snow: 'fffafa',
    springgreen: '00ff7f',
    steelblue: '4682b4',
    tan: 'd2b48c',
    teal: '008080',
    thistle: 'd8bfd8',
    tomato: 'ff6347',
    turquoise: '40e0d0',
    violet: 'ee82ee',
    violetred: 'd02090',
    wheat: 'f5deb3',
    white: 'ffffff',
    whitesmoke: 'f5f5f5',
    yellow: 'ffff00',
    yellowgreen: '9acd32'
  };
  
  // JAVA compatibility TODO: remove after porting MS
  Color.BLACK = new Color( 0, 0, 0 );
  Color.BLUE = new Color( 0, 0, 255 );
  Color.CYAN = new Color( 0, 255, 255 );
  Color.DARK_GRAY = new Color( 64, 64, 64 );
  Color.GRAY = new Color( 128, 128, 128 );
  Color.GREEN = new Color( 0, 255, 0 );
  Color.LIGHT_GRAY = new Color( 192, 192, 192 );
  Color.MAGENTA = new Color( 255, 0, 255 );
  Color.ORANGE = new Color( 255, 200, 0 );
  Color.PINK = new Color( 255, 175, 175 );
  Color.RED = new Color( 255, 0, 0 );
  Color.WHITE = new Color( 255, 255, 255 );
  Color.YELLOW = new Color( 255, 255, 0 );
  
  return Color;
} );
