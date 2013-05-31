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
  
  var fontAttributeMap = {
    weight: 'font-weight',
    family: 'font-family',
    stretch: 'font-stretch',
    style: 'font-style',
    size: 'font-size',
    lineHeight: 'line-height' // NOTE: Canvas spec forces line-height to normal
  };
  
  // span for using the browser to compute font styles
  var $span = $( document.createElement( 'span' ) );
  
  // options from http://www.w3.org/TR/css3-fonts/
  // font-family      v ---
  // font-weight      v normal | bold | bolder | lighter | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900
  // font-stretch     v normal | ultra-condensed | extra-condensed | condensed | semi-condensed | semi-expanded | expanded | extra-expanded | ultra-expanded
  // font-style       v normal | italic | oblique
  // font-size        v <absolute-size> | <relative-size> | <length> | <percentage>
  // font-size-adjust v none | auto | <number>
  // font             v [ [ <‘font-style’> || <font-variant-css21> || <‘font-weight’> || <‘font-stretch’> ]? <‘font-size’> [ / <‘line-height’> ]? <‘font-family’> ] | caption | icon | menu | message-box | small-caption | status-bar
  //                    <font-variant-css21> = [normal | small-caps]
  // font-synthesis   v none | [ weight || style ]
  
  scenery.Font = function Font( options ) {
    // internal string representation
    this._font = '10px sans-serif';
    
    var type = typeof options;
    if ( type === 'string' ) {
      this._font = options;
      $span.css( 'font', this._font ); // properly initialize the font instance
    } else if ( type === 'object' ) {
      // initialize if a 'font' is provided, otherwise use the default
      $span.css( 'font', options.font ? options.font : this._font );
      
      // set any font attributes on our span
      _.each( fontAttributeMap, function( cssAttribute, property ) {
        if ( options[property] ) {
          $span.css( cssAttribute, options[property] );
        }
      } );
      this._font = $span.css( 'font' );
    } else {
      $span.css( 'font', this._font ); // properly initialize the font instance
    }
    
    // cache values for all of the span's properties
    this.cache = $span.css( [
      'font',
      'font-family',
      'font-weight',
      'font-stretch',
      'font-style',
      'font-size',
      'lineHeight'
    ] );
  };
  var Font = scenery.Font;
  
  Font.prototype = {
    constructor: Font,
    
    // direct access to the font string
    getFont: function() {
      return this._font;
    },
    
    getFamily:     function() { return this.cache['font-family']; },
    getWeight:     function() { return this.cache['font-weight']; },
    getStretch:    function() { return this.cache['font-stretch']; },
    getStyle:      function() { return this.cache['font-style']; },
    getSize:       function() { return this.cache['font-size']; },
    getLineHeight: function() { return this.cache['line-height']; },
    
    get font()       { return this.getFont(); },
    get weight()     { return this.getWeight(); },
    get family()     { return this.getFamily(); },
    get stretch()    { return this.getStretch(); },
    get style()      { return this.getStyle(); },
    get size()       { return this.getSize(); },
    get lineHeight() { return this.getLineHeight(); },
    
    copy: function( options ) {
      return new Font( _.extend( { font: this._font }, options ) );
    },
    
    toCSS: function() {
      return this.getFont();
    }
  };
  
  Font.DEFAULT = new Font();
  
  return Font;
} );
