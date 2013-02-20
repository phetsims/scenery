// Copyright 2002-2012, University of Colorado

/**
 * Text
 *
 * TODO: newlines
 *
 * Useful specs:
 * http://www.w3.org/TR/css3-text/
 * http://www.w3.org/TR/css3-fonts/
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

var scenery = scenery || {};

(function(){
  "use strict";
  
  scenery.Text = function( text, params ) {
    this._font         = new scenery.Font(); // default font, usually 10px sans-serif
    this._textAlign    = 'start';            // start, end, left, right, center
    this._textBaseline = 'alphabetic';       // top, hanging, middle, alphabetic, ideographic, bottom
    this._direction    = 'ltr';              // ltr, rtl, inherit -- consider inherit deprecated, due to how we compute text bounds in an off-screen canvas
    
    // ensure we have a parameter object
    params = params || {};
    
    // default to black filled text
    if ( params.fill === undefined ) {
      params.fill = '#000000';
    }
    
    if ( text !== undefined ) {
      // set the text parameter so that setText( text ) is effectively called in the mutator from the super call
      params.text = text;
    }
    scenery.Node.call( this, params );
  };
  var Text = scenery.Text;
  
  Text.prototype = phet.Object.create( scenery.Node.prototype );
  Text.prototype.constructor = Text;
  
  Text.prototype.setText = function( text ) {
    this._text = text;
    this.invalidateText();
    return this;
  };
  
  Text.prototype.getText = function() {
    return this._text;
  };
  
  Text.prototype.invalidateText = function() {
    // TODO: faster bounds determination? getBBox()?
    // investigate http://mudcu.be/journal/2011/01/html5-typographic-metrics/
    this.invalidateSelf( scenery.canvasTextBoundsAccurate( this._text, this ) );
  };

  // TODO: add SVG / DOM support
  Text.prototype.paintCanvas = function( state ) {
    var layer = state.layer;
    var context = layer.context;
    if ( this.hasFill() ) {
      layer.setFillStyle( this.getFill() );
      layer.setFont( this._font.getFont() );
      layer.setTextAlign( this._textAlign );
      layer.setTextBaseline( this._textBaseline );
      layer.setDirection( this._direction );

      context.fillText( this._text, 0, 0 );
    }
  };
  
  Text.prototype.paintWebGL = function( state ) {
    throw new Error( 'Text.prototype.paintWebGL unimplemented' );
  };
  
  /*---------------------------------------------------------------------------*
  * Self setters / getters
  *----------------------------------------------------------------------------*/
  
  Text.prototype.setFont = function( font ) {
    this._font = font instanceof scenery.Font ? font : new scenery.Font( font );
    this.invalidateText();
    return this;
  };
  
  // NOTE: returns mutable copy for now, consider either immutable version, defensive copy, or note about invalidateText()
  Text.prototype.getFont = function() {
    return this._font.getFont();
  };
  
  Text.prototype.setTextAlign = function( textAlign ) {
    this._textAlign = textAlign;
    this.invalidateText();
    return this;
  };
  
  Text.prototype.getTextAlign = function() {
    return this._textAlign;
  };
  
  Text.prototype.setTextBaseline = function( textBaseline ) {
    this._textBaseline = textBaseline;
    this.invalidateText();
    return this;
  };
  
  Text.prototype.getTextBaseline = function() {
    return this._textBaseline;
  };
  
  Text.prototype.setDirection = function( direction ) {
    this._direction = direction;
    this.invalidateText();
    return this;
  };
  
  Text.prototype.getDirection = function() {
    return this._direction;
  };
  
  /*---------------------------------------------------------------------------*
  * Font setters / getters
  *----------------------------------------------------------------------------*/
  
  function addFontForwarding( propertyName, fullCapitalized, shortUncapitalized ) {
    var getterName = 'get' + fullCapitalized;
    var setterName = 'set' + fullCapitalized;
    
    Text.prototype[getterName] = function() {
      // use the ES5 getter to retrieve the property. probably somewhat slow.
      return this._font[ shortUncapitalized ];
    };
    
    Text.prototype[setterName] = function( value ) {
      // use the ES5 setter. probably somewhat slow.
      this._font[ shortUncapitalized ] = value;
      this.invalidateText();
      return this;
    };
    
    Object.defineProperty( Text.prototype, propertyName, { set: Text.prototype[setterName], get: Text.prototype[getterName] } );
  }
  
  addFontForwarding( 'fontWeight', 'fontWeight', 'weight' );
  addFontForwarding( 'fontFamily', 'fontFamily', 'family' );
  addFontForwarding( 'fontStretch', 'fontStretch', 'stretch' );
  addFontForwarding( 'fontStyle', 'fontStyle', 'style' );
  addFontForwarding( 'fontSize', 'fontSize', 'size' );
  addFontForwarding( 'lineHeight', 'LineHeight', 'lineHeight' );
  
  Text.prototype.hasSelf = function() {
    return true;
  };
  
  Text.prototype._mutatorKeys = [ 'text', 'font', 'fontWeight', 'fontFamily', 'fontStretch', 'fontStyle', 'fontSize', 'lineHeight',
                                  'textAlign', 'textBaseline', 'direction' ].concat( scenery.Node.prototype._mutatorKeys );
  
  Text.prototype._supportedLayerTypes = [ scenery.LayerType.Canvas ];
  
  // font-specific ES5 setters and getters are defined using addFontForwarding above
  Object.defineProperty( Text.prototype, 'font', { set: Text.prototype.setFont, get: Text.prototype.getFont } );
  Object.defineProperty( Text.prototype, 'text', { set: Text.prototype.setText, get: Text.prototype.getText } );
  Object.defineProperty( Text.prototype, 'textAlign', { set: Text.prototype.setTextAlign, get: Text.prototype.getTextAlign } );
  Object.defineProperty( Text.prototype, 'textBaseline', { set: Text.prototype.setTextBaseline, get: Text.prototype.getTextBaseline } );
  Object.defineProperty( Text.prototype, 'direction', { set: Text.prototype.setDirection, get: Text.prototype.getDirection } );
  
  // mix in support for fills
  scenery.Fillable( Text );
})();


