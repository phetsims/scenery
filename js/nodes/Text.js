// Copyright 2002-2012, University of Colorado

/**
 * Text
 *
 * TODO: newlines
 *
 * Useful specs:
 * http://www.w3.org/TR/css3-text/
 * http://www.w3.org/TR/css3-fonts/
 * http://www.w3.org/TR/SVG/text.html
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  "use strict";
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var inherit = require( 'PHET_CORE/inherit' );
  var Bounds2 = require( 'DOT/Bounds2' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  var Node = require( 'SCENERY/nodes/Node' ); // inherits from Node
  var Renderer = require( 'SCENERY/layers/Renderer' );
  var fillable = require( 'SCENERY/nodes/Fillable' );
  var strokable = require( 'SCENERY/nodes/Strokable' );
  var objectCreate = require( 'SCENERY/util/Util' ).objectCreate; // i.e. Object.create
  require( 'SCENERY/util/Font' );
  require( 'SCENERY/util/Util' ); // for canvasAccurateBounds
  
  scenery.Text = function Text( text, options ) {
    this._text         = '';                 // filled in with mutator
    this._font         = new scenery.Font(); // default font, usually 10px sans-serif
    this._textAlign    = 'start';            // start, end, left, right, center
    this._textBaseline = 'alphabetic';       // top, hanging, middle, alphabetic, ideographic, bottom
    this._direction    = 'ltr';              // ltr, rtl, inherit -- consider inherit deprecated, due to how we compute text bounds in an off-screen canvas
    
    // ensure we have a parameter object
    options = options || {};
    
    // default to black filled text
    if ( options.fill === undefined ) {
      options.fill = '#000000';
    }
    
    if ( text !== undefined ) {
      // set the text parameter so that setText( text ) is effectively called in the mutator from the super call
      options.text = text;
    }
    
    this.initializeStrokable();
    
    Node.call( this, options );
  };
  var Text = scenery.Text;
  
  inherit( Text, Node, {
    setText: function( text ) {
      if ( text !== this._text ) {
        this._text = text;
        this.invalidateText();
      }
      return this;
    },
    
    getText: function() {
      return this._text;
    },
    
    invalidateText: function() {
      // TODO: faster bounds determination? getBBox()?
      // investigate http://mudcu.be/journal/2011/01/html5-typographic-metrics/
      this.invalidateSelf( this.accurateCanvasBounds() );
    },

    paintCanvas: function( wrapper ) {
      var context = wrapper.context;
      
      // extra parameters we need to set, but should avoid setting if we aren't drawing anything
      if ( this.hasFill() || this.hasStroke() ) {
        wrapper.setFont( this._font.getFont() );
        wrapper.setTextAlign( this._textAlign );
        wrapper.setTextBaseline( this._textBaseline );
        wrapper.setDirection( this._direction );
      }
      
      if ( this.hasFill() ) {
        this.beforeCanvasFill( wrapper ); // defined in Fillable
        context.fillText( this._text, 0, 0 );
        this.afterCanvasFill( wrapper ); // defined in Fillable
      }
      if ( this.hasStroke() ) {
        this.beforeCanvasStroke( wrapper ); // defined in Strokable
        context.strokeText( this._text, 0, 0 );
        this.afterCanvasStroke( wrapper ); // defined in Strokable
      }
    },
    
    paintWebGL: function( state ) {
      throw new Error( 'Text.prototype.paintWebGL unimplemented' );
    },
    
    createSVGFragment: function( svg, defs, group ) {
      return document.createElementNS( 'http://www.w3.org/2000/svg', 'text' );
    },
    
    updateSVGFragment: function( element ) {
      var isRTL = this._direction === 'rtl';
      
      // make the text the only child
      while ( element.hasChildNodes() ) {
        element.removeChild( element.lastChild );
      }
      element.appendChild( document.createTextNode( this._text ) );
      
      element.setAttribute( 'style', this.getSVGFillStyle() + this.getSVGStrokeStyle() );
      
      switch ( this._textAlign ) {
        case 'start':
        case 'end':
          element.setAttribute( 'text-anchor', this._textAlign ); break;
        case 'left':
          element.setAttribute( 'text-anchor', isRTL ? 'end' : 'start' ); break;
        case 'right':
          element.setAttribute( 'text-anchor', !isRTL ? 'end' : 'start' ); break;
        case 'center':
          element.setAttribute( 'text-anchor', 'middle' ); break;
      }
      switch ( this._textBaseline ) {
        case 'alphabetic':
        case 'ideographic':
        case 'hanging':
        case 'middle':
          element.setAttribute( 'dominant-baseline', this._textBaseline ); break;
        default:
          throw new Error( 'impossible to get the SVG approximate bounds for textBaseline: ' + this._textBaseline );
      }
      element.setAttribute( 'direction', this._direction );
      
      // set all of the font attributes, since we can't use the combined one
      element.setAttribute( 'font-family', this._font.getFamily() );
      element.setAttribute( 'font-size', this._font.getSize() );
      element.setAttribute( 'font-style', this._font.getStyle() );
      element.setAttribute( 'font-weight', this._font.getWeight() );
      if ( this._font.getStretch() ) {
        element.setAttribute( 'font-stretch', this._font.getStretch() );
      }
    },
    
    // support patterns, gradients, and anything else we need to put in the <defs> block
    updateSVGDefs: function( svg, defs ) {
      // remove old definitions if they exist
      this.removeSVGDefs( svg, defs );
      
      // add new ones if applicable
      this.addSVGFillDef( svg, defs );
      this.addSVGStrokeDef( svg, defs );
    },
    
    // cleans up references created with udpateSVGDefs()
    removeSVGDefs: function( svg, defs ) {
      this.removeSVGFillDef( svg, defs );
      this.removeSVGStrokeDef( svg, defs );
    },
    
    /*---------------------------------------------------------------------------*
    * Bounds
    *----------------------------------------------------------------------------*/
    
    accurateCanvasBounds: function() {
      var node = this;
      var svgBounds = this.approximateSVGBounds();
      return scenery.Util.canvasAccurateBounds( function( context ) {
        context.font = node.font;
        context.textAlign = node.textAlign;
        context.textBaseline = node.textBaseline;
        context.direction = node.direction;
        context.fillText( node.text, 0, 0 );
      }, {
        precision: 0.5,
        resolution: 128,
        initialScale: 32 / Math.max( Math.abs( svgBounds.minX ), Math.abs( svgBounds.minY ), Math.abs( svgBounds.maxX ), Math.abs( svgBounds.maxY ) )
      } );
    },
    
    approximateCanvasWidth: function() {
      // TODO: consider caching a scratch 1x1 canvas for this purpose
      var context = document.createElement( 'canvas' ).getContext( '2d' );
      context.font = this.font;
      context.textAlign = this.textAlign;
      context.textBaseline = this.textBaseline;
      context.direction = this.direction;
      return context.measureText( this.text ).width;
    },
    
    approximateSVGBounds: function() {
      var isRTL = this._direction === 'rtl';
      
      var svg = document.createElementNS( 'http://www.w3.org/2000/svg', 'svg' );
      svg.setAttribute( 'width', '1024' );
      svg.setAttribute( 'height', '1024' );
      svg.setAttribute( 'style', 'display: hidden;' ); // so we don't flash it in a visible way to the user
      
      var textElement = document.createElementNS( 'http://www.w3.org/2000/svg', 'text' );
      this.updateSVGFragment( textElement );
      
      svg.appendChild( textElement );
      
      document.body.appendChild( svg );
      var rect = textElement.getBBox();
      var result = new Bounds2( rect.x, rect.y, rect.x + rect.width, rect.y + rect.height );
      document.body.removeChild( svg );
      
      return result;
    },
    
    approximateDOMBounds: function() {
      // TODO: we can also technically support 'top' using vertical-align: top and line-height: 0 with the image, but it won't usually render otherwise
      assert && assert( this._textBaseline === 'alphabetic' );
      
      var maxHeight = 1024; // technically this will fail if the font is taller than this!
      var isRTL = this.direction === 'rtl';
      
      // <div style="position: absolute; left: 0; top: 0; padding: 0 !important; margin: 0 !important;"><span id="baselineSpan" style="font-family: Verdana; font-size: 25px;">QuipTaQiy</span><div style="vertical-align: baseline; display: inline-block; width: 0; height: 500px; margin: 0 important!; padding: 0 important!;"></div></div>
      
      var div = document.createElement( 'div' );
      $( div ).css( {
        position: 'absolute',
        left: 0,
        top: 0,
        padding: '0 !important',
        margin: '0 !important',
        display: 'hidden'
      } );
      
      var span = document.createElement( 'span' );
      $( span ).css( 'font', this.getFont() );
      span.appendChild( document.createTextNode( this.text ) );
      span.setAttribute( 'direction', this._direction );
      
      var fakeImage = document.createElement( 'div' );
      $( fakeImage ).css( {
        'vertical-align': 'baseline',
        display: 'inline-block',
        width: 0,
        height: maxHeight + 'px',
        margin: '0 !important',
        padding: '0 !important'
      } );
      
      div.appendChild( span );
      div.appendChild( fakeImage );
      
      document.body.appendChild( div );
      var rect = span.getBoundingClientRect();
      var divRect = div.getBoundingClientRect();
      var result = new Bounds2( rect.left, rect.top - maxHeight, rect.right, rect.bottom - maxHeight ).shifted( -divRect.left, -divRect.top );
      document.body.removeChild( div );
      
      var width = rect.right - rect.left;
      switch ( this._textAlign ) {
        case 'start':
          result = result.shiftedX( isRTL ? -width : 0 );
          break;
        case 'end':
          result = result.shiftedX( !isRTL ? -width : 0 );
          break;
        case 'left':
          break;
        case 'right':
          result = result.shiftedX( -width );
          break;
        case 'center':
          result = result.shiftedX( -width / 2 );
          break;
      }
      
      return result;
    },
    
    /*---------------------------------------------------------------------------*
    * Self setters / getters
    *----------------------------------------------------------------------------*/
    
    setFont: function( font ) {
      // if font is a Font instance, we actually create another copy so that modification on the original will not change this font.
      // in the future we can consider adding listeners to the font to get font change notifications.
      this._font = font instanceof scenery.Font ? new scenery.Font( font.getFont() ) : new scenery.Font( font );
      this.invalidateText();
      return this;
    },
    
    // NOTE: returns mutable copy for now, consider either immutable version, defensive copy, or note about invalidateText()
    getFont: function() {
      return this._font.getFont();
    },
    
    setTextAlign: function( textAlign ) {
      this._textAlign = textAlign;
      this.invalidateText();
      return this;
    },
    
    getTextAlign: function() {
      return this._textAlign;
    },
    
    setTextBaseline: function( textBaseline ) {
      this._textBaseline = textBaseline;
      this.invalidateText();
      return this;
    },
    
    getTextBaseline: function() {
      return this._textBaseline;
    },
    
    setDirection: function( direction ) {
      this._direction = direction;
      this.invalidateText();
      return this;
    },
    
    getDirection: function() {
      return this._direction;
    },
    
    isPainted: function() {
      return true;
    },
    
    getBasicConstructor: function( propLines ) {
      return 'new scenery.Text( \'' + this._text.replace( /'/g, '\\\'' ) + '\', {' + propLines + '} )';
    },
    
    getPropString: function( spaces ) {
      var result = Node.prototype.getPropString.call( this, spaces );
      
      // TODO: if created again, deduplicate with Node's getPropString
      function addProp( key, value, nowrap ) {
        if ( result ) {
          result += ',\n';
        }
        if ( !nowrap && typeof value === 'string' ) {
          result += spaces + key + ': \'' + value + '\'';
        } else {
          result += spaces + key + ': ' + value;
        }
      }
      
      if ( this.font !== new scenery.Font().getFont() ) {
        addProp( 'font', this.font );
      }
      
      if ( this._textAlign !== 'start' ) {
        addProp( 'textAlign', this._textAlign );
      }
      
      if ( this._textBaseline !== 'alphabetic' ) {
        addProp( 'textBaseline', this._textBaseline );
      }
      
      if ( this._direction !== 'ltr' ) {
        addProp( 'direction', this._direction );
      }
      
      return result;
    }
  } );
  
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
  
  addFontForwarding( 'fontWeight', 'FontWeight', 'weight' );
  addFontForwarding( 'fontFamily', 'FontFamily', 'family' );
  addFontForwarding( 'fontStretch', 'FontStretch', 'stretch' );
  addFontForwarding( 'fontStyle', 'FontStyle', 'style' );
  addFontForwarding( 'fontSize', 'FontSize', 'size' );
  addFontForwarding( 'lineHeight', 'LineHeight', 'lineHeight' );
  
  Text.prototype._mutatorKeys = [ 'text', 'font', 'fontWeight', 'fontFamily', 'fontStretch', 'fontStyle', 'fontSize', 'lineHeight',
                                  'textAlign', 'textBaseline', 'direction' ].concat( Node.prototype._mutatorKeys );
  
  Text.prototype._supportedRenderers = [ Renderer.Canvas, Renderer.SVG ];
  
  // font-specific ES5 setters and getters are defined using addFontForwarding above
  Object.defineProperty( Text.prototype, 'font', { set: Text.prototype.setFont, get: Text.prototype.getFont } );
  Object.defineProperty( Text.prototype, 'text', { set: Text.prototype.setText, get: Text.prototype.getText } );
  Object.defineProperty( Text.prototype, 'textAlign', { set: Text.prototype.setTextAlign, get: Text.prototype.getTextAlign } );
  Object.defineProperty( Text.prototype, 'textBaseline', { set: Text.prototype.setTextBaseline, get: Text.prototype.getTextBaseline } );
  Object.defineProperty( Text.prototype, 'direction', { set: Text.prototype.setDirection, get: Text.prototype.getDirection } );
  
  // mix in support for fills and strokes
  fillable( Text );
  strokable( Text );

  return Text;
} );


