// Copyright 2002-2012, University of Colorado

/**
 * Text
 *
 * TODO: newlines (multiline)
 * TODO: htmlText support (and DOM renderer)
 * TODO: don't get bounds until the Text node is fully mutated?
 * TODO: remove some support for centering, since Scenery's Node already handles that better?
 *
 * Useful specs:
 * http://www.w3.org/TR/css3-text/
 * http://www.w3.org/TR/css3-fonts/
 * http://www.w3.org/TR/SVG/text.html
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  'use strict';
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var inherit = require( 'PHET_CORE/inherit' );
  var escapeHTML = require( 'PHET_CORE/escapeHTML' );
  var Bounds2 = require( 'DOT/Bounds2' );
  var Matrix3 = require( 'DOT/Matrix3' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  var Node = require( 'SCENERY/nodes/Node' ); // inherits from Node
  var Renderer = require( 'SCENERY/layers/Renderer' );
  var fillable = require( 'SCENERY/nodes/Fillable' );
  var strokable = require( 'SCENERY/nodes/Strokable' );
  var objectCreate = require( 'SCENERY/util/Util' ).objectCreate; // i.e. Object.create
  require( 'SCENERY/util/Font' );
  require( 'SCENERY/util/Util' ); // for canvasAccurateBounds and CSS transforms
  
  // set up the container and text for testing text bounds quickly (using approximateSVGBounds)
  var svgTextSizeContainer = document.createElementNS( 'http://www.w3.org/2000/svg', 'svg' );
  svgTextSizeContainer.setAttribute( 'width', '2' );
  svgTextSizeContainer.setAttribute( 'height', '2' );
  svgTextSizeContainer.setAttribute( 'style', 'display: hidden; pointer-events: none; position: absolute; left: -65535; right: -65535;' ); // so we don't flash it in a visible way to the user
  var svgTextSizeElement = document.createElementNS( 'http://www.w3.org/2000/svg', 'text' );
  svgTextSizeContainer.appendChild( svgTextSizeElement );
  
  scenery.Text = function Text( text, options ) {
    this._text         = '';                   // filled in with mutator
    this._font         = scenery.Font.DEFAULT; // default font, usually 10px sans-serif
    this._direction    = 'ltr';                // ltr, rtl, inherit -- consider inherit deprecated, due to how we compute text bounds in an off-screen canvas
    this._boundsMethod = 'fast';               // fast (SVG/DOM, no canvas rendering allowed), fastCanvas (SVG/DOM, canvas rendering allowed without dirty regions),
                                               //   or accurate (Canvas accurate recursive)
    
    // whether the text is rendered as HTML or not. if defined (in a subtype constructor), use that value instead
    this._isHTML = this._isHTML === undefined ? false : this._isHTML;
    
    // we will dynamically change renderers, so they are initialized per-instance instead of per-type
    this._supportedRenderers = [ Renderer.Canvas, Renderer.SVG, Renderer.DOM ];
    
    var thisFont = this;
    this.fontListener = function() {
      thisFont.invalidateText();
    };
    this._font.addFontListener( this.fontListener );
    
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
    
    this.updateTextFlags();
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
    
    setBoundsMethod: function( method ) {
      assert && assert( method === 'fast' || method === 'fastCanvas' || method === 'accurate', '"fast" and "accurate" are the only allowed boundsMethod values for Text' );
      if ( method !== this._boundsMethod ) {
        this._boundsMethod = method;
        this.updateTextFlags();
        this.dispatchEvent( 'boundsAccuracy', { node: this } ); // TODO: consider standardizing this, or attaching listeners in a different manner?
        this.invalidateText();
      }
      return this;
    },
    
    getBoundsMethod: function() {
      return this._boundsMethod;
    },
    
    updateTextFlags: function() {
      var thisText = this;
      this.boundsInaccurate = this._boundsMethod !== 'accurate';
      
      var renderersChanged = false;
      function check( predicateValue, renderer ) {
        var inSupportedRenderers = _.contains( thisText._supportedRenderers, renderer );
        if ( predicateValue !== inSupportedRenderers ) {
          renderersChanged = true;
          if ( predicateValue ) {
            // add the renderer
            thisText._supportedRenderers.push( renderer );
          } else {
            // remove the renderer
            thisText._supportedRenderers.splice( _.indexOf( thisText._supportedRenderers, renderer ), 1 );
            if ( thisText.renderer === renderer ) {
              // our set renderer is incompatible. set to null to disable this. TODO: investigate rendering system to prevent overrides like this?
              thisText.renderer = null;
              
              // for now, error out
              throw new Error( 'The explicitly specified Text renderer: ' + renderer.name + ' is not supported by this operation (probably invalid stroke, fill, or boundsMethod)' );
            }
          }
        }
      }
      
      check( !this.boundsInaccurate && !this._isHTML, Renderer.Canvas );
      check( !this._isHTML, Renderer.SVG );
      check( !this.hasStroke() && this.isFillDOMCompatible(), Renderer.DOM );
      
      if ( this._supportedRenderers.length === 0 ) {
        throw new Error( 'No renderers are able to support this Text node (probably HTML text with a stroke or incompatible fill)' );
      }
      
      if ( renderersChanged ) {
        this.markLayerRefreshNeeded();
      }
    },
    
    invalidateText: function() {
      // investigate http://mudcu.be/journal/2011/01/html5-typographic-metrics/
      if ( this._boundsMethod === 'fast' || this._boundsMethod === 'fastCanvas' ) {
        this.invalidateSelf( this._isHTML ? this.approximateDOMBounds() : this.approximateSVGBounds() );
      } else {
        assert && assert( !this._isHTML, 'HTML text is not allowed with the accurate bounds method' );
        this.invalidateSelf( this.accurateCanvasBounds() );
      }
      
      // we may have changed renderers if parameters were changed!
      this.updateTextFlags();
    },
    
    // overrides from Strokable
    invalidateStroke: function() {
      // stroke can change both the bounds and renderer
      this.invalidateText();
    },
    
    // overrides from Fillable
    invalidateFill: function() {
      // fill type can change the renderer (gradient/fill not supported by DOM)
      this.invalidateText();
    },
    
    /*---------------------------------------------------------------------------*
    * Canvas support
    *----------------------------------------------------------------------------*/
    
    paintCanvas: function( wrapper ) {
      var context = wrapper.context;
      
      // extra parameters we need to set, but should avoid setting if we aren't drawing anything
      if ( this.hasFill() || this.hasStroke() ) {
        wrapper.setFont( this._font.getFont() );
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
    
    /*---------------------------------------------------------------------------*
    * WebGL support
    *----------------------------------------------------------------------------*/
    
    paintWebGL: function( state ) {
      throw new Error( 'Text.prototype.paintWebGL unimplemented' );
    },
    
    /*---------------------------------------------------------------------------*
    * SVG support
    *----------------------------------------------------------------------------*/
    
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
      
      // element.setAttribute( 'text-anchor', 'start' ); // not needed right now (default is inherit)
      element.setAttribute( 'dominant-baseline', 'alphabetic' ); // to match Canvas right now
      element.setAttribute( 'direction', this._direction );
      
      // set all of the font attributes, since we can't use the combined one
      element.setAttribute( 'font-family', this._font.getFamily() );
      element.setAttribute( 'font-size', this._font.getSize() );
      element.setAttribute( 'font-style', this._font.getStyle() );
      element.setAttribute( 'font-weight', this._font.getWeight() );
      element.setAttribute( 'font-stretch', this._font.getStretch() );
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
    * DOM support
    *----------------------------------------------------------------------------*/
    
    allowsMultipleDOMInstances: true,
    
    getDOMElement: function() {
      return document.createElement( 'div' );
    },
    
    updateDOMElement: function( div ) {
      var $div = $( div );
      div.style.font = this.getFont();
      div.style.color = this.getFill() ? this.getFill() : 'transparent'; // transparent will make us invisible if the fill is null
      $div.width( this.getSelfBounds().width );
      $div.height( this.getSelfBounds().height );
      $div.empty(); // remove all children, including previously-created text nodes
      div.appendChild( this.getDOMTextNode() );
      div.setAttribute( 'dir', this._direction );
    },
    
    updateCSSTransform: function( transform, element ) {
      // since the DOM origin of the text is at the upper-left, and our Scenery origin is at the lower-left, we need to
      // shift the text vertically, postmultiplied with the entire transform.
      var yOffset = this.getSelfBounds().minY;
      var matrix = transform.getMatrix().timesMatrix( Matrix3.translation( 0, yOffset ) );
      scenery.Util.applyCSSTransform( matrix, element );
    },
    
    // a DOM node (not a Scenery DOM node, but an actual DOM node) with the text
    getDOMTextNode: function() {
      if ( this._isHTML ) {
        var span = document.createElement( 'span' );
        span.innerHTML = this.text;
        return span;
      } else {
        return document.createTextNode( this.text );
      }
    },
    
    /*---------------------------------------------------------------------------*
    * Bounds
    *----------------------------------------------------------------------------*/
    
    accurateCanvasBounds: function() {
      var node = this;
      var svgBounds = this.approximateSVGBounds(); // this seems to be slower than expected, mostly due to Font getters

      //If svgBounds are zero, then return the zero bounds
      if (svgBounds.width===0 && svgBounds.height===0){
        return svgBounds;
      }
      return scenery.Util.canvasAccurateBounds( function( context ) {
        context.font = node.font;
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
      context.direction = this.direction;
      return context.measureText( this.text ).width;
    },
    
    approximateSVGBounds: function() {
      if ( !svgTextSizeContainer.parentNode ) {
        document.body.appendChild( svgTextSizeContainer );
      }
      this.updateSVGFragment( svgTextSizeElement );
      var rect = svgTextSizeElement.getBBox();
      return new Bounds2( rect.x, rect.y, rect.x + rect.width, rect.y + rect.height );
    },
    
    approximateDOMBounds: function() {
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
      span.appendChild( this.getDOMTextNode() );
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
      // console.log( 'rect: ' + rect.toString() );
      // console.log( 'divRect: ' + divRect.toString() );
      // console.log( 'span width from jQuery: ' + $( span ).width() );
      var result = new Bounds2( rect.left, rect.top - maxHeight, rect.right, rect.bottom - maxHeight ).shifted( -divRect.left, -divRect.top );
      // console.log( 'result: ' + result );
      document.body.removeChild( div );
      
      var width = rect.right - rect.left;
      return result.shiftedX( isRTL ? -width : 0 ); // should we even swap here?
    },
    
    /*---------------------------------------------------------------------------*
    * Self setters / getters
    *----------------------------------------------------------------------------*/
    
    setFont: function( font ) {
      if ( this.font !== font ) {
        this._font.removeFontListener( this.fontListener );
        this._font = font instanceof scenery.Font ? font : new scenery.Font( font );
        this._font.addFontListener( this.fontListener );
        this.invalidateText();
      }
      return this;
    },
    
    // NOTE: returns mutable copy for now, consider either immutable version, defensive copy, or note about invalidateText()
    getFont: function() {
      return this._font.getFont();
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
      return 'new scenery.Text( \'' + escapeHTML( this._text.replace( /'/g, '\\\'' ) ) + '\', {' + propLines + '} )';
    },
    
    getPropString: function( spaces, includeChildren ) {
      var result = Node.prototype.getPropString.call( this, spaces, includeChildren );
      result = this.appendFillablePropString( spaces, result );
      result = this.appendStrokablePropString( spaces, result );
      
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
        addProp( 'font', this.font.replace( /'/g, '\\\'' ) );
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
      // create a full copy of our font instance
      var newFont = new scenery.Font( this._font.getFont() );
      
      // use the ES5 setter. probably somewhat slow.
      newFont[ shortUncapitalized ] = value;
      
      // apply the new Font. this should call invalidateText() as normal
      this.setFont( newFont );
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
  
  Text.prototype._mutatorKeys = [ 'boundsMethod', 'text', 'font', 'fontWeight', 'fontFamily', 'fontStretch', 'fontStyle', 'fontSize', 'lineHeight',
                                  'direction' ].concat( Node.prototype._mutatorKeys );
  
  Text.prototype._supportedRenderers = [ Renderer.Canvas, Renderer.SVG, Renderer.DOM ];
  Text.prototype._supportedRenderersWithFastBounds = [ Renderer.SVG, Renderer.DOM ]; // renderers for fast (SVG/DOM) bounds, since canvas dirty regions would present issues
  
  // font-specific ES5 setters and getters are defined using addFontForwarding above
  Object.defineProperty( Text.prototype, 'font', { set: Text.prototype.setFont, get: Text.prototype.getFont } );
  Object.defineProperty( Text.prototype, 'text', { set: Text.prototype.setText, get: Text.prototype.getText } );
  Object.defineProperty( Text.prototype, 'direction', { set: Text.prototype.setDirection, get: Text.prototype.getDirection } );
  Object.defineProperty( Text.prototype, 'boundsMethod', { set: Text.prototype.setBoundsMethod, get: Text.prototype.getBoundsMethod } );
  
  // mix in support for fills and strokes
  fillable( Text );
  strokable( Text );

  return Text;
} );


