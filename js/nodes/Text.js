// Copyright 2002-2013, University of Colorado

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
  
  var inherit = require( 'PHET_CORE/inherit' );
  var escapeHTML = require( 'PHET_CORE/escapeHTML' );
  var Bounds2 = require( 'DOT/Bounds2' );
  var Matrix3 = require( 'DOT/Matrix3' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  var Node = require( 'SCENERY/nodes/Node' ); // inherits from Node
  require( 'SCENERY/layers/Renderer' );
  var Fillable = require( 'SCENERY/nodes/Fillable' );
  var Strokable = require( 'SCENERY/nodes/Strokable' );
  require( 'SCENERY/util/Font' );
  require( 'SCENERY/util/Util' ); // for canvasAccurateBounds and CSS transforms
  
  // set up the container and text for testing text bounds quickly (using approximateSVGBounds)
  var svgTextSizeContainer = document.createElementNS( scenery.svgns, 'svg' );
  svgTextSizeContainer.setAttribute( 'width', '2' );
  svgTextSizeContainer.setAttribute( 'height', '2' );
  svgTextSizeContainer.setAttribute( 'style', 'visibility: hidden; pointer-events: none; position: absolute; left: -65535; right: -65535;' ); // so we don't flash it in a visible way to the user
  // NOTE! copies createSVGElement
  var svgTextSizeElement = document.createElementNS( scenery.svgns, 'text' );
  svgTextSizeElement.appendChild( document.createTextNode( '' ) );
  svgTextSizeElement.setAttribute( 'dominant-baseline', 'alphabetic' ); // to match Canvas right now
  svgTextSizeElement.setAttribute( 'text-rendering', 'geometricPrecision' );
  svgTextSizeElement.setAttribute( 'lengthAdjust', 'spacingAndGlyphs' );
  svgTextSizeContainer.appendChild( svgTextSizeElement );
  
  // SVG bounds seems to be malfunctioning for Safari 5. Since we don't have a reproducible test machine for
  // fast iteration, we'll guess the user agent and use DOM bounds instead of SVG.
  // Hopefully the two contraints rule out any future Safari versions (fairly safe, but not impossible!)
  var useDOMAsFastBounds = window.navigator.userAgent.indexOf( 'like Gecko) Version/5' ) !== -1 &&
                           window.navigator.userAgent.indexOf( 'Safari/' ) !== -1;
  
  var hybridTextNode; // a node that is used to measure SVG text top/height for hybrid caching purposes
  var initializingHybridTextNode = false;
  
  scenery.Text = function Text( text, options ) {
    this._text         = '';                   // filled in with mutator
    this._font         = scenery.Font.DEFAULT; // default font, usually 10px sans-serif
    this._direction    = 'ltr';                // ltr, rtl, inherit -- consider inherit deprecated, due to how we compute text bounds in an off-screen canvas
    this._boundsMethod = 'hybrid';             // fast (SVG/DOM, no canvas rendering allowed), fastCanvas (SVG/DOM, canvas rendering allowed without dirty regions),
                                               // accurate (Canvas accurate recursive), or hybrid (cache SVG height, use canvas measureText for width)
    
    // whether the text is rendered as HTML or not. if defined (in a subtype constructor), use that value instead
    this._isHTML = this._isHTML === undefined ? false : this._isHTML;
    
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
    
    this.initializeFillable();
    this.initializeStrokable();
    
    Node.call( this, options );
    this.updateTextFlags(); // takes care of setting up supported renderers
  };
  var Text = scenery.Text;
  
  inherit( Node, Text, {
    domUpdateTransformOnRepaint: true, // since we have to integrate the baseline offset into the CSS transform, signal to DOMLayer
    
    setText: function( text ) {
      assert && assert( text !== null && text !== undefined, 'Text should be defined and non-null. Use the empty string if needed.' );
      
      // cast it to a string (for numbers, etc.)
      text = '' + text;
      
      if ( text !== this._text ) {
        this._text = text;
        this.invalidateText();
      }
      return this;
    },
    
    getText: function() {
      return this._text;
    },
    
    // Using the non-breaking space (&nbsp;) encoded as 0x00A0 in UTF-8
    getNonBreakingText: function() {
      return this._text.replace( ' ', '\xA0' );
    },
    
    setBoundsMethod: function( method ) {
      assert && assert( method === 'fast' || method === 'fastCanvas' || method === 'accurate' || method === 'hybrid', 'Unknown Text boundsMethod' );
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
    
    // allow more specific path types (Rectangle, Line) to override what restrictions we have
    getTextRendererBitmask: function() {
      var bitmask = 0;
      
      // canvas support (fast bounds may leak out of dirty rectangles)
      if ( this._boundsMethod !== 'fast' && !this._isHTML ) {
        bitmask |= scenery.bitmaskSupportsCanvas;
      }
      if( !this._isHTML ) {
        bitmask |= scenery.bitmaskSupportsSVG;
      }
      
      // fill and stroke will determine whether we have DOM text support
      bitmask |= scenery.bitmaskSupportsDOM;
      
      return bitmask;
    },
    
    invalidateSupportedRenderers: function() {
      this.setRendererBitmask( this.getFillRendererBitmask() & this.getStrokeRendererBitmask() & this.getTextRendererBitmask() );
    },
    
    updateTextFlags: function() {
      this.boundsInaccurate = this._boundsMethod !== 'accurate';
      this.invalidateSupportedRenderers();
    },
    
    invalidateText: function() {
      // investigate http://mudcu.be/journal/2011/01/html5-typographic-metrics/
      if ( this._isHTML || ( useDOMAsFastBounds && this._boundsMethod !== 'accurate' ) ) {
        this.invalidateSelf( this.approximateDOMBounds() );
      } else if ( this._boundsMethod === 'hybrid' ) {
        this.invalidateSelf( this.approximateHybridBounds() );
      } else if ( this._boundsMethod === 'fast' || this._boundsMethod === 'fastCanvas' ) {
        this.invalidateSelf( this.approximateSVGBounds() );
      } else {
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
      // NOTE! reference SVG element at top of file copies createSVGElement!
      var element = document.createElementNS( scenery.svgns, 'text' );
      element.appendChild( document.createTextNode( '' ) );
      element.setAttribute( 'dominant-baseline', 'alphabetic' ); // to match Canvas right now
      element.setAttribute( 'text-rendering', 'geometricPrecision' );
      element.setAttribute( 'lengthAdjust', 'spacingAndGlyphs' );
      element.setAttributeNS( 'http://www.w3.org/XML/1998/namespace', 'xml:space', 'preserve' );
      return element;
    },
    
    updateSVGFragment: function( element ) {
      // update the text-node's value
      element.lastChild.nodeValue = this.getNonBreakingText();
      
      element.setAttribute( 'style', this.getSVGFillStyle() + this.getSVGStrokeStyle() );
      element.setAttribute( 'direction', this._direction );
      
      // text length correction, tested with scenery/tests/text-quality-test.html to determine how to match Canvas/SVG rendering (and overall length)
      if ( isFinite( this._selfBounds.width ) ) {
        element.setAttribute( 'textLength', this._selfBounds.width );
      }
      
      // set all of the font attributes, since we can't use the combined one
      // TODO: optimize so we only set what is changed!!!
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
      var div = document.createElement( 'div' );
      
      // so they are absolutely positioned compared to the containing DOM layer (that is positioned).
      // otherwise, two adjacent HTMLText elements will 'flow' and be positioned incorrectly
      div.style.position = 'absolute';
      return div;
    },
    
    updateDOMElement: function( div ) {
      var $div = $( div );
      div.style.font = this.getFont();
      div.style.color = this.getCSSFill();
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
        span.innerHTML = this._text;
        return span;
      } else {
        return document.createTextNode( this.getNonBreakingText() );
      }
    },
    
    /*---------------------------------------------------------------------------*
    * Bounds
    *----------------------------------------------------------------------------*/
    
    accurateCanvasBounds: function() {
      var node = this;
      var svgBounds = this.approximateSVGBounds(); // this seems to be slower than expected, mostly due to Font getters

      //If svgBounds are zero, then return the zero bounds
      if ( !this._text || svgBounds.width === 0 ) {
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
      var context = scenery.scratchContext;
      context.font = this.font;
      context.direction = this.direction;
      return context.measureText( this.text ).width;
    },
    
    approximateSVGBounds: function() {
      if ( !svgTextSizeContainer.parentNode ) {
        if ( document.body ) {
          document.body.appendChild( svgTextSizeContainer );
        } else {
          // TODO: better way to handle the hybridTextNode being added inside the HEAD? Requiring a body for proper operation might be a problem.
          if ( initializingHybridTextNode ) {
            // if this is almost assuredly the hybridTextNode, return nothing for now. TODO: better way of handling this! it's a hack!
            return Bounds2.NOTHING;
          } else {
            throw new Error( 'No document.body and trying to get approximate SVG bounds of a Text node' );
          }
        }
      }
      this.updateSVGFragment( svgTextSizeElement );
      svgTextSizeElement.removeAttribute( 'textLength' ); // since we may set textLength, remove that so we can get accurate widths
      var rect = svgTextSizeElement.getBBox();
      return new Bounds2( rect.x, rect.y, rect.x + rect.width, rect.y + rect.height );
    },
    
    approximateHybridBounds: function() {
      if ( !hybridTextNode ) {
        return Bounds2.NOTHING; // we are the hybridTextNode, ignore us
      }
      
      if ( this._font._cachedSVGBounds === undefined ) {
        hybridTextNode.setFont( this._font );
        this._font._cachedSVGBounds = hybridTextNode.getBounds();
      }
      
      var canvasWidth = this.approximateCanvasWidth();
      var verticalBounds = this._font._cachedSVGBounds;
      
      // it seems that SVG bounds generally have x=0, so we hard code that here
      return new Bounds2( 0, verticalBounds.minY, canvasWidth, verticalBounds.maxY );
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
      // add 1 pixel to rect.right to prevent HTML text wrapping
      var result = new Bounds2( rect.left, rect.top - maxHeight, rect.right + 1, rect.bottom - maxHeight ).shifted( -divRect.left, -divRect.top );
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
        this._font = font instanceof scenery.Font ? font : new scenery.Font( font );
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
      var ob = {};
      ob[shortUncapitalized] = value;
      var newFont = this._font.copy( ob );
      
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
  
  // font-specific ES5 setters and getters are defined using addFontForwarding above
  Object.defineProperty( Text.prototype, 'font', { set: Text.prototype.setFont, get: Text.prototype.getFont } );
  Object.defineProperty( Text.prototype, 'text', { set: Text.prototype.setText, get: Text.prototype.getText } );
  Object.defineProperty( Text.prototype, 'direction', { set: Text.prototype.setDirection, get: Text.prototype.getDirection } );
  Object.defineProperty( Text.prototype, 'boundsMethod', { set: Text.prototype.setBoundsMethod, get: Text.prototype.getBoundsMethod } );
  
  // mix in support for fills and strokes
  /* jshint -W064 */
  Fillable( Text );
  Strokable( Text );
  
  initializingHybridTextNode = true;
  hybridTextNode = new Text( 'm', { boundsMethod: 'fast' } );
  initializingHybridTextNode = false;

  return Text;
} );


