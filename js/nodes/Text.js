// Copyright 2013-2015, University of Colorado Boulder

/**
 * Text
 *
 * TODO: newlines (multiline)
 * TODO: don't get bounds until the Text node is fully mutated?
 *
 * Useful specs:
 * http://www.w3.org/TR/css3-text/
 * http://www.w3.org/TR/css3-fonts/
 * http://www.w3.org/TR/SVG/text.html
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var escapeHTML = require( 'PHET_CORE/escapeHTML' );
  var platform = require( 'PHET_CORE/platform' );
  var Bounds2 = require( 'DOT/Bounds2' );
  var Matrix3 = require( 'DOT/Matrix3' );

  var scenery = require( 'SCENERY/scenery' );

  var Node = require( 'SCENERY/nodes/Node' ); // inherits from Node
  var Renderer = require( 'SCENERY/display/Renderer' );
  var Paintable = require( 'SCENERY/nodes/Paintable' );
  require( 'SCENERY/util/Font' );
  require( 'SCENERY/util/Util' ); // for canvasAccurateBounds and CSS transforms
  require( 'SCENERY/util/CanvasContextWrapper' );
  var DOMSelfDrawable = require( 'SCENERY/display/DOMSelfDrawable' );
  var SVGSelfDrawable = require( 'SCENERY/display/SVGSelfDrawable' );
  var CanvasSelfDrawable = require( 'SCENERY/display/CanvasSelfDrawable' );
  var SelfDrawable = require( 'SCENERY/display/SelfDrawable' );
  var WebGLSelfDrawable = require( 'SCENERY/display/WebGLSelfDrawable' );

  // TODO: change this based on memory and performance characteristics of the platform
  var keepDOMTextElements = true; // whether we should pool DOM elements for the DOM rendering states, or whether we should free them when possible for memory
  var keepSVGTextElements = true; // whether we should pool SVG elements for the SVG rendering states, or whether we should free them when possible for memory

  // scratch matrix used in DOM rendering
  var scratchMatrix = Matrix3.dirtyFromPool();

  var textSizeContainerId = 'sceneryTextSizeContainer';
  var textSizeElementId = 'sceneryTextSizeElement';
  var svgTextSizeContainer = document.getElementById( textSizeContainerId );
  var svgTextSizeElement = document.getElementById( textSizeElementId );

  // SVG bounds seems to be malfunctioning for Safari 5. Since we don't have a reproducible test machine for
  // fast iteration, we'll guess the user agent and use DOM bounds instead of SVG.
  // Hopefully the two contraints rule out any future Safari versions (fairly safe, but not impossible!)
  var useDOMAsFastBounds = window.navigator.userAgent.indexOf( 'like Gecko) Version/5' ) !== -1 &&
                           window.navigator.userAgent.indexOf( 'Safari/' ) !== -1;

  var hybridTextNode; // a node that is used to measure SVG text top/height for hybrid caching purposes
  var initializingHybridTextNode = false;

  // Some browsers (IE/Edge) can't handle our UTF-8 embedding marks AND SVG textLength/spacingAndGlyphs. We disable
  // using these features, because they aren't necessary on these browsers.
  // See https://github.com/phetsims/scenery/issues/455 for more information.
  var useSVGTextLengthAdjustments = !platform.ie && !platform.edge;

  // Maps CSS {string} => {Bounds2}, so that we can cache the vertical font sizes outside of the Font objects themselves.
  var hybridFontVerticalCache = {};

  function Text( text, options ) {
    this._text = '';                   // filled in with mutator
    this._font = scenery.Font.DEFAULT; // default font, usually 10px sans-serif
    this._direction = 'ltr';           // ltr, rtl, inherit -- consider inherit deprecated, due to how we compute text bounds in an off-screen canvas
    this._boundsMethod = 'hybrid';     // fast (SVG/DOM, no canvas rendering allowed), fastCanvas (SVG/DOM, canvas rendering allowed without dirty regions),
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

    this.initializePaintable();

    Node.call( this, options );
    this.updateTextFlags(); // takes care of setting up supported renderers
  }

  scenery.register( 'Text', Text );

  inherit( Node, Text, {
    domUpdateTransformOnRepaint: true, // since we have to integrate the baseline offset into the CSS transform, signal to DOMLayer

    setText: function( text ) {
      assert && assert( text !== null && text !== undefined, 'Text should be defined and non-null. Use the empty string if needed.' );

      // cast it to a string (for numbers, etc., and do it before the change guard so we don't accidentally trigger on non-changed text)
      text = '' + text;

      if ( text !== this._text ) {
        this._text = text;

        var stateLen = this._drawables.length;
        for ( var i = 0; i < stateLen; i++ ) {
          this._drawables[ i ].markDirtyText();
        }

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

        var stateLen = this._drawables.length;
        for ( var i = 0; i < stateLen; i++ ) {
          this._drawables[ i ].markDirtyBounds();
        }

        this.invalidateText();

        this.trigger0( 'boundsMethod' );

        this.trigger0( 'selfBoundsValid' ); // whether our self bounds are valid may have changed
      }
      return this;
    },

    getBoundsMethod: function() {
      return this._boundsMethod;
    },

    getTextRendererBitmask: function() {
      var bitmask = 0;

      // canvas support (fast bounds may leak out of dirty rectangles)
      if ( this._boundsMethod !== 'fast' && !this._isHTML ) {
        bitmask |= Renderer.bitmaskCanvas;
      }
      if ( !this._isHTML ) {
        bitmask |= Renderer.bitmaskSVG;
      }

      // fill and stroke will determine whether we have DOM text support
      bitmask |= Renderer.bitmaskDOM;

      return bitmask;
    },

    invalidateSupportedRenderers: function() {
      this.setRendererBitmask( this.getFillRendererBitmask() & this.getStrokeRendererBitmask() & this.getTextRendererBitmask() );
    },

    updateTextFlags: function() {
      this.invalidateSupportedRenderers();
    },

    invalidateText: function() {
      this.invalidateSelf();

      // TODO: consider replacing this with a general dirty flag notification, and have DOM update bounds every frame?
      var stateLen = this._drawables.length;
      for ( var i = 0; i < stateLen; i++ ) {
        this._drawables[ i ].markDirtyBounds();
      }

      // we may have changed renderers if parameters were changed!
      this.updateTextFlags();
    },

    /**
     * @override
     *
     * @returns {boolean}
     */
    updateSelfBounds: function() {
      // TODO: don't create another Bounds2 object just for this!
      var selfBounds;

      // investigate http://mudcu.be/journal/2011/01/html5-typographic-metrics/
      if ( this._isHTML || ( useDOMAsFastBounds && this._boundsMethod !== 'accurate' ) ) {
        selfBounds = this.approximateDOMBounds();
      }
      else if ( this._boundsMethod === 'hybrid' ) {
        selfBounds = this.approximateHybridBounds();
      }
      else if ( this._boundsMethod === 'fast' || this._boundsMethod === 'fastCanvas' ) {
        selfBounds = this.approximateSVGBounds();
      }
      else {
        selfBounds = this.accurateCanvasBounds();
      }

      // for now, just add extra on, ignoring the possibility of mitered joints passing beyond
      if ( this.hasStroke() ) {
        selfBounds.dilate( this.getLineWidth() / 2 );
      }

      var changed = !selfBounds.equals( this._selfBounds );
      if ( changed ) {
        this._selfBounds.set( selfBounds );
      }
      return changed;
    },

    // @override from Paintable
    invalidateStroke: function() {
      // stroke can change both the bounds and renderer
      this.invalidateText();
    },

    // @override from Paintable
    invalidateFill: function() {
      // fill type can change the renderer (gradient/fill not supported by DOM)
      this.invalidateText();
    },

    canvasPaintSelf: function( wrapper ) {
      Text.TextCanvasDrawable.prototype.paintCanvas( wrapper, this );
    },

    createDOMDrawable: function( renderer, instance ) {
      return Text.TextDOMDrawable.createFromPool( renderer, instance );
    },

    createSVGDrawable: function( renderer, instance ) {
      return Text.TextSVGDrawable.createFromPool( renderer, instance );
    },

    createCanvasDrawable: function( renderer, instance ) {
      return Text.TextCanvasDrawable.createFromPool( renderer, instance );
    },

    createWebGLDrawable: function( renderer, instance ) {
      return Text.TextWebGLDrawable.createFromPool( renderer, instance );
    },

    // a DOM node (not a Scenery DOM node, but an actual DOM node) with the text
    getDOMTextNode: function() {
      if ( this._isHTML ) {
        var span = document.createElement( 'span' );
        span.innerHTML = this._text;
        return span;
      }
      else {
        return document.createTextNode( this.getNonBreakingText() );
      }
    },

    /*---------------------------------------------------------------------------*
     * Bounds
     *----------------------------------------------------------------------------*/

    getVerticalBounds: function() {
      if ( !hybridTextNode ) {
        return Bounds2.NOTHING; // we are the hybridTextNode, ignore us
      }

      var css = this._font.toCSS();
      var verticalBounds = hybridFontVerticalCache[ css ];
      if ( !verticalBounds ) {
        hybridTextNode.setFont( this._font );
        verticalBounds = hybridFontVerticalCache[ css ] = hybridTextNode.getBounds().copy();
      }

      return verticalBounds;
    },

    accurateCanvasBounds: function() {
      var node = this;
      var svgBounds = this.approximateSVGBounds(); // this seems to be slower than expected, mostly due to Font getters

      //If svgBounds are zero, then return the zero bounds
      if ( !this._text || svgBounds.width === 0 ) {
        return svgBounds;
      }

      // NOTE: should return new instance, so that it can be mutated later
      return scenery.Util.canvasAccurateBounds( function( context ) {
        context.font = node.font;
        context.direction = node.direction;
        context.fillText( node.text, 0, 0 );
        if ( node.hasStroke() ) {
          var fakeWrapper = new scenery.CanvasContextWrapper( null, context );
          node.beforeCanvasStroke( fakeWrapper );
          context.strokeText( node.text, 0, 0 );
          node.afterCanvasStroke( fakeWrapper );
        }
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

    // NOTE: should return new instance, so that it can be mutated later
    approximateSVGBounds: function() {
      if ( !svgTextSizeContainer.parentNode ) {
        if ( document.body ) {
          document.body.appendChild( svgTextSizeContainer );
        }
        else {
          // TODO: better way to handle the hybridTextNode being added inside the HEAD? Requiring a body for proper operation might be a problem.
          if ( initializingHybridTextNode ) {
            // if this is almost assuredly the hybridTextNode, return nothing for now. TODO: better way of handling this! it's a hack!
            return Bounds2.NOTHING;
          }
          else {
            throw new Error( 'No document.body and trying to get approximate SVG bounds of a Text node' );
          }
        }
      }
      updateSVGTextToMeasure( svgTextSizeElement, this );
      var rect = svgTextSizeElement.getBBox();
      return new Bounds2( rect.x, rect.y, rect.x + rect.width, rect.y + rect.height );
    },

    // NOTE: should return new instance, so that it can be mutated later
    approximateHybridBounds: function() {
      var verticalBounds = this.getVerticalBounds();

      var canvasWidth = this.approximateCanvasWidth();

      // it seems that SVG bounds generally have x=0, so we hard code that here
      return new Bounds2( 0, verticalBounds.minY, canvasWidth, verticalBounds.maxY );
    },

    // NOTE: should return new instance, so that it can be mutated later
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

    approximateImprovedDOMBounds: function() {
      // TODO: reuse this div?
      var div = document.createElement( 'div' );
      div.style.display = 'inline-block';
      div.style.font = this.getFont();
      div.style.color = 'transparent';
      div.style.padding = '0 !important';
      div.style.margin = '0 !important';
      div.style.position = 'absolute';
      div.style.left = '0';
      div.style.top = '0';
      div.setAttribute( 'direction', this._direction );
      div.appendChild( this.getDOMTextNode() );

      document.body.appendChild( div );
      var bounds = new Bounds2( div.offsetLeft, div.offsetTop, div.offsetLeft + div.offsetWidth + 1, div.offsetTop + div.offsetHeight + 1 );
      document.body.removeChild( div );

      // Compensate for the baseline alignment
      var verticalBounds = this.getVerticalBounds();
      return bounds.shiftedY( verticalBounds.minY );
    },

    // @override from Node
    getSafeSelfBounds: function() {
      var expansionFactor = 1; // we use a new bounding box with a new size of size * ( 1 + 2 * expansionFactor )

      var selfBounds = this.getSelfBounds();

      // NOTE: we'll keep this as an estimate for the bounds including stroke miters
      return selfBounds.dilatedXY( expansionFactor * selfBounds.width, expansionFactor * selfBounds.height );
    },

    /*---------------------------------------------------------------------------*
     * Self setters / getters
     *----------------------------------------------------------------------------*/

    setFont: function( font ) {
      if ( this.font !== font ) {
        this._font = font instanceof scenery.Font ? font : new scenery.Font( font );

        var stateLen = this._drawables.length;
        for ( var i = 0; i < stateLen; i++ ) {
          this._drawables[ i ].markDirtyFont();
        }

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

      var stateLen = this._drawables.length;
      for ( var i = 0; i < stateLen; i++ ) {
        this._drawables[ i ].markDirtyDirection();
      }

      this.invalidateText();
      return this;
    },

    getDirection: function() {
      return this._direction;
    },

    isPainted: function() {
      return true;
    },

    // @override
    areSelfBoundsValid: function() {
      return this._boundsMethod === 'accurate';
    },

    getDebugHTMLExtras: function() {
      return ' "' + escapeHTML( this.getNonBreakingText() ) + '"' + ( this._isHTML ? ' (html)' : '' );
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
        }
        else {
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

    Text.prototype[ getterName ] = function() {
      // use the ES5 getter to retrieve the property. probably somewhat slow.
      return this._font[ shortUncapitalized ];
    };

    Text.prototype[ setterName ] = function( value ) {
      // create a full copy of our font instance
      var ob = {};
      ob[ shortUncapitalized ] = value;
      var newFont = this._font.copy( ob );

      // apply the new Font. this should call invalidateText() as normal
      // TODO: do more specific font dirty flags in the future, for how SVG does things
      this.setFont( newFont );
      return this;
    };

    Object.defineProperty( Text.prototype, propertyName, {
      set: Text.prototype[ setterName ],
      get: Text.prototype[ getterName ]
    } );
  }

  addFontForwarding( 'fontWeight', 'FontWeight', 'weight' );
  addFontForwarding( 'fontFamily', 'FontFamily', 'family' );
  addFontForwarding( 'fontStretch', 'FontStretch', 'stretch' );
  addFontForwarding( 'fontStyle', 'FontStyle', 'style' );
  addFontForwarding( 'fontSize', 'FontSize', 'size' );
  addFontForwarding( 'lineHeight', 'LineHeight', 'lineHeight' );

  Text.prototype._mutatorKeys = [
    'boundsMethod', 'text', 'font', 'fontWeight', 'fontFamily', 'fontStretch', 'fontStyle', 'fontSize', 'lineHeight', 'direction'
  ].concat( Node.prototype._mutatorKeys );

  // font-specific ES5 setters and getters are defined using addFontForwarding above
  Object.defineProperty( Text.prototype, 'font', { set: Text.prototype.setFont, get: Text.prototype.getFont } );
  Object.defineProperty( Text.prototype, 'text', { set: Text.prototype.setText, get: Text.prototype.getText } );
  Object.defineProperty( Text.prototype, 'direction', {
    set: Text.prototype.setDirection,
    get: Text.prototype.getDirection
  } );
  Object.defineProperty( Text.prototype, 'boundsMethod', {
    set: Text.prototype.setBoundsMethod,
    get: Text.prototype.getBoundsMethod
  } );

  // mix in support for fills and strokes
  Paintable.mixin( Text );

  /*---------------------------------------------------------------------------*
   * Rendering State mixin (DOM/SVG)
   *----------------------------------------------------------------------------*/

  Text.TextStatefulDrawable = {
    mixin: function( drawableType ) {
      var proto = drawableType.prototype;

      // initializes, and resets (so we can support pooled states)
      proto.initializeState = function( renderer, instance ) {
        this.paintDirty = true; // flag that is marked if ANY "paint" dirty flag is set (basically everything except for transforms, so we can accelerated the transform-only case)
        this.dirtyText = true;
        this.dirtyFont = true;
        this.dirtyBounds = true;
        this.dirtyDirection = true;

        // adds fill/stroke-specific flags and state
        this.initializePaintableState( renderer, instance );

        return this; // allow for chaining
      };

      proto.disposeState = function() {
        this.disposePaintableState();
      };

      // catch-all dirty, if anything that isn't a transform is marked as dirty
      proto.markPaintDirty = function() {
        this.paintDirty = true;
        this.markDirty();
      };

      proto.markDirtyText = function() {
        this.dirtyText = true;
        this.markPaintDirty();
      };
      proto.markDirtyFont = function() {
        this.dirtyFont = true;
        this.markPaintDirty();
      };
      proto.markDirtyBounds = function() {
        this.dirtyBounds = true;
        this.markPaintDirty();
      };
      proto.markDirtyDirection = function() {
        this.dirtyDirection = true;
        this.markPaintDirty();
      };

      proto.setToCleanState = function() {
        this.paintDirty = false;
        this.dirtyText = false;
        this.dirtyFont = false;
        this.dirtyBounds = false;
        this.dirtyDirection = false;
      };

      Paintable.PaintableStatefulDrawable.mixin( drawableType );
    }
  };

  /*---------------------------------------------------------------------------*
   * DOM rendering
   *----------------------------------------------------------------------------*/

  var TextDOMDrawable = Text.TextDOMDrawable = inherit( DOMSelfDrawable, function TextDOMDrawable( renderer, instance ) {
    this.initialize( renderer, instance );
  }, {
    initialize: function( renderer, instance ) {
      this.initializeDOMSelfDrawable( renderer, instance );
      this.initializeState( renderer, instance );

      // only create elements if we don't already have them (we pool visual states always, and depending on the platform may also pool the actual elements to minimize
      // allocation and performance costs)
      if ( !this.domElement ) {
        this.domElement = document.createElement( 'div' );
        this.domElement.style.display = 'block';
        this.domElement.style.position = 'absolute';
        this.domElement.style.pointerEvents = 'none';
        this.domElement.style.left = '0';
        this.domElement.style.top = '0';
      }

      scenery.Util.prepareForTransform( this.domElement, this.forceAcceleration );

      return this; // allow for chaining
    },

    updateDOM: function() {
      var node = this.node;

      var div = this.domElement;

      if ( this.paintDirty ) {
        if ( this.dirtyFont ) {
          div.style.font = node.getFont();
        }
        if ( this.dirtyStroke ) {
          div.style.color = node.getCSSFill();
        }
        if ( this.dirtyBounds ) { // TODO: this condition is set on invalidateText, so it's almost always true?
          div.style.width = node.getSelfBounds().width + 'px';
          div.style.height = node.getSelfBounds().height + 'px';
          // TODO: do we require the jQuery versions here, or are they vestigial?
          // $div.width( node.getSelfBounds().width );
          // $div.height( node.getSelfBounds().height );
        }
        if ( this.dirtyText ) {
          // TODO: actually do this in a better way
          div.innerHTML = node.getNonBreakingText();
        }
        if ( this.dirtyDirection ) {
          div.setAttribute( 'dir', node._direction );
        }
      }

      if ( this.transformDirty || this.dirtyText || this.dirtyFont || this.dirtyBounds ) {
        // shift the text vertically, postmultiplied with the entire transform.
        var yOffset = node.getSelfBounds().minY;
        scratchMatrix.set( this.getTransformMatrix() );
        var translation = Matrix3.translation( 0, yOffset );
        scratchMatrix.multiplyMatrix( translation );
        translation.freeToPool();
        scenery.Util.applyPreparedTransform( scratchMatrix, div, this.forceAcceleration );
      }

      // clear all of the dirty flags
      this.setToClean();
    },

    setToClean: function() {
      this.setToCleanState();

      this.cleanPaintableState();

      this.transformDirty = false;
    },

    dispose: function() {
      this.disposeState();

      if ( !keepDOMTextElements ) {
        // clear the references
        this.domElement = null;
      }

      DOMSelfDrawable.prototype.dispose.call( this );
    }
  } );
  Text.TextStatefulDrawable.mixin( TextDOMDrawable );
  SelfDrawable.Poolable.mixin( TextDOMDrawable );

  /*---------------------------------------------------------------------------*
   * SVG rendering
   *----------------------------------------------------------------------------*/

  Text.TextSVGDrawable = function TextSVGDrawable( renderer, instance ) {
    this.initialize( renderer, instance );
  };
  inherit( SVGSelfDrawable, Text.TextSVGDrawable, {
    initialize: function( renderer, instance ) {
      this.initializeSVGSelfDrawable( renderer, instance, true, keepSVGTextElements ); // usesPaint: true

      if ( !this.svgElement ) {
        // NOTE! reference SVG element at top of file copies createSVGElement!
        var text = this.svgElement = document.createElementNS( scenery.svgns, 'text' );
        text.appendChild( document.createTextNode( '' ) );

        // TODO: flag adjustment for SVG qualities
        text.setAttribute( 'dominant-baseline', 'alphabetic' ); // to match Canvas right now
        text.setAttribute( 'text-rendering', 'geometricPrecision' );
        if ( useSVGTextLengthAdjustments ) {
          text.setAttribute( 'lengthAdjust', 'spacingAndGlyphs' );
        }
        text.setAttributeNS( 'http://www.w3.org/XML/1998/namespace', 'xml:space', 'preserve' );
      }

      return this;
    },

    updateSVGSelf: function() {
      var text = this.svgElement;

      if ( this.dirtyDirection ) {
        text.setAttribute( 'direction', this.node._direction );
      }

      // set all of the font attributes, since we can't use the combined one
      if ( this.dirtyFont ) {
        text.setAttribute( 'font-family', this.node._font.getFamily() );
        text.setAttribute( 'font-size', this.node._font.getSize() );
        text.setAttribute( 'font-style', this.node._font.getStyle() );
        text.setAttribute( 'font-weight', this.node._font.getWeight() );
        text.setAttribute( 'font-stretch', this.node._font.getStretch() );
      }

      // update the text-node's value
      if ( this.dirtyText ) {
        text.lastChild.nodeValue = this.node.getNonBreakingText();
      }

      // text length correction, tested with scenery/tests/text-quality-test.html to determine how to match Canvas/SVG rendering (and overall length)
      if ( this.dirtyBounds && isFinite( this.node.selfBounds.width ) && useSVGTextLengthAdjustments ) {
        text.setAttribute( 'textLength', this.node.selfBounds.width );
      }

      this.updateFillStrokeStyle( text );
    }
  } );
  Text.TextStatefulDrawable.mixin( Text.TextSVGDrawable );
  SelfDrawable.Poolable.mixin( Text.TextSVGDrawable );

  /*---------------------------------------------------------------------------*
   * Canvas rendering
   *----------------------------------------------------------------------------*/

  Text.TextCanvasDrawable = function TextCanvasDrawable( renderer, instance ) {
    this.initialize( renderer, instance );
  };
  inherit( CanvasSelfDrawable, Text.TextCanvasDrawable, {
    initialize: function( renderer, instance ) {
      this.initializeCanvasSelfDrawable( renderer, instance );
      this.initializePaintableStateless( renderer, instance );
      return this;
    },

    paintCanvas: function( wrapper, node ) {
      var context = wrapper.context;

      // extra parameters we need to set, but should avoid setting if we aren't drawing anything
      if ( node.hasFill() || node.hasStroke() ) {
        wrapper.setFont( node._font.getFont() );
        wrapper.setDirection( node._direction );
      }

      if ( node.hasFill() ) {
        node.beforeCanvasFill( wrapper ); // defined in Paintable
        context.fillText( node._text, 0, 0 );
        node.afterCanvasFill( wrapper ); // defined in Paintable
      }
      if ( node.hasStroke() ) {
        node.beforeCanvasStroke( wrapper ); // defined in Paintable
        context.strokeText( node._text, 0, 0 );
        node.afterCanvasStroke( wrapper ); // defined in Paintable
      }
    },

    // stateless dirty functions
    markDirtyText: function() { this.markPaintDirty(); },
    markDirtyFont: function() { this.markPaintDirty(); },
    markDirtyBounds: function() { this.markPaintDirty(); },
    markDirtyDirection: function() { this.markPaintDirty(); },

    dispose: function() {
      CanvasSelfDrawable.prototype.dispose.call( this );
      this.disposePaintableStateless();
    }
  } );
  Paintable.PaintableStatelessDrawable.mixin( Text.TextCanvasDrawable );
  SelfDrawable.Poolable.mixin( Text.TextCanvasDrawable );

  /*---------------------------------------------------------------------------*
   * WebGL rendering
   *----------------------------------------------------------------------------*/

  Text.TextWebGLDrawable = inherit( WebGLSelfDrawable, function TextWebGLDrawable( renderer, instance ) {
    this.initialize( renderer, instance );
  }, {
    // called either from the constructor or from pooling
    initialize: function( renderer, instance ) {
      this.initializeWebGLSelfDrawable( renderer, instance );
    },

    onAddToBlock: function( webglBlock ) {
      var self = this;
      this.node.toImageNodeAsynchronous( function( imageNodeContainer ) {
        //toImageNode returns a containerNode with its first child set as ImageNode
        var imageNode = imageNodeContainer.children[ 0 ];
        self.textHandle = webglBlock.webGLRenderer.textureRenderer.createFromImageNode( imageNode, 0.4 );

        // TODO: Don't call this each time a new item is added.
        webglBlock.webGLRenderer.textureRenderer.bindVertexBuffer();
        webglBlock.webGLRenderer.textureRenderer.bindDirtyTextures();
        self.updateText();
      } );

      //TODO: Update the state in the buffer arrays
    },

    onRemoveFromBlock: function( webglBlock ) {

    },

    //Nothing necessary since everything currently handled in the uModelViewMatrix below
    //However, we may switch to dynamic draw, and handle the matrix change only where necessary in the future?
    updateText: function() {
      if ( this.textHandle ) {
        this.textHandle.update();
      }
    },

    render: function( shaderProgram ) {
      // This is handled by the ColorTriangleRenderer
    },

    dispose: function() {
      this.disposeWebGLBuffers();
      // super
      WebGLSelfDrawable.prototype.dispose.call( this );
    },

    disposeWebGLBuffers: function() {
      this.webglBlock.webGLRenderer.colorTriangleRenderer.colorTriangleBufferData.dispose( this.rectangleHandle );
    },

    markDirtyText: function() {
      this.markDirty();
    },

    markDirtyBounds: function() {
      this.markDirty();
    },

    // general flag set on the state, which we forward directly to the drawable's paint flag
    markPaintDirty: function() {
      this.markDirty();
    },

    //TODO: Make sure all of the dirty flags make sense here.  Should we be using fillDirty, paintDirty, dirty, etc?
    update: function() {
      //if ( this.dirty ) {
      this.updateText();
      this.dirty = false;
      //}
    }
  } );

  // include stubs (stateless) for marking dirty stroke and fill (if necessary). we only want one dirty flag, not multiple ones, for WebGL (for now)
  Paintable.PaintableStatefulDrawable.mixin( Text.TextWebGLDrawable );

  // set up pooling
  SelfDrawable.Poolable.mixin( Text.TextWebGLDrawable );

  /*---------------------------------------------------------------------------*
   * Hybrid text setup (for bounds testing)
   *----------------------------------------------------------------------------*/

  function createSVGTextToMeasure() {
    var text = document.createElementNS( scenery.svgns, 'text' );
    text.appendChild( document.createTextNode( '' ) );

    // TODO: flag adjustment for SVG qualities
    text.setAttribute( 'dominant-baseline', 'alphabetic' ); // to match Canvas right now
    text.setAttribute( 'text-rendering', 'geometricPrecision' );
    text.setAttributeNS( 'http://www.w3.org/XML/1998/namespace', 'xml:space', 'preserve' );
    return text;
  }

  function updateSVGTextToMeasure( textElement, textNode ) {
    textElement.setAttribute( 'direction', textNode._direction );
    textElement.setAttribute( 'font-family', textNode._font.getFamily() );
    textElement.setAttribute( 'font-size', textNode._font.getSize() );
    textElement.setAttribute( 'font-style', textNode._font.getStyle() );
    textElement.setAttribute( 'font-weight', textNode._font.getWeight() );
    textElement.setAttribute( 'font-stretch', textNode._font.getStretch() );
    textElement.lastChild.nodeValue = textNode.getNonBreakingText();
  }

  if ( !svgTextSizeContainer ) {
    // set up the container and text for testing text bounds quickly (using approximateSVGBounds)
    svgTextSizeContainer = document.createElementNS( scenery.svgns, 'svg' );
    svgTextSizeContainer.setAttribute( 'width', '2' );
    svgTextSizeContainer.setAttribute( 'height', '2' );
    svgTextSizeContainer.setAttribute( 'id', textSizeContainerId );
    svgTextSizeContainer.setAttribute( 'style', 'visibility: hidden; pointer-events: none; position: absolute; left: -65535px; right: -65535px;' ); // so we don't flash it in a visible way to the user
  }
  // NOTE! copies createSVGElement
  if ( !svgTextSizeElement ) {
    svgTextSizeElement = createSVGTextToMeasure();
    svgTextSizeElement.setAttribute( 'id', textSizeElementId );
    svgTextSizeContainer.appendChild( svgTextSizeElement );
  }

  initializingHybridTextNode = true;
  hybridTextNode = new Text( 'm', { boundsMethod: 'fast' } );
  initializingHybridTextNode = false;

  return Text;
} );


