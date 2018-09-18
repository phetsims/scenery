// Copyright 2016, University of Colorado Boulder

/**
 * Different methods of detection of text bounds.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var Bounds2 = require( 'DOT/Bounds2' );
  var CanvasContextWrapper = require( 'SCENERY/util/CanvasContextWrapper' );
  var Font = require( 'SCENERY/util/Font' );
  var scenery = require( 'SCENERY/scenery' );
  var Util = require( 'SCENERY/util/Util' );

  // @private {string} - ID for a container for our SVG test element (determined to find the size of text elements with SVG)
  var TEXT_SIZE_CONTAINER_ID = 'sceneryTextSizeContainer';

  // @private {string} - ID for our SVG test element (determined to find the size of text elements with SVG)
  var TEXT_SIZE_ELEMENT_ID = 'sceneryTextSizeElement';

  // @private {SVGElement} - Container for our SVG test element (determined to find the size of text elements with SVG)
  var svgTextSizeContainer;

  // @private {SVGElement} - Test SVG element (determined to find the size of text elements with SVG)
  var svgTextSizeElement;

  // Maps CSS {string} => {Bounds2}, so that we can cache the vertical font sizes outside of the Font objects themselves.
  var hybridFontVerticalCache = {};

  var TextBounds = {
    /**
     * Returns a new Bounds2 that is the approximate bounds of a Text node displayed with the specified font and renderedText.
     * @public
     *
     * This method uses an SVG Text element, sets its text, and then determines its size to estimate the size of rendered text.
     *
     * NOTE: Calling code relies on the new Bounds2 instance, as they mutate it.
     *
     * @param {Font} font - The font of the text
     * @param {string} renderedText - Text to display (with any special characters replaced)
     * @returns {Bounds2}
     */
    approximateSVGBounds: function( font, renderedText ) {
      assert && assert( font instanceof Font, 'Font required' );
      assert && assert( typeof renderedText === 'string', 'renderedText required' );

      if ( !svgTextSizeContainer.parentNode ) {
        if ( document.body ) {
          document.body.appendChild( svgTextSizeContainer );
        }
        else {
          throw new Error( 'No document.body and trying to get approximate SVG bounds of a Text node' );
        }
      }
      TextBounds.setSVGTextAttributes( svgTextSizeElement, font, renderedText );
      var rect = svgTextSizeElement.getBBox();
      return new Bounds2( rect.x, rect.y, rect.x + rect.width, rect.y + rect.height );
    },

    /**
     * Returns a new Bounds2 that is the approximate bounds of the specified Text node.
     * @public
     *
     * This method repeatedly renders the text into a Canvas and checks for what pixels are filled. Iteratively doing this for each bound
     * (top/left/bottom/right) until a tolerance results in very accurate bounds of what is displayed.
     *
     * NOTE: Calling code relies on the new Bounds2 instance, as they mutate it.
     *
     * @param {Text} text - The Text node
     * @returns {Bounds2}
     */
    accurateCanvasBounds: function( text ) {
      var svgBounds = TextBounds.approximateSVGBounds( text._font, text.renderedText ); // this seems to be slower than expected, mostly due to Font getters

      //If svgBounds are zero, then return the zero bounds
      if ( !text.renderedText.length || svgBounds.width === 0 ) {
        return svgBounds;
      }

      // NOTE: should return new instance, so that it can be mutated later
      var accurateBounds = Util.canvasAccurateBounds( function( context ) {
        context.font = text._font.toCSS();
        context.direction = 'ltr';
        context.fillText( text.renderedText, 0, 0 );
        if ( text.hasPaintableStroke() ) {
          var fakeWrapper = new CanvasContextWrapper( null, context );
          text.beforeCanvasStroke( fakeWrapper );
          context.strokeText( text.renderedText, 0, 0 );
          text.afterCanvasStroke( fakeWrapper );
        }
      }, {
        precision: 0.5,
        resolution: 128,
        initialScale: 32 / Math.max( Math.abs( svgBounds.minX ), Math.abs( svgBounds.minY ), Math.abs( svgBounds.maxX ), Math.abs( svgBounds.maxY ) )
      } );
      // Try falling back to SVG bounds if our accurate bounds are not finite
      return accurateBounds.isFinite() ? accurateBounds : svgBounds;
    },

    /**
     * Returns a possibly-cached (treat as immutable) Bounds2 for use mainly for vertical parameters, given a specific Font.
     * @public
     *
     * Uses SVG bounds determination for this value.
     *
     * @param {Font} font - The font of the text
     * @returns {Bounds2}
     */
    getVerticalBounds: function( font ) {
      assert && assert( font instanceof Font, 'Font required' );

      var css = font.toCSS();

      // Cache these, as it's more expensive
      var verticalBounds = hybridFontVerticalCache[ css ];
      if ( !verticalBounds ) {
        verticalBounds = hybridFontVerticalCache[ css ] = TextBounds.approximateSVGBounds( font, 'm' );
      }

      return verticalBounds;
    },

    /**
     * Returns an approximate width for text, determined by using Canvas' measureText().
     * @public
     *
     * @param {Font} font - The font of the text
     * @param {string} renderedText - Text to display (with any special characters replaced)
     * @returns {number}
     */
    approximateCanvasWidth: function( font, renderedText ) {
      assert && assert( font instanceof Font, 'Font required' );
      assert && assert( typeof renderedText === 'string', 'renderedText required' );

      var context = scenery.scratchContext;
      context.font = font.toCSS();
      context.direction = 'ltr';
      return context.measureText( renderedText ).width;
    },

    /**
     * Returns a new Bounds2 that is the approximate bounds of a Text node displayed with the specified font and renderedText.
     * @public
     *
     * This method uses a hybrid approach, using SVG measurement to determine the height, but using Canvas to determine the width.
     *
     * NOTE: Calling code relies on the new Bounds2 instance, as they mutate it.
     *
     * @param {Font} font - The font of the text
     * @param {string} renderedText - Text to display (with any special characters replaced)
     * @returns {Bounds2}
     */
    approximateHybridBounds: function( font, renderedText ) {
      assert && assert( font instanceof Font, 'Font required' );
      assert && assert( typeof renderedText === 'string', 'renderedText required' );

      var verticalBounds = TextBounds.getVerticalBounds( font );

      var canvasWidth = TextBounds.approximateCanvasWidth( font, renderedText );

      // it seems that SVG bounds generally have x=0, so we hard code that here
      return new Bounds2( 0, verticalBounds.minY, canvasWidth, verticalBounds.maxY );
    },

    /**
     * Returns a new Bounds2 that is the approximate bounds of a Text node displayed with the specified font, given a DOM element
     * @public
     *
     * NOTE: Calling code relies on the new Bounds2 instance, as they mutate it.
     *
     * @param {Font} font - The font of the text
     * @param {Element} element - DOM element created for the text. This is required, as the text handles HTML and non-HTML text differently.
     * @returns {Bounds2}
     */
    approximateDOMBounds: function( font, element ) {
      assert && assert( font instanceof Font, 'Font required' );

      var maxHeight = 1024; // technically this will fail if the font is taller than this!

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
      $( span ).css( 'font', font.toCSS() );
      span.appendChild( element );
      span.setAttribute( 'direction', 'ltr' );

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
      document.body.removeChild( div );

      return result;
    },

    /**
     * Returns a new Bounds2 that is the approximate bounds of a Text node displayed with the specified font, given a DOM element
     * @public
     *
     * TODO: Can we use this? What are the differences?
     *
     * NOTE: Calling code relies on the new Bounds2 instance, as they mutate it.
     *
     * @param {Font} font - The font of the text
     * @param {Element} element - DOM element created for the text. This is required, as the text handles HTML and non-HTML text differently.
     * @returns {Bounds2}
     */
    approximateImprovedDOMBounds: function( font, element ) {
      assert && assert( font instanceof Font, 'Font required' );

      // TODO: reuse this div?
      var div = document.createElement( 'div' );
      div.style.display = 'inline-block';
      div.style.font = font.toCSS();
      div.style.color = 'transparent';
      div.style.padding = '0 !important';
      div.style.margin = '0 !important';
      div.style.position = 'absolute';
      div.style.left = '0';
      div.style.top = '0';
      div.setAttribute( 'direction', 'ltr' );
      div.appendChild( element );

      document.body.appendChild( div );
      var bounds = new Bounds2( div.offsetLeft, div.offsetTop, div.offsetLeft + div.offsetWidth + 1, div.offsetTop + div.offsetHeight + 1 );
      document.body.removeChild( div );

      // Compensate for the baseline alignment
      var verticalBounds = TextBounds.getVerticalBounds( font );
      return bounds.shiftedY( verticalBounds.minY );
    },

    /**
     * Modifies an SVG text element's properties to match the specified font and text.
     * @public
     *
     * @param {SVGTextElement} textElement
     * @param {Font} font - The font of the text
     * @param {string} renderedText - Text to display (with any special characters replaced)
     */
    setSVGTextAttributes: function( textElement, font, renderedText ) {
      assert && assert( font instanceof Font, 'Font required' );
      assert && assert( typeof renderedText === 'string', 'renderedText required' );

      textElement.setAttribute( 'direction', 'ltr' );
      textElement.setAttribute( 'font-family', font.getFamily() );
      textElement.setAttribute( 'font-size', font.getSize() );
      textElement.setAttribute( 'font-style', font.getStyle() );
      textElement.setAttribute( 'font-weight', font.getWeight() );
      textElement.setAttribute( 'font-stretch', font.getStretch() );
      textElement.lastChild.nodeValue = renderedText;
    },

    /**
     * Initializes containers and elements required for SVG text measurement.
     * @public
     */
    initializeTextBounds: function() {
      svgTextSizeContainer = document.getElementById( TEXT_SIZE_CONTAINER_ID );

      if ( !svgTextSizeContainer ) {
        // set up the container and text for testing text bounds quickly (using approximateSVGBounds)
        svgTextSizeContainer = document.createElementNS( scenery.svgns, 'svg' );
        svgTextSizeContainer.setAttribute( 'width', '2' );
        svgTextSizeContainer.setAttribute( 'height', '2' );
        svgTextSizeContainer.setAttribute( 'id', TEXT_SIZE_CONTAINER_ID );
        svgTextSizeContainer.setAttribute( 'style', 'visibility: hidden; pointer-events: none; position: absolute; left: -65535px; right: -65535px;' ); // so we don't flash it in a visible way to the user
      }

      svgTextSizeElement = document.getElementById( TEXT_SIZE_ELEMENT_ID );

      // NOTE! copies createSVGElement
      if ( !svgTextSizeElement ) {
        svgTextSizeElement = document.createElementNS( scenery.svgns, 'text' );
        svgTextSizeElement.appendChild( document.createTextNode( '' ) );
        svgTextSizeElement.setAttribute( 'dominant-baseline', 'alphabetic' ); // to match Canvas right now
        svgTextSizeElement.setAttribute( 'text-rendering', 'geometricPrecision' );
        svgTextSizeElement.setAttributeNS( 'http://www.w3.org/XML/1998/namespace', 'xml:space', 'preserve' );
        svgTextSizeElement.setAttribute( 'id', TEXT_SIZE_ELEMENT_ID );
        svgTextSizeContainer.appendChild( svgTextSizeElement );
      }
    }
  };

  scenery.register( 'TextBounds', TextBounds );

  return TextBounds;
} );
