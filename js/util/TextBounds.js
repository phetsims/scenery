// Copyright 2016-2023, University of Colorado Boulder

/**
 * Different methods of detection of text bounds.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Bounds2 from '../../../dot/js/Bounds2.js';
import { CanvasContextWrapper, Font, scenery, svgns, Utils } from '../imports.js';

// @private {string} - ID for a container for our SVG test element (determined to find the size of text elements with SVG)
const TEXT_SIZE_CONTAINER_ID = 'sceneryTextSizeContainer';

// @private {string} - ID for our SVG test element (determined to find the size of text elements with SVG)
const TEXT_SIZE_ELEMENT_ID = 'sceneryTextSizeElement';

// @private {SVGElement} - Container for our SVG test element (determined to find the size of text elements with SVG)
let svgTextSizeContainer;

// @private {SVGElement} - Test SVG element (determined to find the size of text elements with SVG)
let svgTextSizeElement;

// Maps CSS {string} => {Bounds2}, so that we can cache the vertical font sizes outside of the Font objects themselves.
const hybridFontVerticalCache = {};

let deliveredWarning = false;

const TextBounds = {
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
  approximateSVGBounds( font, renderedText ) {
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
    const rect = svgTextSizeElement.getBBox();

    if ( rect.width === 0 && rect.height === 0 && renderedText.length > 0 ) {
      if ( !deliveredWarning ) {
        deliveredWarning = true;

        console.log( 'WARNING: Guessing text bounds, is the simulation hidden? See https://github.com/phetsims/chipper/issues/768' );
      }
      return TextBounds.guessSVGBounds( font, renderedText );
    }

    return new Bounds2( rect.x, rect.y, rect.x + rect.width, rect.y + rect.height );
  },

  /**
   * Returns a guess for what the SVG bounds of a font would be, based on PhetFont as an example.
   * @public
   *
   * @param {Font} font
   * @param {string} renderedText
   * @returns {Bounds2}
   */
  guessSVGBounds( font, renderedText ) {
    const px = font.getNumericSize();
    const isBold = font.weight === 'bold';

    // Our best guess, based on PhetFont in macOS Chrome. Things may differ, but hopefully this approximation
    // is useful.
    return new Bounds2( 0, -0.9 * px, ( isBold ? 0.435 : 0.4 ) * px * renderedText.length, 0.22 * px );
  },

  /**
   * Returns a new Bounds2 that is the approximate bounds of the specified Text node.
   * @public
   *
   * NOTE: Calling code relies on the new Bounds2 instance, as they mutate it.
   *
   * @param {scenery.Text} text - The Text node
   * @returns {Bounds2}
   */
  accurateCanvasBounds( text ) {
    const context = scenery.scratchContext;
    context.font = text._font.toCSS();
    context.direction = 'ltr';
    const metrics = context.measureText( text.renderedText );
    return new Bounds2(
      -metrics.actualBoundingBoxLeft,
      -metrics.actualBoundingBoxAscent,
      metrics.actualBoundingBoxRight,
      metrics.actualBoundingBoxDescent
    );
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
   * @param {scenery.Text} text - The Text node
   * @returns {Bounds2}
   */
  accurateCanvasBoundsFallback( text ) {
    // this seems to be slower than expected, mostly due to Font getters
    const svgBounds = TextBounds.approximateSVGBounds( text._font, text.renderedText );

    //If svgBounds are zero, then return the zero bounds
    if ( !text.renderedText.length || svgBounds.width === 0 ) {
      return svgBounds;
    }

    // NOTE: should return new instance, so that it can be mutated later
    const accurateBounds = Utils.canvasAccurateBounds( context => {
      context.font = text._font.toCSS();
      context.direction = 'ltr';
      context.fillText( text.renderedText, 0, 0 );
      if ( text.hasPaintableStroke() ) {
        const fakeWrapper = new CanvasContextWrapper( null, context );
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
  getVerticalBounds( font ) {
    assert && assert( font instanceof Font, 'Font required' );

    const css = font.toCSS();

    // Cache these, as it's more expensive
    let verticalBounds = hybridFontVerticalCache[ css ];
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
  approximateCanvasWidth( font, renderedText ) {
    assert && assert( font instanceof Font, 'Font required' );
    assert && assert( typeof renderedText === 'string', 'renderedText required' );

    const context = scenery.scratchContext;
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
  approximateHybridBounds( font, renderedText ) {
    assert && assert( font instanceof Font, 'Font required' );
    assert && assert( typeof renderedText === 'string', 'renderedText required' );

    const verticalBounds = TextBounds.getVerticalBounds( font );

    const canvasWidth = TextBounds.approximateCanvasWidth( font, renderedText );

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
  approximateDOMBounds( font, element ) {
    assert && assert( font instanceof Font, 'Font required' );

    const maxHeight = 1024; // technically this will fail if the font is taller than this!

    // <div style="position: absolute; left: 0; top: 0; padding: 0 !important; margin: 0 !important;"><span id="baselineSpan" style="font-family: Verdana; font-size: 25px;">QuipTaQiy</span><div style="vertical-align: baseline; display: inline-block; width: 0; height: 500px; margin: 0 important!; padding: 0 important!;"></div></div>

    const div = document.createElement( 'div' );
    $( div ).css( {
      position: 'absolute',
      left: 0,
      top: 0,
      padding: '0 !important',
      margin: '0 !important',
      display: 'hidden'
    } );

    const span = document.createElement( 'span' );
    $( span ).css( 'font', font.toCSS() );
    span.appendChild( element );
    span.setAttribute( 'direction', 'ltr' );

    const fakeImage = document.createElement( 'div' );
    $( fakeImage ).css( {
      'vertical-align': 'baseline',
      display: 'inline-block',
      width: 0,
      height: `${maxHeight}px`,
      margin: '0 !important',
      padding: '0 !important'
    } );

    div.appendChild( span );
    div.appendChild( fakeImage );

    document.body.appendChild( div );
    const rect = span.getBoundingClientRect();
    const divRect = div.getBoundingClientRect();
    // add 1 pixel to rect.right to prevent HTML text wrapping
    const result = new Bounds2( rect.left, rect.top - maxHeight, rect.right + 1, rect.bottom - maxHeight ).shiftedXY( -divRect.left, -divRect.top );
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
  approximateImprovedDOMBounds( font, element ) {
    assert && assert( font instanceof Font, 'Font required' );

    // TODO: reuse this div?
    const div = document.createElement( 'div' );
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
    const bounds = new Bounds2( div.offsetLeft, div.offsetTop, div.offsetLeft + div.offsetWidth + 1, div.offsetTop + div.offsetHeight + 1 );
    document.body.removeChild( div );

    // Compensate for the baseline alignment
    const verticalBounds = TextBounds.getVerticalBounds( font );
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
  setSVGTextAttributes( textElement, font, renderedText ) {
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
  initializeTextBounds() {
    svgTextSizeContainer = document.getElementById( TEXT_SIZE_CONTAINER_ID );

    if ( !svgTextSizeContainer ) {
      // set up the container and text for testing text bounds quickly (using approximateSVGBounds)
      svgTextSizeContainer = document.createElementNS( svgns, 'svg' );
      svgTextSizeContainer.setAttribute( 'width', '2' );
      svgTextSizeContainer.setAttribute( 'height', '2' );
      svgTextSizeContainer.setAttribute( 'id', TEXT_SIZE_CONTAINER_ID );
      svgTextSizeContainer.setAttribute( 'style', 'visibility: hidden; pointer-events: none; position: absolute; left: -65535px; right: -65535px;' ); // so we don't flash it in a visible way to the user
    }

    svgTextSizeElement = document.getElementById( TEXT_SIZE_ELEMENT_ID );

    // NOTE! copies createSVGElement
    if ( !svgTextSizeElement ) {
      svgTextSizeElement = document.createElementNS( svgns, 'text' );
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

export default TextBounds;