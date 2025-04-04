// Copyright 2016-2025, University of Colorado Boulder

/**
 * Different methods of detection of text bounds.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Bounds2 from '../../../dot/js/Bounds2.js';
import scenery from '../scenery.js';
import CanvasContextWrapper from '../util/CanvasContextWrapper.js';
import Font from '../util/Font.js';
import svgns from '../util/svgns.js';
import Utils from '../util/Utils.js';
import type Text from '../nodes/Text.js';
import { scratchContext } from './scratches.js';

// ID for a container for our SVG test element (determined to find the size of text elements with SVG)
const TEXT_SIZE_CONTAINER_ID = 'sceneryTextSizeContainer';

// ID for our SVG test element (determined to find the size of text elements with SVG)
const TEXT_SIZE_ELEMENT_ID = 'sceneryTextSizeElement';

// Container for our SVG test element (determined to find the size of text elements with SVG)
let svgTextSizeContainer: SVGSVGElement;

// Test SVG element (determined to find the size of text elements with SVG)
let svgTextSizeElement: SVGTextElement;

// Maps CSS string => Bounds2, so that we can cache the vertical font sizes outside of the Font objects themselves.
const hybridFontVerticalCache: Record<string, Bounds2> = {};

let deliveredWarning = false;

export default class TextBounds {
  /**
   * Returns a new Bounds2 that is the approximate bounds of a Text node displayed with the specified font and renderedText.
   *
   * This method uses an SVG Text element, sets its text, and then determines its size to estimate the size of rendered text.
   *
   * NOTE: Calling code relies on the new Bounds2 instance, as they mutate it.
   */
  public static approximateSVGBounds( font: Font, renderedText: string ): Bounds2 {

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
  }

  /**
   * Returns a guess for what the SVG bounds of a font would be, based on PhetFont as an example.
   */
  public static guessSVGBounds( font: Font, renderedText: string ): Bounds2 {
    const px = font.getNumericSize();
    const isBold = font.weight === 'bold';

    // Our best guess, based on PhetFont in macOS Chrome. Things may differ, but hopefully this approximation
    // is useful.
    return new Bounds2( 0, -0.9 * px, ( isBold ? 0.435 : 0.4 ) * px * renderedText.length, 0.22 * px );
  }

  /**
   * Returns a new Bounds2 that is the approximate bounds of the specified Text node.
   *
   * NOTE: Calling code relies on the new Bounds2 instance, as they mutate it.
   */
  public static accurateCanvasBounds( text: Text ): Bounds2 {
    const context = scratchContext;
    context.font = text._font.toCSS();
    context.direction = 'ltr';
    const metrics = context.measureText( text.renderedText );
    return new Bounds2(
      -metrics.actualBoundingBoxLeft,
      -metrics.actualBoundingBoxAscent,
      metrics.actualBoundingBoxRight,
      metrics.actualBoundingBoxDescent
    );
  }

  /**
   * Returns a new Bounds2 that is the approximate bounds of the specified Text node.
   *
   * This method repeatedly renders the text into a Canvas and checks for what pixels are filled. Iteratively doing this for each bound
   * (top/left/bottom/right) until a tolerance results in very accurate bounds of what is displayed.
   *
   * NOTE: Calling code relies on the new Bounds2 instance, as they mutate it.
   */
  public static accurateCanvasBoundsFallback( text: Text ): Bounds2 {
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
        const fakeWrapper = new CanvasContextWrapper( null as unknown as HTMLCanvasElement, context );
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
  }

  /**
   * Returns a possibly-cached (treat as immutable) Bounds2 for use mainly for vertical parameters, given a specific Font.
   *
   * Uses SVG bounds determination for this value.
   */
  public static getVerticalBounds( font: Font ): Bounds2 {

    const css = font.toCSS();

    // Cache these, as it's more expensive
    let verticalBounds = hybridFontVerticalCache[ css ];
    if ( !verticalBounds ) {
      verticalBounds = hybridFontVerticalCache[ css ] = TextBounds.approximateSVGBounds( font, 'm' );
    }

    return verticalBounds;
  }

  /**
   * Returns an approximate width for text, determined by using Canvas' measureText().
   *
   * @param font - The font of the text
   * @param renderedText - Text to display (with any special characters replaced)
   */
  public static approximateCanvasWidth( font: Font, renderedText: string ): number {

    const context = scratchContext;
    context.font = font.toCSS();
    context.direction = 'ltr';
    return context.measureText( renderedText ).width;
  }

  /**
   * Returns a new Bounds2 that is the approximate bounds of a Text node displayed with the specified font and renderedText.
   *
   * This method uses a hybrid approach, using SVG measurement to determine the height, but using Canvas to determine the width.
   *
   * NOTE: Calling code relies on the new Bounds2 instance, as they mutate it.
   *
   * @param font - The font of the text
   * @param renderedText - Text to display (with any special characters replaced)
   */
  public static approximateHybridBounds( font: Font, renderedText: string ): Bounds2 {

    const verticalBounds = TextBounds.getVerticalBounds( font );

    const canvasWidth = TextBounds.approximateCanvasWidth( font, renderedText );

    // it seems that SVG bounds generally have x=0, so we hard code that here
    return new Bounds2( 0, verticalBounds.minY, canvasWidth, verticalBounds.maxY );
  }

  /**
   * Returns a new Bounds2 that is the approximate bounds of a Text node displayed with the specified font, given a DOM element
   *
   * NOTE: Calling code relies on the new Bounds2 instance, as they mutate it.
   *
   * @param font - The font of the text
   * @param element - DOM element created for the text. This is required, as the text handles HTML and non-HTML text differently.
   */
  public static approximateDOMBounds( font: Font, element: globalThis.Node ): Bounds2 {

    const maxHeight = 1024; // technically this will fail if the font is taller than this!

    // <div style="position: absolute; left: 0; top: 0; padding: 0 !important; margin: 0 !important;"><span id="baselineSpan" style="font-family: Verdana; font-size: 25px;">QuipTaQiy</span><div style="vertical-align: baseline; display: inline-block; width: 0; height: 500px; margin: 0 important!; padding: 0 important!;"></div></div>

    const div = document.createElement( 'div' );
    div.style.position = 'absolute';
    div.style.left = '0';
    div.style.top = '0';
    div.style.setProperty( 'padding', '0', 'important' );
    div.style.setProperty( 'margin', '0', 'important' );
    div.style.display = 'hidden';

    const span = document.createElement( 'span' );
    span.style.font = font.toCSS();
    span.appendChild( element );
    span.setAttribute( 'direction', 'ltr' );

    const fakeImage = document.createElement( 'div' );
    fakeImage.style.verticalAlign = 'baseline';
    fakeImage.style.display = 'inline-block';
    fakeImage.style.width = '0px';
    fakeImage.style.height = `${maxHeight}px`;
    fakeImage.style.setProperty( 'margin', '0', 'important' );
    fakeImage.style.setProperty( 'padding', '0', 'important' );

    div.appendChild( span );
    div.appendChild( fakeImage );

    document.body.appendChild( div );
    const rect = span.getBoundingClientRect();
    const divRect = div.getBoundingClientRect();
    // add 1 pixel to rect.right to prevent HTML text wrapping
    const result = new Bounds2( rect.left, rect.top - maxHeight, rect.right + 1, rect.bottom - maxHeight ).shiftedXY( -divRect.left, -divRect.top );
    document.body.removeChild( div );

    return result;
  }

  /**
   * Returns a new Bounds2 that is the approximate bounds of a Text node displayed with the specified font, given a DOM element
   *
   * TODO: Can we use this? What are the differences? https://github.com/phetsims/scenery/issues/1581
   *
   * NOTE: Calling code relies on the new Bounds2 instance, as they mutate it.
   *
   * @param font - The font of the text
   * @param element - DOM element created for the text. This is required, as the text handles HTML and non-HTML text differently.
   */
  public static approximateImprovedDOMBounds( font: Font, element: globalThis.Node ): Bounds2 {

    // TODO: reuse this div? https://github.com/phetsims/scenery/issues/1581
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
  }

  /**
   * Modifies an SVG text element's properties to match the specified font and text.
   *
   * @param textElement
   * @param font - The font of the text
   * @param renderedText - Text to display (with any special characters replaced)
   */
  public static setSVGTextAttributes( textElement: SVGTextElement, font: Font, renderedText: string ): void {

    textElement.setAttribute( 'direction', 'ltr' );
    textElement.setAttribute( 'font-family', font.getFamily() );
    textElement.setAttribute( 'font-size', font.getSize() );
    textElement.setAttribute( 'font-style', font.getStyle() );
    textElement.setAttribute( 'font-weight', font.getWeight() );
    textElement.setAttribute( 'font-stretch', font.getStretch() );
    textElement.lastChild!.nodeValue = renderedText;
  }

  /**
   * Initializes containers and elements required for SVG text measurement.
   */
  public static initializeTextBounds(): void {
    svgTextSizeContainer = document.getElementById( TEXT_SIZE_CONTAINER_ID ) as unknown as SVGSVGElement;

    if ( !svgTextSizeContainer ) {
      // set up the container and text for testing text bounds quickly (using approximateSVGBounds)
      svgTextSizeContainer = document.createElementNS( svgns, 'svg' );
      svgTextSizeContainer.setAttribute( 'width', '2' );
      svgTextSizeContainer.setAttribute( 'height', '2' );
      svgTextSizeContainer.setAttribute( 'id', TEXT_SIZE_CONTAINER_ID );
      svgTextSizeContainer.setAttribute( 'style', 'visibility: hidden; pointer-events: none; position: absolute; left: -65535px; right: -65535px;' ); // so we don't flash it in a visible way to the user
    }

    svgTextSizeElement = document.getElementById( TEXT_SIZE_ELEMENT_ID ) as unknown as SVGTextElement;

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
}

scenery.register( 'TextBounds', TextBounds );