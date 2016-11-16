// Copyright 2013-2015, University of Colorado Boulder

/**
 * Different methods of detection of text bounds.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var scenery = require( 'SCENERY/scenery' );
  var Font = require( 'SCENERY/util/Font' );
  var Bounds2 = require( 'DOT/Bounds2' );
  var Util = require( 'SCENERY/util/Util' );
  var CanvasContextWrapper = require( 'SCENERY/util/CanvasContextWrapper' );

  // @private {string} - ID for a container for our SVG test element (determined to find the size of text elements with SVG)
  var TEXT_SIZE_CONTAINER_ID = 'sceneryTextSizeContainer';

  // @private {string} - ID for our SVG test element (determined to find the size of text elements with SVG)
  var TEXT_SIZE_ELEMENT_ID = 'sceneryTextSizeElement';

  // @private {SVGElement} - Container for our SVG test element (determined to find the size of text elements with SVG)
  var svgTextSizeContainer;

  // @private {SVGElement} - Test SVG element (determined to find the size of text elements with SVG)
  var svgTextSizeElement;

  var hybridTextNode; // a node that is used to measure SVG text top/height for hybrid caching purposes
  var initializingHybridTextNode = false;

  // Maps CSS {string} => {Bounds2}, so that we can cache the vertical font sizes outside of the Font objects themselves.
  var hybridFontVerticalCache = {};

  var TextBounds = {
    // NOTE: should return new instance, so that it can be mutated later
    approximateSVGBounds: function( font, renderedText ) {
      assert && assert( font instanceof Font, 'Font required' );
      assert && assert( typeof renderedText === 'string', 'renderedText required' );

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
      TextBounds.setSVGTextAttributes( svgTextSizeElement, font, renderedText );
      var rect = svgTextSizeElement.getBBox();
      return new Bounds2( rect.x, rect.y, rect.x + rect.width, rect.y + rect.height );
    },

    accurateCanvasBounds: function( text ) {
      var svgBounds = TextBounds.approximateSVGBounds( text._font, text.renderedText ); // this seems to be slower than expected, mostly due to Font getters

      //If svgBounds are zero, then return the zero bounds
      if ( !text.renderedText.length || svgBounds.width === 0 ) {
        return svgBounds;
      }

      // NOTE: should return new instance, so that it can be mutated later
      return Util.canvasAccurateBounds( function( context ) {
        context.font = text._font.toCSS();
        context.direction = 'ltr';
        context.fillText( text.renderedText, 0, 0 );
        if ( text.hasStroke() ) {
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
    },

    getVerticalBounds: function( font ) {
      assert && assert( font instanceof Font, 'Font required' );

      if ( !hybridTextNode ) {
        return Bounds2.NOTHING; // we are the hybridTextNode, ignore us
      }

      var css = font.toCSS();
      var verticalBounds = hybridFontVerticalCache[ css ];
      if ( !verticalBounds ) {
        hybridTextNode.setFont( font );
        verticalBounds = hybridFontVerticalCache[ css ] = hybridTextNode.getBounds().copy();
      }

      return verticalBounds;
    },

    approximateCanvasWidth: function( font, renderedText ) {
      assert && assert( font instanceof Font, 'Font required' );
      assert && assert( typeof renderedText === 'string', 'renderedText required' );

      var context = scenery.scratchContext;
      context.font = font.toCSS();
      context.direction = 'ltr';
      return context.measureText( renderedText ).width;
    },

    // NOTE: should return new instance, so that it can be mutated later
    approximateHybridBounds: function( font, renderedText ) {
      assert && assert( font instanceof Font, 'Font required' );
      assert && assert( typeof renderedText === 'string', 'renderedText required' );

      var verticalBounds = TextBounds.getVerticalBounds( font );

      var canvasWidth = TextBounds.approximateCanvasWidth( font, renderedText );

      // it seems that SVG bounds generally have x=0, so we hard code that here
      return new Bounds2( 0, verticalBounds.minY, canvasWidth, verticalBounds.maxY );
    },

    // NOTE: should return new instance, so that it can be mutated later
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

    // TODO: can we use this?
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
      var verticalBounds = Text.getVerticalBounds( font );
      return bounds.shiftedY( verticalBounds.minY );
    },

    // TODO: update name!
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

    setupHybridTextNode: function() {
      function createSVGTextToMeasure() {
        var text = document.createElementNS( scenery.svgns, 'text' );
        text.appendChild( document.createTextNode( '' ) );

        // TODO: flag adjustment for SVG qualities
        text.setAttribute( 'dominant-baseline', 'alphabetic' ); // to match Canvas right now
        text.setAttribute( 'text-rendering', 'geometricPrecision' );
        text.setAttributeNS( 'http://www.w3.org/XML/1998/namespace', 'xml:space', 'preserve' );
        return text;
      }

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
        svgTextSizeElement = createSVGTextToMeasure();
        svgTextSizeElement.setAttribute( 'id', TEXT_SIZE_ELEMENT_ID );
        svgTextSizeContainer.appendChild( svgTextSizeElement );
      }

      initializingHybridTextNode = true;
      hybridTextNode = new scenery.Text( 'm', { boundsMethod: 'fast' } );
      initializingHybridTextNode = false;
    }
  };

  scenery.register( 'TextBounds', TextBounds );

  return TextBounds;
} );
