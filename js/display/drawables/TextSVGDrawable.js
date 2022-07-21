// Copyright 2016-2022, University of Colorado Boulder

/**
 * SVG drawable for Text nodes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import platform from '../../../../phet-core/js/platform.js';
import Poolable from '../../../../phet-core/js/Poolable.js';
import { scenery, svgns, SVGSelfDrawable, TextStatefulDrawable, Utils } from '../../imports.js';

// TODO: change this based on memory and performance characteristics of the platform
const keepSVGTextElements = true; // whether we should pool SVG elements for the SVG rendering states, or whether we should free them when possible for memory

// Some browsers (IE/Edge) can't handle our UTF-8 embedding marks AND SVG textLength/spacingAndGlyphs. We disable
// using these features, because they aren't necessary on these browsers.
// See https://github.com/phetsims/scenery/issues/455 for more information.
const useSVGTextLengthAdjustments = !platform.edge;

class TextSVGDrawable extends TextStatefulDrawable( SVGSelfDrawable ) {
  /**
   * @public
   * @override
   *
   * @param {number} renderer
   * @param {Instance} instance
   */
  initialize( renderer, instance ) {
    super.initialize( renderer, instance, true, keepSVGTextElements ); // usesPaint: true

    // @private {boolean}
    this.hasLength = false;

    if ( !this.svgElement ) {
      const text = document.createElementNS( svgns, 'text' );

      // @protected {SVGTextElement} - Sole SVG element for this drawable, implementing API for SVGSelfDrawable
      this.svgElement = text;

      text.appendChild( document.createTextNode( '' ) );

      // TODO: flag adjustment for SVG qualities
      text.setAttribute( 'dominant-baseline', 'alphabetic' ); // to match Canvas right now
      text.setAttribute( 'text-rendering', 'geometricPrecision' );
      text.setAttributeNS( 'http://www.w3.org/XML/1998/namespace', 'xml:space', 'preserve' );
      text.setAttribute( 'direction', 'ltr' );
    }
  }

  /**
   * Updates the SVG elements so that they will appear like the current node's representation.
   * @protected
   *
   * Implements the interface for SVGSelfDrawable (and is called from the SVGSelfDrawable's update).
   */
  updateSVGSelf() {
    const text = this.svgElement;

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
      text.lastChild.nodeValue = Utils.safariEmbeddingMarkWorkaround( this.node.renderedText );
    }

    // text length correction, tested with scenery/tests/text-quality-test.html to determine how to match Canvas/SVG rendering (and overall length)
    if ( this.dirtyBounds && useSVGTextLengthAdjustments ) {
      const useLengthAdjustment = this.node._boundsMethod !== 'accurate' && isFinite( this.node.selfBounds.width );

      if ( useLengthAdjustment ) {
        if ( !this.hasLength ) {
          this.hasLength = true;
          text.setAttribute( 'lengthAdjust', 'spacingAndGlyphs' );
        }
        text.setAttribute( 'textLength', this.node.selfBounds.width );
      }
      else if ( this.hasLength ) {
        this.hasLength = false;
        text.removeAttribute( 'lengthAdjust' );
        text.removeAttribute( 'textLength' );
      }
    }

    // Apply any fill/stroke changes to our element.
    this.updateFillStrokeStyle( text );
  }
}

scenery.register( 'TextSVGDrawable', TextSVGDrawable );

Poolable.mixInto( TextSVGDrawable );

export default TextSVGDrawable;