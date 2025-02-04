// Copyright 2016-2025, University of Colorado Boulder

/**
 * SVG drawable for Text nodes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import platform from '../../../../phet-core/js/platform.js';
import Poolable from '../../../../phet-core/js/Poolable.js';
import scenery from '../../scenery.js';
import svgns from '../../util/svgns.js';
import SVGSelfDrawable from '../../display/SVGSelfDrawable.js';
import TextStatefulDrawable from '../../display/drawables/TextStatefulDrawable.js';
import Utils from '../../util/Utils.js';

// TODO: change this based on memory and performance characteristics of the platform https://github.com/phetsims/scenery/issues/1581
const keepSVGTextElements = true; // whether we should pool SVG elements for the SVG rendering states, or whether we should free them when possible for memory

// Some browsers (IE/Edge) can't handle our UTF-8 embedding marks AND SVG textLength/spacingAndGlyphs. We disable
// using these features, because they aren't necessary on these browsers.
// See https://github.com/phetsims/scenery/issues/455 for more information.
const useSVGTextLengthAdjustments = !platform.edge;

// Safari seems to have many issues with text and repaint regions, resulting in artifacts showing up when not correctly
// repainted (https://github.com/phetsims/qa/issues/1039#issuecomment-1949196606), and
// cutting off some portions of the text (https://github.com/phetsims/scenery/issues/1610).
// We have persistently created "transparent" rectangles to force repaints (requiring the client to do so), but this
// seems to not work in many cases, and seems to be a usability issue to have to add workarounds.
// If we place it in the same SVG group as the text, we'll get the same transform, but it seems to provide a consistent
// workaround.
const useTransparentSVGTextWorkaround = platform.safari;

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

      // @private {SVGTextElement}
      this.text = text;

      // If we're applying the workaround, we'll nest everything under a group element
      if ( useTransparentSVGTextWorkaround ) {
        const group = document.createElementNS( svgns, 'g' );
        group.appendChild( text );

        this.svgElement = group;

        // "transparent" fill seems to trick Safari into repainting the region correctly.
        this.workaroundRect = document.createElementNS( svgns, 'rect' );
        this.workaroundRect.setAttribute( 'fill', 'transparent' );
        group.appendChild( this.workaroundRect );
      }
      else {
        // @protected {SVGTextElement|SVGGroup} - Sole SVG element for this drawable, implementing API for SVGSelfDrawable
        this.svgElement = text;
      }

      text.appendChild( document.createTextNode( '' ) );

      // TODO: flag adjustment for SVG qualities https://github.com/phetsims/scenery/issues/1581
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
    const text = this.text;

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
      let string = Utils.safariEmbeddingMarkWorkaround( this.node.renderedText );

      // Workaround for Firefox handling of RTL embedding marks, https://github.com/phetsims/scenery/issues/1643
      if ( platform.firefox ) {
        string = '\u200b' + string;
      }

      text.lastChild.nodeValue = string;
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

      if ( useTransparentSVGTextWorkaround ) {
        // Since text can get bigger/smaller, lets make the region larger than the "reported" bounds - this is needed
        // for the usually-problematic locales that have glyphs that extend well past the normal browser-reported
        // bounds. Since this is transparent, we can make it larger than the actual bounds.
        const paddingRatio = 0.2;
        const horizontalPadding = this.node.selfBounds.width * paddingRatio;
        const verticalPadding = this.node.selfBounds.height * paddingRatio;
        this.workaroundRect.setAttribute( 'x', this.node.selfBounds.minX - horizontalPadding );
        this.workaroundRect.setAttribute( 'y', this.node.selfBounds.minY - verticalPadding );
        this.workaroundRect.setAttribute( 'width', this.node.selfBounds.width + 2 * horizontalPadding );
        this.workaroundRect.setAttribute( 'height', this.node.selfBounds.height + 2 * verticalPadding );
      }
    }

    // Apply any fill/stroke changes to our element.
    this.updateFillStrokeStyle( text );
  }
}

scenery.register( 'TextSVGDrawable', TextSVGDrawable );

Poolable.mixInto( TextSVGDrawable );

export default TextSVGDrawable;