// Copyright 2016-2019, University of Colorado Boulder

/**
 * SVG drawable for Text nodes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var platform = require( 'PHET_CORE/platform' );
  var Poolable = require( 'PHET_CORE/Poolable' );
  var scenery = require( 'SCENERY/scenery' );
  var SVGSelfDrawable = require( 'SCENERY/display/SVGSelfDrawable' );
  var TextStatefulDrawable = require( 'SCENERY/display/drawables/TextStatefulDrawable' );

  // TODO: change this based on memory and performance characteristics of the platform
  var keepSVGTextElements = true; // whether we should pool SVG elements for the SVG rendering states, or whether we should free them when possible for memory

  // Some browsers (IE/Edge) can't handle our UTF-8 embedding marks AND SVG textLength/spacingAndGlyphs. We disable
  // using these features, because they aren't necessary on these browsers.
  // See https://github.com/phetsims/scenery/issues/455 for more information.
  var useSVGTextLengthAdjustments = !platform.ie && !platform.edge;

  // Safari seems to have many issues with text and repaint regions, resulting in artifacts showing up when not correctly
  // repainted (https://github.com/phetsims/qa/issues/1039#issuecomment-1949196606), and
  // cutting off some portions of the text (https://github.com/phetsims/scenery/issues/1610).
  // We have persistently created "transparent" rectangles to force repaints (requiring the client to do so), but this
  // seems to not work in many cases, and seems to be a usability issue to have to add workarounds.
  // If we place it in the same SVG group as the text, we'll get the same transform, but it seems to provide a consistent
  // workaround.
  var useTransparentSVGTextWorkaround = platform.safari;

  /**
   * A generated SVGSelfDrawable whose purpose will be drawing our Text. One of these drawables will be created
   * for each displayed instance of a Text node.
   * @constructor
   *
   * @param {number} renderer - Renderer bitmask, see Renderer's documentation for more details.
   * @param {Instance} instance
   */
  function TextSVGDrawable( renderer, instance ) {
    // Super-type initialization
    this.initializeSVGSelfDrawable( renderer, instance, true, keepSVGTextElements ); // usesPaint: true

    if ( !this.svgElement ) {
      var text = document.createElementNS( scenery.svgns, 'text' );
      text.appendChild( document.createTextNode( '' ) );

      // @private {SVGTextElement}
      this.text = text;

      // If we're applying the workaround, we'll nest everything under a group element
      if ( useTransparentSVGTextWorkaround ) {
        var group = document.createElementNS( scenery.svgns, 'g' );
        group.appendChild( text );

        this.svgElement = group;

        // "transparent" fill seems to trick Safari into repainting the region correctly.
        this.workaroundRect = document.createElementNS( scenery.svgns, 'rect' );
        this.workaroundRect.setAttribute( 'fill', 'transparent' );
        group.appendChild( this.workaroundRect );
      }
      else {
        // @protected {SVGTextElement|SVGGroup} - Sole SVG element for this drawable, implementing API for SVGSelfDrawable
        this.svgElement = text;
      }

      // TODO: flag adjustment for SVG qualities
      text.setAttribute( 'dominant-baseline', 'alphabetic' ); // to match Canvas right now
      text.setAttribute( 'text-rendering', 'geometricPrecision' );
      if ( useSVGTextLengthAdjustments ) {
        text.setAttribute( 'lengthAdjust', 'spacingAndGlyphs' );
      }
      text.setAttributeNS( 'http://www.w3.org/XML/1998/namespace', 'xml:space', 'preserve' );
      text.setAttribute( 'direction', 'ltr' );
    }
  }

  scenery.register( 'TextSVGDrawable', TextSVGDrawable );

  inherit( SVGSelfDrawable, TextSVGDrawable, {
    /**
     * Updates the SVG elements so that they will appear like the current node's representation.
     * @protected
     *
     * Implements the interface for SVGSelfDrawable (and is called from the SVGSelfDrawable's update).
     */
    updateSVGSelf: function() {
      var text = this.text;

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
        text.lastChild.nodeValue = this.node.renderedText;
      }

      // text length correction, tested with scenery/tests/text-quality-test.html to determine how to match Canvas/SVG rendering (and overall length)
      if ( this.dirtyBounds && useSVGTextLengthAdjustments && isFinite( this.node.selfBounds.width ) ) {
        text.setAttribute( 'textLength', this.node.selfBounds.width );

        if ( useTransparentSVGTextWorkaround ) {
          // Since text can get bigger/smaller, lets make the region larger than the "reported" bounds - this is needed
          // for the usually-problematic locales that have glyphs that extend well past the normal browser-reported
          // bounds. Since this is transparent, we can make it larger than the actual bounds.
          var paddingRatio = 0.2;
          var horizontalPadding = this.node.selfBounds.width * paddingRatio;
          var verticalPadding = this.node.selfBounds.height * paddingRatio;
          this.workaroundRect.setAttribute( 'x', this.node.selfBounds.minX - horizontalPadding );
          this.workaroundRect.setAttribute( 'y', this.node.selfBounds.minY - verticalPadding );
          this.workaroundRect.setAttribute( 'width', this.node.selfBounds.width + 2 * horizontalPadding );
          this.workaroundRect.setAttribute( 'height', this.node.selfBounds.height + 2 * verticalPadding );
        }
      }

      // Apply any fill/stroke changes to our element.
      this.updateFillStrokeStyle( text );
    }
  } );

  TextStatefulDrawable.mixInto( TextSVGDrawable );

  Poolable.mixInto( TextSVGDrawable );

  return TextSVGDrawable;
} );
