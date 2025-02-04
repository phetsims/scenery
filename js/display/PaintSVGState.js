// Copyright 2016-2021, University of Colorado Boulder

/**
 * Handles SVG <defs> and fill/stroke style for SVG elements (by composition, not a trait or for inheritance).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import scenery from '../scenery.js';

class PaintSVGState {
  constructor() {
    this.initialize();
  }

  /**
   * Initializes the state
   * @public
   */
  initialize() {
    // @public {SVGBlock|null}
    this.svgBlock = null;

    // @public {string} fill/stroke style fragments that are currently used
    this.fillStyle = 'none';
    this.strokeStyle = 'none';

    // @public {PaintDef} - current reference-counted fill/stroke paints (gradients and fills) that will need to be
    // released on changes or disposal
    this.fillPaint = null;
    this.strokePaint = null;

    // these are used by the actual SVG element
    this.updateBaseStyle(); // the main style CSS
    this.strokeDetailStyle = ''; // width/dash/cap/join CSS
  }

  /**
   * Disposes the PaintSVGState, releasing listeners as needed.
   * @public
   */
  dispose() {
    // be cautious, release references
    this.releaseFillPaint();
    this.releaseStrokePaint();
  }

  /**
   * @private
   */
  releaseFillPaint() {
    if ( this.fillPaint ) {
      this.svgBlock.decrementPaint( this.fillPaint );
      this.fillPaint = null;
    }
  }

  /**
   * @private
   */
  releaseStrokePaint() {
    if ( this.strokePaint ) {
      this.svgBlock.decrementPaint( this.strokePaint );
      this.strokePaint = null;
    }
  }

  /**
   * Called when the fill needs to be updated, with the latest defs SVG block
   * @public
   *
   * @param {SVGBlock} svgBlock
   * @param {null|string|Color|LinearGradient|RadialGradient|Pattern} fill
   */
  updateFill( svgBlock, fill ) {
    assert && assert( this.svgBlock === svgBlock );

    // NOTE: If fill.isPaint === true, this should be different if we switched to a different SVG block.
    const fillStyle = paintToSVGStyle( fill, svgBlock );

    // If our fill paint reference changed
    if ( fill !== this.fillPaint ) {
      // release the old reference
      this.releaseFillPaint();

      // only store a new reference if our new fill is a paint
      if ( fill && fill.isPaint ) {
        this.fillPaint = fill;
        svgBlock.incrementPaint( fill );
      }
    }

    // If we need to update the SVG style of our fill
    if ( fillStyle !== this.fillStyle ) {
      this.fillStyle = fillStyle;
      this.updateBaseStyle();
    }
  }

  /**
   * Called when the stroke needs to be updated, with the latest defs SVG block
   * @public
   *
   * @param {SVGBlock} svgBlock
   * @param {null|string|Color|LinearGradient|RadialGradient|Pattern} fill
   */
  updateStroke( svgBlock, stroke ) {
    assert && assert( this.svgBlock === svgBlock );

    // NOTE: If stroke.isPaint === true, this should be different if we switched to a different SVG block.
    const strokeStyle = paintToSVGStyle( stroke, svgBlock );

    // If our stroke paint reference changed
    if ( stroke !== this.strokePaint ) {
      // release the old reference
      this.releaseStrokePaint();

      // only store a new reference if our new stroke is a paint
      if ( stroke && stroke.isPaint ) {
        this.strokePaint = stroke;
        svgBlock.incrementPaint( stroke );
      }
    }

    // If we need to update the SVG style of our stroke
    if ( strokeStyle !== this.strokeStyle ) {
      this.strokeStyle = strokeStyle;
      this.updateBaseStyle();
    }
  }

  /**
   * @private
   */
  updateBaseStyle() {
    this.baseStyle = `fill: ${this.fillStyle}; stroke: ${this.strokeStyle};`;
  }

  /**
   * @private
   *
   * @param {Node} node
   */
  updateStrokeDetailStyle( node ) {
    let strokeDetailStyle = '';

    const lineWidth = node.getLineWidth();
    if ( lineWidth !== 1 ) {
      strokeDetailStyle += `stroke-width: ${lineWidth};`;
    }

    const lineCap = node.getLineCap();
    if ( lineCap !== 'butt' ) {
      strokeDetailStyle += `stroke-linecap: ${lineCap};`;
    }

    const lineJoin = node.getLineJoin();
    if ( lineJoin !== 'miter' ) {
      strokeDetailStyle += `stroke-linejoin: ${lineJoin};`;
    }

    const miterLimit = node.getMiterLimit();
    strokeDetailStyle += `stroke-miterlimit: ${miterLimit};`;

    if ( node.hasLineDash() ) {
      strokeDetailStyle += `stroke-dasharray: ${node.getLineDash().join( ',' )};`;
      strokeDetailStyle += `stroke-dashoffset: ${node.getLineDashOffset()};`;
    }

    this.strokeDetailStyle = strokeDetailStyle;
  }

  /**
   * Called when the defs SVG block is switched (our SVG element was moved to another SVG top-level context)
   * @public
   *
   * @param {SVGBlock} svgBlock
   */
  updateSVGBlock( svgBlock ) {
    // remove paints from the old svgBlock
    const oldSvgBlock = this.svgBlock;
    if ( oldSvgBlock ) {
      if ( this.fillPaint ) {
        oldSvgBlock.decrementPaint( this.fillPaint );
      }
      if ( this.strokePaint ) {
        oldSvgBlock.decrementPaint( this.strokePaint );
      }
    }

    this.svgBlock = svgBlock;

    // add paints to the new svgBlock
    if ( this.fillPaint ) {
      svgBlock.incrementPaint( this.fillPaint );
    }
    if ( this.strokePaint ) {
      svgBlock.incrementPaint( this.strokePaint );
    }
  }
}

/**
 * Returns the SVG style string used to represent a paint.
 *
 * @param {null|string|Color|LinearGradient|RadialGradient|Pattern} paint
 * @param {SVGBlock} svgBlock
 */
function paintToSVGStyle( paint, svgBlock ) {
  if ( !paint ) {
    // no paint
    return 'none';
  }
  else if ( paint.toCSS ) {
    // Color object paint
    return paint.toCSS();
  }
  else if ( paint.isPaint ) {
    // reference the SVG definition with a URL
    return `url(#${paint.id}-${svgBlock ? svgBlock.id : 'noblock'})`;
  }
  else {
    // plain CSS color
    return paint;
  }
}

scenery.register( 'PaintSVGState', PaintSVGState );

export default PaintSVGState;