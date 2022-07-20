// Copyright 2016-2022, University of Colorado Boulder

/**
 * SVG drawable for Line nodes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Poolable from '../../../../phet-core/js/Poolable.js';
import { LineStatefulDrawable, scenery, svgns, SVGSelfDrawable } from '../../imports.js';

// TODO: change this based on memory and performance characteristics of the platform
const keepSVGLineElements = true; // whether we should pool SVG elements for the SVG rendering states, or whether we should free them when possible for memory

/*---------------------------------------------------------------------------*
 * SVG Rendering
 *----------------------------------------------------------------------------*/

class LineSVGDrawable extends LineStatefulDrawable( SVGSelfDrawable ) {
  /**
   * @public
   * @override
   *
   * @param {number} renderer
   * @param {Instance} instance
   */
  initialize( renderer, instance ) {
    super.initialize( renderer, instance, true, keepSVGLineElements ); // usesPaint: true

    this.svgElement = this.svgElement || document.createElementNS( svgns, 'line' );
  }

  /**
   * Updates the SVG elements so that they will appear like the current node's representation.
   * @protected
   * @override
   */
  updateSVGSelf() {
    const line = this.svgElement;

    if ( this.dirtyX1 ) {
      line.setAttribute( 'x1', this.node.x1 );
    }
    if ( this.dirtyY1 ) {
      line.setAttribute( 'y1', this.node.y1 );
    }
    if ( this.dirtyX2 ) {
      line.setAttribute( 'x2', this.node.x2 );
    }
    if ( this.dirtyY2 ) {
      line.setAttribute( 'y2', this.node.y2 );
    }

    // Apply any fill/stroke changes to our element.
    this.updateFillStrokeStyle( line );
  }
}

scenery.register( 'LineSVGDrawable', LineSVGDrawable );

Poolable.mixInto( LineSVGDrawable );

export default LineSVGDrawable;