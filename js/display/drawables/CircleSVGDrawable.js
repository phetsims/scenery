// Copyright 2016-2022, University of Colorado Boulder

/**
 * SVG drawable for Circle nodes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Poolable from '../../../../phet-core/js/Poolable.js';
import { CircleStatefulDrawable, scenery, svgns, SVGSelfDrawable } from '../../imports.js';

// TODO: change this based on memory and performance characteristics of the platform
const keepSVGCircleElements = true; // whether we should pool SVG elements for the SVG rendering states, or whether we should free them when possible for memory

class CircleSVGDrawable extends CircleStatefulDrawable( SVGSelfDrawable ) {
  /**
   * @public
   * @override
   *
   * @param {number} renderer
   * @param {Instance} instance
   */
  initialize( renderer, instance ) {
    super.initialize( renderer, instance, true, keepSVGCircleElements ); // usesPaint: true

    // @protected {SVGCircleElement} - Sole SVG element for this drawable, implementing API for SVGSelfDrawable
    this.svgElement = this.svgElement || document.createElementNS( svgns, 'circle' );
  }


  /**
   * Updates the SVG elements so that they will appear like the current node's representation.
   * @protected
   *
   * Implements the interface for SVGSelfDrawable (and is called from the SVGSelfDrawable's update).
   */
  updateSVGSelf() {
    const circle = this.svgElement;

    if ( this.dirtyRadius ) {
      circle.setAttribute( 'r', this.node._radius );
    }

    // Apply any fill/stroke changes to our element.
    this.updateFillStrokeStyle( circle );
  }
}

scenery.register( 'CircleSVGDrawable', CircleSVGDrawable );

Poolable.mixInto( CircleSVGDrawable );

export default CircleSVGDrawable;