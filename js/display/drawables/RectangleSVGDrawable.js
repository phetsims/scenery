// Copyright 2016-2022, University of Colorado Boulder

/**
 * SVG drawable for Rectangle nodes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Poolable from '../../../../phet-core/js/Poolable.js';
import { RectangleStatefulDrawable, scenery, svgns, SVGSelfDrawable } from '../../imports.js';

// TODO: change this based on memory and performance characteristics of the platform
const keepSVGRectangleElements = true; // whether we should pool SVG elements for the SVG rendering states, or whether we should free them when possible for memory

class RectangleSVGDrawable extends RectangleStatefulDrawable( SVGSelfDrawable ) {
  /**
   * @public
   * @override
   *
   * @param {number} renderer
   * @param {Instance} instance
   */
  initialize( renderer, instance ) {
    super.initialize( renderer, instance, true, keepSVGRectangleElements ); // usesPaint: true

    this.lastArcW = -1; // invalid on purpose
    this.lastArcH = -1; // invalid on purpose

    // @protected {SVGRectElement} - Sole SVG element for this drawable, implementing API for SVGSelfDrawable
    this.svgElement = this.svgElement || document.createElementNS( svgns, 'rect' );
  }

  /**
   * Updates the SVG elements so that they will appear like the current node's representation.
   * @protected
   *
   * Implements the interface for SVGSelfDrawable (and is called from the SVGSelfDrawable's update).
   */
  updateSVGSelf() {
    const rect = this.svgElement;

    if ( this.dirtyX ) {
      rect.setAttribute( 'x', this.node._rectX );
    }
    if ( this.dirtyY ) {
      rect.setAttribute( 'y', this.node._rectY );
    }
    if ( this.dirtyWidth ) {
      rect.setAttribute( 'width', this.node._rectWidth );
    }
    if ( this.dirtyHeight ) {
      rect.setAttribute( 'height', this.node._rectHeight );
    }
    if ( this.dirtyCornerXRadius || this.dirtyCornerYRadius || this.dirtyWidth || this.dirtyHeight ) {
      let arcw = 0;
      let arch = 0;

      // workaround for various browsers if rx=20, ry=0 (behavior is inconsistent, either identical to rx=20,ry=20, rx=0,ry=0. We'll treat it as rx=0,ry=0)
      // see https://github.com/phetsims/scenery/issues/183
      if ( this.node.isRounded() ) {
        const maximumArcSize = this.node.getMaximumArcSize();
        arcw = Math.min( this.node._cornerXRadius, maximumArcSize );
        arch = Math.min( this.node._cornerYRadius, maximumArcSize );
      }
      if ( arcw !== this.lastArcW ) {
        this.lastArcW = arcw;
        rect.setAttribute( 'rx', arcw );
      }
      if ( arch !== this.lastArcH ) {
        this.lastArcH = arch;
        rect.setAttribute( 'ry', arch );
      }
    }

    // Apply any fill/stroke changes to our element.
    this.updateFillStrokeStyle( rect );
  }
}

scenery.register( 'RectangleSVGDrawable', RectangleSVGDrawable );

Poolable.mixInto( RectangleSVGDrawable );

export default RectangleSVGDrawable;