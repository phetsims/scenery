// Copyright 2016-2020, University of Colorado Boulder

/**
 * SVG drawable for Path nodes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import inherit from '../../../../phet-core/js/inherit.js';
import Poolable from '../../../../phet-core/js/Poolable.js';
import scenery from '../../scenery.js';
import SVGSelfDrawable from '../SVGSelfDrawable.js';
import PathStatefulDrawable from './PathStatefulDrawable.js';

// TODO: change this based on memory and performance characteristics of the platform
const keepSVGPathElements = true; // whether we should pool SVG elements for the SVG rendering states, or whether we should free them when possible for memory

/**
 * A generated SVGSelfDrawable whose purpose will be drawing our Path. One of these drawables will be created
 * for each displayed instance of a Path.
 * @constructor
 *
 * @param {number} renderer - Renderer bitmask, see Renderer's documentation for more details.
 * @param {Instance} instance
 */
function PathSVGDrawable( renderer, instance ) {
  // Super-type initialization
  this.initializeSVGSelfDrawable( renderer, instance, true, keepSVGPathElements ); // usesPaint: true

  // @protected {SVGPathElement} - Sole SVG element for this drawable, implementing API for SVGSelfDrawable
  this.svgElement = this.svgElement || document.createElementNS( scenery.svgns, 'path' );
}

scenery.register( 'PathSVGDrawable', PathSVGDrawable );

inherit( SVGSelfDrawable, PathSVGDrawable, {
  /**
   * Updates the SVG elements so that they will appear like the current node's representation.
   * @protected
   *
   * Implements the interface for SVGSelfDrawable (and is called from the SVGSelfDrawable's update).
   */
  updateSVGSelf: function() {
    assert && assert( !this.node.requiresSVGBoundsWorkaround(),
      'No workaround for https://github.com/phetsims/scenery/issues/196 is provided at this time, please add an epsilon' );

    const path = this.svgElement;
    if ( this.dirtyShape ) {
      let svgPath = this.node.hasShape() ? this.node._shape.getSVGPath() : '';

      // temporary workaround for https://bugs.webkit.org/show_bug.cgi?id=78980
      // and http://code.google.com/p/chromium/issues/detail?id=231626 where even removing
      // the attribute can cause this bug
      if ( !svgPath ) { svgPath = 'M0 0'; }

      // only set the SVG path if it's not the empty string
      path.setAttribute( 'd', svgPath );
    }

    // Apply any fill/stroke changes to our element.
    this.updateFillStrokeStyle( path );
  }
} );

PathStatefulDrawable.mixInto( PathSVGDrawable );

Poolable.mixInto( PathSVGDrawable );

export default PathSVGDrawable;