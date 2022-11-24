// Copyright 2016-2022, University of Colorado Boulder

/**
 * SVG drawable for Path nodes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import platform from '../../../../phet-core/js/platform.js';
import Poolable from '../../../../phet-core/js/Poolable.js';
import { PathStatefulDrawable, scenery, svgns, SVGSelfDrawable } from '../../imports.js';

// TODO: change this based on memory and performance characteristics of the platform
const keepSVGPathElements = true; // whether we should pool SVG elements for the SVG rendering states, or whether we should free them when possible for memory

class PathSVGDrawable extends PathStatefulDrawable( SVGSelfDrawable ) {
  /**
   * @public
   * @override
   *
   * @param {number} renderer
   * @param {Instance} instance
   */
  initialize( renderer, instance ) {
    super.initialize( renderer, instance, true, keepSVGPathElements ); // usesPaint: true

    // @protected {SVGPathElement} - Sole SVG element for this drawable, implementing API for SVGSelfDrawable
    this.svgElement = this.svgElement || document.createElementNS( svgns, 'path' );
  }

  /**
   * Updates the SVG elements so that they will appear like the current node's representation.
   * @protected
   *
   * Implements the interface for SVGSelfDrawable (and is called from the SVGSelfDrawable's update).
   */
  updateSVGSelf() {
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

      // We'll conditionally add another M0 0 to the end of the path if we're on Safari, we're running into a bug in
      // https://github.com/phetsims/gravity-and-orbits/issues/472 (debugged in
      // https://github.com/phetsims/geometric-optics-basics/issues/31) where we're getting artifacts.
      path.setAttribute( 'd', `${svgPath}${platform.safari ? ' M0 0' : ''}` );
    }

    // Apply any fill/stroke changes to our element.
    this.updateFillStrokeStyle( path );
  }
}

scenery.register( 'PathSVGDrawable', PathSVGDrawable );

Poolable.mixInto( PathSVGDrawable );

export default PathSVGDrawable;