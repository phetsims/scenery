// Copyright 2013-2020, University of Colorado Boulder

/**
 * Supertype for overlays that display colored shapes (updated every frame).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import scenery from '../scenery.js';
import svgns from '../util/svgns.js';

class ShapeBasedOverlay {
  /**
   * @param {Display} display
   * @param {Node} rootNode
   * @param {string} name
   */
  constructor( display, rootNode, name ) {
    this.display = display;
    this.rootNode = rootNode;

    const svg = document.createElementNS( svgns, 'svg' );
    svg.style.position = 'absolute';
    svg.setAttribute( 'class', name );
    svg.style.top = 0;
    svg.style.left = 0;
    svg.style[ 'pointer-events' ] = 'none';
    this.svg = svg;

    function resize( width, height ) {
      svg.setAttribute( 'width', width );
      svg.setAttribute( 'height', height );
      svg.style.clip = 'rect(0px,' + width + 'px,' + height + 'px,0px)';
    }

    display.sizeProperty.link( dimension => {
      resize( dimension.width, dimension.height );
    } );

    this.domElement = svg;
  }

  /**
   * @public
   *
   * @param {Shape} shape
   * @param {string} color
   * @param {boolean} isOffset
   */
  addShape( shape, color, isOffset ) {
    const path = document.createElementNS( svgns, 'path' );
    let svgPath = shape.getSVGPath();

    // temporary workaround for https://bugs.webkit.org/show_bug.cgi?id=78980
    // and http://code.google.com/p/chromium/issues/detail?id=231626 where even removing
    // the attribute can cause this bug
    if ( !svgPath ) { svgPath = 'M0 0'; }

    if ( svgPath ) {
      // only set the SVG path if it's not the empty string
      path.setAttribute( 'd', svgPath );
    }
    else if ( path.hasAttribute( 'd' ) ) {
      path.removeAttribute( 'd' );
    }

    path.setAttribute( 'style', 'fill: none; stroke: ' + color + '; stroke-dasharray: 5, 3; stroke-dashoffset: ' + ( isOffset ? 5 : 0 ) + '; stroke-width: 3;' );
    this.svg.appendChild( path );
  }

  /**
   * @public
   */
  update() {
    while ( this.svg.childNodes.length ) {
      this.svg.removeChild( this.svg.childNodes[ this.svg.childNodes.length - 1 ] );
    }

    this.addShapes();
  }

  /**
   * @public
   * @abstract
   */
  addShapes() {

  }

  /**
   * Releases references
   * @public
   */
  dispose() {

  }
}

scenery.register( 'ShapeBasedOverlay', ShapeBasedOverlay );
export default ShapeBasedOverlay;