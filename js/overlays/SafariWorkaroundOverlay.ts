// Copyright 2022, University of Colorado Boulder

/**
 * Tricks Safari into forcing SVG rendering, see https://github.com/phetsims/geometric-optics-basics/issues/31
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import dotRandom from '../../../dot/js/dotRandom.js';
import { Display, scenery, svgns, TOverlay } from '../imports.js';

export default class SafariWorkaroundOverlay implements TOverlay {

  public domElement: SVGElement;
  private rect: SVGPathElement;
  private display: Display;

  public constructor( display: Display ) {

    this.display = display;

    // Create an SVG element that will be in front
    const svg = document.createElementNS( svgns, 'svg' );
    this.domElement = svg;
    svg.style.position = 'absolute';
    svg.setAttribute( 'class', 'safari-workaround' );
    svg.style.top = '0';
    svg.style.left = '0';
    // @ts-expect-error
    svg.style[ 'pointer-events' ] = 'none';

    // Make sure it covers our full size
    display.sizeProperty.link( dimension => {
      svg.setAttribute( 'width', '' + dimension.width );
      svg.setAttribute( 'height', '' + dimension.height );
      svg.style.clip = `rect(0px,${dimension.width}px,${dimension.height}px,0px)`;
    } );

    this.rect = document.createElementNS( svgns, 'rect' );

    svg.appendChild( this.rect );

    this.update();
  }

  public update(): void {
    const random = dotRandom.nextDouble();

    // Position the rectangle to take up the full display width/height EXCEPT for being eroded by a random
    // less-than-pixel amount.
    this.rect.setAttribute( 'x', '' + random );
    this.rect.setAttribute( 'y', '' + random );
    this.rect.setAttribute( 'style', 'fill: rgba(255,200,100,0); stroke: none;' );
    if ( this.display.width ) {
      this.rect.setAttribute( 'width', '' + ( this.display.width - random * 2 ) );
    }
    if ( this.display.height ) {
      this.rect.setAttribute( 'height', '' + ( this.display.height - random * 2 ) );
    }
  }
}

scenery.register( 'SafariWorkaroundOverlay', SafariWorkaroundOverlay );
