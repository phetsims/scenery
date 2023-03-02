// Copyright 2022-2023, University of Colorado Boulder

/**
 * Controller that creates and keeps an SVG radial gradient up-to-date with a Scenery RadialGradient
 *
 * SVG gradients, see http://www.w3.org/TR/SVG/pservers.html
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Pool, { TPoolable } from '../../../phet-core/js/Pool.js';
import { RadialGradient, scenery, SVGBlock, SVGGradient, svgns } from '../imports.js';

export default class SVGRadialGradient extends SVGGradient implements TPoolable {

  public constructor( svgBlock: SVGBlock, gradient: RadialGradient ) {
    super( svgBlock, gradient );
  }

  public override initialize( svgBlock: SVGBlock, radialGradient: RadialGradient ): this {
    sceneryLog && sceneryLog.Paints && sceneryLog.Paints( `[SVGRadialGradient] initialize ${radialGradient.id}` );
    sceneryLog && sceneryLog.Paints && sceneryLog.push();

    super.initialize( svgBlock, radialGradient );

    // Radial-specific setup
    this.definition.setAttribute( 'cx', '' + radialGradient.largePoint.x );
    this.definition.setAttribute( 'cy', '' + radialGradient.largePoint.y );
    this.definition.setAttribute( 'r', '' + radialGradient.maxRadius );
    this.definition.setAttribute( 'fx', '' + radialGradient.focalPoint.x );
    this.definition.setAttribute( 'fy', '' + radialGradient.focalPoint.y );

    sceneryLog && sceneryLog.Paints && sceneryLog.pop();

    return this;
  }

  /**
   * Creates the gradient-type-specific definition.
   */
  protected createDefinition(): SVGRadialGradientElement {
    return document.createElementNS( svgns, 'radialGradient' );
  }

  public freeToPool(): void {
    SVGRadialGradient.pool.freeToPool( this );
  }

  public static readonly pool = new Pool( SVGRadialGradient );
}

scenery.register( 'SVGRadialGradient', SVGRadialGradient );
