// Copyright 2017-2023, University of Colorado Boulder

/**
 * Controller that creates and keeps an SVG linear gradient up-to-date with a Scenery LinearGradient
 *
 * SVG gradients, see http://www.w3.org/TR/SVG/pservers.html
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Pool, { TPoolable } from '../../../phet-core/js/Pool.js';
import { LinearGradient, scenery, SVGBlock, SVGGradient, svgns } from '../imports.js';

export default class SVGLinearGradient extends SVGGradient implements TPoolable {

  public constructor( svgBlock: SVGBlock, gradient: LinearGradient ) {
    super( svgBlock, gradient );
  }

  public override initialize( svgBlock: SVGBlock, gradient: LinearGradient ): this {
    sceneryLog && sceneryLog.Paints && sceneryLog.Paints( `[SVGLinearGradient] initialize ${gradient.id}` );
    sceneryLog && sceneryLog.Paints && sceneryLog.push();

    super.initialize( svgBlock, gradient );

    // seems we need the defs: http://stackoverflow.com/questions/7614209/linear-gradients-in-svg-without-defs
    // SVG: spreadMethod 'pad' 'reflect' 'repeat' - find Canvas usage

    /* Approximate example of what we are creating:
     <linearGradient id="grad2" x1="0" y1="0" x2="100" y2="0" gradientUnits="userSpaceOnUse">
     <stop offset="0" style="stop-color:rgb(255,255,0);stop-opacity:1" />
     <stop offset="0.5" style="stop-color:rgba(255,255,0,0);stop-opacity:0" />
     <stop offset="1" style="stop-color:rgb(255,0,0);stop-opacity:1" />
     </linearGradient>
     */

    // Linear-specific setup
    const linearGradient = gradient as unknown as LinearGradient;
    this.definition.setAttribute( 'x1', '' + linearGradient.start.x );
    this.definition.setAttribute( 'y1', '' + linearGradient.start.y );
    this.definition.setAttribute( 'x2', '' + linearGradient.end.x );
    this.definition.setAttribute( 'y2', '' + linearGradient.end.y );

    sceneryLog && sceneryLog.Paints && sceneryLog.pop();

    return this;
  }

  /**
   * Creates the gradient-type-specific definition.
   */
  protected createDefinition(): SVGLinearGradientElement {
    return document.createElementNS( svgns, 'linearGradient' );
  }

  public freeToPool(): void {
    SVGLinearGradient.pool.freeToPool( this );
  }

  public static readonly pool = new Pool( SVGLinearGradient );
}
scenery.register( 'SVGLinearGradient', SVGLinearGradient );
