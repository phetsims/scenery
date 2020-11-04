// Copyright 2017-2020, University of Colorado Boulder

/**
 * Controller that creates and keeps an SVG radial gradient up-to-date with a Scenery RadialGradient
 *
 * SVG gradients, see http://www.w3.org/TR/SVG/pservers.html
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Poolable from '../../../phet-core/js/Poolable.js';
import scenery from '../scenery.js';
import svgns from '../util/svgns.js';
import SVGGradient from './SVGGradient.js';

class SVGRadialGradient extends SVGGradient {
  /**
   * Poolable initializer.
   * @private
   *
   * @param {SVGBlock} svgBlock
   * @param {RadialGradient} radialGradient
   */
  initialize( svgBlock, radialGradient ) {
    sceneryLog && sceneryLog.Paints && sceneryLog.Paints( '[SVGRadialGradient] initialize ' + radialGradient.id );
    sceneryLog && sceneryLog.Paints && sceneryLog.push();

    super.initialize( svgBlock, radialGradient );

    // Radial-specific setup
    this.definition.setAttribute( 'cx', radialGradient.largePoint.x );
    this.definition.setAttribute( 'cy', radialGradient.largePoint.y );
    this.definition.setAttribute( 'r', radialGradient.maxRadius );
    this.definition.setAttribute( 'fx', radialGradient.focalPoint.x );
    this.definition.setAttribute( 'fy', radialGradient.focalPoint.y );

    sceneryLog && sceneryLog.Paints && sceneryLog.pop();

    return this;
  }

  /**
   * Creates the gradient-type-specific definition.
   * @protected
   * @override
   *
   * @returns {SVGRadialGradientElement}
   */
  createDefinition() {
    return document.createElementNS( svgns, 'radialGradient' );
  }
}

scenery.register( 'SVGRadialGradient', SVGRadialGradient );

Poolable.mixInto( SVGRadialGradient );

export default SVGRadialGradient;