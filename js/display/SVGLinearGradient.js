// Copyright 2017-2021, University of Colorado Boulder

/**
 * Controller that creates and keeps an SVG linear gradient up-to-date with a Scenery LinearGradient
 *
 * SVG gradients, see http://www.w3.org/TR/SVG/pservers.html
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Poolable from '../../../phet-core/js/Poolable.js';
import scenery from '../scenery.js';
import svgns from '../util/svgns.js';
import SVGGradient from './SVGGradient.js';

class SVGLinearGradient extends SVGGradient {
  /**
   * Poolable initializer.
   * @private
   *
   * @param {SVGBlock} svgBlock
   * @param {LinearGradient} linearGradient
   */
  initialize( svgBlock, linearGradient ) {
    sceneryLog && sceneryLog.Paints && sceneryLog.Paints( `[SVGLinearGradient] initialize ${linearGradient.id}` );
    sceneryLog && sceneryLog.Paints && sceneryLog.push();

    super.initialize( svgBlock, linearGradient );

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
    this.definition.setAttribute( 'x1', linearGradient.start.x );
    this.definition.setAttribute( 'y1', linearGradient.start.y );
    this.definition.setAttribute( 'x2', linearGradient.end.x );
    this.definition.setAttribute( 'y2', linearGradient.end.y );

    sceneryLog && sceneryLog.Paints && sceneryLog.pop();

    return this;
  }

  /**
   * Creates the gradient-type-specific definition.
   * @protected
   * @override
   *
   * @returns {SVGLinearGradientElement}
   */
  createDefinition() {
    return document.createElementNS( svgns, 'linearGradient' );
  }
}

scenery.register( 'SVGLinearGradient', SVGLinearGradient );

Poolable.mixInto( SVGLinearGradient );

export default SVGLinearGradient;