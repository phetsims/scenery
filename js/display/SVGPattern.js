// Copyright 2017-2021, University of Colorado Boulder

/**
 * Creates an SVG pattern element for a given pattern.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Poolable from '../../../phet-core/js/Poolable.js';
import scenery from '../scenery.js';
import svgns from '../util/svgns.js';
import xlinkns from '../util/xlinkns.js';

class SVGPattern {
  /**
   * @param {Pattern} pattern
   */
  constructor( pattern ) {
    this.initialize( pattern );
  }

  /**
   * Poolable initializer.
   * @private
   *
   * @param {Pattern} pattern
   */
  initialize( pattern ) {
    sceneryLog && sceneryLog.Paints && sceneryLog.Paints( `[SVGPattern] initialize: ${pattern.id}` );
    sceneryLog && sceneryLog.Paints && sceneryLog.push();

    const hasPreviousDefinition = this.definition !== undefined;

    // @public {SVGPatternElement} - persistent
    this.definition = this.definition || document.createElementNS( svgns, 'pattern' );

    if ( !hasPreviousDefinition ) {
      // so we don't depend on the bounds of the object being drawn with the pattern
      this.definition.setAttribute( 'patternUnits', 'userSpaceOnUse' );

      //TODO: is this needed?
      this.definition.setAttribute( 'patternContentUnits', 'userSpaceOnUse' );
    }

    if ( pattern.transformMatrix ) {
      this.definition.setAttribute( 'patternTransform', pattern.transformMatrix.getSVGTransform() );
    }
    else {
      this.definition.removeAttribute( 'patternTransform' );
    }

    this.definition.setAttribute( 'x', '0' );
    this.definition.setAttribute( 'y', '0' );
    this.definition.setAttribute( 'width', pattern.image.width );
    this.definition.setAttribute( 'height', pattern.image.height );

    // @private {SVGImageElement} - persistent
    this.imageElement = this.imageElement || document.createElementNS( svgns, 'image' );
    this.imageElement.setAttribute( 'x', '0' );
    this.imageElement.setAttribute( 'y', '0' );
    this.imageElement.setAttribute( 'width', `${pattern.image.width}px` );
    this.imageElement.setAttribute( 'height', `${pattern.image.height}px` );
    this.imageElement.setAttributeNS( xlinkns, 'xlink:href', pattern.image.src );
    if ( !hasPreviousDefinition ) {
      this.definition.appendChild( this.imageElement );
    }

    sceneryLog && sceneryLog.Paints && sceneryLog.pop();

    return this;
  }

  /**
   * Called from SVGBlock, matches other paints.
   * @public
   */
  update() {
    // Nothing
  }

  /**
   * Disposes, so that it can be reused from the pool.
   * @public
   */
  dispose() {
    this.freeToPool();
  }
}

scenery.register( 'SVGPattern', SVGPattern );

Poolable.mixInto( SVGPattern );

export default SVGPattern;