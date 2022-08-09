// Copyright 2017-2022, University of Colorado Boulder

/**
 * Creates an SVG pattern element for a given pattern.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Pool, { TPoolable } from '../../../phet-core/js/Pool.js';
import { Pattern, scenery, svgns, xlinkns } from '../imports.js';

export default class SVGPattern implements TPoolable {

  // persistent
  public definition!: SVGPatternElement;
  private imageElement!: SVGImageElement;

  public constructor( pattern: Pattern ) {
    this.initialize( pattern );
  }

  public initialize( pattern: Pattern ): this {
    sceneryLog && sceneryLog.Paints && sceneryLog.Paints( `[SVGPattern] initialize: ${pattern.id}` );
    sceneryLog && sceneryLog.Paints && sceneryLog.push();

    const hasPreviousDefinition = this.definition !== undefined;

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
    this.definition.setAttribute( 'width', '' + pattern.image.width );
    this.definition.setAttribute( 'height', '' + pattern.image.height );

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
   */
  public update(): void {
    // Nothing
  }

  /**
   * Disposes, so that it can be reused from the pool.
   */
  public dispose(): void {
    this.freeToPool();
  }

  public freeToPool(): void {
    SVGPattern.pool.freeToPool( this );
  }

  public static readonly pool = new Pool( SVGPattern );
}

scenery.register( 'SVGPattern', SVGPattern );
