// Copyright 2023, University of Colorado Boulder

/**
 * Represents a path that controls what regions things are drawn in.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { FillRule, scenery } from '../../../imports.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Matrix3 from '../../../../../dot/js/Matrix3.js';

let globalPathId = 0;

export default class RenderPath {

  public readonly id = globalPathId++;

  public constructor( public readonly fillRule: FillRule, public readonly subpaths: Vector2[][] ) {}

  public transformed( transform: Matrix3 ): RenderPath {
    return new RenderPath( this.fillRule, this.subpaths.map( subpath => subpath.map( point => transform.timesVector2( point ) ) ) );
  }

  public serialize(): SerializedRenderPath {
    return {
      fillRule: this.fillRule,
      subpaths: this.subpaths.map( subpath => subpath.map( point => ( { x: point.x, y: point.y } ) ) )
    };
  }

  public static deserialize( obj: SerializedRenderPath ): RenderPath {
    return new RenderPath( obj.fillRule, obj.subpaths.map( subpath => subpath.map( point => new Vector2( point.x, point.y ) ) ) );
  }
}

scenery.register( 'RenderPath', RenderPath );

export type SerializedRenderPath = {
  fillRule: FillRule;
  subpaths: { x: number; y: number }[][];
};