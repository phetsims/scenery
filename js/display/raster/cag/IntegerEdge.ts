// Copyright 2023, University of Colorado Boulder

/**
 * A line-segment edge with integer coordinates, as part of the rendering
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { RationalIntersection, scenery } from '../../../imports.js';
import { RenderPath } from '../render-program/RenderProgram.js';
import Bounds2 from '../../../../../dot/js/Bounds2.js';
import Utils from '../../../../../dot/js/Utils.js';
import Vector2 from '../../../../../dot/js/Vector2.js';

export default class IntegerEdge {

  public readonly bounds: Bounds2;
  public readonly intersections: RationalIntersection[] = [];

  public constructor(
    public readonly renderPath: RenderPath,
    public readonly x0: number,
    public readonly y0: number,
    public readonly x1: number,
    public readonly y1: number
  ) {
    assert && assert( Number.isInteger( x0 ) );
    assert && assert( Number.isInteger( y0 ) );
    assert && assert( Number.isInteger( x1 ) );
    assert && assert( Number.isInteger( y1 ) );

    // TODO: maybe don't compute this here? Can we compute it in the other work?
    this.bounds = new Bounds2(
      Math.min( x0, x1 ),
      Math.min( y0, y1 ),
      Math.max( x0, x1 ),
      Math.max( y0, y1 )
    );
  }

  public static fromUnscaledPoints( path: RenderPath, scale: number, p0: Vector2, p1: Vector2 ): IntegerEdge {
    const x0 = Utils.roundSymmetric( p0.x * scale );
    const y0 = Utils.roundSymmetric( p0.y * scale );
    const x1 = Utils.roundSymmetric( p1.x * scale );
    const y1 = Utils.roundSymmetric( p1.y * scale );
    return new IntegerEdge( path, x0, y0, x1, y1 );
  }
}

scenery.register( 'IntegerEdge', IntegerEdge );
