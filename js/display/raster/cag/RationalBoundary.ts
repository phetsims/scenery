// Copyright 2023, University of Colorado Boulder

/**
 * A loop of rational half-edges
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { BigRationalVector2, LinearEdge, RationalHalfEdge, scenery } from '../../../imports.js';
import Bounds2 from '../../../../../dot/js/Bounds2.js';
import Vector2 from '../../../../../dot/js/Vector2.js';

export default class RationalBoundary {
  public readonly edges: RationalHalfEdge[] = [];
  public signedArea!: number;
  public bounds!: Bounds2;
  public minimalXRationalPoint!: BigRationalVector2;

  public computeProperties(): void {
    let signedArea = 0;
    const bounds = Bounds2.NOTHING.copy();
    let minimalXP0Edge = this.edges[ 0 ];

    for ( let i = 0; i < this.edges.length; i++ ) {
      const edge = this.edges[ i % this.edges.length ];

      if ( edge.p0.x.compareCrossMul( minimalXP0Edge.p0.x ) < 0 ) {
        minimalXP0Edge = edge;
      }

      const p0float = edge.p0float;
      const p1float = edge.p1float;

      bounds.addPoint( p0float );

      // PolygonIntegrals.evaluateShoelaceArea( p0.x, p0.y, p1.x, p1.y );
      signedArea += 0.5 * ( p1float.x + p0float.x ) * ( p1float.y - p0float.y );
    }

    this.minimalXRationalPoint = minimalXP0Edge.p0;
    this.bounds = bounds;
    this.signedArea = signedArea;
  }

  // TODO: just have a matrix?
  public toTransformedPolygon( scale = 1, translation = Vector2.ZERO ): Vector2[] {
    const result: Vector2[] = [];
    for ( let i = 0; i < this.edges.length; i++ ) {
      result.push( this.edges[ i ].p0float.timesScalar( scale ).plus( translation ) );
    }
    return result;
  }

  public toTransformedLinearEdges( scale = 1, translation = Vector2.ZERO ): LinearEdge[] {
    const result: LinearEdge[] = [];
    for ( let i = 0; i < this.edges.length; i++ ) {
      const edge = this.edges[ i ];
      result.push( new LinearEdge(
        edge.p0float.timesScalar( scale ).plus( translation ),
        edge.p1float.timesScalar( scale ).plus( translation )
      ) );
    }
    return result;
  }
}

scenery.register( 'RationalBoundary', RationalBoundary );
