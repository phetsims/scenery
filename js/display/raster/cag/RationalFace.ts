// Copyright 2023, University of Colorado Boulder

/**
 * Represents a face with a main (positive-oriented) boundary and zero or more (negative-oriented) holes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { EdgedFace, LinearEdge, PolygonalFace, RationalBoundary, RenderPath, RenderProgram, scenery, WindingMap } from '../../../imports.js';
import Bounds2 from '../../../../../dot/js/Bounds2.js';
import Matrix3 from '../../../../../dot/js/Matrix3.js';

export default class RationalFace {
  public readonly holes: RationalBoundary[] = [];
  public windingMapMap = new Map<RationalFace, WindingMap>();
  public windingMap: WindingMap | null = null;
  public inclusionSet: Set<RenderPath> = new Set<RenderPath>();
  public renderProgram: RenderProgram | null = null;

  public constructor( public readonly boundary: RationalBoundary ) {}

  public toPolygonalFace( matrix: Matrix3 ): PolygonalFace {
    return new PolygonalFace( [
      this.boundary.toTransformedPolygon( matrix ),
      ...this.holes.map( hole => hole.toTransformedPolygon( matrix ) )
    ] );
  }

  public toEdgedFace( matrix: Matrix3 ): EdgedFace {
    return new EdgedFace( this.toLinearEdges( matrix ) );
  }

  public toLinearEdges( matrix: Matrix3 ): LinearEdge[] {
    return [
      ...this.boundary.toTransformedLinearEdges( matrix ),
      ...this.holes.flatMap( hole => hole.toTransformedLinearEdges( matrix ) )
    ];
  }

  public getBounds( matrix: Matrix3 ): Bounds2 {
    const polygonalBounds = Bounds2.NOTHING.copy();
    polygonalBounds.includeBounds( this.boundary.bounds );
    for ( let i = 0; i < this.holes.length; i++ ) {
      polygonalBounds.includeBounds( this.holes[ i ].bounds );
    }
    polygonalBounds.transform( matrix );

    return polygonalBounds;
  }
}

scenery.register( 'RationalFace', RationalFace );
