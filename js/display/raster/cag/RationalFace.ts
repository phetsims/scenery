// Copyright 2023, University of Colorado Boulder

/**
 * Represents a face with a main (positive-oriented) boundary and zero or more (negative-oriented) holes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { EdgedFace, LinearEdge, PolygonalFace, RationalBoundary, RenderProgram, scenery, WindingMap } from '../../../imports.js';
import { RenderPath } from '../render-program/RenderProgram.js';
import Bounds2 from '../../../../../dot/js/Bounds2.js';
import Vector2 from '../../../../../dot/js/Vector2.js';

export default class RationalFace {
  public readonly holes: RationalBoundary[] = [];
  public windingMapMap = new Map<RationalFace, WindingMap>();
  public windingMap: WindingMap | null = null;
  public inclusionSet: Set<RenderPath> = new Set<RenderPath>();
  public renderProgram: RenderProgram | null = null;

  public constructor( public readonly boundary: RationalBoundary ) {}

  public toPolygonalFace( inverseScale = 1, translation: Vector2 = Vector2.ZERO ): PolygonalFace {
    return new PolygonalFace( [
      this.boundary.toTransformedPolygon( inverseScale, translation ),
      ...this.holes.map( hole => hole.toTransformedPolygon( inverseScale, translation ) )
    ] );
  }

  public toEdgedFace( inverseScale = 1, translation: Vector2 = Vector2.ZERO ): EdgedFace {
    return new EdgedFace( this.toLinearEdges( inverseScale, translation ) );
  }

  public toLinearEdges( inverseScale = 1, translation: Vector2 = Vector2.ZERO ): LinearEdge[] {
    return [
      ...this.boundary.toTransformedLinearEdges( inverseScale, translation ),
      ...this.holes.flatMap( hole => hole.toTransformedLinearEdges( inverseScale, translation ) )
    ];
  }

  public getBounds( inverseScale = 1, translation: Vector2 = Vector2.ZERO ): Bounds2 {
    const polygonalBounds = Bounds2.NOTHING.copy();
    polygonalBounds.includeBounds( this.boundary.bounds );
    for ( let i = 0; i < this.holes.length; i++ ) {
      polygonalBounds.includeBounds( this.holes[ i ].bounds );
    }
    polygonalBounds.minX = polygonalBounds.minX * inverseScale + translation.x;
    polygonalBounds.minY = polygonalBounds.minY * inverseScale + translation.y;
    polygonalBounds.maxX = polygonalBounds.maxX * inverseScale + translation.x;
    polygonalBounds.maxY = polygonalBounds.maxY * inverseScale + translation.y;

    return polygonalBounds;
  }
}

scenery.register( 'RationalFace', RationalFace );
