// Copyright 2023, University of Colorado Boulder

/**
 * Represents a face with a main (positive-oriented) boundary and zero or more (negative-oriented) holes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { EdgedFace, LinearEdge, PolygonalFace, RationalBoundary, RationalHalfEdge, RenderPath, RenderProgram, scenery, WindingMap } from '../../../imports.js';
import Bounds2 from '../../../../../dot/js/Bounds2.js';
import Matrix3 from '../../../../../dot/js/Matrix3.js';

export default class RationalFace {
  public readonly holes: RationalBoundary[] = [];
  public windingMapMap = new Map<RationalFace, WindingMap>();
  public windingMap: WindingMap | null = null;
  public renderProgram: RenderProgram | null = null;
  public processed = false;

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

  public getEdges(): RationalHalfEdge[] {
    // TODO: create less garbage with iteration?
    return [
      ...this.boundary.edges,
      ...this.holes.flatMap( hole => hole.edges )
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

  /**
   * Returns a set of render paths that are included in this face (based on winding numbers).
   *
   * REQUIRED: Should have computed the winding map first.
   */
  public getIncludedRenderPaths(): Set<RenderPath> {
    const inclusionSet = new Set<RenderPath>();

    for ( const renderPath of this.windingMap!.map.keys() ) {
      const windingNumber = this.windingMap!.getWindingNumber( renderPath );
      const included = renderPath.fillRule === 'nonzero' ? windingNumber !== 0 : windingNumber % 2 !== 0;
      if ( included ) {
        inclusionSet.add( renderPath );
      }
    }

    return inclusionSet;
  }

  public postWindingRenderProgram( renderProgram: RenderProgram ): void {
    const inclusionSet = this.getIncludedRenderPaths();

    // TODO: for extracting a non-render-program based setup, can we create a new class here?
    this.renderProgram = renderProgram.simplify( renderPath => inclusionSet.has( renderPath ) );
  }
}

scenery.register( 'RationalFace', RationalFace );
