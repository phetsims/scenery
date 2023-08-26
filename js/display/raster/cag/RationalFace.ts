// Copyright 2023, University of Colorado Boulder

/**
 * Represents a face with a main (positive-oriented) boundary and zero or more (negative-oriented) holes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { BigRational, EdgedFace, LinearEdge, PolygonalFace, RationalBoundary, RationalHalfEdge, RenderPath, RenderProgram, scenery, WindingMap } from '../../../imports.js';
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
    this.renderProgram = renderProgram.withPathInclusion( renderPath => inclusionSet.has( renderPath ) ).simplified();
  }

  public static traceBoundaries(
    filteredRationalHalfEdges: RationalHalfEdge[],
    innerBoundaries: RationalBoundary[],
    outerBoundaries: RationalBoundary[],
    faces: RationalFace[]
  ): void {
    for ( let i = 0; i < filteredRationalHalfEdges.length; i++ ) {
      const firstEdge = filteredRationalHalfEdges[ i ];
      if ( !firstEdge.boundary ) {
        const boundary = new RationalBoundary();
        boundary.edges.push( firstEdge );
        firstEdge.boundary = boundary;

        let edge = firstEdge.nextEdge!;
        while ( edge !== firstEdge ) {
          edge.boundary = boundary;
          boundary.edges.push( edge );
          edge = edge.nextEdge!;
        }

        boundary.computeProperties();

        const signedArea = boundary.signedArea;
        if ( Math.abs( signedArea ) > 1e-8 ) {
          if ( signedArea > 0 ) {
            innerBoundaries.push( boundary );
            const face = new RationalFace( boundary );
            faces.push( face );
            for ( let j = 0; j < boundary.edges.length; j++ ) {
              const edge = boundary.edges[ j ];
              edge.face = face;
            }
          }
          else {
            outerBoundaries.push( boundary );
          }
        }
      }
    }
  }

  // Returns the fully exterior boundary (should be singular, since we added the exterior rectangle)
  // NOTE: mutates faces order
  public static computeFaceHolesWithOrderedWindingNumbers(
    outerBoundaries: RationalBoundary[],
    faces: RationalFace[]
  ): RationalBoundary {
    let exteriorBoundary: RationalBoundary | null = null;

    // Sort faces by their exterior signed areas (when we find our point is included in the smallest face boundary,
    // we'll be able to exit without testing others).
    faces.sort( ( a, b ) => a.boundary.signedArea - b.boundary.signedArea );

    for ( let i = 0; i < outerBoundaries.length; i++ ) {
      const outerBoundary = outerBoundaries[ i ];
      const outerBounds = outerBoundary.bounds;
      const outerArea = -outerBoundary.signedArea;
      assert && assert( outerArea > 0 );

      const minimalRationalPoint = outerBoundary.minimalXRationalPoint;

      let found = false;

      for ( let j = 0; j < faces.length; j++ ) {
        const face = faces[ j ];
        const innerBoundary = face.boundary;

        // Skip through faces that are smaller than our hole
        if ( outerArea > innerBoundary.signedArea ) {
          continue;
        }

        const innerBounds = innerBoundary.bounds;

        // Check if the "inner" bounds actually fully contains (strictly) our "outer" bounds.
        // This is a constraint that has to be satisfied for the outer boundary to be a hole.
        if (
          outerBounds.minX > innerBounds.minX &&
          outerBounds.minY > innerBounds.minY &&
          outerBounds.maxX < innerBounds.maxX &&
          outerBounds.maxY < innerBounds.maxY
        ) {
          // Now we check for inclusion!
          if ( innerBoundary.containsPoint( minimalRationalPoint ) ) {
            // FOUND IT!
            face.holes.push( outerBoundary );

            // Fill in face data for holes, so we can traverse nicely
            for ( let k = 0; k < outerBoundary.edges.length; k++ ) {
              outerBoundary.edges[ k ].face = face;
            }

            found = true;
            break;
          }
        }
      }

      // We should only find one exterior boundary
      if ( !found ) {
        assert && assert( !exteriorBoundary );
        exteriorBoundary = outerBoundary;
      }
    }

    assert && assert( exteriorBoundary );
    return exteriorBoundary!;
  }

  // Returns the fully exterior boundary (should be singular, since we added the exterior rectangle)
  // TODO: DOUBTS on the correctness of this, the filtering of boundaries seems sketchy. Probably not as high-performance
  // TODO: BUT perhaps it is more parallelizable?
  public static computeFaceHoles(
    integerBounds: Bounds2,
    outerBoundaries: RationalBoundary[],
    faces: RationalFace[],
    faceHoleLog: FaceHoleLog | null = null
  ): RationalBoundary {
    let exteriorBoundary: RationalBoundary | null = null;

    for ( let i = 0; i < outerBoundaries.length; i++ ) {
      const outerBoundary = outerBoundaries[ i ];
      const outerBounds = outerBoundary.bounds;

      const logEntry: FaceHoleLogEntry | null = faceHoleLog ? {
        outerBoundary: outerBoundary
      } : null;
      if ( logEntry ) {
        faceHoleLog!.entries.push( logEntry );
      }

      const minimalRationalPoint = outerBoundary.minimalXRationalPoint;

      let maxIntersectionX = new BigRational( integerBounds.left - 1, 1 );
      let maxIntersectionEdge: RationalHalfEdge | null = null;
      let maxIntersectionIsVertex = false;

      for ( let j = 0; j < faces.length; j++ ) {
        const face = faces[ j ];
        const innerBoundary = face.boundary;
        const innerBounds = innerBoundary.bounds;

        // Check if the "inner" bounds actually fully contains (strictly) our "outer" bounds.
        // This is a constraint that has to be satisfied for the outer boundary to be a hole.
        if (
          outerBounds.minX > innerBounds.minX &&
          outerBounds.minY > innerBounds.minY &&
          outerBounds.maxX < innerBounds.maxX &&
          outerBounds.maxY < innerBounds.maxY
        ) {
          for ( let k = 0; k < innerBoundary.edges.length; k++ ) {
            const edge = innerBoundary.edges[ k ];

            // TODO: This will require a lot of precision, how do we handle this?
            // TODO: we'll need to handle these anyway!
            const dx0 = edge.p0.x.minus( minimalRationalPoint.x );
            const dx1 = edge.p1.x.minus( minimalRationalPoint.x );

            // If both x values of the segment are at or to the right, there will be no intersection
            if ( dx0.isNegative() || dx1.isNegative() ) {

              const dy0 = edge.p0.y.minus( minimalRationalPoint.y );
              const dy1 = edge.p1.y.minus( minimalRationalPoint.y );

              const bothPositive = dy0.isPositive() && dy1.isPositive();
              const bothNegative = dy0.isNegative() && dy1.isNegative();

              if ( !bothPositive && !bothNegative ) {
                const isZero0 = dy0.isZero();
                const isZero1 = dy1.isZero();

                let candidateMaxIntersectionX: BigRational;
                let isVertex: boolean;
                if ( isZero0 && isZero1 ) {
                  // NOTE: on a vertex
                  const is0Less = edge.p0.x.compareCrossMul( edge.p1.x ) < 0;
                  candidateMaxIntersectionX = is0Less ? edge.p1.x : edge.p0.x;
                  isVertex = true;
                }
                else if ( isZero0 ) {
                  // NOTE: on a vertex
                  candidateMaxIntersectionX = edge.p0.x;
                  isVertex = true;
                }
                else if ( isZero1 ) {
                  // NOTE: on a vertex
                  candidateMaxIntersectionX = edge.p1.x;
                  isVertex = true;
                }
                else {
                  // p0.x + ( p1.x - p0.x ) * ( minimalRationalPoint.y - p0.y ) / ( p1.y - p0.y );
                  // TODO: could simplify by reversing sign and using dy1
                  candidateMaxIntersectionX = edge.p0.x.plus( edge.p1.x.minus( edge.p0.x ).times( minimalRationalPoint.y.minus( edge.p0.y ) ).dividedBy( edge.p1.y.minus( edge.p0.y ) ) );
                  isVertex = false;
                }

                // TODO: add less-than, etc.
                if (
                  maxIntersectionX.compareCrossMul( candidateMaxIntersectionX ) < 0 &&
                  // NOTE: Need to ensure that our actual intersection is to the left of our minimal point!!!
                  candidateMaxIntersectionX.compareCrossMul( minimalRationalPoint.x ) < 0
                ) {
                  maxIntersectionX = candidateMaxIntersectionX;
                  maxIntersectionEdge = edge;
                  maxIntersectionIsVertex = isVertex;
                }
              }
            }
          }
        }
      }

      if ( logEntry ) {
        logEntry.maxIntersectionX = maxIntersectionX;
        logEntry.maxIntersectionEdge = maxIntersectionEdge;
        logEntry.maxIntersectionIsVertex = maxIntersectionIsVertex;
      }

      let connectedFace: RationalFace | null = null;
      if ( maxIntersectionEdge ) {
        const edge0 = maxIntersectionEdge;
        const edge1 = maxIntersectionEdge.reversed;
        if ( !edge0.face ) {
          connectedFace = edge1.face!;
        }
        else if ( !edge1.face ) {
          connectedFace = edge0.face!;
        }
        else if ( maxIntersectionIsVertex ) {
          // We'll need to traverse around the vertex to find the face we need.

          // Get a starting edge with p0 = intersection
          const startEdge = ( edge0.p0.x.equalsCrossMul( maxIntersectionX ) && edge0.p0.y.equalsCrossMul( minimalRationalPoint.y ) ) ? edge0 : edge1;

          assert && assert( startEdge.p0.x.equalsCrossMul( maxIntersectionX ) );
          assert && assert( startEdge.p0.y.equalsCrossMul( minimalRationalPoint.y ) );

          // TODO: for testing this, remember we'll need multiple "fully surrounding" boundaries?
          // TODO: wait, no we won't
          let bestEdge = startEdge;
          let edge = startEdge.previousEdge!.reversed;
          while ( edge !== startEdge ) {
            if ( edge.compare( bestEdge ) < 0 ) {
              bestEdge = edge;
            }
            edge = edge.previousEdge!.reversed;
          }
          connectedFace = edge.face!; // TODO: why do we NOT reverse it here?!? reversed issues?
        }
        else {
          // non-vertex, a bit easier
          // TODO: could grab this value stored from earlier
          const isP0YLess = edge0.p0.y.compareCrossMul( edge0.p1.y ) < 0;
          // Because it should have a "positive" orientation, we want the "negative-y-facing edge"
          connectedFace = isP0YLess ? edge1.face : edge0.face;
        }

        assert && assert( connectedFace );
        connectedFace.holes.push( outerBoundary );

        // Fill in face data for holes, so we can traverse nicely
        for ( let k = 0; k < outerBoundary.edges.length; k++ ) {
          outerBoundary.edges[ k ].face = connectedFace;
        }
      }
      else {
        exteriorBoundary = outerBoundary;
      }

      if ( logEntry ) {
        logEntry.connectedFace = connectedFace;
      }
    }

    assert && assert( exteriorBoundary );

    return exteriorBoundary!;
  }

  public static createUnboundedFace( exteriorBoundary: RationalBoundary ): RationalFace {
    const unboundedFace = new RationalFace( exteriorBoundary );

    for ( let i = 0; i < exteriorBoundary.edges.length; i++ ) {
      exteriorBoundary.edges[ i ].face = unboundedFace;
    }
    return unboundedFace;
  }

  public static computeWindingMaps( filteredRationalHalfEdges: RationalHalfEdge[], unboundedFace: RationalFace ): void {
    for ( let i = 0; i < filteredRationalHalfEdges.length; i++ ) {
      const edge = filteredRationalHalfEdges[ i ];

      const face = edge.face!;
      const otherFace = edge.reversed.face!;

      assert && assert( face );
      assert && assert( otherFace );

      // TODO: possibly reverse this, check to see which winding map is correct
      if ( !face.windingMapMap.has( otherFace ) ) {
        face.windingMapMap.set( otherFace, edge.windingMap );
      }
    }

    unboundedFace.windingMap = new WindingMap(); // no windings, empty!
    const recursiveWindingMap = ( solvedFace: RationalFace ) => {
      // TODO: no recursion, could blow recursion limits
      for ( const [ otherFace, windingMap ] of solvedFace.windingMapMap ) {
        const needsNewWindingMap = !otherFace.windingMap;

        if ( needsNewWindingMap || assert ) {
          const newWindingMap = new WindingMap();
          const existingMap = solvedFace.windingMap!;
          const deltaMap = windingMap;

          newWindingMap.addWindingMap( existingMap );
          newWindingMap.addWindingMap( deltaMap );

          if ( assert ) {
            // TODO: object for the winding map?
          }
          otherFace.windingMap = newWindingMap;

          if ( needsNewWindingMap ) {
            recursiveWindingMap( otherFace );
          }
        }
      }
    };
    recursiveWindingMap( unboundedFace );
  }
}

scenery.register( 'RationalFace', RationalFace );

export type FaceHoleLog = {
  entries: FaceHoleLogEntry[];
};

export type FaceHoleLogEntry = {
  outerBoundary: RationalBoundary;
  maxIntersectionX?: BigRational;
  maxIntersectionEdge?: RationalHalfEdge | null;
  maxIntersectionIsVertex?: boolean;
  connectedFace?: RationalFace | null;
};
