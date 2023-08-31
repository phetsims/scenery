// Copyright 2023, University of Colorado Boulder

/**
 * Represents a face with a main (positive-oriented) boundary and zero or more (negative-oriented) holes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { BigRational, ClipSimplifier, EdgedFace, LinearEdge, PolygonalFace, RationalBoundary, RationalHalfEdge, RenderPath, RenderProgram, scenery, WindingMap } from '../../../imports.js';
import Bounds2 from '../../../../../dot/js/Bounds2.js';
import Matrix3 from '../../../../../dot/js/Matrix3.js';
import Vector2 from '../../../../../dot/js/Vector2.js';

const traceSimplifier = new ClipSimplifier();

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

  // NOTE: Returns better-filtered rational half edges, with zero-area boundaries removed
  public static traceBoundaries(
    filteredRationalHalfEdges: RationalHalfEdge[],
    innerBoundaries: RationalBoundary[],
    outerBoundaries: RationalBoundary[],
    faces: RationalFace[]
  ): RationalHalfEdge[] {
    const zeroAreaEdges: RationalHalfEdge[] = [];

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
        else {
          for ( let j = 0; j < boundary.edges.length; j++ ) {
            zeroAreaEdges.push( boundary.edges[ j ] );
          }
        }
      }
    }

    // TODO: a better way of filtering out zero-area boundaries beforehand?
    return filteredRationalHalfEdges.filter( edge => !zeroAreaEdges.includes( edge ) );
  }

  // Returns the fully exterior boundaries (should be singular in the rendering case, since we added the exterior,
  // rectangle, HOWEVER can be multiples otherwise)
  // NOTE: mutates faces order
  public static computeFaceHolesWithOrderedWindingNumbers(
    outerBoundaries: RationalBoundary[],
    faces: RationalFace[]
  ): RationalBoundary[] {
    const exteriorBoundaries: RationalBoundary[] = [];

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
        exteriorBoundaries.push( outerBoundary );
      }
    }

    assert && assert( exteriorBoundaries.length );
    return exteriorBoundaries;
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

  public static createUnboundedFace( ...exteriorBoundaries: RationalBoundary[] ): RationalFace {
    const unboundedFace = new RationalFace( new RationalBoundary() );

    for ( let i = 0; i < exteriorBoundaries.length; i++ ) {
      const boundary = exteriorBoundaries[ i ];

      unboundedFace.holes.push( boundary );

      for ( let j = 0; j < boundary.edges.length; j++ ) {
        boundary.edges[ j ].face = unboundedFace;
      }
    }

    unboundedFace.holes.push( ...exteriorBoundaries );

    return unboundedFace;
  }

  /**
   * Computes winding maps for all of the faces
   */
  public static computeWindingMaps( filteredRationalHalfEdges: RationalHalfEdge[], unboundedFace: RationalFace ): void {

    // TODO: how can we do this more efficiently? (prevent double scan, and can we avoid computing/visiting some edges?)
    // Scan through the edges, and detect adjacent faces (storing the winding map difference between faces).
    for ( let i = 0; i < filteredRationalHalfEdges.length; i++ ) {
      const edge = filteredRationalHalfEdges[ i ];

      const face = edge.face!;
      const otherFace = edge.reversed.face!;

      assert && assert( face );
      assert && assert( otherFace );

      if ( !face.windingMapMap.has( otherFace ) ) {
        face.windingMapMap.set( otherFace, edge.reversed.windingMap );
      }
    }

    unboundedFace.windingMap = new WindingMap(); // no windings, empty!

    // Iterate through adjacent faces until we've reached everything (... everything should be adjacent, since our
    // unbounded face connects every outer boundary).
    const pendingFaces: RationalFace[] = [ unboundedFace ];
    while ( pendingFaces.length ) {
      const solvedFace = pendingFaces.pop()!;

      for ( const [ otherFace, windingMap ] of solvedFace.windingMapMap ) {
        const needsNewWindingMap = !otherFace.windingMap;

        if ( needsNewWindingMap ) {
          const newWindingMap = new WindingMap();
          const existingMap = solvedFace.windingMap!;
          const deltaMap = windingMap;

          // New winding map is a combination of our existing map and the delta map (based on the edge)
          newWindingMap.addWindingMap( existingMap );
          newWindingMap.addWindingMap( deltaMap );

          otherFace.windingMap = newWindingMap;

          pendingFaces.push( otherFace );
        }
      }
    }
  }

  /**
   * Combines faces that have equivalent face data IFF they border each other (leaving separate programs with
   * equivalent face data separate if they don't border). It will also remove edges that border between faces
   * that we combine, and will connect edges to keep things polygonal!
   */
  public static traceCombineFaces<FaceData, OutputFace>(
    faces: RationalFace[],
    fromIntegerMatrix: Matrix3,
    getFaceData: ( face: RationalFace ) => FaceData,
    createOutputFace: ( face: PolygonalFace, faceData: FaceData, bounds: Bounds2 ) => OutputFace,
    // null is for the unbounded face
    isFaceDataCompatible: ( faceData1: FaceData, faceData2: FaceData | null ) => boolean
  ): OutputFace[] {

    // In summary, we'll find an edge between incompatible faces, and then we'll trace that edge (staying only on edges
    // between incompatible faces) until we get back to the starting edge. Once we've done this, we have constructed one
    // polygon.
    //
    // For this algorithm, we mark faces and edges as processed when we've already handled their contribution (or marked
    // them to contribute).
    //
    // We'll naturally run into "compatible" faces as we trace the edges. Whenever we run into a new compatible face,
    // (in isFaceCompatible), we'll add its edges to the pool of edges to trace for this OutputFace.
    //
    // When we trace edges, we'll always start with one between "incompatible" faces. If its nextEdge is between
    // compatible faces, we can skip it (and use the winged edge data structure to try the "next" edge). Thus we'll
    // skip visiting edges between compatible faces. IF they are all compatible, we'll end back up to our starting
    // edge's reversed edge, and we'll essentially trace backward. If this happens, our simplifier will remove the
    // evidence of this. So it should handle "degenerate" cases (e.g. same face on both sides, or a vertex with only
    // one incident edge) just fine.

    const outputFaces: OutputFace[] = [];

    if ( assert ) {
      // Ensure that things aren't marked as processed yet
      for ( let i = 0; i < faces.length; i++ ) {
        assert( !faces[ i ].processed );
        assert( faces[ i ].getEdges().every( edge => !edge.processed ) );
      }
    }

    // Compute face data up front
    const dataMap = new Map<RationalFace, FaceData>();
    for ( let i = 0; i < faces.length; i++ ) {
      dataMap.set( faces[ i ], getFaceData( faces[ i ] ) );
    }

    for ( let i = 0; i < faces.length; i++ ) {
      const startingFace = faces[ i ];

      if ( !startingFace.processed ) {
        startingFace.processed = true;

        // A list of polygons we'll append into (for our OutputFace).
        const polygons: Vector2[][] = [];

        // A list of edges remaining to process. NOTE: some of these may be marked as "processed", we will just ignore
        // those. Any time we run across a new compatible face, we'll dump its edges in here.
        const edges: RationalHalfEdge[] = [
          ...startingFace.getEdges() // defensive copy, could remove sometime
        ];

        // We'll need to pass bounds to the OutputFace constructor, we'll accumulate them here.
        const bounds = startingFace.getBounds( fromIntegerMatrix ).copy(); // we'll mutate this

        // All RenderPrograms should be equivalent, so we'll just use the first one
        const faceData = dataMap.get( startingFace )!;

        // Cache whether faces or compatible or not
        const compatibleFaces = new Set<RationalFace>();
        const incompatibleFaces = new Set<RationalFace>();
        compatibleFaces.add( startingFace );

        // NOTE: side effects!
        const isFaceCompatible = ( candidateFace: RationalFace ): boolean => {
          if ( compatibleFaces.has( candidateFace ) ) {
            return true;
          }
          else if ( incompatibleFaces.has( candidateFace ) ) {
            return false;
          }
          else {
            const candidateFaceData = dataMap.get( candidateFace ) || null;
            // Not in either place, we need to test (also, the unbounded face won't have a RenderProgram)
            if ( isFaceDataCompatible( faceData, candidateFaceData ) ) {
              // ADD it to the current renderable face
              assert && assert( !candidateFace.processed, 'We should have already found this' );
              candidateFace.processed = true;
              bounds.includeBounds( candidateFace.getBounds( fromIntegerMatrix ) );
              edges.push( ...candidateFace.getEdges() );

              compatibleFaces.add( candidateFace );
              return true;
            }
            else {
              incompatibleFaces.add( candidateFace );
              return false;
            }
          }
        };

        // We'll have edges appended sometimes in isFaceCompatible() checks.
        while ( edges.length ) {
          const startingEdge = edges.pop()!;

          // If the edge is processed OR both faces are compatible, we'll just skip it anyway. We don't want to start
          // tracing on a loop that we will completely remove
          if ( !startingEdge.processed && ( !isFaceCompatible( startingEdge.face! ) || !isFaceCompatible( startingEdge.reversed.face! ) ) ) {
            // Start an edge trace!

            // We'll use the simplifier to remove duplicate or walked-back points.
            // TODO: check to see if removing arbitrary collinear points helps us a lot here. It might be good, but
            // TODO: we don't want to introduce a lot of error. Probably is additional cost

            // Add the first edge
            let currentEdge = startingEdge;
            traceSimplifier.addTransformed( fromIntegerMatrix, currentEdge.p0float.x, currentEdge.p0float.y );
            currentEdge.processed = true;

            do {
              // Walk edges
              let nextEdge = currentEdge.nextEdge!;
              assert && assert( nextEdge.face === currentEdge.face );

              // BUT don't walk edges that are between compatible faces. Instead, wind around until we find either
              // an incompatible face, OR we walk back our edge (the simplifier will take care of removing this)
              while ( nextEdge !== currentEdge.reversed && isFaceCompatible( nextEdge.reversed.face! ) ) {
                nextEdge = nextEdge.reversed.nextEdge!;
              }

              assert && assert( isFaceCompatible( nextEdge.face! ) || isFaceCompatible( nextEdge.reversed.face! ) );

              // Add subsequent edges
              currentEdge = nextEdge;
              traceSimplifier.addTransformed( fromIntegerMatrix, currentEdge.p0float.x, currentEdge.p0float.y );
              currentEdge.processed = true;
            } while ( currentEdge !== startingEdge );

            const polygon = traceSimplifier.finalize();
            if ( polygon.length >= 3 ) {
              polygons.push( polygon );
            }
          }
        }

        outputFaces.push( createOutputFace(
          new PolygonalFace( polygons ),
          faceData,
          bounds
        ) );
      }
    }

    return outputFaces;
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
