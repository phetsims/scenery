// Copyright 2023, University of Colorado Boulder

/**
 * Test rasterization
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { BigRational, ClippableFace, CombinedRaster, EdgedFace, IntegerEdge, LinearEdge, LineIntersector, LineSplitter, OutputRaster, PolygonClipping, RationalBoundary, RationalFace, RationalHalfEdge, RenderableFace, RenderColor, RenderPath, RenderPathProgram, RenderProgram, scenery, WindingMap } from '../../../imports.js';
import Bounds2 from '../../../../../dot/js/Bounds2.js';
import Utils from '../../../../../dot/js/Utils.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import IntentionalAny from '../../../../../phet-core/js/types/IntentionalAny.js';
import Vector4 from '../../../../../dot/js/Vector4.js';
import { optionize3 } from '../../../../../phet-core/js/optionize.js';

export type RasterizationOptions = {

  edgeIntersectionMethod?: 'quadratic' | 'boundsTree' | 'arrayBoundsTree';

  renderableFaceMethod?: 'polygonal' | 'edged' | 'fullyCombined' | 'simplifyingCombined';

  splitLinearGradients?: boolean;
  splitRadialGradients?: boolean;
};

const DEFAULT_OPTIONS = {
  edgeIntersectionMethod: 'arrayBoundsTree',
  renderableFaceMethod: 'simplifyingCombined',
  splitLinearGradients: true,
  splitRadialGradients: true
} as const;

let debugData: Record<string, IntentionalAny> | null = null;

const scratchFullAreaVector = new Vector2( 0, 0 );

class AccumulatingFace {
  public faces = new Set<RationalFace>();
  public facesToProcess: RationalFace[] = [];
  public renderProgram: RenderProgram | null = null;
  public bounds: Bounds2 = Bounds2.NOTHING.copy();
  public clippedEdges: LinearEdge[] = [];
}

export default class Rasterize {

  private static clipScaleToIntegerEdges( paths: RenderPath[], bounds: Bounds2, scale: number ): IntegerEdge[] {

    const translation = new Vector2( -bounds.minX, -bounds.minY );

    const integerEdges: IntegerEdge[] = [];
    for ( let i = 0; i < paths.length; i++ ) {
      const path = paths[ i ];

      for ( let j = 0; j < path.subpaths.length; j++ ) {
        const subpath = path.subpaths[ j ];
        const clippedSubpath = PolygonClipping.boundsClipPolygon( subpath, bounds );

        for ( let k = 0; k < clippedSubpath.length; k++ ) {
          // TODO: when micro-optimizing, improve this pattern so we only have one access each iteration
          const p0 = clippedSubpath[ k ];
          const p1 = clippedSubpath[ ( k + 1 ) % clippedSubpath.length ];
          const edge = IntegerEdge.fromUnscaledPoints( path, scale, translation, p0, p1 );
          if ( edge !== null ) {
            integerEdges.push( edge );
          }
        }
      }
    }
    return integerEdges;
  }

  private static filterAndConnectHalfEdges( rationalHalfEdges: RationalHalfEdge[] ): RationalHalfEdge[] {
    // Do filtering for duplicate half-edges AND connecting edge linked list in the same traversal
    // NOTE: We don't NEED to filter "low-order" vertices (edge whose opposite is its next edge), but we could at
    // some point in the future. Note that removing a low-order edge then might create ANOTHER low-order edge, so
    // it would need to chase these.
    // NOTE: We could also remove "composite" edges that have no winding contribution (degenerate "touching" in the
    // source path), however it's probably not too common so it's not done here.
    let firstEdge = rationalHalfEdges[ 0 ];
    let lastEdge = rationalHalfEdges[ 0 ];
    const filteredRationalHalfEdges = [ lastEdge ];
    for ( let i = 1; i < rationalHalfEdges.length; i++ ) {
      const edge = rationalHalfEdges[ i ];

      if ( edge.p0.equals( lastEdge.p0 ) ) {
        if ( edge.p1.equals( lastEdge.p1 ) ) {
          lastEdge.addWindingFrom( edge );
        }
        else {
          filteredRationalHalfEdges.push( edge );
          edge.reversed.nextEdge = lastEdge;
          lastEdge.previousEdge = edge.reversed;
          lastEdge = edge;
        }
      }
      else {
        firstEdge.reversed.nextEdge = lastEdge;
        lastEdge.previousEdge = firstEdge.reversed;
        filteredRationalHalfEdges.push( edge );
        firstEdge = edge;
        lastEdge = edge;
      }
    }
    // last connection
    firstEdge.reversed.nextEdge = lastEdge;
    lastEdge.previousEdge = firstEdge.reversed;
    return filteredRationalHalfEdges;
  }

  private static traceBoundaries(
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
  private static computeFaceHoles(
    integerBounds: Bounds2,
    outerBoundaries: RationalBoundary[],
    faces: RationalFace[]
  ): RationalBoundary {
    let exteriorBoundary: RationalBoundary | null = null;
    if ( assert ) {
      debugData!.exteriorBoundary = exteriorBoundary;
    }
    for ( let i = 0; i < outerBoundaries.length; i++ ) {
      const outerBoundary = outerBoundaries[ i ];
      const outerBounds = outerBoundary.bounds;

      const boundaryDebugData: IntentionalAny = assert ? {
        outerBoundary: outerBoundary
      } : null;
      if ( assert ) {
        debugData!.boundaryDebugData = debugData!.boundaryDebugData || [];
        debugData!.boundaryDebugData.push( boundaryDebugData );
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

      if ( assert ) {
        boundaryDebugData.maxIntersectionX = maxIntersectionX;
        boundaryDebugData.maxIntersectionEdge = maxIntersectionEdge;
        boundaryDebugData.maxIntersectionIsVertex = maxIntersectionIsVertex;
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

      if ( assert ) {
        boundaryDebugData.connectedFace = connectedFace;
      }
    }

    assert && assert( exteriorBoundary );

    return exteriorBoundary!;
  }

  private static createUnboundedFace( exteriorBoundary: RationalBoundary ): RationalFace {
    const unboundedFace = new RationalFace( exteriorBoundary );

    for ( let i = 0; i < exteriorBoundary.edges.length; i++ ) {
      exteriorBoundary.edges[ i ].face = unboundedFace;
    }
    return unboundedFace;
  }

  private static computeWindingMaps( filteredRationalHalfEdges: RationalHalfEdge[], unboundedFace: RationalFace ): void {
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

  private static getRenderProgrammedFaces( renderProgram: RenderProgram, faces: RationalFace[] ): RationalFace[] {
    const renderProgrammedFaces: RationalFace[] = [];

    for ( let i = 0; i < faces.length; i++ ) {
      const face = faces[ i ];

      face.inclusionSet = new Set<RenderPath>();
      for ( const renderPath of face.windingMap!.map.keys() ) {
        const windingNumber = face.windingMap!.getWindingNumber( renderPath );
        const included = renderPath.fillRule === 'nonzero' ? windingNumber !== 0 : windingNumber % 2 !== 0;
        if ( included ) {
          face.inclusionSet.add( renderPath );
        }
      }
      const faceRenderProgram = renderProgram.simplify( renderPath => face.inclusionSet.has( renderPath ) );
      face.renderProgram = faceRenderProgram;

      // Drop faces that will be fully transparent
      const isFullyTransparent = faceRenderProgram instanceof RenderColor && faceRenderProgram.color.w <= 1e-8;

      if ( !isFullyTransparent ) {
        renderProgrammedFaces.push( face );
      }
    }

    return renderProgrammedFaces;
  }

  private static toPolygonalRenderableFaces(
    faces: RationalFace[],
    scale: number,
    translation: Vector2
  ): RenderableFace[] {

    // TODO: naming with above!!
    const renderableFaces: RenderableFace[] = [];
    for ( let i = 0; i < faces.length; i++ ) {
      const face = faces[ i ];
      renderableFaces.push( new RenderableFace(
        face.toPolygonalFace( 1 / scale, translation ),
        face.renderProgram!,
        face.getBounds( 1 / scale, translation )
      ) );
    }
    return renderableFaces;
  }

  private static toEdgedRenderableFaces(
    faces: RationalFace[],
    scale: number,
    translation: Vector2
  ): RenderableFace[] {

    // TODO: naming with above!!
    const renderableFaces: RenderableFace[] = [];
    for ( let i = 0; i < faces.length; i++ ) {
      const face = faces[ i ];
      renderableFaces.push( new RenderableFace(
        face.toEdgedFace( 1 / scale, translation ),
        face.renderProgram!,
        face.getBounds( 1 / scale, translation )
      ) );
    }
    return renderableFaces;
  }

  private static toFullyCombinedRenderableFaces(
    faces: RationalFace[],
    scale: number,
    translation: Vector2
  ): RenderableFace[] {

    const faceEquivalenceClasses: Set<RationalFace>[] = [];

    for ( let i = 0; i < faces.length; i++ ) {
      const face = faces[ i ];
      let found = false;

      for ( let j = 0; j < faceEquivalenceClasses.length; j++ ) {
        const faceEquivalenceClass = faceEquivalenceClasses[ j ];
        const representative: RationalFace = faceEquivalenceClass.values().next().value;
        if ( face.renderProgram!.equals( representative.renderProgram! ) ) {
          faceEquivalenceClass.add( face );
          found = true;
          break;
        }
      }

      if ( !found ) {
        const newSet = new Set<RationalFace>();
        newSet.add( face );
        faceEquivalenceClasses.push( newSet );
      }
    }

    const inverseScale = 1 / scale;

    const renderableFaces: RenderableFace[] = [];
    for ( let i = 0; i < faceEquivalenceClasses.length; i++ ) {
      const faces = faceEquivalenceClasses[ i ];

      const clippedEdges: LinearEdge[] = [];
      let renderProgram: RenderProgram | null = null;
      const bounds = Bounds2.NOTHING.copy();

      for ( const face of faces ) {
        renderProgram = face.renderProgram!;
        bounds.includeBounds( face.getBounds( inverseScale, translation ) );

        for ( const boundary of [
          face.boundary,
          ...face.holes
        ] ) {
          for ( const edge of boundary.edges ) {
            if ( !faces.has( edge.reversed.face! ) ) {
              clippedEdges.push( new LinearEdge(
                edge.p0float.timesScalar( inverseScale ).plus( translation ),
                edge.p1float.timesScalar( inverseScale ).plus( translation )
              ) );
            }
          }
        }
      }

      renderableFaces.push( new RenderableFace( new EdgedFace( clippedEdges ), renderProgram!, bounds ) );
    }

    return renderableFaces;
  }

  // Will combine faces that have equivalent RenderPrograms IFF they border each other (leaving separate programs with
  // equivalent RenderPrograms separate if they don't border). It will also remove edges that border between faces
  // that we combine (thus switching to EdgedFaces with unsorted edges).
  private static toSimplifyingCombinedRenderableFaces(
    faces: RationalFace[],
    scale: number,
    translation: Vector2
  ): RenderableFace[] {

    const inverseScale = 1 / scale;

    const accumulatedFaces: AccumulatingFace[] = [];

    // TODO: see if we need micro-optimizations here
    faces.forEach( face => {
      if ( accumulatedFaces.every( accumulatedFace => !accumulatedFace.faces.has( face ) ) ) {
        const newAccumulatedFace = new AccumulatingFace();
        newAccumulatedFace.faces.add( face );
        newAccumulatedFace.facesToProcess.push( face );
        newAccumulatedFace.renderProgram = face.renderProgram!;
        newAccumulatedFace.bounds.includeBounds( face.getBounds( inverseScale, translation ) );

        const incompatibleFaces = new Set<RationalFace>();

        // NOTE: side effects!
        const isFaceCompatible = ( face: RationalFace ): boolean => {
          if ( incompatibleFaces.has( face ) ) {
            return false;
          }
          if ( newAccumulatedFace.faces.has( face ) ) {
            return true;
          }

          // Not in either place, we need to test
          if ( face.renderProgram && newAccumulatedFace.renderProgram!.equals( face.renderProgram ) ) {
            newAccumulatedFace.faces.add( face );
            newAccumulatedFace.facesToProcess.push( face );
            newAccumulatedFace.bounds.includeBounds( face.getBounds( inverseScale, translation ) );
            return true;
          }
          else {
            incompatibleFaces.add( face );
            return false;
          }
        };

        accumulatedFaces.push( newAccumulatedFace );

        while ( newAccumulatedFace.facesToProcess.length ) {
          const faceToProcess = newAccumulatedFace.facesToProcess.pop()!;

          for ( const boundary of [
            faceToProcess.boundary,
            ...faceToProcess.holes
          ] ) {
            for ( const edge of boundary.edges ) {
              if ( !isFaceCompatible( edge.reversed.face! ) ) {
                newAccumulatedFace.clippedEdges.push( new LinearEdge(
                  edge.p0float.timesScalar( inverseScale ).plus( translation ),
                  edge.p1float.timesScalar( inverseScale ).plus( translation )
                ) );
              }
            }
          }
        }
      }
    } );

    return accumulatedFaces.map( accumulatedFace => new RenderableFace(
      new EdgedFace( accumulatedFace.clippedEdges ),
      accumulatedFace.renderProgram!,
      accumulatedFace.bounds
    ) );
  }

  // TODO: inline eventually
  private static addPartialPixel(
    outputRaster: OutputRaster,
    renderProgram: RenderProgram,
    constColor: Vector4 | null,
    translation: Vector2,
    pixelFace: ClippableFace,
    area: number,
    x: number,
    y: number
  ): void {
    if ( assert ) {
      debugData!.areas.push( new Bounds2( x, y, x + 1, y + 1 ) );
    }

    // TODO: potentially cache the centroid, if we have multiple overlapping gradients?
    // pixelFace.getCentroid( area )
    const color = constColor || renderProgram.evaluate(
      pixelFace,
      area,
      pixelFace.getCentroid( area ),
      x,
      y,
      x + 1,
      y + 1
    );
    outputRaster.addPartialPixel( color.timesScalar( area ), x + translation.x, y + translation.y );
  }

  // TODO: inline eventually
  private static addFullArea(
    outputRaster: OutputRaster,
    renderProgram: RenderProgram,
    constColor: Vector4 | null,
    translation: Vector2,
    minX: number,
    minY: number,
    maxX: number,
    maxY: number
  ): void {
    if ( assert ) {
      debugData!.areas.push( new Bounds2( minX, minY, maxX, maxY ) );
    }
    if ( constColor ) {
      outputRaster.addFullRegion( constColor, minX, minY, maxX - minX, maxY - minY );
    }
    else {
      for ( let y = minY; y < maxY; y++ ) {
        for ( let x = minX; x < maxX; x++ ) {
          const centroid = scratchFullAreaVector.setXY( x + 0.5, y + 0.5 );
          const color = renderProgram.evaluate(
            null,
            1,
            centroid,
            x,
            y,
            x + 1,
            y + 1
          );
          outputRaster.addFullPixel( color, x + translation.x, y + translation.y );
        }
      }
    }
  }

  private static fullRasterize(
    outputRaster: OutputRaster,
    renderProgram: RenderProgram,
    clippableFace: ClippableFace,
    constColor: Vector4 | null,
    bounds: Bounds2, // TODO: check it's integral
    translation: Vector2
  ): void {
    const pixelBounds = Bounds2.NOTHING.copy();
    const minX = bounds.minX;
    const minY = bounds.minY;
    const maxX = bounds.maxX;
    const maxY = bounds.maxY;

    for ( let y = minY; y < maxY; y++ ) {
      pixelBounds.minY = y;
      pixelBounds.maxY = y + 1;
      for ( let x = minX; x < maxX; x++ ) {
        pixelBounds.minX = x;
        pixelBounds.maxX = x + 1;

        const pixelFace = clippableFace.getClipped( pixelBounds );
        const area = pixelFace.getArea();
        if ( area > 1e-8 ) {
          Rasterize.addPartialPixel(
            outputRaster, renderProgram, constColor, translation,
            pixelFace, area, x, y
          );
        }
      }
    }
  }

  private static binaryInternalRasterize(
    outputRaster: OutputRaster,
    renderProgram: RenderProgram,
    constColor: Vector4 | null,
    translation: Vector2,
    clippableFace: ClippableFace,
    area: number,
    minX: number,
    minY: number,
    maxX: number,
    maxY: number
  ): void {

    // TODO: more advanced handling

    // TODO: potential filtering!!!

    // TODO TODO TODO TODO TODO: non-zero-centered bounds! Verify everything

    const xDiff = maxX - minX;
    const yDiff = maxY - minY;
    if ( area > 1e-8 ) {
      if ( area >= ( maxX - minX ) * ( maxY - minY ) - 1e-8 ) {
        Rasterize.addFullArea(
          outputRaster, renderProgram, constColor, translation,
          minX, minY, maxX, maxY
        );
      }
      else if ( xDiff === 1 && yDiff === 1 ) {
        Rasterize.addPartialPixel(
          outputRaster, renderProgram, constColor, translation,
          clippableFace, area, minX, minY
        );
      }
      else {
        if ( xDiff > yDiff ) {
          const xSplit = Math.floor( ( minX + maxX ) / 2 );

          const { minFace, maxFace } = clippableFace.getBinaryXClip( xSplit, ( minY + maxY ) / 2 );

          if ( assertSlow ) {
            const oldMinFace = clippableFace.getClipped( new Bounds2( minX, minY, xSplit, maxY ) );
            const oldMaxFace = clippableFace.getClipped( new Bounds2( xSplit, minY, maxX, maxY ) );

            if ( Math.abs( minFace.getArea() - oldMinFace.getArea() ) > 1e-8 || Math.abs( maxFace.getArea() - oldMaxFace.getArea() ) > 1e-8 ) {
              assertSlow( false, 'binary X clip issue' );
            }
          }

          const minArea = minFace.getArea();
          const maxArea = maxFace.getArea();

          if ( minArea > 1e-8 ) {
            Rasterize.binaryInternalRasterize(
              outputRaster, renderProgram, constColor, translation,
              minFace, minArea, minX, minY, xSplit, maxY
            );
          }
          if ( maxArea > 1e-8 ) {
            Rasterize.binaryInternalRasterize(
              outputRaster, renderProgram, constColor, translation,
              maxFace, maxArea, xSplit, minY, maxX, maxY
            );
          }
        }
        else {
          const ySplit = Math.floor( ( minY + maxY ) / 2 );

          const { minFace, maxFace } = clippableFace.getBinaryYClip( ySplit, ( minX + maxX ) / 2 );

          if ( assertSlow ) {
            const oldMinFace = clippableFace.getClipped( new Bounds2( minX, minY, maxX, ySplit ) );
            const oldMaxFace = clippableFace.getClipped( new Bounds2( minX, ySplit, maxX, maxY ) );

            if ( Math.abs( minFace.getArea() - oldMinFace.getArea() ) > 1e-8 || Math.abs( maxFace.getArea() - oldMaxFace.getArea() ) > 1e-8 ) {
              assertSlow( false, 'binary Y clip issue' );
            }
          }

          const minArea = minFace.getArea();
          const maxArea = maxFace.getArea();

          if ( minArea > 1e-8 ) {
            Rasterize.binaryInternalRasterize(
              outputRaster, renderProgram, constColor, translation,
              minFace, minArea, minX, minY, maxX, ySplit
            );
          }
          if ( maxArea > 1e-8 ) {
            Rasterize.binaryInternalRasterize(
              outputRaster, renderProgram, constColor, translation,
              maxFace, maxArea, minX, ySplit, maxX, maxY
            );
          }
        }
      }
    }
  }

  private static rasterize(
    outputRaster: OutputRaster,
    renderProgram: RenderProgram,
    clippableFace: ClippableFace,
    constColor: Vector4 | null,
    bounds: Bounds2, // TODO: check it's integral
    translation: Vector2
  ): void {
    Rasterize.binaryInternalRasterize(
      outputRaster, renderProgram, constColor, translation,
      clippableFace, clippableFace.getArea(), bounds.minX, bounds.minY, bounds.maxX, bounds.maxY
    );
  }

  private static rasterizeAccumulate(
    outputRaster: OutputRaster,
    renderableFaces: RenderableFace[],
    bounds: Bounds2,
    translation: Vector2
  ): void {
    const rasterWidth = bounds.width;
    const rasterHeight = bounds.height;

    for ( let i = 0; i < renderableFaces.length; i++ ) {
      const renderableFace = renderableFaces[ i ];
      const face = renderableFace.face;
      const renderProgram = renderableFace.renderProgram;
      const polygonalBounds = renderableFace.bounds;
      const clippableFace = renderableFace.face;

      const faceDebugData: IntentionalAny = assert ? {
        face: face,
        pixels: [],
        areas: []
      } : null;
      if ( assert ) {
        debugData!.faceDebugData = debugData!.faceDebugData || [];
        debugData!.faceDebugData.push( faceDebugData );
      }
      if ( assert ) {
        faceDebugData.clippableFace = clippableFace;
      }

      const minX = Math.max( Math.floor( polygonalBounds.minX ), 0 );
      const minY = Math.max( Math.floor( polygonalBounds.minY ), 0 );
      const maxX = Math.min( Math.ceil( polygonalBounds.maxX ), rasterWidth );
      const maxY = Math.min( Math.ceil( polygonalBounds.maxY ), rasterHeight );

      const faceBounds = new Bounds2( minX, minY, maxX, maxY );

      const constColor = renderProgram instanceof RenderColor ? renderProgram.color : null;

      Rasterize.rasterize(
        outputRaster,
        renderProgram,
        clippableFace,
        constColor,
        faceBounds,
        translation
      );
    }
  }

  public static rasterizeRenderProgram( renderProgram: RenderProgram, bounds: Bounds2, providedOptions?: RasterizationOptions ): ImageData {

    const options = optionize3<RasterizationOptions>()( {}, DEFAULT_OPTIONS, providedOptions );

    if ( assert ) {
      debugData = {
        areas: []
      };

      // NOTE: find a better way of doing this?
      // @ts-expect-error
      window.debugData = debugData;
    }

    assert && assert( Number.isInteger( bounds.left ) && Number.isInteger( bounds.top ) && Number.isInteger( bounds.right ) && Number.isInteger( bounds.bottom ) );

    const scale = Math.pow( 2, 20 - Math.ceil( Math.log2( Math.max( bounds.width, bounds.height ) ) ) );
    if ( assert ) {
      debugData!.scale = scale;
    }

    const paths: RenderPath[] = [];
    renderProgram.depthFirst( program => {
      if ( program instanceof RenderPathProgram && program.path !== null ) {
        paths.push( program.path );
      }
    } );
    const backgroundPath = new RenderPath( 'nonzero', [
      [
        bounds.leftTop,
        bounds.rightTop,
        bounds.rightBottom,
        bounds.leftBottom
      ]
    ] );
    paths.push( backgroundPath );

    // TODO: These are WRONG given our translation! We should fix these.
    const integerBounds = new Bounds2(
      Utils.roundSymmetric( bounds.minX * scale ),
      Utils.roundSymmetric( bounds.minY * scale ),
      Utils.roundSymmetric( bounds.maxX * scale ),
      Utils.roundSymmetric( bounds.maxY * scale )
    );
    if ( assert ) { debugData!.integerBounds = integerBounds; }

    const integerEdges = Rasterize.clipScaleToIntegerEdges( paths, bounds, scale );
    if ( assert ) { debugData!.integerEdges = integerEdges; }

    // TODO: optional hilbert space-fill sort here?

    if ( options.edgeIntersectionMethod === 'quadratic' ) {
      LineIntersector.edgeIntersectionQuadratic( integerEdges );
    }
    else if ( options.edgeIntersectionMethod === 'boundsTree' ) {
      LineIntersector.edgeIntersectionBoundsTree( integerEdges );
    }
    else if ( options.edgeIntersectionMethod === 'arrayBoundsTree' ) {
      LineIntersector.edgeIntersectionArrayBoundsTree( integerEdges );
    }
    else {
      throw new Error( `unknown edgeIntersectionMethod: ${options.edgeIntersectionMethod}` );
    }

    const rationalHalfEdges = LineSplitter.splitIntegerEdges( integerEdges );

    rationalHalfEdges.sort( ( a, b ) => a.compare( b ) );

    const filteredRationalHalfEdges = Rasterize.filterAndConnectHalfEdges( rationalHalfEdges );
    if ( assert ) { debugData!.filteredRationalHalfEdges = filteredRationalHalfEdges; }

    const innerBoundaries: RationalBoundary[] = [];
    const outerBoundaries: RationalBoundary[] = [];
    const faces: RationalFace[] = [];
    if ( assert ) {
      debugData!.innerBoundaries = innerBoundaries;
      debugData!.outerBoundaries = outerBoundaries;
      debugData!.faces = faces;
    }
    Rasterize.traceBoundaries( filteredRationalHalfEdges, innerBoundaries, outerBoundaries, faces );

    const exteriorBoundary = Rasterize.computeFaceHoles(
      integerBounds,
      outerBoundaries,
      faces
    );

    // For ease of use, an unbounded face (it is essentially fake)
    const unboundedFace = Rasterize.createUnboundedFace( exteriorBoundary );
    if ( assert ) {
      debugData!.unboundedFace = unboundedFace;
    }

    Rasterize.computeWindingMaps( filteredRationalHalfEdges, unboundedFace );

    const renderedFaces = Rasterize.getRenderProgrammedFaces( renderProgram, faces );

    // TODO: translation is... just based on the bounds, right? Can we avoid passing it in?
    // TODO: really test the translated (dirty region) bit
    const translation = new Vector2( -bounds.minX, -bounds.minY );

    // TODO: naming with above!!
    let renderableFaces: RenderableFace[];
    if ( options.renderableFaceMethod === 'polygonal' ) {
      renderableFaces = Rasterize.toPolygonalRenderableFaces( renderedFaces, scale, translation );
    }
    else if ( options.renderableFaceMethod === 'edged' ) {
      renderableFaces = Rasterize.toEdgedRenderableFaces( renderedFaces, scale, translation );
    }
    else if ( options.renderableFaceMethod === 'fullyCombined' ) {
      renderableFaces = Rasterize.toFullyCombinedRenderableFaces( renderedFaces, scale, translation );
    }
    else if ( options.renderableFaceMethod === 'simplifyingCombined' ) {
      renderableFaces = Rasterize.toSimplifyingCombinedRenderableFaces( renderedFaces, scale, translation );
    }
    else {
      throw new Error( 'unknown renderableFaceMethod' );
    }

    if ( options.splitLinearGradients ) {
      renderableFaces = renderableFaces.flatMap( face => face.splitLinearGradients() );
    }
    if ( options.splitRadialGradients ) {
      renderableFaces = renderableFaces.flatMap( face => face.splitRadialGradients() );
    }

    const rasterWidth = bounds.width;
    const rasterHeight = bounds.height;

    // const outputRaster = new AccumulationRaster( rasterWidth, rasterHeight );
    const outputRaster = new CombinedRaster( rasterWidth, rasterHeight );

    Rasterize.rasterizeAccumulate(
      outputRaster,
      renderableFaces,
      bounds,
      translation
    );

    return outputRaster.toImageData();
  }

  public static imageDataToCanvas( imageData: ImageData ): HTMLCanvasElement {
    const canvas = document.createElement( 'canvas' );
    canvas.width = imageData.width;
    canvas.height = imageData.height;
    const context = canvas.getContext( '2d' )!;
    context.putImageData( imageData, 0, 0 );
    return canvas;
  }
}

scenery.register( 'Rasterize', Rasterize );
