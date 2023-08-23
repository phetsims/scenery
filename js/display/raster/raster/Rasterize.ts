// Copyright 2023, University of Colorado Boulder

/**
 * Test rasterization
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { BigRational, ClippableFace, ClipSimplifier, FaceConversion, getPolygonFilterExtraPixels, getPolygonFilterGridOffset, IntegerEdge, LinearEdge, LineIntersector, LineSplitter, OutputRaster, PolygonClipping, PolygonFilterType, PolygonMitchellNetravali, RationalBoundary, RationalFace, RationalHalfEdge, RenderableFace, RenderColor, RenderPath, RenderPathProgram, RenderProgram, RenderProgramNeeds, scenery, WindingMap } from '../../../imports.js';
import Bounds2 from '../../../../../dot/js/Bounds2.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import IntentionalAny from '../../../../../phet-core/js/types/IntentionalAny.js';
import Vector4 from '../../../../../dot/js/Vector4.js';
import { optionize3 } from '../../../../../phet-core/js/optionize.js';
import Matrix3 from '../../../../../dot/js/Matrix3.js';

export type RasterizationOptions = {

  outputRasterOffset?: Vector2;
  polygonFiltering?: PolygonFilterType;

  edgeIntersectionMethod?: 'quadratic' | 'boundsTree' | 'arrayBoundsTree';

  renderableFaceMethod?: 'polygonal' | 'edged' | 'fullyCombined' | 'simplifyingCombined' | 'traced';

  splitLinearGradients?: boolean;
  splitRadialGradients?: boolean;
};

const DEFAULT_OPTIONS = {
  outputRasterOffset: Vector2.ZERO,
  polygonFiltering: PolygonFilterType.Box,
  edgeIntersectionMethod: 'arrayBoundsTree',
  renderableFaceMethod: 'traced',
  splitLinearGradients: true,
  splitRadialGradients: true
} as const;

let debugData: Record<string, IntentionalAny> | null = null;

const scratchFullAreaVector = new Vector2( 0, 0 );

const nanVector = new Vector2( NaN, NaN );

export default class Rasterize {

  private static clipScaleToIntegerEdges( paths: RenderPath[], bounds: Bounds2, toIntegerMatrix: Matrix3 ): IntegerEdge[] {
    const integerEdges: IntegerEdge[] = [];
    for ( let i = 0; i < paths.length; i++ ) {
      const path = paths[ i ];

      for ( let j = 0; j < path.subpaths.length; j++ ) {
        const subpath = path.subpaths[ j ];

        // NOTE: This is a variant that will fully optimize out "doesn't contribute anything" bits to an empty array
        // If a path is fully outside of the clip region, we won't create integer edges out of it.
        const clippedSubpath = PolygonClipping.boundsClipPolygon( subpath, bounds );

        for ( let k = 0; k < clippedSubpath.length; k++ ) {
          // TODO: when micro-optimizing, improve this pattern so we only have one access each iteration
          const p0 = clippedSubpath[ k ];
          const p1 = clippedSubpath[ ( k + 1 ) % clippedSubpath.length ];
          const edge = IntegerEdge.createTransformed( path, toIntegerMatrix, p0, p1 );
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

  // TODO: reconstitute this into an alternative non-binary setup! (filtering will be a bit more annoying?)
  // private static fullRasterize(
  //   outputRaster: OutputRaster,
  //   renderProgram: RenderProgram,
  //   clippableFace: ClippableFace,
  //   constColor: Vector4 | null,
  //   bounds: Bounds2, // TODO: check it's integral
  //   translation: Vector2
  // ): void {
  //   const pixelBounds = Bounds2.NOTHING.copy();
  //   const minX = bounds.minX;
  //   const minY = bounds.minY;
  //   const maxX = bounds.maxX;
  //   const maxY = bounds.maxY;
  //
  //   for ( let y = minY; y < maxY; y++ ) {
  //     pixelBounds.minY = y;
  //     pixelBounds.maxY = y + 1;
  //     for ( let x = minX; x < maxX; x++ ) {
  //       pixelBounds.minX = x;
  //       pixelBounds.maxX = x + 1;
  //
  //       const pixelFace = clippableFace.getClipped( pixelBounds );
  //       const area = pixelFace.getArea();
  //       if ( area > 1e-8 ) {
  //         Rasterize.addPartialPixel(
  //           outputRaster, renderProgram, constColor, translation,
  //           pixelFace, area, x, y
  //         );
  //       }
  //     }
  //   }
  // }

  private static addFilterPixel(
    outputRaster: OutputRaster,
    outputRasterOffset: Vector2,
    bounds: Bounds2,
    polygonFiltering: PolygonFilterType,
    pixelFace: ClippableFace | null,
    area: number,
    x: number,
    y: number,
    color: Vector4
  ): void {

    assert && assert( polygonFiltering === PolygonFilterType.Bilinear || polygonFiltering === PolygonFilterType.MitchellNetravali,
      'Only supports these filters currently' );

    const expand = polygonFiltering === PolygonFilterType.MitchellNetravali ? 2 : 1;

    // e.g. x - px is 0,-1 for bilinear, 1,-2 for mitchell-netravali
    // px = x + (0,1) for bilinear, x + (-1,2) for mitchell-netravali

    const subIterMinX = x - expand + 1;
    const subIterMinY = y - expand + 1;
    const subIterMaxX = x + expand + 1;
    const subIterMaxY = y + expand + 1;

    for ( let py = subIterMinY; py < subIterMaxY; py++ ) {

      const pixelY = py - 0.5;
      // TODO: put these in the subIter values above
      if ( pixelY < bounds.minY || pixelY >= bounds.maxY ) {
        continue;
      }

      for ( let px = subIterMinX; px < subIterMaxX; px++ ) {

        const pixelX = px - 0.5;
        if ( pixelX < bounds.minX || pixelX >= bounds.maxX ) {
          continue;
        }

        let contribution;

        // If it has a full pixel of area, we can simplify computation SIGNIFICANTLY
        if ( area > 1 - 1e-8 ) {
          // only bilinear and mitchell-netravali
          contribution = polygonFiltering === PolygonFilterType.MitchellNetravali ? PolygonMitchellNetravali.evaluateFull( px, py, x, y ) : 0.25;
        }
        else {
          assert && assert( pixelFace );

          if ( assertSlow ) {
            // TODO: implement these for polygonal faces
            const edges = pixelFace!.toEdgedFace().edges;

            assertSlow( edges.every( edge => {
              return edge.startPoint.x >= x && edge.startPoint.x <= x + 1 &&
                     edge.startPoint.y >= y && edge.startPoint.y <= y + 1 &&
                     edge.endPoint.x >= x && edge.endPoint.x <= x + 1 &&
                     edge.endPoint.y >= y && edge.endPoint.y <= y + 1;
            } ) );
          }

          contribution = polygonFiltering === PolygonFilterType.MitchellNetravali ?
                         pixelFace!.getMitchellNetravaliFiltered( px, py, x, y ) :
                         pixelFace!.getBilinearFiltered( px, py, x, y );
        }

        outputRaster.addPartialPixel( color.timesScalar( contribution ), pixelX + outputRasterOffset.x, pixelY + outputRasterOffset.y );
      }
    }
  }

  // TODO: inline eventually
  private static addPartialPixel(
    outputRaster: OutputRaster,
    renderProgram: RenderProgram,
    constColor: Vector4 | null,
    outputRasterOffset: Vector2,
    bounds: Bounds2,
    polygonFiltering: PolygonFilterType,
    needs: RenderProgramNeeds,
    pixelFace: ClippableFace,
    area: number,
    x: number,
    y: number
  ): void {
    if ( assert ) {
      debugData!.areas.push( new Bounds2( x, y, x + 1, y + 1 ) );
    }

    // TODO: potentially cache the centroid, if we have multiple overlapping gradients?
    const color = constColor || renderProgram.evaluate(
      pixelFace,
      needs.needsArea ? area : NaN, // NaNs to hopefully hard-error
      needs.needsCentroid ? pixelFace.getCentroid( area ) : nanVector,
      x,
      y,
      x + 1,
      y + 1
    );

    if ( polygonFiltering === PolygonFilterType.Box ) {
      outputRaster.addPartialPixel( color.timesScalar( area ), x + outputRasterOffset.x, y + outputRasterOffset.y );
    }
    else {
      Rasterize.addFilterPixel(
        outputRaster, outputRasterOffset, bounds, polygonFiltering,
        pixelFace, area, x, y, color
      );
    }
  }

  // TODO: inline eventually
  private static addFullArea(
    outputRaster: OutputRaster,
    renderProgram: RenderProgram,
    constColor: Vector4 | null,
    outputRasterOffset: Vector2,
    bounds: Bounds2,
    polygonFiltering: PolygonFilterType,
    needs: RenderProgramNeeds,
    minX: number,
    minY: number,
    maxX: number,
    maxY: number
  ): void {
    if ( assert ) {
      debugData!.areas.push( new Bounds2( minX, minY, maxX, maxY ) );
    }

    if ( constColor ) {
      assert && assert( !needs.needsArea && !needs.needsCentroid );

      if ( polygonFiltering === PolygonFilterType.Box ) {
        outputRaster.addFullRegion(
          constColor,
          minX + outputRasterOffset.x,
          minY + outputRasterOffset.y,
          maxX - minX,
          maxY - minY
        );
      }
      else {
        // TODO: ideally we can optimize this if it has a significant number of contained pixels. We only need to
        // "filter" the outside ones (the inside will be a constant color)
        for ( let y = minY; y < maxY; y++ ) {
          for ( let x = minX; x < maxX; x++ ) {
            Rasterize.addFilterPixel(
              outputRaster, outputRasterOffset, bounds, polygonFiltering,
              null, 1, x, y, constColor
            );
          }
        }
      }
    }
    else {
      const pixelArea = needs.needsArea ? 1 : NaN; // NaNs to hopefully hard-error
      for ( let y = minY; y < maxY; y++ ) {
        for ( let x = minX; x < maxX; x++ ) {
          const color = renderProgram.evaluate(
            null,
            pixelArea,
            needs.needsCentroid ? scratchFullAreaVector.setXY( x + 0.5, y + 0.5 ) : nanVector,
            x,
            y,
            x + 1,
            y + 1
          );
          if ( polygonFiltering === PolygonFilterType.Box ) {
            outputRaster.addFullPixel( color, x + outputRasterOffset.x, y + outputRasterOffset.y );
          }
          else {
            Rasterize.addFilterPixel(
              outputRaster, outputRasterOffset, bounds, polygonFiltering,
              null, 1, x, y, color
            );
          }
        }
      }
    }
  }

  private static binaryInternalRasterize(
    outputRaster: OutputRaster,
    renderProgram: RenderProgram,
    constColor: Vector4 | null,
    outputRasterOffset: Vector2,
    bounds: Bounds2,
    polygonFiltering: PolygonFilterType,
    needs: RenderProgramNeeds,
    clippableFace: ClippableFace,
    area: number,
    minX: number,
    minY: number,
    maxX: number,
    maxY: number
  ): void {

    // TODO: more advanced handling

    const xDiff = maxX - minX;
    const yDiff = maxY - minY;

    assert && assert( xDiff >= 1 && yDiff >= 1 );
    assert && assert( Number.isInteger( xDiff ) && Number.isInteger( yDiff ) );
    assert && assert( polygonFiltering === PolygonFilterType.Box ? Number.isInteger( minX ) : minX - Math.floor( minX ) === 0.5 );
    assert && assert( polygonFiltering === PolygonFilterType.Box ? Number.isInteger( minY ) : minY - Math.floor( minY ) === 0.5 );
    assert && assert( polygonFiltering === PolygonFilterType.Box ? Number.isInteger( maxX ) : maxX - Math.floor( maxX ) === 0.5 );
    assert && assert( polygonFiltering === PolygonFilterType.Box ? Number.isInteger( maxY ) : maxY - Math.floor( maxY ) === 0.5 );

    if ( area > 1e-8 ) {
      if ( area >= ( maxX - minX ) * ( maxY - minY ) - 1e-8 ) {
        Rasterize.addFullArea(
          outputRaster, renderProgram, constColor, outputRasterOffset, bounds, polygonFiltering, needs,
          minX, minY, maxX, maxY
        );
      }
      else if ( xDiff === 1 && yDiff === 1 ) {
        Rasterize.addPartialPixel(
          outputRaster, renderProgram, constColor, outputRasterOffset, bounds, polygonFiltering, needs,
          clippableFace, area, minX, minY
        );
      }
      else {
        const averageX = ( minX + maxX ) / 2;
        const averageY = ( minY + maxY ) / 2;

        if ( xDiff > yDiff ) {
          const xSplit = minX + Math.floor( 0.5 * xDiff );

          assert && assert( xSplit !== minX && xSplit !== maxX );

          // TODO: If this is the LAST level of clipping, can we perhaps skip the actual face output (and only get
          // TODO: area output)?
          const { minFace, maxFace } = clippableFace.getBinaryXClip( xSplit, averageY );

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
              outputRaster, renderProgram, constColor, outputRasterOffset, bounds, polygonFiltering, needs,
              minFace, minArea, minX, minY, xSplit, maxY
            );
          }
          if ( maxArea > 1e-8 ) {
            Rasterize.binaryInternalRasterize(
              outputRaster, renderProgram, constColor, outputRasterOffset, bounds, polygonFiltering, needs,
              maxFace, maxArea, xSplit, minY, maxX, maxY
            );
          }
        }
        else {
          const ySplit = minY + Math.floor( 0.5 * yDiff );

          const { minFace, maxFace } = clippableFace.getBinaryYClip( ySplit, averageX );

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
              outputRaster, renderProgram, constColor, outputRasterOffset, bounds, polygonFiltering, needs,
              minFace, minArea, minX, minY, maxX, ySplit
            );
          }
          if ( maxArea > 1e-8 ) {
            Rasterize.binaryInternalRasterize(
              outputRaster, renderProgram, constColor, outputRasterOffset, bounds, polygonFiltering, needs,
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
    faceBounds: Bounds2,
    bounds: Bounds2,
    outputRasterOffset: Vector2,
    polygonFiltering: PolygonFilterType
  ): void {
    const gridOffset = getPolygonFilterGridOffset( polygonFiltering );
    faceBounds = faceBounds.shiftedXY( gridOffset, gridOffset ).roundedOut().shiftedXY( -gridOffset, -gridOffset );

    // TODO: Can we avoid having to do this? Can we include this in places where it could cause this? (gradient splits?)
    clippableFace = clippableFace.getClipped( faceBounds );

    const needs = renderProgram.getNeeds();

    Rasterize.binaryInternalRasterize(
      outputRaster, renderProgram, constColor, outputRasterOffset, bounds, polygonFiltering, needs,
      clippableFace, clippableFace.getArea(), faceBounds.minX, faceBounds.minY, faceBounds.maxX, faceBounds.maxY
    );
  }

  private static rasterizeAccumulate(
    outputRaster: OutputRaster,
    renderableFaces: RenderableFace[],
    bounds: Bounds2,
    gridBounds: Bounds2,
    outputRasterOffset: Vector2,
    polygonFiltering: PolygonFilterType
  ): void {
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

      // TODO: would the bounds show up OUTSIDE of this?
      const faceBounds = polygonalBounds.intersection( gridBounds );

      const constColor = renderProgram instanceof RenderColor ? renderProgram.color : null;

      Rasterize.rasterize(
        outputRaster,
        renderProgram,
        clippableFace,
        constColor,
        faceBounds,
        bounds,
        outputRasterOffset,
        polygonFiltering
      );
    }
  }

  // TODO: name change?
  public static rasterizeRenderProgram( renderProgram: RenderProgram, outputRaster: OutputRaster, bounds: Bounds2, providedOptions?: RasterizationOptions ): void {

    // Coordinate frames:
    //
    // First, we start with the RenderProgram coordinate frame (the coordinate frame of the paths inside the renderPrograms,
    // and the bounds provided to us.
    //
    // We will then transfer over to the "integer" coordinate frame, where each of our input edges will have
    // integer-valued coordinates. Additionally, we've shifted/scaled this, so that the integers lie within about
    // 20-bits of precision (and are centered in the integer grid). We do this so that we can do exact operations for
    // intersection/etc., which solves most of the robustness issues (the intersection point x,y between two of these
    // line segments can be stored each with a 64-bit numerator and a 64-bit denominator, and we can do the needed
    // arithmetic with the rationals).
    //
    // Once we have determined the intersections AND connected half-edges (which requires sorting the half-edges with
    // the exact values), we can then transfer back to the RenderProgram coordinate frame, and rasterize the faces
    // in this coordinate frame.
    //
    // Of note, when we're filtering with bilinear or Mitchell-Netravali filters, we'll be cutting up the faces into
    // half-pixel offset expanded regions, so that we can evaluate the filters AT the pixel centers.

    assert && assert( bounds.isValid() && !bounds.isEmpty(), 'Rasterization bounds should be valid and non-empty' );
    assert && assert( Number.isInteger( bounds.left ) && Number.isInteger( bounds.top ) && Number.isInteger( bounds.right ) && Number.isInteger( bounds.bottom ) );

    const options = optionize3<RasterizationOptions>()( {}, DEFAULT_OPTIONS, providedOptions );

    if ( assert ) {
      debugData = {
        areas: []
      };

      // NOTE: find a better way of doing this?
      // @ts-expect-error
      window.debugData = debugData;
    }

    // TODO: outputRasterOffset totally broken?!!!
    const outputRasterOffset: Vector2 = options.outputRasterOffset;
    const polygonFiltering: PolygonFilterType = options.polygonFiltering;

    const filterAdditionalPixels = getPolygonFilterExtraPixels( polygonFiltering );
    const filterGridOffset = getPolygonFilterGridOffset( polygonFiltering );

    const outputWidth = bounds.width;
    const outputHeight = bounds.height;
    const gridWidth = outputWidth + filterAdditionalPixels;
    const gridHeight = outputHeight + filterAdditionalPixels;

    // in RenderProgram coordinate frame
    const gridBounds = new Bounds2(
      bounds.minX + filterGridOffset,
      bounds.minY + filterGridOffset,
      bounds.maxX + filterGridOffset + filterAdditionalPixels,
      bounds.maxY + filterGridOffset + filterAdditionalPixels
    );

    // Keep us at 20 bits of precision (after rounding)
    const scale = Math.pow( 2, 20 - Math.ceil( Math.log2( Math.max( gridWidth, gridHeight ) ) ) );
    if ( assert ) { debugData!.scale = scale; }

    // -( scale * ( bounds.minX + filterGridOffset.x ) + translation.x ) = scale * ( bounds.maxX + filterGridOffset.x ) + translation.x
    const translation = new Vector2(
      -0.5 * scale * ( 2 * filterGridOffset + filterAdditionalPixels + bounds.minX + bounds.maxX ),
      -0.5 * scale * ( 2 * filterGridOffset + filterAdditionalPixels + bounds.minY + bounds.maxY )
    );
    if ( assert ) { debugData!.translation = translation; }

    const toIntegerMatrix = Matrix3.affine( scale, 0, translation.x, 0, scale, translation.y );
    if ( assert ) { debugData!.toIntegerMatrix = toIntegerMatrix; }

    const fromIntegerMatrix = toIntegerMatrix.inverted();
    if ( assert ) { debugData!.fromIntegerMatrix = fromIntegerMatrix; }

    // Verify our math! Make sure we will be perfectly centered in our integer grid!
    assert && assert( Math.abs( ( scale * gridBounds.minX + translation.x ) + ( scale * gridBounds.maxX + translation.x ) ) < 1e-10 );
    assert && assert( Math.abs( ( scale * gridBounds.minY + translation.y ) + ( scale * gridBounds.maxY + translation.y ) ) < 1e-10 );

    const integerBounds = gridBounds.transformed( toIntegerMatrix );
    if ( assert ) { debugData!.integerBounds = integerBounds; }

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

    const integerEdges = Rasterize.clipScaleToIntegerEdges( paths, gridBounds, toIntegerMatrix );
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

    // TODO: a good way to optimize this? For certain scenes we might be computing a lot of intersections that we
    // TODO: don't need.
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
    // const translation = new Vector2( -bounds.minX, -bounds.minY );

    // TODO: naming with above!!
    let renderableFaces: RenderableFace[];
    if ( options.renderableFaceMethod === 'polygonal' ) {
      renderableFaces = FaceConversion.toPolygonalRenderableFaces( renderedFaces, fromIntegerMatrix );
    }
    else if ( options.renderableFaceMethod === 'edged' ) {
      renderableFaces = FaceConversion.toEdgedRenderableFaces( renderedFaces, fromIntegerMatrix );
    }
    else if ( options.renderableFaceMethod === 'fullyCombined' ) {
      renderableFaces = FaceConversion.toFullyCombinedRenderableFaces( renderedFaces, fromIntegerMatrix );
    }
    else if ( options.renderableFaceMethod === 'simplifyingCombined' ) {
      renderableFaces = FaceConversion.toSimplifyingCombinedRenderableFaces( renderedFaces, fromIntegerMatrix );
    }
    else if ( options.renderableFaceMethod === 'traced' ) {
      renderableFaces = FaceConversion.toTracedRenderableFaces( renderedFaces, fromIntegerMatrix );
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

    Rasterize.rasterizeAccumulate(
      outputRaster,
      renderableFaces,
      bounds,
      gridBounds,
      outputRasterOffset,
      polygonFiltering
    );
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
