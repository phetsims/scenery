// Copyright 2023, University of Colorado Boulder

/**
 * Represents a face with a main (positive-oriented) boundary and zero or more (negative-oriented) holes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { IntegerEdge, LineIntersector, LineSplitter, PolygonalFace, RationalBoundary, RationalFace, RationalHalfEdge, RenderPath, scenery } from '../../../imports.js';
import Bounds2 from '../../../../../dot/js/Bounds2.js';
import Matrix3 from '../../../../../dot/js/Matrix3.js';
import Vector2 from '../../../../../dot/js/Vector2.js';

const defaultLineIntersector = LineIntersector.edgeIntersectionArrayBoundsTree;

export default class PolygonalBoolean {

  public static union( ...paths: RenderPath[] ): Vector2[][] {
    return PolygonalBoolean.cag(
      paths,
      face => face.getIncludedRenderPaths().size > 0,
      ( face, faceData, bounds ) => faceData ? face.polygons : [],
      ( faceData1, faceData2 ) => faceData1 === faceData2
    ).flat();
  }

  public static intersection( ...paths: RenderPath[] ): Vector2[][] {
    return PolygonalBoolean.cag(
      paths,
      face => face.getIncludedRenderPaths().size === paths.length,
      ( face, faceData, bounds ) => faceData ? face.polygons : [],
      ( faceData1, faceData2 ) => faceData1 === faceData2
    ).flat();
  }

  public static difference( pathA: RenderPath, pathB: RenderPath ): Vector2[][] {
    return PolygonalBoolean.cag(
      [ pathA, pathB ],
      face => {
        const set = face.getIncludedRenderPaths();
        return set.has( pathA ) && !set.has( pathB );
      },
      ( face, faceData, bounds ) => faceData ? face.polygons : [],
      ( faceData1, faceData2 ) => faceData1 === faceData2
    ).flat();
  }

  public static getOverlaps( pathA: RenderPath, pathB: RenderPath ): { intersection: Vector2[][]; aOnly: Vector2[][]; bOnly: Vector2[][] } {
    const taggedPolygonsList = PolygonalBoolean.cag(
      [ pathA, pathB ],
      ( face: RationalFace ): 'intersection' | 'aOnly' | 'bOnly' | null => {
        const set = face.getIncludedRenderPaths();

        if ( set.has( pathA ) && set.has( pathB ) ) {
          return 'intersection';
        }
        else if ( set.has( pathA ) ) {
          return 'aOnly';
        }
        else if ( set.has( pathB ) ) {
          return 'bOnly';
        }
        else {
          return null;
        }
      },
      ( face, faceData, bounds ) => ( { tag: faceData, polygons: face.polygons } ),
      ( faceData1, faceData2 ) => faceData1 === faceData2
    );

    const result: { intersection: Vector2[][]; aOnly: Vector2[][]; bOnly: Vector2[][] } = {
      intersection: [],
      aOnly: [],
      bOnly: []
    };

    taggedPolygonsList.forEach( taggedPolygon => {
      if ( taggedPolygon.tag !== null ) {
        result[ taggedPolygon.tag ].push( ...taggedPolygon.polygons );
      }
    } );

    return result;
  }

  // TODO: ideally handle the fully collinear simplification?

  public static cag<FaceData, OutputFace>(
    paths: RenderPath[],
    getFaceData: ( face: RationalFace ) => FaceData,
    createOutputFace: ( face: PolygonalFace, faceData: FaceData, bounds: Bounds2 ) => OutputFace,
    // null is for the unbounded face
    isFaceDataCompatible: ( faceData1: FaceData, faceData2: FaceData | null ) => boolean
  ): OutputFace[] {

    const bounds = Bounds2.NOTHING.copy();
    for ( let i = 0; i < paths.length; i++ ) {
      bounds.includeBounds( paths[ i ].getBounds() );
    }

    // Keep us at 20 bits of precision (after rounding)
    const scale = Math.pow( 2, 20 - Math.ceil( Math.log2( Math.max( bounds.width, bounds.height ) ) ) );

    const translation = new Vector2(
      -0.5 * scale * ( bounds.minX + bounds.maxX ),
      -0.5 * scale * ( bounds.minY + bounds.maxY )
    );

    const toIntegerMatrix = Matrix3.affine( scale, 0, translation.x, 0, scale, translation.y );
    const fromIntegerMatrix = toIntegerMatrix.inverted();

    // Verify our math! Make sure we will be perfectly centered in our integer grid!
    assert && assert( Math.abs( ( scale * bounds.minX + translation.x ) + ( scale * bounds.maxX + translation.x ) ) < 1e-10 );
    assert && assert( Math.abs( ( scale * bounds.minY + translation.y ) + ( scale * bounds.maxY + translation.y ) ) < 1e-10 );

    const integerEdges = IntegerEdge.scaleToIntegerEdges( paths, toIntegerMatrix );

    // TODO: optional hilbert space-fill sort here?

    defaultLineIntersector( integerEdges );

    const rationalHalfEdges = LineSplitter.splitIntegerEdges( integerEdges );

    rationalHalfEdges.sort( ( a, b ) => a.compare( b ) );

    const filteredRationalHalfEdges = RationalHalfEdge.filterAndConnectHalfEdges( rationalHalfEdges );

    const innerBoundaries: RationalBoundary[] = [];
    const outerBoundaries: RationalBoundary[] = [];
    const faces: RationalFace[] = [];
    RationalFace.traceBoundaries( filteredRationalHalfEdges, innerBoundaries, outerBoundaries, faces );

    const exteriorBoundary = RationalFace.computeFaceHolesWithOrderedWindingNumbers(
      outerBoundaries,
      faces
    );

    // For ease of use, an unbounded face (it is essentially fake)
    const unboundedFace = RationalFace.createUnboundedFace( exteriorBoundary );

    RationalFace.computeWindingMaps( filteredRationalHalfEdges, unboundedFace );

    return RationalFace.traceCombineFaces(
      faces,
      fromIntegerMatrix,
      getFaceData,
      createOutputFace,
      isFaceDataCompatible
    );
  }
}

scenery.register( 'PolygonalBoolean', PolygonalBoolean );
