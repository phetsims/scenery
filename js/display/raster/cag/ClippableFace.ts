// Copyright 2023, University of Colorado Boulder

/**
 * An interface for clippable/subdivide-able faces, with defined bounds and area.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Bounds2 from '../../../../../dot/js/Bounds2.js';
import Range from '../../../../../dot/js/Range.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Matrix3 from '../../../../../dot/js/Matrix3.js';
import IntentionalAny from '../../../../../phet-core/js/types/IntentionalAny.js';
import { EdgedFace, GridClipCallback, PolygonalFace, PolygonCompleteCallback, SerializedEdgedFace, SerializedPolygonalFace } from '../../../imports.js';
import { Shape } from '../../../../../kite/js/imports.js';

type ClippableFace = {
  /**
   * Returns the bounds of the face (ignoring any "fake" edges, if the type supports them)
   */
  getBounds(): Bounds2;

  /**
   * Returns the range of values for the dot product of the given normal with any point contained within the face
   * (for polygons, this is the same as the range of values for the dot product of the normal with any vertex).
   */
  getDotRange( normal: Vector2 ): Range;

  /**
   * Returns the range of distances from the given point to every point along the edges of the face.
   * For instance, if the face was the unit cube, the range would be 1/2 to sqrt(2), for distances to the middles of
   * the edges and the corners respectively.
   */
  getDistanceRangeToEdges( point: Vector2 ): Range;

  /**
   * Returns the range of distances from the given point to every point inside the face. The upper bound should be
   * the same as getDistanceRangeToEdges, however the lower bound may be 0 if the point is inside the face.
   */
  getDistanceRangeToInside( point: Vector2 ): Range;

  /**
   * Returns the signed area of the face (positive if the vertices are in counter-clockwise order, negative if clockwise)
   */
  getArea(): number;

  /**
   * Returns the centroid of the face (area is required for the typical integral required to evaluate)
   */
  getCentroid( area: number ): Vector2;

  /**
   * Returns the partial for the centroid computation. These should be summed up, divided by 6, and divided by the area
   * to give the full centroid
   */
  getCentroidPartial(): Vector2;

  /**
   * Returns the evaluation of an integral that will be zero if the boundaries of the face are correctly closed.
   * It is designed so that if there is a "gap" and we have open boundaries, the result will likely be non-zero.
   *
   * NOTE: This is only used for debugging, so performance is not a concern.
   */
  getZero(): number;

  /**
   * Returns the average distance from the given point to every point inside the face. The integral evaluation requires
   * the area (similarly to the centroid computation).
   */
  getAverageDistance( point: Vector2, area: number ): number;

  /**
   * Returns the average distance from the origin to every point inside the face transformed by the given matrix.
   */
  getAverageDistanceTransformedToOrigin( transform: Matrix3, area: number ): number;

  /**
   * Returns an affine-transformed version of the face.
   */
  getTransformed( transform: Matrix3 ): ClippableFace;

  /**
   * Returns a rounded version of the face, where [-epsilon/2, epsilon/2] rounds to 0, etc.
   */
  getRounded( epsilon: number ): ClippableFace;

  /**
   * Returns a copy of the face that is clipped to be within the given axis-aligned bounding box.
   *
   * TODO: consider a binary clip for this, using duality.
   */
  getClipped( minX: number, minY: number, maxX: number, maxY: number ): ClippableFace;

  /**
   * Returns two copies of the face, one that is clipped to be to the left of the given x value, and one that is
   * clipped to be to the right of the given x value.
   *
   * The fakeCornerY is used to determine the "fake" corner that is used for unsorted-edge clipping.
   */
  getBinaryXClip( x: number, fakeCornerY: number ): { minFace: ClippableFace; maxFace: ClippableFace };

  /**
   * Returns two copies of the face, one that is clipped to y values less than the given y value, and one that is
   * clipped to values greater than the given y value.
   *
   * The fakeCornerX is used to determine the "fake" corner that is used for unsorted-edge clipping.
   */
  getBinaryYClip( y: number, fakeCornerX: number ): { minFace: ClippableFace; maxFace: ClippableFace };

  /**
   * Returns two copies of the face, one that is clipped to contain points where dot( normal, point ) < value,
   * and one that is clipped to contain points where dot( normal, point ) > value.
   *
   * The fake corner perpendicular is used to determine the "fake" corner that is used for unsorted-edge clipping
   */
  getBinaryLineClip(
    normal: Vector2,
    value: number,
    fakeCornerPerpendicular: number
  ): { minFace: ClippableFace; maxFace: ClippableFace };

  /**
   * Returns an array of faces, clipped similarly to getBinaryLineClip, but with more than one (parallel) split line at
   * a time. The first face will be the one with dot( normal, point ) < values[0], the second one with
   * values[ 0 ] < dot( normal, point ) < values[1], etc.
   */
  getStripeLineClip(
    normal: Vector2,
    values: number[],
    fakeCornerPerpendicular: number
  ): ClippableFace[];

  /**
   * Returns two copies of the face, one that is clipped to contain points inside the circle defined by the given
   * center and radius, and one that is clipped to contain points outside the circle.
   *
   * NOTE: maxAngleSplit is used to determine the polygonal approximation of the circle. The returned result will not
   * have a chord with an angle greater than maxAngleSplit.
   */
  getBinaryCircularClip(
    center: Vector2,
    radius: number,
    maxAngleSplit: number
  ): { insideFace: ClippableFace; outsideFace: ClippableFace };

  /**
   * Given an integral bounding box and step sizes (which define the grid), this will clip the face to each cell in the
   * grid, calling the callback for each cell's contributing edges (in order, if we are a PolygonalFace).
   * polygonCompleteCallback will be called whenever a polygon is completed (if we are a polygonal type of face).
   */
  gridClipIterate(
    minX: number, minY: number, maxX: number, maxY: number,
    stepX: number, stepY: number, stepWidth: number, stepHeight: number,
    callback: GridClipCallback,
    polygonCompleteCallback: PolygonCompleteCallback
  ): void;

  /**
   * Returns the evaluation of the bilinear (tent) filter integrals for the given point, ASSUMING that the face
   * is clipped to the transformed unit square of x: [minX,minX+1], y: [minY,minY+1].
   */
  getBilinearFiltered( pointX: number, pointY: number, minX: number, minY: number ): number;

  /**
   * Returns the evaluation of the Mitchell-Netravali (1/3,1/3) filter integrals for the given point, ASSUMING that the
   * face is clipped to the transformed unit square of x: [minX,minX+1], y: [minY,minY+1].
   */
  getMitchellNetravaliFiltered( pointX: number, pointY: number, minX: number, minY: number ): number;

  /**
   * Returns whether the face contains the given point.
   */
  containsPoint( point: Vector2 ): boolean;

  /**
   * Converts the face to a polygonal face. Epsilon is used to determine whether start/end points match.
   *
   * NOTE: This is likely a low-performance method, and should only be used for debugging.
   */
  toPolygonalFace( epsilon?: number ): PolygonalFace;

  /**
   * Converts the face to an edged face.
   */
  toEdgedFace(): EdgedFace;

  /**
   * Returns a singleton accumulator for this type of face.
   */
  getScratchAccumulator(): ClippableFaceAccumulator;

  /**
   * Returns a new accumulator for this type of face.
   */
  getAccumulator(): ClippableFaceAccumulator;

  /**
   * Returns a Shape for the face.
   *
   * NOTE: This is likely a low-performance method, and should only be used for debugging.
   */
  getShape( epsilon?: number ): Shape;

  /**
   * Calls the callback with points for each edge in the face.
   */
  forEachEdge( callback: ( startPoint: Vector2, endPoint: Vector2 ) => void ): void;

  /**
   * Returns a debugging string.
   */
  toString(): string;

  /**
   * Returns a serialized version of the face, that should be able to be deserialized into the same type of face.
   * See {FaceType}.deserialize.
   *
   * NOTE: If you don't know what type of face this is, use serializeClippableFace instead.
   */
  serialize(): IntentionalAny;
};

export default ClippableFace;

// A type for building up a face from edges and new-polygon markers
export type ClippableFaceAccumulator = {
  addEdge( startX: number, startY: number, endX: number, endY: number, startPoint: Vector2 | null, endPoint: Vector2 | null ): void;
  markNewPolygon(): void;

  setAccumulationBounds( minX: number, minY: number, maxX: number, maxY: number ): void;

  // Will reset it to the initial state also
  finalizeFace(): ClippableFace | null;

  // Will reset without creating a face
  reset(): void;
};

export type SerializedClippableFace = {
  type: 'PolygonalFace';
  face: SerializedPolygonalFace;
} | {
  type: 'EdgedFace';
  face: SerializedEdgedFace;
};

export const serializeClippableFace = ( clippableFace: ClippableFace ): SerializedClippableFace => {
  // We are not checking the given type! We're wrapping these
  // eslint-disable-next-line no-simple-type-checking-assertions
  assert && assert( clippableFace instanceof PolygonalFace || clippableFace instanceof EdgedFace );

  return {
    type: clippableFace instanceof PolygonalFace ? 'PolygonalFace' : 'EdgedFace',
    face: clippableFace.serialize()
  };
};

export const deserializeClippableFace = ( serialized: SerializedClippableFace ): ClippableFace => {
  return serialized.type === 'PolygonalFace' ? PolygonalFace.deserialize( serialized.face ) : EdgedFace.deserialize( serialized.face );
};
