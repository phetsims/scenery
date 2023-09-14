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
  // TODO: add full docs here
  getBounds(): Bounds2;
  getDotRange( normal: Vector2 ): Range;
  getDistanceRangeToEdges( point: Vector2 ): Range;
  getDistanceRangeToInside( point: Vector2 ): Range;
  getArea(): number;
  getCentroid( area: number ): Vector2;

  // Returns the partial for the centroid computation. These should be summed up, divided by 6, and divided by the area
  // to give the full centroid
  getCentroidPartial(): Vector2;

  getZero(): number;
  getAverageDistance( point: Vector2, area: number ): number;
  getAverageDistanceTransformedToOrigin( transform: Matrix3, area: number ): number;
  getTransformed( transform: Matrix3 ): ClippableFace;
  getRounded( epsilon: number ): ClippableFace; // rounds [-epsilon/2, epsilon/2] to 0, etc.
  getClipped( minX: number, minY: number, maxX: number, maxY: number ): ClippableFace;
  getBinaryXClip( x: number, fakeCornerY: number ): { minFace: ClippableFace; maxFace: ClippableFace };
  getBinaryYClip( y: number, fakeCornerX: number ): { minFace: ClippableFace; maxFace: ClippableFace };
  getBinaryLineClip(
    normal: Vector2,
    value: number,
    fakeCornerPerpendicular: number
  ): { minFace: ClippableFace; maxFace: ClippableFace };
  getStripeLineClip(
    normal: Vector2,
    values: number[],
    fakeCornerPerpendicular: number
  ): ClippableFace[];
  getBinaryCircularClip(
    center: Vector2,
    radius: number,
    maxAngleSplit: number
  ): { insideFace: ClippableFace; outsideFace: ClippableFace };
  gridClipIterate(
    minX: number, minY: number, maxX: number, maxY: number,
    stepX: number, stepY: number, stepWidth: number, stepHeight: number,
    callback: GridClipCallback,
    polygonCompleteCallback: PolygonCompleteCallback
  ): void;
  getBilinearFiltered( pointX: number, pointY: number, minX: number, minY: number ): number;
  getMitchellNetravaliFiltered( pointX: number, pointY: number, minX: number, minY: number ): number;
  containsPoint( point: Vector2 ): boolean;
  toPolygonalFace( epsilon?: number ): PolygonalFace;
  toEdgedFace(): EdgedFace;
  getScratchAccumulator(): ClippableFaceAccumulator;
  getAccumulator(): ClippableFaceAccumulator;
  getShape( epsilon?: number ): Shape; // NOTE: this is only for debugging, performance won't be great
  forEachEdge( callback: ( startPoint: Vector2, endPoint: Vector2 ) => void ): void;
  toString(): string;
  serialize(): IntentionalAny;
};

export default ClippableFace;

export type ClippableFaceAccumulator = {
  addEdge( startX: number, startY: number, endX: number, endY: number, startPoint: Vector2 | null, endPoint: Vector2 | null ): void;
  markNewPolygon(): void;

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
