// Copyright 2023, University of Colorado Boulder

/**
 * Test rasterization
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { BigIntVector2, BigRational, BigRationalVector2, ClippableFace, EdgedFace, IntegerEdge, IntersectionPoint, LinearEdge, PolygonClipping, RationalBoundary, RationalFace, RationalHalfEdge, RationalIntersection, RenderColor, RenderExtend, RenderGradientStop, RenderLinearBlend, RenderLinearGradient, RenderPathProgram, RenderProgram, RenderRadialGradient, scenery, WindingMap } from '../../../imports.js';
import { RenderPath } from '../render-program/RenderProgram.js';
import Bounds2 from '../../../../../dot/js/Bounds2.js';
import Utils from '../../../../../dot/js/Utils.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import IntentionalAny from '../../../../../phet-core/js/types/IntentionalAny.js';
import Vector4 from '../../../../../dot/js/Vector4.js';

let debugData: Record<string, IntentionalAny> | null = null;

const scratchFullAreaVector = new Vector2( 0, 0 );

class RenderLinearRange {
  public constructor(
    public readonly start: number,
    public readonly end: number,

    // NOTE: equal start/end programs means constant!
    public readonly startProgram: RenderProgram,
    public readonly endProgram: RenderProgram
  ) {}

  public getUnitReversed(): RenderLinearRange {
    return new RenderLinearRange( 1 - this.end, 1 - this.start, this.endProgram, this.startProgram );
  }

  public withOffset( offset: number ): RenderLinearRange {
    return new RenderLinearRange( this.start + offset, this.end + offset, this.startProgram, this.endProgram );
  }
}

// TODO: naming, omg
class RenderableFace {
  public constructor(
    public readonly face: ClippableFace,
    public readonly renderProgram: RenderProgram,
    public readonly bounds: Bounds2
  ) {}

  public static getGradientLinearRanges( min: number, max: number, offset: number, extend: RenderExtend, stops: RenderGradientStop[] ): RenderLinearRange[] {
    const unitRanges: RenderLinearRange[] = [];
    if ( stops[ 0 ].ratio > 0 ) {
      unitRanges.push( new RenderLinearRange(
        0,
        stops[ 0 ].ratio,
        stops[ 0 ].program,
        stops[ 0 ].program
      ) );
    }
    for ( let i = 0; i < stops.length - 1; i++ ) {
      const start = stops[ i ];
      const end = stops[ i + 1 ];

      unitRanges.push( new RenderLinearRange(
        start.ratio,
        end.ratio,
        start.program,
        end.program
      ) );
    }
    if ( stops[ stops.length - 1 ].ratio < 1 ) {
      unitRanges.push( new RenderLinearRange(
        stops[ stops.length - 1 ].ratio,
        1,
        stops[ stops.length - 1 ].program,
        stops[ stops.length - 1 ].program
      ) );
    }
    const reversedUnitRanges: RenderLinearRange[] = unitRanges.map( r => r.getUnitReversed() ).reverse();

    // Our used linear ranges
    const linearRanges: RenderLinearRange[] = [];

    const addSectionRanges = ( sectionOffset: number, reversed: boolean ): void => {
      // now relative to our section!
      const sectionMin = min - sectionOffset;
      const sectionMax = max - sectionOffset;

      const ranges = reversed ? reversedUnitRanges : unitRanges;
      ranges.forEach( range => {
        if ( sectionMin < range.end && sectionMax > range.start ) {
          linearRanges.push( range.withOffset( sectionOffset + offset ) );
        }
      } );
    };

    if ( extend === RenderExtend.Pad ) {
      if ( min < 0 ) {
        const firstProgram = stops[ 0 ].program;
        linearRanges.push( new RenderLinearRange(
          Number.NEGATIVE_INFINITY,
          offset,
          firstProgram,
          firstProgram
        ) );
      }
      if ( min <= 1 && max >= 0 ) {
        addSectionRanges( 0, false );
      }
      if ( max > 1 ) {
        const lastProgram = stops[ stops.length - 1 ].program;
        linearRanges.push( new RenderLinearRange(
          1 + offset,
          Number.POSITIVE_INFINITY,
          lastProgram,
          lastProgram
        ) );
      }
    }
    else {
      const isReflect = extend === RenderExtend.Reflect;

      const floorMin = Math.floor( min );
      const ceilMax = Math.ceil( max );

      for ( let i = floorMin; i < ceilMax; i++ ) {
        addSectionRanges( i, isReflect ? i % 2 === 0 : false );
      }
    }

    // Merge adjacent ranges with the same (constant) program
    for ( let i = 0; i < linearRanges.length - 1; i++ ) {
      const range = linearRanges[ i ];
      const nextRange = linearRanges[ i + 1 ];

      if ( range.startProgram === range.endProgram && nextRange.startProgram === nextRange.endProgram && range.startProgram === nextRange.startProgram ) {
        linearRanges.splice( i, 2, new RenderLinearRange(
          range.start,
          nextRange.end,
          range.startProgram,
          range.startProgram
        ) );
        i--;
      }
    }

    return linearRanges;
  }

  public splitLinearGradients(): RenderableFace[] {
    const processedFaces: RenderableFace[] = [];
    const unprocessedFaces: RenderableFace[] = [ this ];

    const findLinearGradient = ( renderProgram: RenderProgram ): RenderLinearGradient | null => {
      let result: RenderLinearGradient | null = null;

      renderProgram.depthFirst( subProgram => {
        // TODO: early exit?
        if ( subProgram instanceof RenderLinearGradient ) {
          result = subProgram;
        }
      } );

      return result;
    };

    while ( unprocessedFaces.length ) {
      const face = unprocessedFaces.pop()!;

      const linearGradient = findLinearGradient( face.renderProgram );

      if ( linearGradient ) {

        const start = linearGradient.transform.timesVector2( linearGradient.start );
        const end = linearGradient.transform.timesVector2( linearGradient.end );

        const delta = end.minus( start );
        const normal = delta.timesScalar( 1 / delta.magnitudeSquared );
        const offset = normal.dot( start );

        // Should evaluate to 1 at the end
        assert && assert( Math.abs( normal.dot( end ) - offset - 1 ) < 1e-8 );

        const dotRange = face.face.getDotRange( normal );

        // relative to gradient "origin"
        const min = dotRange.min - offset;
        const max = dotRange.max - offset;

        const linearRanges = RenderableFace.getGradientLinearRanges( min, max, offset, linearGradient.extend, linearGradient.stops );

        if ( linearRanges.length < 2 ) {
          processedFaces.push( face );
        }
        else {
          const splitValues = linearRanges.map( range => range.start ).slice( 1 );
          const clippedFaces = face.face.getStripeLineClip( normal, splitValues, 0 );

          const renderableFaces = linearRanges.map( ( range, i ) => {
            const clippedFace = clippedFaces[ i ];

            const replacer = ( renderProgram: RenderProgram ): RenderProgram | null => {
              if ( renderProgram !== linearGradient ) {
                return null;
              }

              if ( range.startProgram === range.endProgram ) {
                return range.startProgram.replace( replacer );
              }
              else {
                // We need to rescale our normal for the linear blend, and then adjust our offset to point to the
                // "start":
                // From our original formulation:
                //   normal.dot( startPoint ) = range.start
                //   normal.dot( endPoint ) = range.end
                // with a difference of (range.end - range.start). We want this to be zero, so we rescale our normal:
                //   newNormal.dot( startPoint ) = range.start / ( range.end - range.start )
                //   newNormal.dot( endPoint ) = range.end / ( range.end - range.start )
                // And then we can adjust our offset such that:
                //   newNormal.dot( startPoint ) - offset = 0
                //   newNormal.dot( endPoint ) - offset = 1
                const scaledNormal = normal.timesScalar( 1 / ( range.end - range.start ) );
                const scaledOffset = range.start / ( range.end - range.start );

                return new RenderLinearBlend(
                  null,
                  scaledNormal,
                  scaledOffset,
                  range.startProgram.replace( replacer ),
                  range.endProgram.replace( replacer ),
                  linearGradient.colorSpace
                );
              }
            };

            return new RenderableFace( clippedFace, face.renderProgram.replace( replacer ), clippedFace.getBounds() );
          } ).filter( face => face.face.getArea() > 1e-8 );

          unprocessedFaces.push( ...renderableFaces );
        }
      }
      else {
        processedFaces.push( face );
      }
    }

    return processedFaces;
  }

  public splitRadialGradients(): RenderableFace[] {
    const processedFaces: RenderableFace[] = [];
    const unprocessedFaces: RenderableFace[] = [ this ];

    const findCircularRadialGradient = ( renderProgram: RenderProgram ): RenderRadialGradient | null => {
      let result: RenderRadialGradient | null = null;

      renderProgram.depthFirst( subProgram => {
        // TODO: early exit?
        if ( subProgram instanceof RenderRadialGradient && subProgram.start.equals( subProgram.end ) ) {
          result = subProgram;
        }
      } );

      return result;
    };

    while ( unprocessedFaces.length ) {
      const face = unprocessedFaces.pop()!;

      const radialGradient = findCircularRadialGradient( face.renderProgram );

      if ( radialGradient ) {
        const localClippableFace = face.face.getTransformed( radialGradient.transform.inverted() );

        const center = radialGradient.start;

        const distanceRange = localClippableFace.getDistanceRange( center );
      }
      else {
        processedFaces.push( face );
      }
    }

    return processedFaces;
  }
}

class AccumulatingFace {
  public faces = new Set<RationalFace>();
  public facesToProcess: RationalFace[] = [];
  public renderProgram: RenderProgram | null = null;
  public bounds: Bounds2 = Bounds2.NOTHING.copy();
  public clippedEdges: LinearEdge[] = [];
}

type OutputRaster = {
  addPartialPixel( color: Vector4, x: number, y: number ): void;
  addFullPixel( color: Vector4, x: number, y: number ): void;
  addFullRegion( color: Vector4, x: number, y: number, width: number, height: number ): void;
};

// TODO: type of raster that applies itself to a rectangle in the future?
class AccumulationRaster implements OutputRaster {
  public readonly accumulationBuffer: Vector4[] = [];

  public constructor( public readonly width: number, public readonly height: number ) {
    for ( let i = 0; i < width * height; i++ ) {
      this.accumulationBuffer.push( Vector4.ZERO.copy() );
    }
  }

  public addPartialPixel( color: Vector4, x: number, y: number ): void {
    const index = y * this.width + x;
    this.accumulationBuffer[ index ].add( color );
  }

  public addFullPixel( color: Vector4, x: number, y: number ): void {
    const index = y * this.width + x;
    this.accumulationBuffer[ index ].set( color );
  }

  public addFullRegion( color: Vector4, x: number, y: number, width: number, height: number ): void {
    for ( let j = 0; j < height; j++ ) {
      const rowIndex = ( y + j ) * this.width + x;
      for ( let i = 0; i < width; i++ ) {
        this.accumulationBuffer[ rowIndex + i ].set( color );
      }
    }
  }

  public toImageData(): ImageData {
    const imageData = new ImageData( this.width, this.height, { colorSpace: 'srgb' } );
    if ( assert ) {
      debugData!.imageData = imageData;
    }

    for ( let i = 0; i < this.accumulationBuffer.length; i++ ) {
      const accumulation = this.accumulationBuffer[ i ];
      const a = accumulation.w;

      // unpremultiply
      if ( a > 0 ) {
        const x = accumulation.x / a;
        const y = accumulation.y / a;
        const z = accumulation.z / a;

        // linear to sRGB
        const r = x <= 0.00313066844250063 ? x * 12.92 : 1.055 * Math.pow( x, 1 / 2.4 ) - 0.055;
        const g = y <= 0.00313066844250063 ? y * 12.92 : 1.055 * Math.pow( y, 1 / 2.4 ) - 0.055;
        const b = z <= 0.00313066844250063 ? z * 12.92 : 1.055 * Math.pow( z, 1 / 2.4 ) - 0.055;

        const index = 4 * i;
        imageData.data[ index ] = r * 255;
        imageData.data[ index + 1 ] = g * 255;
        imageData.data[ index + 2 ] = b * 255;
        imageData.data[ index + 3 ] = a * 255;
      }
    }

    return imageData;
  }
}

const scratchCombinedVector = new Vector4( 0, 0, 0, 0 );

// TODO: consider implementing a raster that JUST uses ImageData, and does NOT do linear (proper) blending
class CombinedRaster implements OutputRaster {
  public readonly accumulationArray: Float64Array;
  public readonly imageData: ImageData;
  private combined = false;

  public constructor( public readonly width: number, public readonly height: number ) {
    this.accumulationArray = new Float64Array( width * height * 4 );
    this.imageData = new ImageData( this.width, this.height, { colorSpace: 'srgb' } );
  }

  public addPartialPixel( color: Vector4, x: number, y: number ): void {
    const baseIndex = 4 * ( y * this.width + x );
    this.accumulationArray[ baseIndex ] += color.x;
    this.accumulationArray[ baseIndex + 1 ] += color.y;
    this.accumulationArray[ baseIndex + 2 ] += color.z;
    this.accumulationArray[ baseIndex + 3 ] += color.w;
  }

  public addFullPixel( color: Vector4, x: number, y: number ): void {
    // Be lazy, we COULD convert here, but we'll just do it at the end
    this.addPartialPixel( color, x, y );
  }

  public addFullRegion( color: Vector4, x: number, y: number, width: number, height: number ): void {
    const sRGB = CombinedRaster.convertToSRGB( color );
    for ( let j = 0; j < height; j++ ) {
      const rowIndex = ( y + j ) * this.width + x;
      for ( let i = 0; i < width; i++ ) {
        const baseIndex = 4 * ( rowIndex + i );
        const data = this.imageData.data;
        data[ baseIndex ] = sRGB.x;
        data[ baseIndex + 1 ] = sRGB.y;
        data[ baseIndex + 2 ] = sRGB.z;
        data[ baseIndex + 3 ] = sRGB.w;
      }
    }
  }

  // TODO: can we combine these methods of sRGB conversion without losing performance?
  // TODO: move this somewhere?
  private static convertToSRGB( color: Vector4 ): Vector4 {
    const accumulation = color;
    const a = accumulation.w;

    if ( a > 0 ) {
      // unpremultiply
      const x = accumulation.x / a;
      const y = accumulation.y / a;
      const z = accumulation.z / a;

      // linear to sRGB
      const r = x <= 0.00313066844250063 ? x * 12.92 : 1.055 * Math.pow( x, 1 / 2.4 ) - 0.055;
      const g = y <= 0.00313066844250063 ? y * 12.92 : 1.055 * Math.pow( y, 1 / 2.4 ) - 0.055;
      const b = z <= 0.00313066844250063 ? z * 12.92 : 1.055 * Math.pow( z, 1 / 2.4 ) - 0.055;

      return scratchCombinedVector.setXYZW(
        r * 255,
        g * 255,
        b * 255,
        a * 255
      );
    }
    else {
      return scratchCombinedVector.setXYZW( 0, 0, 0, 0 );
    }
  }

  public toImageData(): ImageData {
    if ( !this.combined ) {
      const quantity = this.accumulationArray.length / 4;
      for ( let i = 0; i < quantity; i++ ) {
        const baseIndex = i * 4;
        const a = this.accumulationArray[ baseIndex + 3 ];

        if ( a > 0 ) {
          // unpremultiply
          const x = this.accumulationArray[ baseIndex ] / a;
          const y = this.accumulationArray[ baseIndex + 1 ] / a;
          const z = this.accumulationArray[ baseIndex + 2 ] / a;

          // linear to sRGB
          const r = x <= 0.00313066844250063 ? x * 12.92 : 1.055 * Math.pow( x, 1 / 2.4 ) - 0.055;
          const g = y <= 0.00313066844250063 ? y * 12.92 : 1.055 * Math.pow( y, 1 / 2.4 ) - 0.055;
          const b = z <= 0.00313066844250063 ? z * 12.92 : 1.055 * Math.pow( z, 1 / 2.4 ) - 0.055;

          const index = 4 * i;
          // NOTE: ADDING HERE!!!! Don't change (we've set this for some pixels already)
          // Also, if we have a weird case where something sneaks in that is above an epsilon, so that we have a
          // barely non-zero linear value, we DO NOT want to wipe away something that saw a "almost full" pixel and
          // wrote into the imageData.
          this.imageData.data[ index ] += r * 255;
          this.imageData.data[ index + 1 ] += g * 255;
          this.imageData.data[ index + 2 ] += b * 255;
          this.imageData.data[ index + 3 ] += a * 255;
        }
      }
      this.combined = true;
    }

    return this.imageData;
  }
}

export default class Rasterize {

  private static clipScaleToIntegerEdges( paths: RenderPath[], bounds: Bounds2, scale: number ): IntegerEdge[] {
    const integerEdges = [];
    for ( let i = 0; i < paths.length; i++ ) {
      const path = paths[ i ];

      for ( let j = 0; j < path.subpaths.length; j++ ) {
        const subpath = path.subpaths[ j ];
        const clippedSubpath = PolygonClipping.boundsClipPolygon( subpath, bounds );

        for ( let k = 0; k < clippedSubpath.length; k++ ) {
          // TODO: when micro-optimizing, improve this pattern so we only have one access each iteration
          const p0 = clippedSubpath[ k ];
          const p1 = clippedSubpath[ ( k + 1 ) % clippedSubpath.length ];
          integerEdges.push( IntegerEdge.fromUnscaledPoints( path, scale, p0, p1 ) );
        }
      }
    }
    return integerEdges;
  }

  private static processIntegerEdgeIntersection( edgeA: IntegerEdge, edgeB: IntegerEdge ): void {
    const intersectionPoints = IntersectionPoint.intersectLineSegments(
      new BigIntVector2( BigInt( edgeA.x0 ), BigInt( edgeA.y0 ) ),
      new BigIntVector2( BigInt( edgeA.x1 ), BigInt( edgeA.y1 ) ),
      new BigIntVector2( BigInt( edgeB.x0 ), BigInt( edgeB.y0 ) ),
      new BigIntVector2( BigInt( edgeB.x1 ), BigInt( edgeB.y1 ) )
    );

    for ( let i = 0; i < intersectionPoints.length; i++ ) {
      const intersectionPoint = intersectionPoints[ i ];

      const t0 = intersectionPoint.t0;
      const t1 = intersectionPoint.t1;
      const point = intersectionPoint.point;

      // TODO: in WGSL, use atomicExchange to write a linked list of these into each edge
      // NOTE: We filter out endpoints of lines, since they wouldn't trigger a split in the segment anyway
      if ( !t0.equals( BigRational.ZERO ) && !t0.equals( BigRational.ONE ) ) {
        edgeA.intersections.push( new RationalIntersection( t0, point ) );
      }
      if ( !t1.equals( BigRational.ZERO ) && !t1.equals( BigRational.ONE ) ) {
        edgeB.intersections.push( new RationalIntersection( t1, point ) );
      }
    }
  }

  private static edgeIntersectionQuadratic( integerEdges: IntegerEdge[] ): void {
    // Compute intersections
    // TODO: improve on the quadratic!!!!
    // similar to BoundsIntersectionFilter.quadraticIntersect( integerBounds, integerEdges, ( edgeA, edgeB ) => {
    for ( let i = 0; i < integerEdges.length; i++ ) {
      const edgeA = integerEdges[ i ];
      const boundsA = edgeA.bounds;
      const xAEqual = edgeA.x0 === edgeA.x1;
      const yAEqual = edgeA.y0 === edgeA.y1;

      for ( let j = i + 1; j < integerEdges.length; j++ ) {
        const edgeB = integerEdges[ j ];
        const boundsB = edgeB.bounds;
        const someXEqual = xAEqual || edgeB.x0 === edgeB.x1;
        const someYEqual = yAEqual || edgeB.y0 === edgeB.y1;

        // Bounds min/max for overlap checks
        const minX = Math.max( boundsA.minX, boundsB.minX );
        const minY = Math.max( boundsA.minY, boundsB.minY );
        const maxX = Math.min( boundsA.maxX, boundsB.maxX );
        const maxY = Math.min( boundsA.maxY, boundsB.maxY );

        // If one of the segments is (e.g.) vertical, we'll need to allow checks for overlap ONLY on the x value, otherwise
        // we can have a strict inequality check. This also applies to horizontal segments and the y value.
        // The reason this is OK is because if the segments are both (e.g.) non-vertical, then if the bounds only meet
        // at a single x value (and not a continuos area of overlap), THEN the only intersection would be at the
        // endpoints (which we would filter out and not want anyway).
        if (
          someXEqual ? ( maxX >= minX ) : ( maxX > minX ) &&
          someYEqual ? ( maxY >= minY ) : ( maxY > minY )
        ) {
          Rasterize.processIntegerEdgeIntersection( edgeA, edgeB );
        }
      }
    }
  }

  private static splitIntegerEdges( integerEdges: IntegerEdge[] ): RationalHalfEdge[] {
    let edgeIdCounter = 0;
    const rationalHalfEdges: RationalHalfEdge[] = [];

    // TODO: reduce closures
    integerEdges.forEach( integerEdge => {
      const points = [
        new BigRationalVector2( BigRational.whole( integerEdge.x0 ), BigRational.whole( integerEdge.y0 ) )
      ];

      let lastT = BigRational.ZERO;

      integerEdge.intersections.sort( ( a, b ) => {
        // TODO: we'll need to map this over with functions
        return a.t.compareCrossMul( b.t );
      } );

      // Deduplicate
      integerEdge.intersections.forEach( intersection => {
        if ( !lastT.equals( intersection.t ) ) {
          points.push( intersection.point );
        }
        lastT = intersection.t;
      } );

      points.push( new BigRationalVector2( BigRational.whole( integerEdge.x1 ), BigRational.whole( integerEdge.y1 ) ) );

      for ( let i = 0; i < points.length - 1; i++ ) {
        const p0 = points[ i ];
        const p1 = points[ i + 1 ];

        // We will remove degenerate edges now, so during the deduplication we won't collapse them together
        if ( !p0.equals( p1 ) ) {
          const edgeId = edgeIdCounter++;
          const forwardEdge = new RationalHalfEdge( edgeId, p0, p1 );
          const reverseEdge = new RationalHalfEdge( edgeId, p1, p0 );
          forwardEdge.reversed = reverseEdge;
          reverseEdge.reversed = forwardEdge;
          forwardEdge.windingMap.addWindingNumber( integerEdge.renderPath, 1 );
          reverseEdge.windingMap.addWindingNumber( integerEdge.renderPath, -1 );

          const deltaX = integerEdge.x1 - integerEdge.x0;
          const deltaY = integerEdge.y1 - integerEdge.y0;

          // We compute slope here due to rational precision (while it would be possible to create a larger rational
          // number later and reduce it, here we're starting with integers, so we don't have to do as much).
          const discriminator = deltaY === 0 ? ( deltaX > 0 ? 0 : 2 ) : ( deltaY > 0 ? 1 : 3 );
          const slope = deltaY === 0 ? BigRational.ZERO : new BigRational( deltaX, deltaY ).reduced();

          // We store the slope and discriminator here, as that allows us to tell the order-difference between two
          // edges that have one point the same. This works here, because we have already broken lines up at the
          // endpoints in the case of overlap, so that if it has the same start point, discriminator and slope, then it
          // WILL have the same end point, and thus will be the same effective edge.
          forwardEdge.discriminator = discriminator;
          reverseEdge.discriminator = ( discriminator + 2 ) % 4;
          forwardEdge.slope = slope;
          reverseEdge.slope = slope;

          rationalHalfEdges.push( forwardEdge );
          rationalHalfEdges.push( reverseEdge );
        }
      }
    } );
    return rationalHalfEdges;
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
                if ( maxIntersectionX.compareCrossMul( candidateMaxIntersectionX ) < 0 ) {
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

    const color = constColor || renderProgram.evaluate( pixelFace.getCentroid( area ).minus( translation ) );
    outputRaster.addPartialPixel( color.timesScalar( area ), x, y );
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
          const centroid = scratchFullAreaVector.setXY( x + 0.5, y + 0.5 ).minus( translation );
          outputRaster.addFullPixel( renderProgram.evaluate( centroid ), x, y );
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

          if ( assert ) {
            const oldMinFace = clippableFace.getClipped( new Bounds2( minX, minY, xSplit, maxY ) );
            const oldMaxFace = clippableFace.getClipped( new Bounds2( xSplit, minY, maxX, maxY ) );

            if ( Math.abs( minFace.getArea() - oldMinFace.getArea() ) > 1e-8 || Math.abs( maxFace.getArea() - oldMaxFace.getArea() ) > 1e-8 ) {
              assert( false, 'binary X clip issue' );
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

          if ( assert ) {
            const oldMinFace = clippableFace.getClipped( new Bounds2( minX, minY, maxX, ySplit ) );
            const oldMaxFace = clippableFace.getClipped( new Bounds2( minX, ySplit, maxX, maxY ) );

            if ( Math.abs( minFace.getArea() - oldMinFace.getArea() ) > 1e-8 || Math.abs( maxFace.getArea() - oldMaxFace.getArea() ) > 1e-8 ) {
              assert( false, 'binary Y clip issue' );
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

  public static rasterizeRenderProgram( renderProgram: RenderProgram, bounds: Bounds2 ): Record<string, IntentionalAny> | null {

    if ( assert ) {
      debugData = {
        areas: []
      };
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

    const integerBounds = new Bounds2(
      Utils.roundSymmetric( bounds.minX * scale ),
      Utils.roundSymmetric( bounds.minY * scale ),
      Utils.roundSymmetric( bounds.maxX * scale ),
      Utils.roundSymmetric( bounds.maxY * scale )
    );
    if ( assert ) { debugData!.integerBounds = integerBounds; }

    const integerEdges = Rasterize.clipScaleToIntegerEdges( paths, bounds, scale );
    if ( assert ) { debugData!.integerEdges = integerEdges; }

    Rasterize.edgeIntersectionQuadratic( integerEdges );

    const rationalHalfEdges = Rasterize.splitIntegerEdges( integerEdges );

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
    // let renderableFaces = Rasterize.toPolygonalRenderableFaces( renderedFaces, scale, translation );
    // let renderableFaces = Rasterize.toEdgedRenderableFaces( renderedFaces, scale, translation );
    // let renderableFaces = Rasterize.toFullyCombinedRenderableFaces( renderedFaces, scale, translation );
    let renderableFaces = Rasterize.toSimplifyingCombinedRenderableFaces( renderedFaces, scale, translation );

    // TODO: FLESH OUT
    const SIMPLIFY_GRADIENTS = true;
    if ( SIMPLIFY_GRADIENTS ) {
      renderableFaces = renderableFaces.flatMap( face => face.splitLinearGradients() );
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

    const imageData = outputRaster.toImageData();

    if ( assert ) {
      const canvas = document.createElement( 'canvas' );
      canvas.width = rasterWidth;
      canvas.height = rasterHeight;
      const context = canvas.getContext( '2d' )!;
      context.putImageData( imageData, 0, 0 );
      debugData!.canvas = canvas;
    }

    return ( debugData! ) || null;
  }
}

scenery.register( 'Rasterize', Rasterize );
