// Copyright 2023, University of Colorado Boulder

/**
 * Maillot '92 polygon clipping algorithm, using Cohen-Sutherland clipping
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */
import Vector2 from '../../../../dot/js/Vector2.js';
import Bounds2 from '../../../../dot/js/Bounds2.js';
import { scenery } from '../../imports.js';

type Code = number;

// TODO: parallelize this (should be possible)

const X_MAX_CODE = 0x1;
const Y_MAX_CODE = 0x2;
const X_MIN_CODE = 0x4;
const Y_MIN_CODE = 0x8;
const TWO_BITS_CODE = 0x10;
const TWO_BITS_MASK = 0xf;

const turningPointOffset: Record<number, number> = { 1: -3, 2: -6, 4: 3, 8: 6 };
const codeToCorner: Record<number, number> = { 3: 2, 6: 3, 9: 1, 12: 0 };

class CodedVector2 extends Vector2 {
  public constructor( x: number, y: number, public code: Code ) {
    super( x, y );
  }

  public override copy(): CodedVector2 {
    return new CodedVector2( this.x, this.y, this.code );
  }

  public toVector2(): Vector2 {
    return new Vector2( this.x, this.y );
  }

  public updateCode( bounds: Bounds2 ): void {
    this.code = PolygonClipping.getCode( this.x, this.y, bounds );
  }

  public static create( p: Vector2, bounds: Bounds2 ): CodedVector2 {
    return new CodedVector2( p.x, p.y, PolygonClipping.getCode( p.x, p.y, bounds ) );
  }
}

class Simplifier {

  private points: Vector2[] = [];

  public constructor() {
    // NOTHING NEEDED
  }

  public reset(): void {
    this.points = [];
  }

  public add( x: number, y: number ): void {
    if ( this.points.length >= 1 ) {

      const lastPoint = this.points[ this.points.length - 1 ];
      const xEquals = lastPoint.x === x;
      const yEquals = lastPoint.y === y;

      // If we are equal to the last point, NO-OP
      if ( xEquals && yEquals ) {
        return;
      }

      if ( this.points.length >= 2 ) {
        const secondLastPoint = this.points[ this.points.length - 2 ];
        const secondXEquals = secondLastPoint.x === x;
        const secondYEquals = secondLastPoint.y === y;

        // If we are equal to the second-to-last point, we can just undo our last point
        if ( secondXEquals && secondYEquals ) {
          this.points.pop(); // TODO: pooling freeToPool?
          return;
        }

        // X-collinearity check (if we would have 3 points with the same X, we can just remove the middle one)
        if ( xEquals && secondXEquals ) {
          // Instead of adding new one and removing the middle one, we can just update the last one
          lastPoint.y = y;
          return;
        }

        // Y-collinearity check (if we would have 3 points with the same Y, we can just remove the middle one)
        if ( yEquals && secondYEquals ) {
          // Instead of adding new one and removing the middle one, we can just update the last one
          lastPoint.x = x;
          return;
        }
      }
    }

    // TODO: pooling?
    this.points.push( new Vector2( x, y ) );
  }

  public finalize(): Vector2[] {
    // TODO: add more comprehensive testing for this! Tested a simple example

    // TODO: is this complexity worth porting to WGSL?
    // We'll handle our equality and collinearity checks. Because we could have a situation where the end of our points
    // retraces the start of our points (but backwards, is possible since the middle could be fine), we'll actually need
    // iteration to rewind this. Repeatedly check equality/collinearity until we don't change anything.
    let changed: boolean;
    do {
      changed = false;
      // Equality check (start/end)
      if ( this.points.length >= 2 ) {
        const firstPoint = this.points[ 0 ];
        const lastPoint = this.points[ this.points.length - 1 ];

        // If the first and last points are equal, remove the last point
        if ( firstPoint.equals( lastPoint ) ) {
          this.points.pop(); // TODO: pooling freeToPool?
          changed = true;
        }
      }

      // Collinearity check (the first two points, and last two points)
      if ( this.points.length >= 3 ) {
        // NOTE: It is technically possible that this happens with exactly three points left (that are collinear).
        // This should still work to reduce it, but will "garble" the order. We don't care, since the resulting
        // polygon would have no area.
        const firstPoint = this.points[ 0 ];
        const lastPoint = this.points[ this.points.length - 1 ];

        const xEquals = firstPoint.x === lastPoint.x;
        const yEquals = firstPoint.y === lastPoint.y;

        if ( xEquals || yEquals ) {
          const secondPoint = this.points[ 1 ];
          const secondLastPoint = this.points[ this.points.length - 2 ];

          if (
            ( xEquals && firstPoint.x === secondPoint.x ) ||
            ( yEquals && firstPoint.y === secondPoint.y )
          ) {
            // TODO: We can record the "starting" index, and avoid repeated shifts (that are probably horrible for perf)
            // TODO: See if this is significant, or needed for WGSL
            this.points.shift(); // TODO: pooling freeToPool?
            changed = true;
          }

          if (
            ( xEquals && lastPoint.x === secondLastPoint.x ) ||
            ( yEquals && lastPoint.y === secondLastPoint.y )
          ) {
            this.points.pop(); // TODO: pooling freeToPool?
            changed = true;
          }
        }
      }
    } while ( changed );

    // Clear out to an empty array if we won't have enough points to have any area
    if ( this.points.length <= 2 ) {
      this.points.length = 0;
    }

    return this.points;
  }
}

const scratchStartPoint = new Vector2( 0, 0 );
const scratchEndPoint = new Vector2( 0, 0 );
const simplifier = new Simplifier();

export default class PolygonClipping {

  // TODO: This is a homebrew algorithm that for now generates a bunch of extra points, but is hopefully pretty simple
  public static boundsClipPolygon( polygon: Vector2[], bounds: Bounds2 ): Vector2[] {

    simplifier.reset();

    const centerX = bounds.centerX;
    const centerY = bounds.centerY;

    // TODO: optimize this
    for ( let i = 0; i < polygon.length; i++ ) {
      const startPoint = polygon[ i ];
      const endPoint = polygon[ ( i + 1 ) % polygon.length ];

      const clippedStartPoint = scratchStartPoint.set( startPoint );
      const clippedEndPoint = scratchEndPoint.set( endPoint );

      const clipped = PolygonClipping.matthesDrakopoulosClip( clippedStartPoint, clippedEndPoint, bounds );

      let startXLess;
      let startYLess;
      let endXLess;
      let endYLess;

      // TODO: we're adding all sorts of duplicate unnecessary points!

      const needsStartCorner = !clipped || !startPoint.equals( clippedStartPoint );
      const needsEndCorner = !clipped || !endPoint.equals( clippedEndPoint );

      if ( needsStartCorner ) {
        startXLess = startPoint.x < centerX;
        startYLess = startPoint.y < centerY;
      }
      if ( needsEndCorner ) {
        endXLess = endPoint.x < centerX;
        endYLess = endPoint.y < centerY;
      }

      // TODO: don't rely on the simplifier so much! (especially with arrays?)
      if ( needsStartCorner ) {
        simplifier.add(
          startXLess ? bounds.minX : bounds.maxX,
          startYLess ? bounds.minY : bounds.maxY
        );
      }
      if ( clipped ) {
        simplifier.add( clippedStartPoint.x, clippedStartPoint.y );
        simplifier.add( clippedEndPoint.x, clippedEndPoint.y );
      }
      else {
        if ( startXLess !== endXLess && startYLess !== endYLess ) {
          // we crossed from one corner to the opposite, but didn't hit. figure out which corner we passed
          // we're diagonal, so solving for y=centerY should give us the info we need
          const y = startPoint.y + ( endPoint.y - startPoint.y ) * ( centerX - startPoint.x ) / ( endPoint.x - startPoint.x );

          // Based on whether we are +x+y => -x-y or -x+y => +x-y
          const startSame = startXLess === startYLess;
          const yGreater = y > centerY;
          simplifier.add(
            startSame === yGreater ? bounds.minX : bounds.maxX,
            yGreater ? bounds.maxY : bounds.minY
          );
        }
      }
      if ( needsEndCorner ) {
        simplifier.add(
          endXLess ? bounds.minX : bounds.maxX,
          endYLess ? bounds.minY : bounds.maxY
        );
      }
    }

    const result = simplifier.finalize();;

    simplifier.reset();

    return result;
  }

  /**
   * From "Another Simple but Faster Method for 2D Line Clipping" (2019)
   * by Dimitrios Matthes and Vasileios Drakopoulos
   */
  private static matthesDrakopoulosClip( p0: Vector2, p1: Vector2, bounds: Bounds2 ): boolean {
    const x1 = p0.x;
    const y1 = p0.y;
    const x2 = p1.x;
    const y2 = p1.y;
    // TODO: a version without requiring a Bounds2?
    const minX = bounds.minX;
    const minY = bounds.minY;
    const maxX = bounds.maxX;
    const maxY = bounds.maxY;

    if ( !( x1 < minX && x2 < minX ) && !( x1 > maxX && x2 > maxX ) ) {
      if ( !( y1 < minY && y2 < minY ) && !( y1 > maxY && y2 > maxY ) ) {
        // TODO: consider NOT computing these if we don't need them? We probably won't use both?
        const ma = ( y2 - y1 ) / ( x2 - x1 );
        const mb = ( x2 - x1 ) / ( y2 - y1 );

        // TODO: on GPU, consider if we should extract out partial subexpressions below

        // Unrolled (duplicated essentially)
        if ( p0.x < minX ) {
          p0.x = minX;
          p0.y = ma * ( minX - x1 ) + y1;
        }
        else if ( p0.x > maxX ) {
          p0.x = maxX;
          p0.y = ma * ( maxX - x1 ) + y1;
        }
        if ( p0.y < minY ) {
          p0.y = minY;
          p0.x = mb * ( minY - y1 ) + x1;
        }
        else if ( p0.y > maxY ) {
          p0.y = maxY;
          p0.x = mb * ( maxY - y1 ) + x1;
        }
        // Second unrolled form
        if ( p1.x < minX ) {
          p1.x = minX;
          p1.y = ma * ( minX - x1 ) + y1;
        }
        else if ( p1.x > maxX ) {
          p1.x = maxX;
          p1.y = ma * ( maxX - x1 ) + y1;
        }
        if ( p1.y < minY ) {
          p1.y = minY;
          p1.x = mb * ( minY - y1 ) + x1;
        }
        else if ( p1.y > maxY ) {
          p1.y = maxY;
          p1.x = mb * ( maxY - y1 ) + x1;
        }
        if ( !( p0.x < minX && p1.x < minX ) && !( p0.x > maxX && p1.x > maxX ) ) {
          return true;
        }
      }
    }

    return false;
  }

  // The Maillot extension of the Cohen-Sutherland encoding of points
  public static getCode( x: number, y: number, bounds: Bounds2 ): Code {
    if ( x < bounds.minX ) {
      if ( y > bounds.maxY ) {
        return 0x16;
      }
      else if ( y < bounds.minY ) {
        return 0x1c;
      }
      else {
        return 0x4;
      }
    }
    else if ( x > bounds.maxX ) {
      if ( y > bounds.maxY ) {
        return 0x13;
      }
      else if ( y < bounds.minY ) {
        return 0x19;
      }
      else {
        return 0x1;
      }
    }
    else if ( y > bounds.maxY ) {
      return 0x2;
    }
    else if ( y < bounds.minY ) {
      return 0x8;
    }
    else {
      return 0;
    }
  }

  // Mutates the vectors, returns whether the line segment is at least partially within the bounds
  private static cohenSutherlandClip( p0: CodedVector2, p1: CodedVector2, bounds: Bounds2 ): boolean {
    while ( ( p0.code | p1.code ) !== 0 ) {
      if ( ( p0.code & p1.code & ~TWO_BITS_CODE ) !== 0 ) {
        return false;
      }

      // Choose the first point not in the window
      const c = p0.code === 0 ? p1.code : p0.code;

      let x: number;
      let y: number;

      // Now clip against line corresponding to first nonzero bit
      if ( ( X_MIN_CODE & c ) !== 0 ) {
        x = bounds.left;
        y = p0.y + ( p1.y - p0.y ) * ( bounds.left - p0.x ) / ( p1.x - p0.x );
      }
      else if ( ( X_MAX_CODE & c ) !== 0 ) {
        x = bounds.right;
        y = p0.y + ( p1.y - p0.y ) * ( bounds.right - p0.x ) / ( p1.x - p0.x );
      }
      else if ( ( Y_MIN_CODE & c ) !== 0 ) {
        x = p0.x + ( p1.x - p0.x ) * ( bounds.top - p0.y ) / ( p1.y - p0.y );
        y = bounds.top;
      }
      else if ( ( Y_MAX_CODE & c ) !== 0 ) {
        x = p0.x + ( p1.x - p0.x ) * ( bounds.bottom - p0.y ) / ( p1.y - p0.y );
        y = bounds.bottom;
      }
      else {
        throw new Error( 'cohenSutherlandClip: Unknown case' );
      }
      if ( c === p0.code ) {
        p0.x = x;
        p0.y = y;
        p0.updateCode( bounds );
      }
      else {
        p1.x = x;
        p1.y = y;
        p1.updateCode( bounds );
      }
    }

    return true;
  }

  // Will mutate the vectors!
  public static boundsClipSegment( p0: Vector2, p1: Vector2, bounds: Bounds2 ): boolean {
    const cp0 = CodedVector2.create( p0, bounds );
    const cp1 = CodedVector2.create( p1, bounds );
    const result = PolygonClipping.cohenSutherlandClip( cp0, cp1, bounds );
    if ( result ) {
      p0.set( cp0 );
      p1.set( cp1 );
    }
    return result;
  }

  private static isLeftOfLine( l1: Vector2, l2: Vector2, p: Vector2 ): boolean {
    return ( l2.x - l1.x ) * ( p.y - l1.y ) > ( l2.y - l1.y ) * ( p.x - l1.x );
  }

  // TODO: bad case phet.scenery.PolygonClipping.boundsClipPolygon( [ phet.dot.v2( 500, 500 ), phet.dot.v2( 700, 500 ), phet.dot.v2( 700, 600 ) ], new phet.dot.Bounds2( 600, 505, 601, 506 ) )
  /*
     Adds points:
     Vector2 {x: 601, y: 505}
     Vector2 {x: 600, y: 505}
     Vector2 {x: 601, y: 505}
     Vector2 {x: 601, y: 506}
   */
  // Maillot '92 polygon clipping algorithm, using Cohen-Sutherland clipping
  // TODO: get rid of if we're not using it
  public static buggyMaillotClipping( polygon: Vector2[], bounds: Bounds2 ): Vector2[] {

    const minMin = new Vector2( bounds.minX, bounds.minY );
    const minMax = new Vector2( bounds.minX, bounds.maxY );
    const maxMin = new Vector2( bounds.maxX, bounds.minY );
    const maxMax = new Vector2( bounds.maxX, bounds.maxY );

    const clippingWindow = [ minMin, maxMin, maxMax, minMax ]; // 0x1c, 0x19, 0x13, 0x16 codes for turning points

    const simplifier = new Simplifier();

    let startPoint = CodedVector2.create( polygon[ polygon.length - 1 ], bounds );
    let turningPointCode;
    for ( let i = 0; i < polygon.length; i++ ) {
      const endPoint = CodedVector2.create( polygon[ i ], bounds );
      turningPointCode = endPoint.code;

      const clippedStartPoint = startPoint.copy();
      const clippedEndPoint = endPoint.copy();
      const clipped = PolygonClipping.cohenSutherlandClip( clippedStartPoint, clippedEndPoint, bounds );

      // If the edge is at least partially within our bounds, we can handle the simple case
      if ( clipped ) {
        if ( !clippedStartPoint.equals( startPoint ) ) {
          simplifier.add( clippedStartPoint.x, clippedStartPoint.y );
        }
        simplifier.add( clippedEndPoint.x, clippedStartPoint.y );
      }
      else {
        // Resolve cases
        const startCorner = ( startPoint.code & TWO_BITS_CODE ) !== 0;
        const endCorner = ( endPoint.code & TWO_BITS_CODE ) !== 0;

        if (
          startCorner && endCorner && ( startPoint.code & endPoint.code & TWO_BITS_MASK ) === 0
        ) {
          // 2-2 case
          const startToTheLeft = PolygonClipping.isLeftOfLine( maxMin, minMax, startPoint );
          const endToTheLeft = PolygonClipping.isLeftOfLine( maxMin, minMax, endPoint );

          let turningPoint;
          if ( startToTheLeft && endToTheLeft ) {
            turningPoint = minMin;
          }
          else if ( !startToTheLeft && !endToTheLeft ) {
            turningPoint = maxMax;
          }
          else if ( PolygonClipping.isLeftOfLine( minMin, maxMax, startPoint ) ) {
            turningPoint = minMax;
          }
          else {
            turningPoint = maxMin;
          }
          simplifier.add( turningPoint.x, turningPoint.y );
        }
        else if ( !startCorner && endCorner && ( startPoint.code & endPoint.code ) === 0 ) {
          // 1-2 case
          const code = endPoint.code + turningPointOffset[ startPoint.code ];
          const turningPoint = clippingWindow[ codeToCorner[ code & TWO_BITS_MASK ] ];
          simplifier.add( turningPoint.x, turningPoint.y );
        }
        else if ( startCorner && !endCorner && ( startPoint.code & endPoint.code ) === 0 ) {
          // 2-1 case
          turningPointCode = startPoint.code + turningPointOffset[ endPoint.code ];
        }
        else if ( !startCorner && !endCorner && startPoint.code !== endPoint.code ) {
          // 1-1 case
          turningPointCode |= startPoint.code | TWO_BITS_CODE;
        }
      }

      // Basic turning point test
      if ( ( turningPointCode & TWO_BITS_CODE ) !== 0 ) {
        const turningPoint = clippingWindow[ codeToCorner[ turningPointCode & TWO_BITS_MASK ] ];
        simplifier.add( turningPoint.x, turningPoint.y );
      }

      startPoint = endPoint;
    }

    return simplifier.finalize();
  }
}

scenery.register( 'PolygonClipping', PolygonClipping );
