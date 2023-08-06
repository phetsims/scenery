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

  private simplify(): boolean {
    // equality check
    if ( this.points.length >= 2 ) {
      const p0 = this.points[ this.points.length - 1 ];
      const p1 = this.points[ this.points.length - 2 ];
      if ( p0.equals( p1 ) ) {
        this.points.pop();
        return true;
      }
    }

    // axis-aligned collinear check
    if ( this.points.length >= 3 ) {
      const p0 = this.points[ this.points.length - 1 ];
      const p1 = this.points[ this.points.length - 2 ];
      const p2 = this.points[ this.points.length - 3 ];

      if ( ( p0.x === p1.x && p1.x === p2.x ) || ( p0.y === p1.y && p1.y === p2.y ) ) {
        this.points.pop();
        this.points.pop();
        this.points.push( p0 );
        return true;
      }
    }

    return false;
  }

  private simplifyLoop(): void {
    let needsSimplify: boolean;
    do {
      needsSimplify = this.simplify();
    } while ( needsSimplify );
  }

  public add( point: Vector2 ): void {
    this.points.push( point );
    this.simplifyLoop();
  }

  public finalize(): Vector2[] {
    // If we don't have 3 points, we won't have any area
    if ( this.points.length >= 3 ) {
      // TODO: better logic that wouldn't be such bad performance?
      // Handle collinearity/equality between the start/end
      this.points.push( this.points.shift()! );
      this.simplifyLoop();
      this.points.push( this.points.shift()! );
      this.simplifyLoop();
      return this.points;
    }
    else {
      return [];
    }
  }
}

export default class PolygonClipping {

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

  // TODO: This is a homebrew algorithm that for now generates a bunch of extra points, but is hopefully pretty simple
  public static boundsClipPolygon( polygon: Vector2[], bounds: Bounds2 ): Vector2[] {
    const simplifier = new Simplifier();

    const center = bounds.center;

    // TODO: optimize this
    for ( let i = 0; i < polygon.length; i++ ) {
      const startPoint = polygon[ i ];
      const endPoint = polygon[ ( i + 1 ) % polygon.length ];

      const clippedStartPoint = CodedVector2.create( startPoint, bounds );
      const clippedEndPoint = CodedVector2.create( endPoint, bounds );

      // TODO: liang-barsky or something better!!!
      const clipped = PolygonClipping.cohenSutherlandClip( clippedStartPoint, clippedEndPoint, bounds );

      let startXLess;
      let startYLess;
      let endXLess;
      let endYLess;

      // TODO: we're adding all sorts of duplicate unnecessary points!

      const needsStartCorner = !clipped || !startPoint.equals( clippedStartPoint );
      const needsEndCorner = !clipped || !endPoint.equals( clippedEndPoint );

      if ( needsStartCorner ) {
        startXLess = startPoint.x < center.x;
        startYLess = startPoint.y < center.y;
      }
      if ( needsEndCorner ) {
        endXLess = endPoint.x < center.x;
        endYLess = endPoint.y < center.y;
      }

      // TODO: don't rely on the simplifier so much! (especially with arrays?)
      if ( needsStartCorner ) {
        simplifier.add( new Vector2(
          startXLess ? bounds.minX : bounds.maxX,
          startYLess ? bounds.minY : bounds.maxY
        ) );
      }
      if ( clipped ) {
        simplifier.add( clippedStartPoint.toVector2() );
        simplifier.add( clippedEndPoint.toVector2() );
      }
      else {
        if ( startXLess !== endXLess && startYLess !== endYLess ) {
          // we crossed from one corner to the opposite, but didn't hit. figure out which corner we passed
          // we're diagonal, so solving for y=centerY should give us the info we need
          const y = startPoint.y + ( endPoint.y - startPoint.y ) * ( center.x - startPoint.x ) / ( endPoint.x - startPoint.x );

          simplifier.add(
            // Based on whether we are +x+y => -x-y or -x+y => +x-y
            ( startXLess === startYLess ) ? (
              y > 0 ? new Vector2( bounds.minX, bounds.maxY ) : new Vector2( bounds.maxX, bounds.minY )
            ) : (
              y > 0 ? new Vector2( bounds.maxX, bounds.maxY ) : new Vector2( bounds.minX, bounds.minY )
            )
          );
        }
      }
      if ( needsEndCorner ) {
        simplifier.add( new Vector2(
          endXLess ? bounds.minX : bounds.maxX,
          endYLess ? bounds.minY : bounds.maxY
        ) );
      }
    }

    return simplifier.finalize();
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
          simplifier.add( clippedStartPoint.toVector2() );
        }
        simplifier.add( clippedEndPoint.toVector2() );
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
          simplifier.add( turningPoint );
        }
        else if ( !startCorner && endCorner && ( startPoint.code & endPoint.code ) === 0 ) {
          // 1-2 case
          const code = endPoint.code + turningPointOffset[ startPoint.code ];
          const turningPoint = clippingWindow[ codeToCorner[ code & TWO_BITS_MASK ] ];
          simplifier.add( turningPoint );
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
        simplifier.add( turningPoint );
      }

      startPoint = endPoint;
    }

    return simplifier.finalize();
  }
}

scenery.register( 'PolygonClipping', PolygonClipping );
