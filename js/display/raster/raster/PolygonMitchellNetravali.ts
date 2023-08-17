// Copyright 2023, University of Colorado Boulder

/**
 * Mitchell-Netravali filter (B=1/3, C=1/3 ) contribution given a polygon
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */
import { LinearEdge, PolygonClipping, scenery } from '../../../imports.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Bounds2 from '../../../../../dot/js/Bounds2.js';

const bounds00 = new Bounds2( 0, 0, 1, 1 );
const bounds10 = bounds00.shiftedXY( 1, 0 );
const bounds01 = bounds00.shiftedXY( 0, 1 );
const bounds11 = bounds00.shiftedXY( 1, 1 );
const boundsn00 = bounds00.shiftedXY( -1, 0 );
const boundsn10 = bounds00.shiftedXY( -2, 0 );
const boundsn01 = bounds00.shiftedXY( -1, 1 );
const boundsn11 = bounds00.shiftedXY( -2, 1 );
const bounds0n0 = bounds00.shiftedXY( 0, -1 );
const bounds1n0 = bounds00.shiftedXY( 1, -1 );
const bounds0n1 = bounds00.shiftedXY( 0, -2 );
const bounds1n1 = bounds00.shiftedXY( 1, -2 );
const boundsn0n0 = bounds00.shiftedXY( -1, -1 );
const boundsn1n0 = bounds00.shiftedXY( -2, -1 );
const boundsn0n1 = bounds00.shiftedXY( -1, -2 );
const boundsn1n1 = bounds00.shiftedXY( -2, -2 );

type Case = ( p0x: number, p0y: number, p1x: number, p1y: number ) => number;

export default class PolygonMitchellNetravali {
  // Values for the three cases, if presented with a full "pixel", e.g.
  // PolygonMitchellNetravali.evaluateCase00( 0, 0, 1, 0 ) +
  // PolygonMitchellNetravali.evaluateCase00( 1, 0, 1, 1 ) +
  // PolygonMitchellNetravali.evaluateCase00( 1, 1, 0, 1 ) +
  // PolygonMitchellNetravali.evaluateCase00( 0, 1, 0, 0 )
  // 0.2640817901234568
  // PolygonMitchellNetravali.evaluateCase10( 1, 0, 2, 0 ) +
  // PolygonMitchellNetravali.evaluateCase10( 2, 0, 2, 1 ) +
  // PolygonMitchellNetravali.evaluateCase10( 2, 1, 1, 1 ) +
  // PolygonMitchellNetravali.evaluateCase10( 1, 1, 1, 0 )
  // -0.007137345679012345
  // PolygonMitchellNetravali.evaluateCase11( 1, 1, 2, 1 ) +
  // PolygonMitchellNetravali.evaluateCase11( 2, 1, 2, 2 ) +
  // PolygonMitchellNetravali.evaluateCase11( 2, 2, 1, 2 ) +
  // PolygonMitchellNetravali.evaluateCase11( 1, 2, 1, 1 )
  // 0.0001929012345679021
  public static full00 = 0.2640817901234568;
  public static full10 = -0.007137345679012345;
  public static full11 = 0.0001929012345679021;

  public static evaluateFull( pointX: number, pointY: number, minX: number, minY: number ): number {
    const offsetX = minX - pointX;
    const offsetY = minY - pointY;

    assert && assert( offsetX === -2 || offsetX === -1 || offsetX === 0 || offsetX === 1 );
    assert && assert( offsetY === -2 || offsetY === -1 || offsetY === 0 || offsetY === 1 );

    const xCentral = offsetX === 0 || offsetX === -1;
    const yCentral = offsetY === 0 || offsetY === -1;

    if ( xCentral && yCentral ) {
      return PolygonMitchellNetravali.full00;
    }
    else if ( !xCentral && !yCentral ) {
      return PolygonMitchellNetravali.full11;
    }
    else {
      return PolygonMitchellNetravali.full10;
    }
  }

  /**
   * Evaluates the contribution of the (clipped) polygon to the filter at the given point. minX/minY note the lower
   * coordinates of the clipped polygon unit pixel.
   */
  public static evaluateClippedEdges( edges: LinearEdge[], pointX: number, pointY: number, minX: number, minY: number ): number {
    const offsetX = minX - pointX;
    const offsetY = minY - pointY;

    // TODO: hardcode things more, so we don't need this logic

    assert && assert( offsetX === -2 || offsetX === -1 || offsetX === 0 || offsetX === 1 );
    assert && assert( offsetY === -2 || offsetY === -1 || offsetY === 0 || offsetY === 1 );

    const xCentral = offsetX === 0 || offsetX === -1;
    const yCentral = offsetY === 0 || offsetY === -1;
    const xPositive = offsetX >= 0;
    const yPositive = offsetY >= 0;

    const sign = ( xPositive === yPositive ) ? 1 : -1;
    let evaluator: ( p0x: number, p0y: number, p1x: number, p1y: number ) => number;
    if ( xCentral && yCentral ) {
      evaluator = PolygonMitchellNetravali.evaluateCase00;
    }
    else if ( !xCentral && !yCentral ) {
      evaluator = PolygonMitchellNetravali.evaluateCase11;
    }
    else {
      evaluator = PolygonMitchellNetravali.evaluateCase10;
    }
    const transpose = xCentral && !yCentral;

    let sum = 0;

    for ( let i = 0; i < edges.length; i++ ) {
      const edge = edges[ i ];

      const p0x = Math.abs( edge.startPoint.x - pointX );
      const p0y = Math.abs( edge.startPoint.y - pointY );
      const p1x = Math.abs( edge.endPoint.x - pointX );
      const p1y = Math.abs( edge.endPoint.y - pointY );

      sum += transpose ? evaluator( p0y, p0x, p1y, p1x ) : evaluator( p0x, p0y, p1x, p1y );
    }

    return sum * sign;
  }

  public static evaluate( polygon: Vector2[], point: Vector2 ): number {
    const relativePolygon = polygon.map( p => p.minus( point ) );

    const case00 = PolygonMitchellNetravali.evaluateCase00;
    const case10 = PolygonMitchellNetravali.evaluateCase10;
    const case11 = PolygonMitchellNetravali.evaluateCase11;

    const f = ( cas: Case, transpose: boolean, bounds: Bounds2 ) => {
      return PolygonMitchellNetravali.evaluateWith( relativePolygon, cas, transpose, bounds );
    };

    // TODO: NOTE that this is suboptimal performance-wise, since we are over-clipping if this is executed in a grid,
    // TODO: among other things

    // TODO: be able to apply point offsets WITHIN the evaluation code, so we don't need to do it here (if we can clip
    // TODO: each individual part of the polygon THEN distribute them...)
    const x0y0 = f( case00, false, bounds00 );
    const x1y0 = f( case10, false, bounds10 );
    const x0y1 = f( case10, true, bounds01 );
    const x1y1 = f( case11, false, bounds11 );
    const nx0y0 = -f( case00, false, boundsn00 );
    const nx1y0 = -f( case10, false, boundsn10 );
    const nx0y1 = -f( case10, true, boundsn01 );
    const nx1y1 = -f( case11, false, boundsn11 );
    const x0ny0 = -f( case00, false, bounds0n0 );
    const x1ny0 = -f( case10, false, bounds1n0 );
    const x0ny1 = -f( case10, true, bounds0n1 );
    const x1ny1 = -f( case11, false, bounds1n1 );
    const nx0ny0 = f( case00, false, boundsn0n0 );
    const nx1ny0 = f( case10, false, boundsn1n0 );
    const nx0ny1 = f( case10, true, boundsn0n1 );
    const nx1ny1 = f( case11, false, boundsn1n1 );

    return x0y0 + x1y0 + x0y1 + x1y1 +
           nx0y0 + nx1y0 + nx0y1 + nx1y1 +
           x0ny0 + x1ny0 + x0ny1 + x1ny1 +
           nx0ny0 + nx1ny0 + nx0ny1 + nx1ny1;
  }

  private static evaluateWith( polygon: Vector2[], evaluator: Case, transpose: boolean, bounds: Bounds2 ): number {
    let sum = 0;

    const clippedPolygon = PolygonClipping.boundsClipPolygon( polygon, bounds );

    for ( let i = 0; i < clippedPolygon.length; i++ ) {
      const p0 = clippedPolygon[ i % clippedPolygon.length ];
      const p1 = clippedPolygon[ ( i + 1 ) % clippedPolygon.length ];

      if ( transpose ) {
        // flip reverses orientation
        sum -= evaluator( Math.abs( p0.y ), Math.abs( p0.x ), Math.abs( p1.y ), Math.abs( p1.x ) );
      }
      else {
        sum += evaluator( Math.abs( p0.x ), Math.abs( p0.y ), Math.abs( p1.x ), Math.abs( p1.y ) );
      }
    }

    return sum;
  }

  public static evaluateCase00( p0x: number, p0y: number, p1x: number, p1y: number ): number {
    const p0x2 = p0x * p0x;
    const p0x3 = p0x2 * p0x;
    const p0y2 = p0y * p0y;
    const p1x2 = p1x * p1x;
    const p1x3 = p1x2 * p1x;
    const p1y3 = p1y * p1y * p1y;
    const p0x7 = 7 * p0x;
    const p1x7 = 7 * p1x;
    const p1y7 = 7 * p1y;
    const p0x720 = p0x7 - 20;
    const p1x720 = p1x7 - 20;
    const p0x732 = p0x7 - 32;
    const p1x732 = p1x7 - 32;
    const p1xy7 = p1x7 * p1y;
    const p01x = p0x * p1x;
    const p1x716 = p1x7 - 16;

    return ( 1 / 51840 ) * ( p0x - p1x ) * (
      3 * p0y2 * p0y2 * ( 896 + 735 * p0x3 + 45 * p0x2 * p1x732 + 15 * p01x * p1x732 + 3 * p1x2 * p1x732 ) +
      128 * ( 160 + 3 * p0x2 * p0x720 + 6 * p0x * p0x720 * p1x + 9 * p0x720 * p1x2 + 84 * p1x3 ) * p1y -
      96 * ( 80 + 3 * ( p0x - 4 ) * p0x2 + 12 * ( p0x - 4 ) * p01x + 30 * ( p0x - 4 ) * p1x2 + 60 * p1x3 ) * p1y3 +
      3 * ( 896 + 3 * p0x2 * p0x732 + 15 * p0x * p0x732 * p1x + 45 * p0x732 * p1x2 + 735 * p1x3 ) * p1y3 * p1y +
      6 * p0y2 * p1y * (
        -16 * ( 80 + 30 * p0x3 + 36 * p0x2 * ( p1x - 2 ) + 12 * ( p1x - 3 ) * p1x2 + 9 * p01x * ( 3 * p1x - 8 ) ) +
        ( 448 + 3 * ( 35 * p0x3 + 9 * p01x * p1x716 + p1x2 * ( 35 * p1x - 96 ) + p0x2 * ( 63 * p1x - 96 ) ) ) * p1y
      ) +
      4 * p0y * (
        32 * ( 160 + 84 * p0x3 + 9 * p0x2 * p1x720 + 6 * p01x * p1x720 + 3 * p1x2 * p1x720 ) -
        24 * ( 80 + 12 * ( p0x - 3 ) * p0x2 + 9 * p0x * ( 3 * p0x - 8 ) * p1x + 36 * ( p0x - 2 ) * p1x2 + 30 * p1x3 ) * p1y * p1y +
        3 * ( 224 + 21 * p0x3 + 15 * p1x2 * p1x716 + 9 * p0x2 * ( p1x7 - 8 ) + 3 * p01x * ( 35 * p1x - 64 ) ) * p1y3
      ) +
      12 * p0y2 * p0y * (
        32 * ( p1y7 - 20 ) +
        3 * (
          5 * p0x3 * ( p1y7 - 32 ) +
          p1x2 * ( 32 - 8 * p1x - 24 * p1y + p1xy7 ) +
          5 * p0x2 * ( 64 - 16 * p1x - 16 * p1y + p1xy7 ) +
          p01x * ( 128 - 32 * p1x - 64 * p1y + 3 * p1xy7 )
        )
      )
    );
  }

  public static evaluateCase10( p0x: number, p0y: number, p1x: number, p1y: number ): number {
    const p0x2 = p0x * p0x;
    const p0x3 = p0x2 * p0x;
    const p0y2 = p0y * p0y;
    const p0y3 = p0y2 * p0y;
    const p0y4 = p0y3 * p0y;
    const p1x2 = p1x * p1x;
    const p1x3 = p1x2 * p1x;
    const p1y2 = p1y * p1y;
    const p1y3 = p1y2 * p1y;
    const p1y4 = p1y3 * p1y;
    const p0x73 = 7 * p0x3;
    const p1x7 = 7 * p1x;
    const p1x796 = p1x7 - 96;
    const p1x760 = p1x7 - 60;
    const p560 = 560 + p1x * p1x796;
    const p1xp1x760200 = 200 + p1x * p1x760;

    return -( 1 / 51840 ) * ( p0x - p1x ) * (
      3 * p0y4 * ( -1792 + 245 * p0x3 + 15 * p0x2 * p1x796 + 5 * p0x * p560 + p1x * p560 ) +
      128 * ( -320 + 200 * p0x + p0x73 + 3 * p0x * p1x * ( p1x7 - 40 ) + 2 * p0x2 * ( p1x7 - 30 ) + 4 * p1x * ( 100 + p1x * ( p1x7 - 45 ) ) ) * p1y -
      96 * ( p0x3 + 4 * p0x2 * ( p1x - 3 ) + 20 * Math.pow( p1x - 2, 3 ) + 2 * p0x * ( 30 + p1x * ( 5 * p1x - 24 ) ) ) * p1y3 +
      3 * ( -1792 + p0x73 + p0x2 * ( 35 * p1x - 96 ) + 5 * p0x * ( 112 + 3 * p1x * ( p1x7 - 32 ) ) + 5 * p1x * ( 560 + p1x * ( 49 * p1x - 288 ) ) ) * p1y4 +
      12 * p0y3 * (
        8 * ( -20 * Math.pow( p0x - 2, 3 ) - 2 * ( 30 + p0x * ( 5 * p0x - 24 ) ) * p1x - 4 * ( p0x - 3 ) * p1x2 - p1x3 ) +
        ( -448 + 35 * p0x3 + 5 * p0x2 * ( p1x7 - 48 ) + p1x * ( 280 + p1x * ( -72 + p1x7 ) ) + p0x * ( 560 + 3 * p1x * ( p1x7 - 64 ) ) ) * p1y
      ) +
      6 * p0y2 * p1y * (
        16 * ( -10 * p0x3 - 12 * p0x2 * ( p1x - 6 ) - 9 * p0x * ( 20 + ( p1x - 8 ) * p1x ) - 4 * ( p1x - 4 ) * ( 10 + ( p1x - 5 ) * p1x ) ) +
        ( -896 + 35 * p0x3 + 9 * p0x2 * ( p1x7 - 32 ) + p0x * ( 840 + 9 * p1x * ( p1x7 - 48 ) ) + p1x * ( 840 + p1x * ( 35 * p1x - 288 ) ) ) * p1y
      ) +
      4 * p0y * (
        32 * ( -320 + 28 * p0x3 + 3 * p0x2 * p1x760 + 2 * p0x * p1xp1x760200 + p1x * p1xp1x760200 ) -
      24 * ( 4 * ( p0x - 4 ) * ( 10 + ( p0x - 5 ) * p0x ) + 9 * ( 20 + ( p0x - 8 ) * p0x ) * p1x + 12 * ( p0x - 6 ) * p1x2 + 10 * p1x3 ) * p1y2 +
      3 * ( -448 + p0x73 + 3 * p0x2 * ( p1x7 - 24 ) + 5 * p1x * ( 112 + p1x * ( p1x7 - 48 ) ) + p0x * ( 280 + p1x * ( 35 * p1x - 192 ) ) ) * p1y3
      )
    );
  }

  public static evaluateCase11( p0x: number, p0y: number, p1x: number, p1y: number ): number {
    const p0x2 = p0x * p0x;
    const p0x3 = p0x2 * p0x;
    const p0y2 = p0y * p0y;
    const p0y3 = p0y2 * p0y;
    const p0y4 = p0y3 * p0y;
    const p1x2 = p1x * p1x;
    const p1x3 = p1x2 * p1x;
    const p1y2 = p1y * p1y;
    const p1y3 = p1y2 * p1y;
    const p0x73 = 7 * p0x3;
    const p0x7 = 7 * p0x;
    const p1x7 = 7 * p1x;
    const p1x796 = p1x7 - 96;
    const p1x760 = p1x7 - 60;
    const p560 = 560 + p1x * p1x796;
    const p1xp1x760200 = 200 + p1x * p1x760;

    return ( 1 / 51840 ) * ( p0x - p1x ) * (
      p0y4 * ( -1792 + 245 * p0x3 + 15 * p0x2 * p1x796 + 5 * p0x * p560 + p1x * p560 ) +
      4 * p0y3 * (
        24 * ( -20 * Math.pow( p0x - 2, 3 ) - 2 * ( 30 + p0x * ( 5 * p0x - 24 ) ) * p1x - 4 * ( p0x - 3 ) * p1x2 - p1x3 ) +
        ( -448 + 35 * p0x3 + 5 * p0x2 * ( p1x7 - 48 ) + p1x * ( 280 + p1x * ( -72 + p1x7 ) ) + p0x * ( 560 + 3 * p1x * ( p1x7 - 64 ) ) ) * p1y
      ) +
      2 * p0y2 * (
        40 * ( -640 + 70 * p0x3 + 6 * p0x2 * ( -72 + p1x7 ) + 3 * p0x * ( 300 + p1x * ( -72 + p1x7 ) ) + p1x * ( 300 + p1x * ( -72 + p1x7 ) ) ) -
        48 * ( 10 * p0x3 + 12 * p0x2 * ( p1x - 6 ) + 9 * p0x * ( 20 + ( p1x - 8 ) * p1x ) + 4 * ( p1x - 4 ) * ( 10 + ( p1x - 5 ) * p1x ) ) * p1y +
        ( -896 + 35 * p0x3 + 9 * p0x2 * ( p1x7 - 32 ) + p0x * ( 840 + 9 * p1x * ( p1x7 - 48 ) ) + p1x * ( 840 + p1x * ( 35 * p1x - 288 ) ) ) * p1y2
      ) +
      4 * p0y * (
        -64 * ( -320 + 28 * p0x3 + 3 * p0x2 * p1x760 + 2 * p0x * p1xp1x760200 + p1x * p1xp1x760200 ) +
        40 * ( -320 + 14 * p0x3 + 3 * p0x2 * ( -36 + p1x7 ) + 2 * p1x * ( 150 + p1x * ( -54 + p1x7 ) ) + 3 * p0x * ( 100 + p1x * ( p1x7 - 48 ) ) ) * p1y -
        24 * ( 4 * ( p0x - 4 ) * ( 10 + ( p0x - 5 ) * p0x ) + 9 * ( 20 + ( p0x - 8 ) * p0x ) * p1x + 12 * ( p0x - 6 ) * p1x2 + 10 * p1x3 ) * p1y2 +
        ( -448 + p0x73 + 3 * p0x2 * ( p1x7 - 24 ) + 5 * p1x * ( 112 + p1x * ( p1x7 - 48 ) ) + p0x * ( 280 + p1x * ( 35 * p1x - 192 ) ) ) * p1y3
      ) +
      p1y * (
        -256 * ( -320 + 200 * p0x + p0x73 + 3 * p0x * p1x * ( p1x7 - 40 ) + 2 * p0x2 * ( p1x7 - 30 ) + 4 * p1x * ( 100 + p1x * ( p1x7 - 45 ) ) ) +
        80 * ( -640 + p0x * ( 300 + p0x * ( -72 + p0x7 ) ) + 900 * p1x + 3 * p0x * ( -72 + p0x7 ) * p1x + 6 * ( -72 + p0x7 ) * p1x2 + 70 * p1x3 ) * p1y -
        96 * ( p0x3 + 4 * p0x2 * ( p1x - 3 ) + 20 * Math.pow( p1x - 2, 3 ) + 2 * p0x * ( 30 + p1x * ( 5 * p1x - 24 ) ) ) * p1y2 +
        ( -1792 + p0x73 + p0x2 * ( 35 * p1x - 96 ) + 5 * p0x * ( 112 + 3 * p1x * ( p1x7 - 32 ) ) + 5 * p1x * ( 560 + p1x * ( 49 * p1x - 288 ) ) ) * p1y3
      )
    );
  }
}

scenery.register( 'PolygonMitchellNetravali', PolygonMitchellNetravali );
