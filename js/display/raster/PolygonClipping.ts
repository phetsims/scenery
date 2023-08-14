// Copyright 2023, University of Colorado Boulder

/**
 * Maillot '92 polygon clipping algorithm, using Cohen-Sutherland clipping
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */
import Vector2 from '../../../../dot/js/Vector2.js';
import Bounds2 from '../../../../dot/js/Bounds2.js';
import { LinearEdge, scenery } from '../../imports.js';

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
const minSimplifier = new Simplifier();
const maxSimplifier = new Simplifier();

export default class PolygonClipping {

  // Returns if all done
  private static binaryInitialPush(
    startPoint: Vector2,
    endPoint: Vector2,
    startCmp: number,
    endCmp: number,
    minLinearEdges: LinearEdge[],
    maxLinearEdges: LinearEdge[]
  ): boolean {

    // both values less than the split
    if ( startCmp === -1 && endCmp === -1 ) {
      minLinearEdges.push( new LinearEdge( startPoint, endPoint ) );
      return true;
    }

    // both values greater than the split
    if ( startCmp === 1 && endCmp === 1 ) {
      maxLinearEdges.push( new LinearEdge( startPoint, endPoint ) );
      return true;
    }

    // both values equal to the split
    if ( startCmp === 0 && endCmp === 0 ) {
      // vertical/horizontal line ON our clip point. It is considered "inside" both, so we can just simply push it to both
      minLinearEdges.push( new LinearEdge( startPoint, endPoint ) );
      maxLinearEdges.push( new LinearEdge( startPoint, endPoint ) );
      return true;
    }

    return false;
  }

  private static binaryPushClipEdges(
    startPoint: Vector2,
    endPoint: Vector2,
    startCmp: number,
    endCmp: number,
    fakeCorner: Vector2,
    intersection: Vector2,
    minLinearEdges: LinearEdge[],
    maxLinearEdges: LinearEdge[]
  ): void {
    const startLess = startCmp === -1;
    const startGreater = startCmp === 1;
    const endLess = endCmp === -1;
    const endGreater = endCmp === 1;

    const minResultStartPoint = startLess ? startPoint : intersection;
    const minResultEndPoint = endLess ? endPoint : intersection;
    const maxResultStartPoint = startGreater ? startPoint : intersection;
    const maxResultEndPoint = endGreater ? endPoint : intersection;

    // min-start corner
    if ( startGreater && !fakeCorner.equals( minResultStartPoint ) ) {
      minLinearEdges.push( new LinearEdge( fakeCorner, minResultStartPoint ) );
    }

    // main min section
    if ( !minResultStartPoint.equals( minResultEndPoint ) ) {
      minLinearEdges.push( new LinearEdge( minResultStartPoint, minResultEndPoint ) );
    }

    // min-end corner
    if ( endGreater && !fakeCorner.equals( minResultEndPoint ) ) {
      minLinearEdges.push( new LinearEdge( minResultEndPoint, fakeCorner ) );
    }

    // max-start corner
    if ( startLess && !fakeCorner.equals( maxResultStartPoint ) ) {
      maxLinearEdges.push( new LinearEdge( fakeCorner, maxResultStartPoint ) );
    }

    // main max section
    if ( !maxResultStartPoint.equals( maxResultEndPoint ) ) {
      maxLinearEdges.push( new LinearEdge( maxResultStartPoint, maxResultEndPoint ) );
    }

    // max-end corner
    if ( endLess && !fakeCorner.equals( maxResultEndPoint ) ) {
      maxLinearEdges.push( new LinearEdge( maxResultEndPoint, fakeCorner ) );
    }
  }

  public static binaryXClipEdge(
    startPoint: Vector2,
    endPoint: Vector2,
    x: number,
    fakeCornerY: number,
    minLinearEdges: LinearEdge[], // Will append into this (for performance)
    maxLinearEdges: LinearEdge[] // Will append into this (for performance)
  ): void {

    const startCmp = Math.sign( startPoint.x - x );
    const endCmp = Math.sign( endPoint.x - x );

    const handled = this.binaryInitialPush(
      startPoint, endPoint,
      startCmp, endCmp,
      minLinearEdges, maxLinearEdges
    );
    if ( handled ) {
      return;
    }

    // There is a single crossing of our x.
    const y = startPoint.y + ( endPoint.y - startPoint.y ) * ( x - startPoint.x ) / ( endPoint.x - startPoint.x );
    const intersection = new Vector2( x, y );
    const fakeCorner = new Vector2( x, fakeCornerY );

    PolygonClipping.binaryPushClipEdges(
      startPoint, endPoint,
      startCmp, endCmp,
      fakeCorner,
      intersection,
      minLinearEdges, maxLinearEdges
    );
  }

  public static binaryYClipEdge(
    startPoint: Vector2,
    endPoint: Vector2,
    y: number,
    fakeCornerX: number,
    minLinearEdges: LinearEdge[], // Will append into this (for performance)
    maxLinearEdges: LinearEdge[] // Will append into this (for performance)
  ): void {

    const startCmp = Math.sign( startPoint.y - y );
    const endCmp = Math.sign( endPoint.y - y );

    const handled = this.binaryInitialPush(
      startPoint, endPoint,
      startCmp, endCmp,
      minLinearEdges, maxLinearEdges
    );
    if ( handled ) {
      return;
    }

    // There is a single crossing of our y.
    const x = startPoint.x + ( endPoint.x - startPoint.x ) * ( y - startPoint.y ) / ( endPoint.y - startPoint.y );
    const intersection = new Vector2( x, y );
    const fakeCorner = new Vector2( fakeCornerX, y );

    PolygonClipping.binaryPushClipEdges(
      startPoint, endPoint,
      startCmp, endCmp,
      fakeCorner,
      intersection,
      minLinearEdges, maxLinearEdges
    );
  }

  // line where dot( normal, point ) - value = 0. "min" side is dot-products < value, "max" side is dot-products > value
  public static binaryLineClipEdge(
    startPoint: Vector2,
    endPoint: Vector2,
    normal: Vector2, // NOTE: does NOT need to be a unit vector
    value: number,
    fakeCornerPerpendicular: number,
    minLinearEdges: LinearEdge[], // Will append into this (for performance)
    maxLinearEdges: LinearEdge[] // Will append into this (for performance)
  ): void {

    const startDot = normal.dot( startPoint );
    const endDot = normal.dot( endPoint );

    const startCmp = Math.sign( startDot - value );
    const endCmp = Math.sign( endDot - value );

    const handled = this.binaryInitialPush(
      startPoint, endPoint,
      startCmp, endCmp,
      minLinearEdges, maxLinearEdges
    );
    if ( handled ) {
      return;
    }

    const perpendicular = normal.perpendicular;

    const startPerp = perpendicular.dot( startPoint );
    const endPerp = perpendicular.dot( endPoint );
    const perpPerp = perpendicular.dot( perpendicular );

    // There is a single crossing of our line
    const intersectionPerp = startPerp + ( endPerp - startPerp ) * ( value - startDot ) / ( endDot - startDot );

    // TODO: pass in the fake corner / basePoint for efficiency?
    const basePoint = normal.timesScalar( value / normal.dot( normal ) );

    const intersection = perpendicular.timesScalar( intersectionPerp / perpPerp ).add( basePoint );
    const fakeCorner = perpendicular.timesScalar( fakeCornerPerpendicular ).add( basePoint );

    PolygonClipping.binaryPushClipEdges(
      startPoint, endPoint,
      startCmp, endCmp,
      fakeCorner,
      intersection,
      minLinearEdges, maxLinearEdges
    );
  }

  // line where dot( normal, point ) - value = 0. "min" side is dot-products < value, "max" side is dot-products > value
  public static binaryStripeClipEdge(
    startPoint: Vector2,
    endPoint: Vector2,
    normal: Vector2, // NOTE: does NOT need to be a unit vector
    values: number[],
    fakeCornerPerpendicular: number,
    clippedEdgeCollection: LinearEdge[][] // Will append into this (for performance)
  ): void {

    const startDot = normal.dot( startPoint );
    const endDot = normal.dot( endPoint );

    const perpendicular = normal.perpendicular;

    const startPerp = perpendicular.dot( startPoint );
    const endPerp = perpendicular.dot( endPoint );
    const perpPerp = perpendicular.dot( perpendicular );
    const basePoints = values.map( value => normal.timesScalar( value / normal.dot( normal ) ) );
    const fakeCorners = basePoints.map( basePoint => perpendicular.timesScalar( fakeCornerPerpendicular ).add( basePoint ) );

    // TODO: don't recompute things twice that don't need to be computed twice (reuse, cycle)
    // TODO: ALSO can we just... jump forward instead of checking each individual one? Perhaps we can find it faster?
    for ( let j = 0; j < values.length + 1; j++ ) {
      const minValue = j > 0 ? values[ j - 1 ] : Number.NEGATIVE_INFINITY;
      const maxValue = j < values.length ? values[ j ] : Number.POSITIVE_INFINITY;

      const clippedEdges = clippedEdgeCollection[ j ];

      // Ignore lines that are completely outside of this stripe
      if (
        ( startDot < minValue && endDot < minValue ) ||
        ( startDot > maxValue && endDot > maxValue )
      ) {
        continue;
      }

      // Fully-internal case
      if ( startDot > minValue && startDot < maxValue && endDot > minValue && endDot < maxValue ) {
        clippedEdges.push( new LinearEdge( startPoint, endPoint ) );
        continue;
      }

      // if ON one of the clip lines, consider it "inside"
      if ( startDot === endDot && ( startDot === minValue || startDot === maxValue ) ) {
        clippedEdges.push( new LinearEdge( startPoint, endPoint ) );
        continue;
      }

      // TODO: don't be recomputing intersections like this
      // TODO: also don't recompute if not needed
      // TODO: we should get things from earlier

      let resultStartPoint = startPoint.copy();
      let resultEndPoint = endPoint.copy();
      let minIntersection: Vector2 | null = null;
      let maxIntersection: Vector2 | null = null;
      let startIntersection: Vector2 | null = null;
      let endIntersection: Vector2 | null = null;
      let minFakeCorner: Vector2 | null = null;
      let maxFakeCorner: Vector2 | null = null;

      if ( startDot < minValue || endDot < minValue ) {
        const value = minValue;
        const basePoint = basePoints[ j - 1 ];
        const intersectionPerp = startPerp + ( endPerp - startPerp ) * ( value - startDot ) / ( endDot - startDot );
        const intersection = perpendicular.timesScalar( intersectionPerp / perpPerp ).add( basePoint );

        minIntersection = intersection;
        if ( startDot < minValue ) {
          resultStartPoint = intersection;
          startIntersection = intersection;
        }
        if ( endDot < minValue ) {
          resultEndPoint = intersection;
          endIntersection = intersection;
        }
      }
      if ( startDot > maxValue || endDot > maxValue ) {
        const value = maxValue;
        const basePoint = basePoints[ j ];
        const intersectionPerp = startPerp + ( endPerp - startPerp ) * ( value - startDot ) / ( endDot - startDot );
        const intersection = perpendicular.timesScalar( intersectionPerp / perpPerp ).add( basePoint );

        maxIntersection = intersection;
        if ( startDot > maxValue ) {
          resultStartPoint = intersection;
          startIntersection = intersection;
        }
        if ( endDot > maxValue ) {
          resultEndPoint = intersection;
          endIntersection = intersection;
        }
      }
      if ( minIntersection ) {
        minFakeCorner = fakeCorners[ j - 1 ];
      }
      if ( maxIntersection ) {
        maxFakeCorner = fakeCorners[ j ];
      }

      // TODO: omg, test against those tricky cases, and UNIT TESTS.

      if ( startIntersection ) {
        if ( startIntersection === minIntersection && !startIntersection.equals( minFakeCorner! ) ) {
          clippedEdges.push( new LinearEdge( minFakeCorner!, resultStartPoint ) );
        }
        if ( startIntersection === maxIntersection && !startIntersection.equals( maxFakeCorner! ) ) {
          clippedEdges.push( new LinearEdge( maxFakeCorner!, resultStartPoint ) );
        }
      }

      if ( !resultStartPoint.equals( resultEndPoint ) ) {
        clippedEdges.push( new LinearEdge( resultStartPoint, resultEndPoint ) );
      }

      if ( endIntersection ) {
        if ( endIntersection === minIntersection && !endIntersection.equals( minFakeCorner! ) ) {
          clippedEdges.push( new LinearEdge( resultEndPoint, minFakeCorner! ) );
        }
        if ( endIntersection === maxIntersection && !endIntersection.equals( maxFakeCorner! ) ) {
          clippedEdges.push( new LinearEdge( resultEndPoint, maxFakeCorner! ) );
        }
      }
    }
  }

  public static binaryXClipPolygon(
    polygon: Vector2[],
    x: number,
    minPolygon: Vector2[], // Will append into this (for performance)
    maxPolygon: Vector2[] // Will append into this (for performance)
  ): void {
    for ( let i = 0; i < polygon.length; i++ ) {
      const startPoint = polygon[ i ];
      const endPoint = polygon[ ( i + 1 ) % polygon.length ];

      if ( startPoint.x < x && endPoint.x < x ) {
        minSimplifier.add( endPoint.x, endPoint.y );
        continue;
      }
      else if ( startPoint.x > x && endPoint.x > x ) {
        maxSimplifier.add( endPoint.x, endPoint.y );
        continue;
      }
      else if ( startPoint.x === x && endPoint.x === x ) {
        // vertical line ON our clip point. It is considered "inside" both, so we can just simply push it to both
        minSimplifier.add( endPoint.x, endPoint.y );
        maxSimplifier.add( endPoint.x, endPoint.y );
        continue;
      }

      // There is a single crossing of our x.
      const y = startPoint.y + ( endPoint.y - startPoint.y ) * ( x - startPoint.x ) / ( endPoint.x - startPoint.x );

      const startSimplifier = startPoint.x < endPoint.x ? minSimplifier : maxSimplifier;
      const endSimplifier = startPoint.x < endPoint.x ? maxSimplifier : minSimplifier;

      startSimplifier.add( x, y );
      endSimplifier.add( x, y );
      endSimplifier.add( endPoint.x, endPoint.y );
    }

    minPolygon.push( ...minSimplifier.finalize() );
    maxPolygon.push( ...maxSimplifier.finalize() );

    minSimplifier.reset();
    maxSimplifier.reset();
  }

  public static binaryYClipPolygon(
    polygon: Vector2[],
    y: number,
    minPolygon: Vector2[], // Will append into this (for performance)
    maxPolygon: Vector2[] // Will append into this (for performance)
  ): void {
    for ( let i = 0; i < polygon.length; i++ ) {
      const startPoint = polygon[ i ];
      const endPoint = polygon[ ( i + 1 ) % polygon.length ];

      if ( startPoint.y < y && endPoint.y < y ) {
        minSimplifier.add( endPoint.x, endPoint.y );
        continue;
      }
      else if ( startPoint.y > y && endPoint.y > y ) {
        maxSimplifier.add( endPoint.x, endPoint.y );
        continue;
      }
      else if ( startPoint.y === y && endPoint.y === y ) {
        // horizontal line ON our clip point. It is considered "inside" both, so we can just simply push it to both
        minSimplifier.add( endPoint.x, endPoint.y );
        maxSimplifier.add( endPoint.x, endPoint.y );
        continue;
      }

      // There is a single crossing of our y.
      const x = startPoint.x + ( endPoint.x - startPoint.x ) * ( y - startPoint.y ) / ( endPoint.y - startPoint.y );

      const startSimplifier = startPoint.y < endPoint.y ? minSimplifier : maxSimplifier;
      const endSimplifier = startPoint.y < endPoint.y ? maxSimplifier : minSimplifier;

      startSimplifier.add( x, y );
      endSimplifier.add( x, y );
      endSimplifier.add( endPoint.x, endPoint.y );
    }

    minPolygon.push( ...minSimplifier.finalize() );
    maxPolygon.push( ...maxSimplifier.finalize() );

    minSimplifier.reset();
    maxSimplifier.reset();
  }

  // line where dot( normal, point ) - value = 0. "min" side is dot-products < value, "max" side is dot-products > value
  public static binaryLineClipPolygon(
    polygon: Vector2[],
    normal: Vector2, // NOTE: does NOT need to be a unit vector
    value: number,
    minPolygon: Vector2[], // Will append into this (for performance)
    maxPolygon: Vector2[] // Will append into this (for performance)
  ): void {

    const perpendicular = normal.perpendicular;
    const basePoint = normal.timesScalar( value / normal.dot( normal ) );
    const perpPerp = perpendicular.dot( perpendicular );

    for ( let i = 0; i < polygon.length; i++ ) {
      const startPoint = polygon[ i ];
      const endPoint = polygon[ ( i + 1 ) % polygon.length ];

      const startDot = normal.dot( startPoint );
      const endDot = normal.dot( endPoint );

      if ( startDot < value && endDot < value ) {
        minSimplifier.add( endPoint.x, endPoint.y );
        continue;
      }
      else if ( startDot > value && endDot > value ) {
        maxSimplifier.add( endPoint.x, endPoint.y );
        continue;
      }
      else if ( startDot === value && endDot === value ) {
        // line ON our clip point. It is considered "inside" both, so we can just simply push it to both
        minSimplifier.add( endPoint.x, endPoint.y );
        maxSimplifier.add( endPoint.x, endPoint.y );
        continue;
      }

      const startPerp = perpendicular.dot( startPoint );
      const endPerp = perpendicular.dot( endPoint );

      const intersectionPerp = startPerp + ( endPerp - startPerp ) * ( value - startDot ) / ( endDot - startDot );

      // There is a single crossing of our line.
      const intersection = perpendicular.timesScalar( intersectionPerp / perpPerp ).add( basePoint );

      const startSimplifier = startDot < endDot ? minSimplifier : maxSimplifier;
      const endSimplifier = startDot < endDot ? maxSimplifier : minSimplifier;

      startSimplifier.add( intersection.x, intersection.y );
      endSimplifier.add( intersection.x, intersection.y );
      endSimplifier.add( endPoint.x, endPoint.y );
    }

    minPolygon.push( ...minSimplifier.finalize() );
    maxPolygon.push( ...maxSimplifier.finalize() );

    minSimplifier.reset();
    maxSimplifier.reset();
  }

  // line where dot( normal, point ) - value = 0. "min" side is dot-products < value, "max" side is dot-products > value
  public static binaryStripeClipPolygon(
    polygon: Vector2[],
    normal: Vector2, // NOTE: does NOT need to be a unit vector
    values: number[] // SHOULD BE SORTED from low to high -- no duplicates (TODO verify, enforce in gradients)
  ): Vector2[][] {
    const perpendicular = normal.perpendicular;
    const basePoints = values.map( value => normal.timesScalar( value / normal.dot( normal ) ) );
    const perpPerp = perpendicular.dot( perpendicular );

    const simplifiers = _.range( values.length + 1 ).map( () => new Simplifier() );

    // TODO: export the bounds of each polygon (ignoring the fake corners)?
    // TODO: this is helpful, since currently we'll need to rasterize the "full" bounds?

    for ( let i = 0; i < polygon.length; i++ ) {
      const startPoint = polygon[ i ];
      const endPoint = polygon[ ( i + 1 ) % polygon.length ];

      const startDot = normal.dot( startPoint );
      const endDot = normal.dot( endPoint );

      for ( let j = 0; j < simplifiers.length; j++ ) {
        const simplifier = simplifiers[ j ];
        const minValue = j > 0 ? values[ j - 1 ] : Number.NEGATIVE_INFINITY;
        const maxValue = j < values.length ? values[ j ] : Number.POSITIVE_INFINITY;

        // Ignore lines that are completely outside of this stripe
        if (
          ( startDot < minValue && endDot < minValue ) ||
          ( startDot > maxValue && endDot > maxValue )
        ) {
          continue;
        }

        // Fully-internal case
        if ( startDot > minValue && startDot < maxValue && endDot > minValue && endDot < maxValue ) {
          simplifier.add( startPoint.x, startPoint.y );
          continue;
        }

        // if ON one of the clip lines, consider it "inside"
        if ( startDot === endDot && ( startDot === minValue || startDot === maxValue ) ) {
          simplifier.add( startPoint.x, startPoint.y );
          continue;
        }

        const startPerp = perpendicular.dot( startPoint );
        const endPerp = perpendicular.dot( endPoint );

        // TODO: don't be recomputing intersections like this
        // TODO: also don't recompute if not needed
        // TODO: we should get things from earlier
        if ( startDot <= minValue ) {
          const minIntersectionPerp = startPerp + ( endPerp - startPerp ) * ( minValue - startDot ) / ( endDot - startDot );
          const minIntersection = perpendicular.timesScalar( minIntersectionPerp / perpPerp ).add( basePoints[ j - 1 ] );
          simplifier.add( minIntersection.x, minIntersection.y );
        }
        else if ( startDot >= maxValue ) {
          const maxIntersectionPerp = startPerp + ( endPerp - startPerp ) * ( maxValue - startDot ) / ( endDot - startDot );
          const maxIntersection = perpendicular.timesScalar( maxIntersectionPerp / perpPerp ).add( basePoints[ j ] );
          simplifier.add( maxIntersection.x, maxIntersection.y );
        }
        else {
          simplifier.add( startPoint.x, startPoint.y );
        }

        if ( endDot <= minValue ) {
          const minIntersectionPerp = startPerp + ( endPerp - startPerp ) * ( minValue - startDot ) / ( endDot - startDot );
          const minIntersection = perpendicular.timesScalar( minIntersectionPerp / perpPerp ).add( basePoints[ j - 1 ] );
          simplifier.add( minIntersection.x, minIntersection.y );
        }
        else if ( endDot >= maxValue ) {
          const maxIntersectionPerp = startPerp + ( endPerp - startPerp ) * ( maxValue - startDot ) / ( endDot - startDot );
          const maxIntersection = perpendicular.timesScalar( maxIntersectionPerp / perpPerp ).add( basePoints[ j ] );
          simplifier.add( maxIntersection.x, maxIntersection.y );
        }
        else {
          simplifier.add( endPoint.x, endPoint.y );
        }
      }
    }

    return simplifiers.map( simplifier => {
      const polygon = simplifier.finalize();

      simplifier.reset();

      return polygon;
    } );
  }

  public static boundsClipEdge(
    startPoint: Vector2,
    endPoint: Vector2,
    bounds: Bounds2,
    result: LinearEdge[] = [] // Will append into this (for performance)
  ): LinearEdge[] {

    const centerX = bounds.centerX;
    const centerY = bounds.centerY;

    const clippedStartPoint = scratchStartPoint.set( startPoint );
    const clippedEndPoint = scratchEndPoint.set( endPoint );

    const clipped = PolygonClipping.matthesDrakopoulosClip( clippedStartPoint, clippedEndPoint, bounds );

    let startXLess;
    let startYLess;
    let endXLess;
    let endYLess;

    const needsStartCorner = !clipped || !startPoint.equals( clippedStartPoint );
    const needsEndCorner = !clipped || !endPoint.equals( clippedEndPoint );
    let startCorner: Vector2;
    let endCorner: Vector2;

    if ( needsStartCorner ) {
      startXLess = startPoint.x < centerX;
      startYLess = startPoint.y < centerY;
      startCorner = new Vector2(
        startXLess ? bounds.minX : bounds.maxX,
        startYLess ? bounds.minY : bounds.maxY
      );
    }
    if ( needsEndCorner ) {
      endXLess = endPoint.x < centerX;
      endYLess = endPoint.y < centerY;
      endCorner = new Vector2(
        endXLess ? bounds.minX : bounds.maxX,
        endYLess ? bounds.minY : bounds.maxY
      );
    }

    if ( clipped ) {
      const resultStartPoint = clippedStartPoint.copy();
      const resultEndPoint = clippedEndPoint.copy();

      if ( needsStartCorner && !startCorner!.equals( resultStartPoint ) ) {
        assert && assert( startCorner! );

        result.push( new LinearEdge( startCorner!, resultStartPoint ) );
      }

      if ( !resultStartPoint.equals( resultEndPoint ) ) {
        result.push( new LinearEdge( resultStartPoint, resultEndPoint ) );
      }

      if ( needsEndCorner && !endCorner!.equals( resultEndPoint ) ) {
        assert && assert( endCorner! );

        result.push( new LinearEdge( resultEndPoint, endCorner! ) );
      }
    }
    else {
      assert && assert( startCorner! && endCorner! );

      if ( startXLess !== endXLess && startYLess !== endYLess ) {
        // we crossed from one corner to the opposite, but didn't hit. figure out which corner we passed
        // we're diagonal, so solving for y=centerY should give us the info we need
        const y = startPoint.y + ( endPoint.y - startPoint.y ) * ( centerX - startPoint.x ) / ( endPoint.x - startPoint.x );

        // Based on whether we are +x+y => -x-y or -x+y => +x-y
        const startSame = startXLess === startYLess;
        const yGreater = y > centerY;

        const middlePoint = new Vector2(
          startSame === yGreater ? bounds.minX : bounds.maxX,
          yGreater ? bounds.maxY : bounds.minY
        );

        result.push( new LinearEdge( startCorner!, middlePoint ) );
        result.push( new LinearEdge( middlePoint, endCorner! ) );
      }
      else if ( !startCorner!.equals( endCorner! ) ) {
        result.push( new LinearEdge( startCorner!, endCorner! ) );
      }
    }

    return result;
  }

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

    const result = simplifier.finalize();

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
