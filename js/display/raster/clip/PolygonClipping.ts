// Copyright 2023, University of Colorado Boulder

/**
 * Maillot '92 polygon clipping algorithm, using Cohen-Sutherland clipping
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */
import Vector2 from '../../../../../dot/js/Vector2.js';
import { ClipSimplifier, LinearEdge, PolygonalFace, scenery } from '../../../imports.js';
import Utils from '../../../../../dot/js/Utils.js';

// TODO: parallelize this (should be possible)

const scratchStartPoint = new Vector2( 0, 0 );
const scratchEndPoint = new Vector2( 0, 0 );
const simplifier = new ClipSimplifier();
const minSimplifier = new ClipSimplifier();
const maxSimplifier = new ClipSimplifier();
const xIntercepts: number[] = [];
const yIntercepts: number[] = [];

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
      minLinearEdges.push( new LinearEdge( fakeCorner, minResultStartPoint, true ) );
    }

    // main min section
    if ( !minResultStartPoint.equals( minResultEndPoint ) ) {
      minLinearEdges.push( new LinearEdge( minResultStartPoint, minResultEndPoint ) );
    }

    // min-end corner
    if ( endGreater && !fakeCorner.equals( minResultEndPoint ) ) {
      minLinearEdges.push( new LinearEdge( minResultEndPoint, fakeCorner, true ) );
    }

    // max-start corner
    if ( startLess && !fakeCorner.equals( maxResultStartPoint ) ) {
      maxLinearEdges.push( new LinearEdge( fakeCorner, maxResultStartPoint, true ) );
    }

    // main max section
    if ( !maxResultStartPoint.equals( maxResultEndPoint ) ) {
      maxLinearEdges.push( new LinearEdge( maxResultStartPoint, maxResultEndPoint ) );
    }

    // max-end corner
    if ( endLess && !fakeCorner.equals( maxResultEndPoint ) ) {
      maxLinearEdges.push( new LinearEdge( maxResultEndPoint, fakeCorner, true ) );
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
          clippedEdges.push( new LinearEdge( minFakeCorner!, resultStartPoint, true ) );
        }
        if ( startIntersection === maxIntersection && !startIntersection.equals( maxFakeCorner! ) ) {
          clippedEdges.push( new LinearEdge( maxFakeCorner!, resultStartPoint, true ) );
        }
      }

      if ( !resultStartPoint.equals( resultEndPoint ) ) {
        clippedEdges.push( new LinearEdge( resultStartPoint, resultEndPoint ) );
      }

      if ( endIntersection ) {
        if ( endIntersection === minIntersection && !endIntersection.equals( minFakeCorner! ) ) {
          clippedEdges.push( new LinearEdge( resultEndPoint, minFakeCorner!, true ) );
        }
        if ( endIntersection === maxIntersection && !endIntersection.equals( maxFakeCorner! ) ) {
          clippedEdges.push( new LinearEdge( resultEndPoint, maxFakeCorner!, true ) );
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

    const simplifiers = _.range( values.length + 1 ).map( () => new ClipSimplifier() );

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

    return simplifiers.map( simplifier => simplifier.finalize() );
  }

  /**
   * Clips a polygon to a grid.
   *
   * @param polygon
   * @param minX
   * @param minY
   * @param maxX
   * @param maxY
   * @param stepX
   * @param stepY
   * @param simplifiers - Will append into this, for performance. Won't finalize them. Should be indexed by ( y * stepWidth + x )
   */
  public static gridClipPolygon(
    polygon: Vector2[],
    minX: number, minY: number, maxX: number, maxY: number,
    stepX: number, stepY: number,
    simplifiers: ClipSimplifier[]
    // TODO: potentially use callbacks?

    // TODO TODO: Ideally simplify logic for polygonal and edged, since we are effectively writing the "edge" code here?
    // TODO: is that performance loss worth it?

    // TODO: That would mean taking startPoint/endPoint, and an array of callbacks that take (x0,y0,x1,y1).
    // TODO: HEY, what if each simplifier gets its own bound methods? We could just pass the bound methods here
    // TODO: We would want to future-proof and pass (x0,y0,x1,y1,p0?,p1?) for GC-friendliness?
    // TODO: Could have "linear edge accumulators" (with the bound methods) similar to the ClipSimplifier bound methods.
  ): void {

    // TODO: in the caller, assert total area is the same!

    // TODO: can we have the caller pass in things like this? In the edge case, we'd want to do the same
    const width = maxX - minX;
    const height = maxY - minY;
    const stepWidth = width / stepX;
    const stepHeight = height / stepY;
    // TODO: should we assert that we're dealing with integers?
    assert && assert( stepWidth % 1 === 0 && stepWidth > 0 );
    assert && assert( stepHeight % 1 === 0 && stepHeight > 0 );
    assert && assert( stepWidth * stepHeight === simplifiers.length );

    if ( stepWidth === 1 && stepHeight === 1 ) {
      for ( let i = 0; i < polygon.length; i++ ) {
        simplifiers[ 0 ].addPoint( polygon[ i ] );
      }
      return;
    }

    // TODO: get rid of these functions (inline)
    const toStepX = ( x: number ) => ( x - minX ) / stepX;
    const toStepY = ( y: number ) => ( y - minY ) / stepY;
    const fromStepX = ( x: number ) => x * stepX + minX;
    const fromStepY = ( y: number ) => y * stepY + minY;

    // TODO: optimize this
    for ( let i = 0; i < polygon.length; i++ ) {
      const startPoint = polygon[ i ];
      const endPoint = polygon[ ( i + 1 ) % polygon.length ];

      assert && assert( startPoint.isFinite() );
      assert && assert( endPoint.isFinite() );
      assert && assert( startPoint.x >= minX && startPoint.x <= maxX && startPoint.y >= minY && startPoint.y <= maxY );
      assert && assert( endPoint.x >= minX && endPoint.x <= maxX && endPoint.y >= minY && endPoint.y <= maxY );

      const startXLess = startPoint.x < endPoint.x;
      const startYLess = startPoint.y < endPoint.y;

      // const lineMinX = Math.min( startPoint.x, endPoint.x );
      // const lineMinY = Math.min( startPoint.y, endPoint.y );
      // const lineMaxX = Math.max( startPoint.x, endPoint.x );
      // const lineMaxY = Math.max( startPoint.y, endPoint.y );

      // In "step" coordinates, in the ranges [0,stepWidth], [0,stepHeight]
      const rawStartStepX = toStepX( startPoint.x );
      const rawStartStepY = toStepY( startPoint.y );
      const rawEndStepX = toStepX( endPoint.x );
      const rawEndStepY = toStepY( endPoint.y );

      const rawMinStepX = Math.min( rawStartStepX, rawEndStepX );
      const rawMinStepY = Math.min( rawStartStepY, rawEndStepY );
      const rawMaxStepX = Math.max( rawStartStepX, rawEndStepX );
      const rawMaxStepY = Math.max( rawStartStepY, rawEndStepY );

      // Integral "step" coordinates
      const startStepX = Math.floor( rawStartStepX );
      const startStepY = Math.floor( rawStartStepY );
      const endStepX = Math.ceil( rawEndStepX );
      const endStepY = Math.ceil( rawEndStepY );

      const minStepX = Math.min( startStepX, endStepX );
      const minStepY = Math.min( startStepY, endStepY );
      const maxStepX = Math.max( startStepX, endStepX );
      const maxStepY = Math.max( startStepY, endStepY );

      const lineStepWidth = maxStepX - minStepX;
      const lineStepHeight = maxStepY - minStepY;

      if ( lineStepWidth > 1 ) {
        const firstY = startPoint.y + ( endPoint.y - startPoint.y ) * ( fromStepX( minStepX + 1 ) - startPoint.x ) / ( endPoint.x - startPoint.x );
        yIntercepts.push( firstY );

        if ( lineStepWidth > 2 ) {
          const slopeIncrement = stepX * ( endPoint.y - startPoint.y ) / ( endPoint.x - startPoint.x );
          let y = firstY;
          for ( let j = minStepX + 2; j < maxStepX; j++ ) {
            y += slopeIncrement;
            yIntercepts.push( y );
          }
        }
      }
      if ( lineStepHeight > 1 ) {
        const firstX = startPoint.x + ( endPoint.x - startPoint.x ) * ( fromStepY( minStepY + 1 ) - startPoint.y ) / ( endPoint.y - startPoint.y );
        xIntercepts.push( firstX );

        if ( lineStepHeight > 2 ) {
          const slopeIncrement = stepY * ( endPoint.x - startPoint.x ) / ( endPoint.y - startPoint.y );
          let x = firstX;
          for ( let j = minStepY + 2; j < maxStepY; j++ ) {
            x += slopeIncrement;
            xIntercepts.push( x );
          }
        }
      }

      // xxxx is the line segment (edge)
      // | and - notes the "clipped along cell bounds" sections
      //
      // minX  minStepX                   maxStepX        maxX
      //   ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┐maxY
      //   │      │  x   │  x   │  x   │  x   │      │      │
      //   │  no  │ past │intern│intern│ not  │  no  │  no  │
      //   │effect│ half │      │      │ past │effect│effect│
      //   │corner│------│------│------│ half │corner│corner│
      //   ├──────┼──────┼──────┼──────┴──────┼──────┼──────┤maxStepY
      //   │  y  |│     |│     |│     |xx     │| y   │| y   │
      //   │ past|│     |│     |│    xx│|     │|past │|past │
      //   │ half|│     |│     |│  xx  │|     │|half │|half │
      //   │     |│------│------│xx    │|     │|     │|     │
      //   ├──────┼──────┼─────xx──────┼──────┼──────┼──────┤
      //   │  y  |│     |│   xx │------│|     │| y   │| y   │
      //   │inter|│     |│ xx   │|     │|     │|ntern│|ntern│
      //   │     |│      xx     │|     │|     │|     │|     │
      //   │     |│----xx│|     │|     │|     │|     │|     │
      //   ├──────┼──xx──┼──────┼──────┼──────┼──────┼──────┤
      //   │  y   │xx----│------│------│      │  y   │  y   │
      //   │ not  x      │      │      │      │ not  │ not  │
      //   │ past │      │      │      │      │ past │ past │
      //   │ half │      │      │      │      │ half │ half │
      //   ├──────┼──────┼──────┼──────┼──────┼──────┼──────┤minStepY
      //   │      │------│------│------│  x   │      │      │
      //   │  no  │  x   │      │      │ not  │  no  │  no  │
      //   │effect│ past │  x   │  x   │ past │effect│effect│
      //   │corner│ half │intern│intern│ half │corner│corner│
      //   └──────┴──────┴──────┴──────┴──────┴──────┴──────┘minY

      const roundedMinStepX = Utils.roundSymmetric( rawMinStepX );
      const roundedMinStepY = Utils.roundSymmetric( rawMinStepY );
      const roundedMaxStepX = Utils.roundSymmetric( rawMaxStepX );
      const roundedMaxStepY = Utils.roundSymmetric( rawMaxStepY );

      // TODO: assertions that we're outputting points INSIDE each range to each simplifier

      // Handle "internal" cases (the step rectangle that overlaps the line)
      if ( lineStepWidth === 1 && lineStepHeight === 1 ) {
        // If we only take up one cell, we can do a much more optimized form (AND in the future hopefully the clip
        // simplifier will be able to pass through vertices without GC

        simplifier.addPoint( startPoint );
        simplifier.addPoint( endPoint );
      }
      else {
        // Do the "internal" grid
        for ( let iy = minStepY; iy < maxStepY; iy++ ) {
          // TODO: this could be optimized
          const cellMinY = fromStepY( iy );
          const cellMaxY = fromStepY( iy + 1 );
          const cellCenterY = ( cellMinY + cellMaxY ) / 2;

          const isFirstY = iy === minStepY;
          const isLastY = iy === maxStepY - 1;

          // The x intercepts for the minimal-y and maximal-y sides of the cell (or if we're on the first or last y cell, the endpoint)
          const minYXIntercept = isFirstY ? ( startYLess ? startPoint.x : endPoint.x ) : xIntercepts[ iy - 1 ];
          const maxYXIntercept = isLastY ? ( startYLess ? endPoint.x : startPoint.x ) : xIntercepts[ iy ];

          // Our range of intercepts (so we can quickly check in the inner iteration)
          const minXIntercept = Math.min( minYXIntercept, maxYXIntercept );
          const maxXIntercept = Math.max( minYXIntercept, maxYXIntercept );

          for ( let ix = minStepX; ix < maxStepX; ix++ ) {
            const simplifier = simplifiers[ iy * stepWidth + ix ];

            const cellMinX = fromStepX( ix );
            const cellMaxX = fromStepX( ix + 1 );
            const cellCenterX = ( cellMinX + cellMaxX ) / 2;

            const isFirstX = ix === minStepX;
            const isLastX = ix === maxStepX - 1;

            const minXYIntercept = isFirstX ? ( startXLess ? startPoint.y : endPoint.y ) : yIntercepts[ ix - 1 ];
            const maxXYIntercept = isLastX ? ( startXLess ? endPoint.y : startPoint.y ) : yIntercepts[ ix ];

            const minYIntercept = Math.min( minXYIntercept, maxXYIntercept );
            // const maxYIntercept = Math.max( minXYIntercept, maxXYIntercept );

            const isLessThanMinX = cellMaxX <= minXIntercept;
            const isGreaterThanMaxX = cellMinX >= maxXIntercept;
            const isLessThanMinY = cellMaxY <= minYIntercept;
            // const isGreaterThanMaxY = cellMinY >= maxYIntercept;

            // If this condition is true, the line does NOT pass through this cell. We just have to handle the corners.
            if ( isLessThanMinX || isGreaterThanMaxX ) {
              // TODO: simplify logic
              const hasHorizontal = ( isFirstX || isLastX ) ? ix >= roundedMinStepX && ( ix + 1 ) <= roundedMaxStepX : true;
              const hasVertical = ( isFirstY || isLastY ) ? iy >= roundedMinStepY && ( iy + 1 ) <= roundedMaxStepY : true;

              if ( hasHorizontal && hasVertical ) {
                const cornerX = isLessThanMinX ? cellMaxX : cellMinX;
                const cornerY = isLessThanMinY ? cellMaxY : cellMinY;
                const otherX = isLessThanMinX ? cellMinX : cellMaxX;
                const otherY = isLessThanMinY ? cellMinY : cellMaxY;
                const xFirst = isLessThanMinX ? startXLess : !startXLess;
                // TODO: is the four ternary expressions better here?
                simplifier.add( xFirst ? otherX : cornerX, xFirst ? cornerY : otherX );
                simplifier.add( cornerX, cornerY );
                simplifier.add( xFirst ? cornerX : otherX, xFirst ? otherY : cornerY );
              }
              else if ( hasHorizontal ) {
                const y = isLessThanMinY ? cellMaxY : cellMinY;
                simplifier.add( startXLess ? cellMinX : cellMaxX, y );
                simplifier.add( startXLess ? cellMaxX : cellMinX, y );
              }
              else if ( hasVertical ) {
                const x = isLessThanMinX ? cellMaxX : cellMinX;
                simplifier.add( x, startYLess ? cellMinY : cellMaxY );
                simplifier.add( x, startYLess ? cellMaxY : cellMinY );
              }
            }
            else {
              // We go through the cell! Additionally due to previous filtering, we are pretty much guaranteed to touch
              // a cell side.

              const minYX = Utils.clamp( minYXIntercept, cellMinX, cellMaxX );
              const maxYX = Utils.clamp( maxYXIntercept, cellMinX, cellMaxX );
              const minXY = Utils.clamp( minXYIntercept, cellMinY, cellMaxY );
              const maxXY = Utils.clamp( maxXYIntercept, cellMinY, cellMaxY );

              const startX = startXLess ? minYX : maxYX;
              const startY = startYLess ? minXY : maxXY;
              const endX = startXLess ? maxYX : minYX;
              const endY = startYLess ? maxXY : minXY;

              // Ensure we have the correct direction (and our logic is correct)
              assert && assert( new Vector2( endX - startX, endY - startY ).normalized().equalsEpsilon( endPoint.minus( startPoint ).normalized(), 1e-8 ) );

              const needsStartCorner = startX !== startPoint.x || startY !== startPoint.y;
              const needsEndCorner = endX !== endPoint.x || endY !== endPoint.y;

              if ( needsStartCorner ) {
                simplifier.add(
                  startPoint.x < cellCenterX ? cellMinX : cellMaxX,
                  startPoint.y < cellCenterY ? cellMinY : cellMaxY
                );
              }
              simplifier.add( startX, startY );
              simplifier.add( endX, endY );
              if ( needsEndCorner ) {
                simplifier.add(
                  endPoint.x < cellCenterX ? cellMinX : cellMaxX,
                  endPoint.y < cellCenterY ? cellMinY : cellMaxY
                );
              }
            }
          }
        }
      }

      // x internal, y external
      for ( let ix = roundedMinStepX; ix < roundedMaxStepX; ix++ ) {
        const x0 = fromStepX( ix );
        const x1 = fromStepX( ix + 1 );

        // min-y side
        for ( let iy = 0; iy < minStepY; iy++ ) {
          const simplifier = simplifiers[ iy * stepWidth + ix ];
          const y = fromStepY( iy + 1 );
          simplifier.add( startXLess ? x0 : x1, y );
          simplifier.add( startXLess ? x1 : x0, y );
        }
        // max-y side
        for ( let iy = maxStepY; iy < stepHeight; iy++ ) {
          const simplifier = simplifiers[ iy * stepWidth + ix ];
          const y = fromStepY( iy );
          simplifier.add( startXLess ? x0 : x1, y );
          simplifier.add( startXLess ? x1 : x0, y );
        }
      }

      // y internal, x external
      for ( let iy = roundedMinStepY; iy < roundedMaxStepY; iy++ ) {
        const y0 = fromStepY( iy );
        const y1 = fromStepY( iy + 1 );

        // min-x side
        for ( let ix = 0; ix < minStepX; ix++ ) {
          const simplifier = simplifiers[ iy * stepWidth + ix ];
          const x = fromStepX( ix + 1 );
          simplifier.add( x, startYLess ? y0 : y1 );
          simplifier.add( x, startYLess ? y1 : y0 );
        }
        // max-x side
        for ( let ix = maxStepX; ix < stepWidth; ix++ ) {
          const simplifier = simplifiers[ iy * stepWidth + ix ];
          const x = fromStepX( ix );
          simplifier.add( x, startYLess ? y0 : y1 );
          simplifier.add( x, startYLess ? y1 : y0 );
        }
      }

      xIntercepts.length = 0;
      yIntercepts.length = 0;
    }
  }

  public static boundsClipEdge(
    startPoint: Vector2, endPoint: Vector2,
    // Properties of the bounds
    minX: number, minY: number, maxX: number, maxY: number, centerX: number, centerY: number,
    result: LinearEdge[] = [] // Will append into this (for performance)
  ): LinearEdge[] {

    const clippedStartPoint = scratchStartPoint.set( startPoint );
    const clippedEndPoint = scratchEndPoint.set( endPoint );

    const clipped = PolygonClipping.matthesDrakopoulosClip( clippedStartPoint, clippedEndPoint, minX, minY, maxX, maxY );

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
        startXLess ? minX : maxX,
        startYLess ? minY : maxY
      );
    }
    if ( needsEndCorner ) {
      endXLess = endPoint.x < centerX;
      endYLess = endPoint.y < centerY;
      endCorner = new Vector2(
        endXLess ? minX : maxX,
        endYLess ? minY : maxY
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
          startSame === yGreater ? minX : maxX,
          yGreater ? maxY : minY
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
  public static boundsClipPolygon(
    polygon: Vector2[],
    // Properties of the bounds
    minX: number, minY: number, maxX: number, maxY: number, centerX: number, centerY: number
  ): Vector2[] {

    // TODO: optimize this
    for ( let i = 0; i < polygon.length; i++ ) {
      const startPoint = polygon[ i ];
      const endPoint = polygon[ ( i + 1 ) % polygon.length ];

      const clippedStartPoint = scratchStartPoint.set( startPoint );
      const clippedEndPoint = scratchEndPoint.set( endPoint );

      const clipped = PolygonClipping.matthesDrakopoulosClip( clippedStartPoint, clippedEndPoint, minX, minY, maxX, maxY );

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
          startXLess ? minX : maxX,
          startYLess ? minY : maxY
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
            startSame === yGreater ? minX : maxX,
            yGreater ? maxY : minY
          );
        }
      }
      if ( needsEndCorner ) {
        simplifier.add(
          endXLess ? minX : maxX,
          endYLess ? minY : maxY
        );
      }
    }

    return simplifier.finalize();
  }

  /**
   * From "Another Simple but Faster Method for 2D Line Clipping" (2019)
   * by Dimitrios Matthes and Vasileios Drakopoulos
   */
  private static matthesDrakopoulosClip(
    p0: Vector2, p1: Vector2,
    minX: number, minY: number, maxX: number, maxY: number
  ): boolean {
    const x0 = p0.x;
    const y0 = p0.y;
    const x1 = p1.x;
    const y1 = p1.y;

    if (
      !( x0 < minX && x1 < minX ) &&
      !( x0 > maxX && x1 > maxX ) &&
      !( y0 < minY && y1 < minY ) &&
      !( y0 > maxY && y1 > maxY )
    ) {
      // TODO: consider NOT computing these if we don't need them? We probably won't use both?
      const ma = ( y1 - y0 ) / ( x1 - x0 );
      const mb = ( x1 - x0 ) / ( y1 - y0 );

      // TODO: on GPU, consider if we should extract out partial subexpressions below

      // Unrolled (duplicated essentially)
      if ( p0.x < minX ) {
        p0.x = minX;
        p0.y = ma * ( minX - x0 ) + y0;
      }
      else if ( p0.x > maxX ) {
        p0.x = maxX;
        p0.y = ma * ( maxX - x0 ) + y0;
      }
      if ( p0.y < minY ) {
        p0.y = minY;
        p0.x = mb * ( minY - y0 ) + x0;
      }
      else if ( p0.y > maxY ) {
        p0.y = maxY;
        p0.x = mb * ( maxY - y0 ) + x0;
      }

      // Second unrolled form
      if ( p1.x < minX ) {
        p1.x = minX;
        p1.y = ma * ( minX - x0 ) + y0;
      }
      else if ( p1.x > maxX ) {
        p1.x = maxX;
        p1.y = ma * ( maxX - x0 ) + y0;
      }
      if ( p1.y < minY ) {
        p1.y = minY;
        p1.x = mb * ( minY - y0 ) + x0;
      }
      else if ( p1.y > maxY ) {
        p1.y = maxY;
        p1.x = mb * ( maxY - y0 ) + x0;
      }

      if ( !( p0.x < minX && p1.x < minX ) && !( p0.x > maxX && p1.x > maxX ) ) {
        return true;
      }
    }

    return false;
  }

  /**
   * Clips a polygon (represented by unsorted LinearEdges) by a circle. This will output both the inside and outside,
   * appending LinearEdges to the given arrays.
   *
   * @param edges - the edges of the polygon to clip
   * @param center - the center of the circle
   * @param radius - the radius of the circle
   * @param maxAngleSplit - the maximum angle of a circular arc that will be converted into a linear edge
   * @param inside - (OUTPUT) the edges that are inside the circle (will be appended to)
   * @param outside - (OUTPUT) the edges that are outside the circle (will be appended to)
   */
  public static binaryCircularClipEdges(
    edges: LinearEdge[],
    center: Vector2,
    radius: number,
    maxAngleSplit: number,
    inside: LinearEdge[],
    outside: LinearEdge[]
  ): void {

    // If we inscribed a circle inside a regular polygon split at angle `maxAngleSplit`, we'd have this radius.
    // Because we're turning our circular arcs into line segments at the end, we need to make sure that content inside
    // the circle doesn't go OUTSIDE the "inner" polygon (in that sliver between the circle and regular polygon).
    // We'll do that by adding "critical angles" for any points between the radius and inradus, so that our polygonal
    // approximation of the circle has a split there.
    // inradius = r cos( pi / n ) for n segments
    // n = 2pi / maxAngleSplit
    const inradius = radius * Math.cos( 0.5 * maxAngleSplit );

    // Our general plan will be to clip by keeping things "inside" the circle, and using the duality of clipping with
    // edges to also get the "outside" edges.
    // The duality follows from the fact that if we have a "full" polygon represented by edges, and then we have a
    // "subset" of it also represented by edges, then the "full - subset" difference can be represented by including
    // both all the edges of the "full" polygon PLUS all of the edges of the "subset" polygon with their direction
    // reversed.
    // Additionally in general, instead of "appending" both of those lists, we can do MUCH better! Instead whenever
    // we INCLUDE part of an original edge in the "subset", we DO NOT include it in the other disjoint polygon, and
    // vice versa. Additionally, when we add in "new" edges (or fake ones), we need to add the REVERSE to the
    // disjoint polygon.
    // Thus we essentially get "dual" binary polygons for free.

    // Because we are clipping to "keep the inside", any edges outside we can actually just "project" down to the circle
    // (imagine wrapping the exterior edge around the circle). For the duality, we can output the internal/external
    // "parts" directly to the inside/outside result arrays, but these wrapped circular projections will need to be
    // stored for later here.
    // Each "edge" in our input will have between 0 and 1 "internal" edges, and 0 and 2 "external" edges.
    //
    // NOTE: We also need to store the start/end points, so that we output exact start/end points (instead of numerically
    // slightly-different points based on the radius/angles), for our later clipping stages to work nicely.
    const insideCircularEdges: CircularEdgeWithPoints[] = [];

    // We'll also need to store "critical" angles for the future polygonalization of the circles. If we were outputting
    // true circular edges, we could just include `insideCircularEdges`, however we want to convert it to line segments
    // so that future stages don't have to deal with this.
    // We'll need the angles so that those points on the circle will be exact (for ALL of the circular edges).
    // This is because we may be wrapping back-and-forth across the circle multiple times, with different start/end
    // angles, and we need the polygonal parts of these overlaps to be identical (to avoid precision issues later,
    // and ESPECIALLY to avoid little polygonal bits with "negative" area where the winding orientation is flipped.
    // There are two types of points where we'll need to store the angles:
    // 1. Intersections with our circle (where we'll need to "split" the edge into two at that point)
    // 2. Points where we are between the circumradius and inradius of the roughest "regular" polygon we might generate.

    // between [-pi,pi], from atan2, tracked so we can turn the arcs piecewise-linear in a consistent fashion later
    let angles: number[] = [];

    // Process a fully-inside-the-circle part of an edge
    const processInternal = ( edge: LinearEdge ) => {
      inside.push( edge );

      const localStart = edge.startPoint.minus( center );
      const localEnd = edge.endPoint.minus( center );

      // We're already inside the circle, so the circumradius check isn't needed. If we're inside the inradius,
      // ensure the critical angles are added.
      if ( localStart.magnitude > inradius ) {
        angles.push( localStart.angle );
      }
      if ( localEnd.magnitude > inradius ) {
        angles.push( localEnd.angle );
      }
    };

    // Process a fully-outside-the-circle part of an edge
    const processExternal = ( edge: LinearEdge, startInside: boolean, endInside: boolean ) => {
      outside.push( edge );

      const localStart = edge.startPoint.minus( center );
      const localEnd = edge.endPoint.minus( center );

      // Modify (project) them into points of the given radius.
      localStart.multiplyScalar( radius / localStart.magnitude );
      localEnd.multiplyScalar( radius / localEnd.magnitude );

      // Handle projecting the edge to the circle.
      // We'll only need to do extra work if the projected points are not equal. If we had a line that was pointed
      // toward the center of the circle, it would project down to a single point, and we wouldn't have any contribution.
      if ( !localStart.equalsEpsilon( localEnd, 1e-8 ) ) {
        // Check to see which way we went "around" the circle

        // (y, -x) perpendicular, so a clockwise pi/2 rotation
        const isClockwise = localStart.perpendicular.dot( localEnd ) > 0;

        const startAngle = localStart.angle;
        const endAngle = localEnd.angle;

        angles.push( startAngle );
        angles.push( endAngle );

        insideCircularEdges.push( new CircularEdgeWithPoints(
          startInside ? edge.startPoint : null,
          endInside ? edge.endPoint : null,
          startAngle,
          endAngle,
          !isClockwise
        ) );
      }
      else {
        // NOTE: We need to do our "fixing" of coordinate matching in this case. It's possible we may need to add
        // a very small "infinitesimal" edge.
        let projectedStart = Vector2.createPolar( radius, localStart.angle ).add( center );
        let projectedEnd = Vector2.createPolar( radius, localEnd.angle ).add( center );

        if ( startInside ) {
          assert && assert( projectedStart.distanceSquared( edge.startPoint ) < 1e-8 );
          projectedStart = edge.startPoint;
        }
        if ( endInside ) {
          assert && assert( projectedEnd.distanceSquared( edge.endPoint ) < 1e-8 );
          projectedEnd = edge.endPoint;
        }

        if ( !projectedStart.equals( projectedEnd ) ) {
          inside.push( new LinearEdge( projectedStart, projectedEnd ) );
          outside.push( new LinearEdge( projectedEnd, projectedStart ) );
        }
      }
    };

    for ( let i = 0; i < edges.length; i++ ) {
      const edge = edges[ i ];

      const startInside = edge.startPoint.distance( center ) <= radius;
      const endInside = edge.endPoint.distance( center ) <= radius;

      // If the endpoints are within the circle, the entire contents will be also (shortcut)
      if ( startInside && endInside ) {
        processInternal( edge );
        continue;
      }

      // Now, we'll solve for the t-values of the intersection of the line and the circle.
      // e.g. p0 + t * ( p1 - p0 ) will be on the circle. This is solvable with a quadratic equation.
      const p0x = edge.startPoint.x - center.x;
      const p0y = edge.startPoint.y - center.y;
      const p1x = edge.endPoint.x - center.x;
      const p1y = edge.endPoint.y - center.y;
      const dx = p1x - p0x;
      const dy = p1y - p0y;

      // quadratic to solve
      const a = dx * dx + dy * dy;
      const b = 2 * ( p0x * dx + p0y * dy );
      const c = p0x * p0x + p0y * p0y - radius * radius;

      assert && assert( a > 0, 'We should have a delta, assumed in code below' );

      const roots = Utils.solveQuadraticRootsReal( a, b, c );

      let isFullyExternal = false;

      // If we have no roots, we're fully outside the circle!
      if ( !roots || roots.length === 0 ) {
        isFullyExternal = true;
      }
      else {
        if ( roots.length === 1 ) {
          roots.push( roots[ 0 ] );
        }
        assert && assert( roots[ 0 ] <= roots[ 1 ], 'Easier for us to assume root ordering' );
        const rootA = roots[ 0 ];
        const rootB = roots[ 1 ];

        if ( rootB <= 0 || rootA >= 1 ) {
          isFullyExternal = true;
        }

        // If our roots are identical, we are TANGENT to the circle. We can consider this to be fully external, since
        // there will not be an internal section.
        if ( rootA === rootB ) {
          isFullyExternal = true;
        }
      }

      if ( isFullyExternal ) {
        processExternal( edge, startInside, endInside );
        continue;
      }

      assert && assert( roots![ 0 ] <= roots![ 1 ], 'Easier for us to assume root ordering' );
      const rootA = roots![ 0 ];
      const rootB = roots![ 1 ];

      // Compute intersection points (when the t values are in the range [0,1])
      const rootAInSegment = rootA > 0 && rootA < 1;
      const rootBInSegment = rootB > 0 && rootB < 1;
      const deltaPoints = edge.endPoint.minus( edge.startPoint );
      const rootAPoint = rootAInSegment ? ( edge.startPoint.plus( deltaPoints.timesScalar( rootA ) ) ) : Vector2.ZERO; // ignore the zero, it's mainly for typing
      const rootBPoint = rootBInSegment ? ( edge.startPoint.plus( deltaPoints.timesScalar( rootB ) ) ) : Vector2.ZERO; // ignore the zero, it's mainly for typing

      if ( rootAInSegment && rootBInSegment ) {
        processExternal( new LinearEdge( edge.startPoint, rootAPoint ), startInside, true );
        processInternal( new LinearEdge( rootAPoint, rootBPoint ) );
        processExternal( new LinearEdge( rootBPoint, edge.endPoint ), true, endInside );
      }
      else if ( rootAInSegment ) {
        processExternal( new LinearEdge( edge.startPoint, rootAPoint ), startInside, true );
        processInternal( new LinearEdge( rootAPoint, edge.endPoint ) );
      }
      else if ( rootBInSegment ) {
        processInternal( new LinearEdge( edge.startPoint, rootBPoint ) );
        processExternal( new LinearEdge( rootBPoint, edge.endPoint ), true, endInside );
      }
      else {
        assert && assert( false, 'Should not reach this point, due to the boolean constraints above' );
      }
    }

    // Sort our critical angles, so we can iterate through unique values in-order
    angles = _.uniq( angles.sort( ( a, b ) => a - b ) );

    for ( let i = 0; i < insideCircularEdges.length; i++ ) {
      const edge = insideCircularEdges[ i ];

      const startIndex = angles.indexOf( edge.startAngle );
      const endIndex = angles.indexOf( edge.endAngle );

      const subAngles: number[] = [];

      // Iterate (in the specific direction) through the angles we cover, and add them to our subAngles list.
      const dirSign = edge.counterClockwise ? 1 : -1;
      for ( let index = startIndex; index !== endIndex; index = ( index + dirSign + angles.length ) % angles.length ) {
        subAngles.push( angles[ index ] );
      }
      subAngles.push( angles[ endIndex ] );

      for ( let j = 0; j < subAngles.length - 1; j++ ) {
        const startAngle = subAngles[ j ];
        const endAngle = subAngles[ j + 1 ];

        // Put our end angle in the dirSign direction from our startAngle (if we're counterclockwise and our angle increases,
        // our relativeEndAngle should be greater than our startAngle, and similarly if we're clockwise and our angle decreases,
        // our relativeEndAngle should be less than our startAngle)
        const relativeEndAngle = ( edge.counterClockwise === ( startAngle < endAngle ) ) ? endAngle : endAngle + dirSign * Math.PI * 2;

        // Split our circular arc into segments!
        const angleDiff = relativeEndAngle - startAngle;
        const numSegments = Math.ceil( Math.abs( angleDiff ) / maxAngleSplit );
        for ( let k = 0; k < numSegments; k++ ) {
          const startTheta = startAngle + angleDiff * ( k / numSegments );
          const endTheta = startAngle + angleDiff * ( ( k + 1 ) / numSegments );

          let startPoint = Vector2.createPolar( radius, startTheta ).add( center );
          let endPoint = Vector2.createPolar( radius, endTheta ).add( center );

          if ( edge.startPoint && j === 0 && k === 0 ) {
            // First "point" of a insideCircularEdge, let's replace with our actual start point for exact precision
            assert && assert( startPoint.distanceSquared( edge.startPoint ) < 1e-8 );
            startPoint = edge.startPoint;
          }
          if ( edge.endPoint && j === subAngles.length - 2 && k === numSegments - 1 ) {
            // Last "point" of an insideCircularEdge, let's replace with our actual end point for exact precision
            assert && assert( endPoint.distanceSquared( edge.endPoint ) < 1e-8 );
            endPoint = edge.endPoint;
          }

          // We might have tiny angle/etc. distances, so we could come into edges that we need to strip
          if ( !startPoint.equals( endPoint ) ) {
            inside.push( new LinearEdge( startPoint, endPoint ) );
            outside.push( new LinearEdge( endPoint, startPoint ) );
          }
        }
      }
    }
  }

  /**
   * Clips a polygon (represented by polygonal vertex lists) by a circle. This will output both the inside and outside,
   * appending vertices to the arrays
   *
   * @param polygons
   * @param center - the center of the circle
   * @param radius - the radius of the circle
   * @param maxAngleSplit - the maximum angle of a circular arc that will be converted into a linear edge
   * @param inside - (OUTPUT) the polygon that is inside the circle (will be appended to)
   * @param outside - (OUTPUT) the polygon that is outside the circle (will be appended to)
   */
  public static binaryCircularClipPolygon(
    polygons: Vector2[][],
    center: Vector2,
    radius: number,
    maxAngleSplit: number,
    inside: Vector2[][],
    outside: Vector2[][]
  ): void {

    const radiusSquared = radius * radius;

    // If we inscribed a circle inside a regular polygon split at angle `maxAngleSplit`, we'd have this radius.
    // Because we're turning our circular arcs into line segments at the end, we need to make sure that content inside
    // the circle doesn't go OUTSIDE the "inner" polygon (in that sliver between the circle and regular polygon).
    // We'll do that by adding "critical angles" for any points between the radius and inradus, so that our polygonal
    // approximation of the circle has a split there.
    // inradius = r cos( pi / n ) for n segments
    // n = 2pi / maxAngleSplit
    const inradius = radius * Math.cos( 0.5 * maxAngleSplit );

    // Our general plan will be to clip by keeping things "inside" the circle, and using the duality of clipping with
    // edges to also get the "outside" edges.
    // The duality follows from the fact that if we have a "full" polygon represented by edges, and then we have a
    // "subset" of it also represented by edges, then the "full - subset" difference can be represented by including
    // both all the edges of the "full" polygon PLUS all of the edges of the "subset" polygon with their direction
    // reversed.
    // Additionally in general, instead of "appending" both of those lists, we can do MUCH better! Instead whenever
    // we INCLUDE part of an original edge in the "subset", we DO NOT include it in the other disjoint polygon, and
    // vice versa. Additionally, when we add in "new" edges (or fake ones), we need to add the REVERSE to the
    // disjoint polygon.
    // Thus we essentially get "dual" binary polygons for free.

    // Because we are clipping to "keep the inside", any edges outside we can actually just "project" down to the circle
    // (imagine wrapping the exterior edge around the circle). For the duality, we can output the internal/external
    // "parts" directly to the inside/outside result arrays, but these wrapped circular projections will need to be
    // stored for later here.
    // Each "edge" in our input will have between 0 and 1 "internal" edges, and 0 and 2 "external" edges.

    // Because we're handling the polygonal form, we'll need to do some complicated handling for the outside. Whenever
    // we have a transition to the outside (at a specific point), we'll start recording the "outside" edges in one
    // "forward" list, and the corresponding circular movements in the "reverse" list (it will be in the wrong order,
    // and will be reversed later). Once our polygon goes back inside, we'll be able to stitch these together to create
    // an "outside" polygon (forward edges + reversed reverse edges).

    // This gets TRICKIER because if we start outside, we'll have an "unclosed" section of a polygon. We'll need to
    // store THOSE edges in the "outsideStartOutside" list, so that once we finish the polygon, we can rejoin them with
    // the other unprocessed "outside" edges.

    // We'll need to detect crossings of the circle, so that we can "join" the outside edges together. This is somewhat
    // complicated by the fact that the endpoints of a segment may be on the circle, so one edge might be fully
    // internal, and the next might be fully external. We'll use an epsilon to detect this.

    // -------------

    // Our edges of output polygons (that will need to be "split up" if they are circular) will be stored here. These
    // are in "final" form, except for the splitting.
    const insideCandidatePolygons: ( LinearEdge | CircularEdge )[][] = [];
    const outsideCandidatePolygons: ( LinearEdge | CircularEdge )[][] = [];

    // Our "inside" edges are always stored in the "forward" order. For every input polygon, we'll push here and then
    // put this into the insideCandidatePolygons array (one input polygon to one potential output polygon).
    const insideCandidateEdges: ( LinearEdge | CircularEdge )[] = [];

    // The arrays we push outside edges when hasOutsideStartPoint = false. When we have a crossing, we'll have a
    // complete outside polygon to push to outsideCandidatePolygons.
    const outsideCandidateForwardEdges: LinearEdge[] = [];
    const outsideCandidateReversedEdges: CircularEdge[] = [];

    // We'll need to handle the cases where we start "outside", and thus don't have the matching "outside" edges yet.
    // If we have an outside start point, we'll need to store the edges until we are completely done with that input
    // polygon, then will connect them up!
    let hasOutsideStartPoint = false;
    let hasInsidePoint = false;
    const outsideStartOutsideCandidateForwardEdges: LinearEdge[] = [];
    const outsideStartOutsideCandidateReversedEdges: CircularEdge[] = [];

    // We'll also need to store "critical" angles for the future polygonalization of the circles. If we were outputting
    // true circular edges, we could just include `insideCircularEdges`, however we want to convert it to line segments
    // so that future stages don't have to deal with this.
    // We'll need the angles so that those points on the circle will be exact (for ALL of the circular edges).
    // This is because we may be wrapping back-and-forth across the circle multiple times, with different start/end
    // angles, and we need the polygonal parts of these overlaps to be identical (to avoid precision issues later,
    // and ESPECIALLY to avoid little polygonal bits with "negative" area where the winding orientation is flipped.
    // There are two types of points where we'll need to store the angles:
    // 1. Intersections with our circle (where we'll need to "split" the edge into two at that point)
    // 2. Points where we are between the circumradius and inradius of the roughest "regular" polygon we might generate.

    // Because we need to output polygon data in order, we'll need to process ALL of the data, determine the angles,
    // and then output all of it.

    // between [-pi,pi], from atan2, tracked so we can turn the arcs piecewise-linear in a consistent fashion later
    let angles: number[] = [];

    const processCrossing = () => {
      // We crossed! Now our future "outside" handling will have a "joined" start point
      hasOutsideStartPoint = false;

      if ( outsideCandidateForwardEdges.length ) {
        outsideCandidateReversedEdges.reverse();

        // Ensure that our start and end points match up
        if ( assert ) {
          const startEdgePoint = outsideCandidateForwardEdges[ 0 ].startPoint;
          const endEdgePoint = outsideCandidateForwardEdges[ outsideCandidateForwardEdges.length - 1 ].endPoint;
          const startRadialPoint = Vector2.createPolar( radius, outsideCandidateReversedEdges[ 0 ].startAngle ).add( center );
          const endRadialPoint = Vector2.createPolar( radius, outsideCandidateReversedEdges[ outsideCandidateReversedEdges.length - 1 ].endAngle ).add( center );

          assert( startEdgePoint.equalsEpsilon( endRadialPoint, 1e-6 ) );
          assert( endEdgePoint.equalsEpsilon( startRadialPoint, 1e-6 ) );
        }

        const candidatePolygon = [
          ...outsideCandidateForwardEdges,
          ...outsideCandidateReversedEdges
        ];
        outsideCandidatePolygons.push( candidatePolygon );

        outsideCandidateForwardEdges.length = 0;
        outsideCandidateReversedEdges.length = 0;
      }
    };

    // Process a fully-inside-the-circle part of an edge
    const processInternal = ( start: Vector2, end: Vector2 ) => {
      insideCandidateEdges.push( new LinearEdge( start, end ) );

      const localStart = start.minus( center );
      const localEnd = end.minus( center );

      // We're already inside the circle, so the circumradius check isn't needed. If we're inside the inradius,
      // ensure the critical angles are added.
      if ( localStart.magnitude > inradius ) {
        angles.push( localStart.angle );
      }
      if ( localEnd.magnitude > inradius ) {
        angles.push( localEnd.angle );
      }
    };

    // Process a fully-outside-the-circle part of an edge
    const processExternal = ( start: Vector2, end: Vector2 ) => {

      if ( hasOutsideStartPoint ) {
        outsideStartOutsideCandidateForwardEdges.push( new LinearEdge( start, end ) );
      }
      else {
        outsideCandidateForwardEdges.push( new LinearEdge( start, end ) );
      }

      const localStart = start.minus( center );
      const localEnd = end.minus( center );

      // Modify (project) them into points of the given radius.
      localStart.multiplyScalar( radius / localStart.magnitude );
      localEnd.multiplyScalar( radius / localEnd.magnitude );

      // Handle projecting the edge to the circle.
      // We'll only need to do extra work if the projected points are not equal. If we had a line that was pointed
      // toward the center of the circle, it would project down to a single point, and we wouldn't have any contribution.
      if ( !localStart.equalsEpsilon( localEnd, 1e-8 ) ) {
        // Check to see which way we went "around" the circle

        // (y, -x) perpendicular, so a clockwise pi/2 rotation
        const isClockwise = localStart.perpendicular.dot( localEnd ) > 0;

        const startAngle = localStart.angle;
        const endAngle = localEnd.angle;

        angles.push( startAngle );
        angles.push( endAngle );

        insideCandidateEdges.push( new CircularEdge( startAngle, endAngle, !isClockwise ) );
        if ( hasOutsideStartPoint ) {
          // TODO: fish out this circular edge, we're using it for both
          outsideStartOutsideCandidateReversedEdges.push( new CircularEdge( endAngle, startAngle, isClockwise ) );
        }
        else {
          outsideCandidateReversedEdges.push( new CircularEdge( endAngle, startAngle, isClockwise ) );
        }
      }
    };

    // Stage to process the edges into the insideCandidatesPolygons/outsideCandidatesPolygons arrays.
    for ( let i = 0; i < polygons.length; i++ ) {
      const polygon = polygons[ i ];

      for ( let j = 0; j < polygon.length; j++ ) {
        const start = polygon[ j ];
        const end = polygon[ ( j + 1 ) % polygon.length ];

        const p0x = start.x - center.x;
        const p0y = start.y - center.y;
        const p1x = end.x - center.x;
        const p1y = end.y - center.y;

        // We'll use squared comparisons to avoid square roots
        const startDistanceSquared = p0x * p0x + p0y * p0y;
        const endDistanceSquared = p1x * p1x + p1y * p1y;

        const startInside = startDistanceSquared <= radiusSquared;
        const endInside = endDistanceSquared <= radiusSquared;

        // If we meet these thresholds, we'll process a crossing
        const startOnCircle = Math.abs( startDistanceSquared - radiusSquared ) < 1e-8;
        const endOnCircle = Math.abs( endDistanceSquared - radiusSquared ) < 1e-8;

        // If we're the first edge, set up our starting conditions
        if ( j === 0 ) {
          hasOutsideStartPoint = !startInside && !startOnCircle;
          hasInsidePoint = startInside || endInside;
        }
        else {
          hasInsidePoint = hasInsidePoint || startInside || endInside;
        }

        // If the endpoints are within the circle, the entire contents will be also (shortcut)
        if ( startInside && endInside ) {
          processInternal( start, end );
          if ( startOnCircle || endOnCircle ) {
            processCrossing();
          }
          continue;
        }

        // Now, we'll solve for the t-values of the intersection of the line and the circle.
        // e.g. p0 + t * ( p1 - p0 ) will be on the circle. This is solvable with a quadratic equation.

        const dx = p1x - p0x;
        const dy = p1y - p0y;

        // quadratic to solve
        const a = dx * dx + dy * dy;
        const b = 2 * ( p0x * dx + p0y * dy );
        const c = p0x * p0x + p0y * p0y - radius * radius;

        assert && assert( a > 0, 'We should have a delta, assumed in code below' );

        const roots = Utils.solveQuadraticRootsReal( a, b, c );

        let isFullyExternal = false;

        // If we have no roots, we're fully outside the circle!
        if ( !roots || roots.length === 0 ) {
          isFullyExternal = true;
        }
        else {
          if ( roots.length === 1 ) {
            roots.push( roots[ 0 ] );
          }
          assert && assert( roots[ 0 ] <= roots[ 1 ], 'Easier for us to assume root ordering' );
          const rootA = roots[ 0 ];
          const rootB = roots[ 1 ];

          if ( rootB <= 0 || rootA >= 1 ) {
            isFullyExternal = true;
          }

          // If our roots are identical, we are TANGENT to the circle. We can consider this to be fully external, since
          // there will not be an internal section.
          if ( rootA === rootB ) {
            isFullyExternal = true;
          }
        }

        if ( isFullyExternal ) {
          processExternal( start, end );
          continue;
        }

        assert && assert( roots![ 0 ] <= roots![ 1 ], 'Easier for us to assume root ordering' );
        const rootA = roots![ 0 ];
        const rootB = roots![ 1 ];

        // Compute intersection points (when the t values are in the range [0,1])
        const rootAInSegment = rootA > 0 && rootA < 1;
        const rootBInSegment = rootB > 0 && rootB < 1;
        const deltaPoints = end.minus( start );
        const rootAPoint = rootAInSegment ? ( start.plus( deltaPoints.timesScalar( rootA ) ) ) : Vector2.ZERO; // ignore the zero, it's mainly for typing
        const rootBPoint = rootBInSegment ? ( start.plus( deltaPoints.timesScalar( rootB ) ) ) : Vector2.ZERO; // ignore the zero, it's mainly for typing

        if ( rootAInSegment && rootBInSegment ) {
          processExternal( start, rootAPoint );
          processCrossing();
          processInternal( rootAPoint, rootBPoint );
          processCrossing();
          processExternal( rootBPoint, end );
        }
        else if ( rootAInSegment ) {
          processExternal( start, rootAPoint );
          processCrossing();
          processInternal( rootAPoint, end );
          if ( endOnCircle ) {
            processCrossing();
          }
        }
        else if ( rootBInSegment ) {
          if ( startOnCircle ) {
            processCrossing();
          }
          processInternal( start, rootBPoint );
          processCrossing();
          processExternal( rootBPoint, end );
        }
        else {
          assert && assert( false, 'Should not reach this point, due to the boolean constraints above' );
        }
      }

      // We finished the input polygon! Now we need to connect up things if we started outside.
      if ( outsideCandidateForwardEdges.length || outsideStartOutsideCandidateForwardEdges.length ) {
        // We... really should have both? Let's be permissive with epsilon checks?

        outsideCandidateReversedEdges.reverse();
        outsideStartOutsideCandidateReversedEdges.reverse();

        if ( hasInsidePoint ) {
          const candidatePolygon = [
            ...outsideCandidateForwardEdges,
            ...outsideStartOutsideCandidateForwardEdges,
            ...outsideStartOutsideCandidateReversedEdges,
            ...outsideCandidateReversedEdges
          ];
          outsideCandidatePolygons.push( candidatePolygon );

          // Ensure that all of our points must match up
          if ( assertSlow ) {
            for ( let i = 0; i < candidatePolygon.length; i++ ) {
              const edge = candidatePolygon[ i ];
              const nextEdge = candidatePolygon[ ( i + 1 ) % candidatePolygon.length ];

              const endPoint = edge instanceof LinearEdge ? edge.endPoint : Vector2.createPolar( radius, edge.endAngle ).add( center );
              const startPoint = nextEdge instanceof LinearEdge ? nextEdge.startPoint : Vector2.createPolar( radius, nextEdge.startAngle ).add( center );

              assertSlow( endPoint.equalsEpsilon( startPoint, 1e-6 ) );
            }
          }
        }
        else {
          // If we're fully external, we'll need to create two paths
          outsideCandidatePolygons.push( [
            ...outsideStartOutsideCandidateForwardEdges
          ] );
          outsideCandidatePolygons.push( [
            ...outsideStartOutsideCandidateReversedEdges
          ] );

          // Ensure match-ups
          if ( assertSlow ) {
            // Just check this for now
            assertSlow( outsideStartOutsideCandidateForwardEdges[ 0 ].startPoint.equalsEpsilon( outsideStartOutsideCandidateForwardEdges[ outsideStartOutsideCandidateForwardEdges.length - 1 ].endPoint, 1e-6 ) );
          }
        }

        outsideCandidateForwardEdges.length = 0;
        outsideCandidateReversedEdges.length = 0;
        outsideStartOutsideCandidateForwardEdges.length = 0;
        outsideStartOutsideCandidateReversedEdges.length = 0;
      }

      // TODO: should we assertion-check that these match up?
      if ( insideCandidateEdges.length ) {
        insideCandidatePolygons.push( insideCandidateEdges.slice() );
        insideCandidateEdges.length = 0;
      }
    }

    // Sort our critical angles, so we can iterate through unique values in-order
    angles = _.uniq( angles.sort( ( a, b ) => a - b ) );

    // We'll just add the start point(s)
    const addEdgeTo = ( edge: LinearEdge | CircularEdge, simplifier: ClipSimplifier ) => {
      if ( edge instanceof LinearEdge ) {
        simplifier.addPoint( edge.startPoint );
      }
      else {
        const startIndex = angles.indexOf( edge.startAngle );
        const endIndex = angles.indexOf( edge.endAngle );

        const subAngles: number[] = [];

        // Iterate (in the specific direction) through the angles we cover, and add them to our subAngles list.
        const dirSign = edge.counterClockwise ? 1 : -1;
        for ( let index = startIndex; index !== endIndex; index = ( index + dirSign + angles.length ) % angles.length ) {
          subAngles.push( angles[ index ] );
        }
        subAngles.push( angles[ endIndex ] );

        for ( let j = 0; j < subAngles.length - 1; j++ ) {
          const startAngle = subAngles[ j ];
          const endAngle = subAngles[ j + 1 ];

          // Skip "negligible" angles
          const absDiff = Math.abs( startAngle - endAngle );
          if ( absDiff < 1e-9 || Math.abs( absDiff - Math.PI * 2 ) < 1e-9 ) {
            continue;
          }

          // Put our end angle in the dirSign direction from our startAngle (if we're counterclockwise and our angle increases,
          // our relativeEndAngle should be greater than our startAngle, and similarly if we're clockwise and our angle decreases,
          // our relativeEndAngle should be less than our startAngle)
          const relativeEndAngle = ( edge.counterClockwise === ( startAngle < endAngle ) ) ? endAngle : endAngle + dirSign * Math.PI * 2;

          // Split our circular arc into segments!
          const angleDiff = relativeEndAngle - startAngle;
          const numSegments = Math.ceil( Math.abs( angleDiff ) / maxAngleSplit );
          for ( let k = 0; k < numSegments; k++ ) {
            const startTheta = startAngle + angleDiff * ( k / numSegments );
            const startPoint = Vector2.createPolar( radius, startTheta ).add( center );

            simplifier.addPoint( startPoint );
          }
        }
      }
    };

    let totalArea = 0; // For assertions

    const addPolygonTo = ( edges: ( LinearEdge | CircularEdge )[], polygons: Vector2[][] ) => {

      for ( let j = 0; j < edges.length; j++ ) {
        addEdgeTo( edges[ j ], simplifier );
      }

      const polygon = simplifier.finalize();

      if ( polygon.length >= 3 ) {
        polygons.push( polygon );

        if ( assertSlow ) {
          totalArea += new PolygonalFace( [ polygon ] ).getArea();
        }
      }
    };

    for ( let i = 0; i < insideCandidatePolygons.length; i++ ) {
      addPolygonTo( insideCandidatePolygons[ i ], inside );
    }

    for ( let i = 0; i < outsideCandidatePolygons.length; i++ ) {
      addPolygonTo( outsideCandidatePolygons[ i ], outside );
    }

    if ( assertSlow ) {
      const beforeArea = new PolygonalFace( polygons ).getArea();

      assertSlow( Math.abs( totalArea - beforeArea ) < 1e-5 );
    }
  }
}

// Stores data for binaryCircularClipPolygon
class CircularEdge {
  public constructor(
    public readonly startAngle: number,
    public readonly endAngle: number,
    public readonly counterClockwise: boolean
  ) {}
}

// Stores data for binaryCircularClipEdges
class CircularEdgeWithPoints {
  public constructor(
    public readonly startPoint: Vector2 | null,
    public readonly endPoint: Vector2 | null,
    public readonly startAngle: number,
    public readonly endAngle: number,
    public readonly counterClockwise: boolean
  ) {}
}

scenery.register( 'PolygonClipping', PolygonClipping );
