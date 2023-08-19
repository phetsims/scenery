// Copyright 2023, University of Colorado Boulder

/**
 * Handles finding intersections between IntegerEdges (will push RationalIntersections into the edge's intersections
 * arrays)
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { BigIntVector2, BigRational, IntegerEdge, IntersectionPoint, RationalIntersection, scenery } from '../../../imports.js';
import Bounds2 from '../../../../../dot/js/Bounds2.js';

export default class LineIntersector {
  public static processIntegerEdgeIntersection( edgeA: IntegerEdge, edgeB: IntegerEdge ): void {
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

  public static edgeIntersectionBoundsTree( integerEdges: IntegerEdge[] ): void {
    BoundsTreeNode.fromIntegerEdges( integerEdges ).selfIntersect();
  }

  public static edgeIntersectionArrayBoundsTree( integerEdges: IntegerEdge[] ): void {
    // Probably more micro-efficient ways, but this is a proof-of-concept

    // TODO: probably just implement a struct-based approach, so it's a bit easier?

    const boundsTree: Bounds2[][] = [];
    const hasHorizontalTree: boolean[][] = [];
    const hasVerticalTree: boolean[][] = [];

    // Code for the first layer TODO: should be combined with the other logic?
    const secondLayer = [];
    const secondLayerHasHorizontal = [];
    const secondLayerHasVertical = [];
    for ( let i = 0; i < integerEdges.length; i += 2 ) {
      const edgeA = integerEdges[ i ];

      let bounds;
      let hasHorizontal = edgeA.y0 === edgeA.y1;
      let hasVertical = edgeA.x0 === edgeA.x1;

      if ( i + 1 < integerEdges.length ) {
        const edgeB = integerEdges[ i + 1 ];
        bounds = edgeA.bounds.union( edgeB.bounds );
        hasHorizontal = hasHorizontal || edgeB.y0 === edgeB.y1;
        hasVertical = hasVertical || edgeB.x0 === edgeB.x1;
      }
      else {
        bounds = edgeA.bounds;
      }
      secondLayer.push( bounds );
      secondLayerHasHorizontal.push( hasHorizontal );
      secondLayerHasVertical.push( hasVertical );
    }
    boundsTree.push( secondLayer );
    hasHorizontalTree.push( secondLayerHasHorizontal );
    hasVerticalTree.push( secondLayerHasVertical );

    while ( boundsTree[ boundsTree.length - 1 ].length > 1 ) {
      const nextLayer = [];
      const nextLayerHasHorizontal = [];
      const nextLayerHasVertical = [];
      const lastLayer = boundsTree[ boundsTree.length - 1 ];
      const lastLayerHasHorizontal = hasHorizontalTree[ hasHorizontalTree.length - 1 ];
      const lastLayerHasVertical = hasVerticalTree[ hasVerticalTree.length - 1 ];
      for ( let i = 0; i < lastLayer.length; i += 2 ) {
        let bounds;
        let hasHorizontal = lastLayerHasHorizontal[ i ];
        let hasVertical = lastLayerHasVertical[ i ];
        if ( i + 1 < lastLayer.length ) {
          bounds = lastLayer[ i ].union( lastLayer[ i + 1 ] );
          hasHorizontal = hasHorizontal || lastLayerHasHorizontal[ i + 1 ];
          hasVertical = hasVertical || lastLayerHasVertical[ i + 1 ];
        }
        else {
          bounds = lastLayer[ i ];
        }
        nextLayer.push( bounds );
        nextLayerHasHorizontal.push( hasHorizontal );
        nextLayerHasVertical.push( hasVertical );
      }
      boundsTree.push( nextLayer );
      hasHorizontalTree.push( nextLayerHasHorizontal );
      hasVerticalTree.push( nextLayerHasVertical );
    }

    // TODO: integrate this scanning into our traversal of the tree
    const handleSelfIntersection = ( level: number, index: number ): void => {
      const baseIndex = 2 * index;
      const nextIndex = baseIndex + 1;

      const hasNext = nextIndex < ( level === 0 ? integerEdges.length : boundsTree[ level - 1 ].length );

      if ( level === 0 ) {
        if ( hasNext ) {
          const edgeA = integerEdges[ baseIndex ];
          const edgeB = integerEdges[ nextIndex ];

          if ( edgeA.hasBoundsIntersectionWith( edgeB ) ) {
            LineIntersector.processIntegerEdgeIntersection( edgeA, edgeB );
          }
        }
      }
      else {
        handleSelfIntersection( level - 1, baseIndex );
        if ( hasNext ) {
          handleSelfIntersection( level - 1, nextIndex );
          handleCrossIntersection( level - 1, baseIndex, nextIndex );
        }
      }
    };

    const handleCrossIntersection = ( level: number, indexA: number, indexB: number ): void => {
      const boundsLevel = boundsTree[ level ];
      const hasHorizontalLevel = hasHorizontalTree[ level ];
      const hasVerticalLevel = hasVerticalTree[ level ];

      const boundsA = boundsLevel[ indexA ];
      const boundsB = boundsLevel[ indexB ];

      // TODO: renaming? we have swapping vertical/horizontal which might be confusing
      const someIsHorizontal = hasHorizontalLevel[ indexA ] || hasHorizontalLevel[ indexB ];
      const someIsVertical = hasVerticalLevel[ indexA ] || hasVerticalLevel[ indexB ];

      if ( IntegerEdge.hasBoundsIntersection( boundsA, boundsB, someIsVertical, someIsHorizontal ) ) {
        const baseIndexA = 2 * indexA;
        const baseIndexB = 2 * indexB;

        const nextIndexA = baseIndexA + 1;
        const nextIndexB = baseIndexB + 1;

        const nextLevelCount = level === 0 ? integerEdges.length : boundsTree[ level - 1 ].length;

        const hasA1 = nextIndexA < nextLevelCount;
        const hasB1 = nextIndexB < nextLevelCount;

        if ( level === 0 ) {
          const edgeA0 = integerEdges[ baseIndexA ];
          const edgeB0 = integerEdges[ baseIndexB ];
          const edgeA1 = hasA1 ? integerEdges[ nextIndexA ] : null;
          const edgeB1 = hasB1 ? integerEdges[ nextIndexB ] : null;

          // TODO: optimization-wise, could collapse some conditionals
          if ( edgeA0.hasBoundsIntersectionWith( edgeB0 ) ) {
            LineIntersector.processIntegerEdgeIntersection( edgeA0, edgeB0 );
          }
          if ( edgeA1 && edgeA1.hasBoundsIntersectionWith( edgeB0 ) ) {
            LineIntersector.processIntegerEdgeIntersection( edgeA1, edgeB0 );
          }
          if ( edgeB1 && edgeA0.hasBoundsIntersectionWith( edgeB1 ) ) {
            LineIntersector.processIntegerEdgeIntersection( edgeA0, edgeB1 );
          }
          if ( edgeA1 && edgeB1 && edgeA1.hasBoundsIntersectionWith( edgeB1 ) ) {
            LineIntersector.processIntegerEdgeIntersection( edgeA1, edgeB1 );
          }
        }
        else {
          handleCrossIntersection( level - 1, baseIndexA, baseIndexB );
          // TODO: optimization-wise, could collapse some conditionals
          if ( hasA1 ) {
            handleCrossIntersection( level - 1, nextIndexA, baseIndexB );
          }
          if ( hasB1 ) {
            handleCrossIntersection( level - 1, baseIndexA, nextIndexB );
          }
          if ( hasA1 && hasB1 ) {
            handleCrossIntersection( level - 1, nextIndexA, nextIndexB );
          }
        }
      }
    };

    handleSelfIntersection( boundsTree.length - 1, 0 );
  }

  public static edgeIntersectionQuadratic( integerEdges: IntegerEdge[] ): void {
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

        if ( IntegerEdge.hasBoundsIntersection( boundsA, boundsB, someXEqual, someYEqual ) ) {
          LineIntersector.processIntegerEdgeIntersection( edgeA, edgeB );
        }
      }
    }
  }
}

abstract class BoundsTreeNode {
  protected constructor( public readonly bounds: Bounds2, public readonly hasHorizontal: boolean, public readonly hasVertical: boolean ) {}

  public abstract selfIntersect(): void;
  public abstract crossIntersect( other: BoundsTreeNode ): void;

  public static fromIntegerEdges( integerEdges: IntegerEdge[] ): BoundsTreeNode {
    if ( integerEdges.length === 0 ) {
      throw new Error();
    }
    let nodes: BoundsTreeNode[] = integerEdges.map( edge => new BoundsTreeLeaf( edge ) );

    // Recursively pair up nodes
    while ( nodes.length > 1 ) {
      const nextNodes: BoundsTreeNode[] = [];

      // Pair up the nodes
      for ( let i = 0; i < nodes.length; i += 2 ) {
        const nodeA = nodes[ i ];
        if ( i + 1 < nodes.length ) {
          const nodeB = nodes[ i + 1 ];
          nextNodes.push( new BoundsTreeBinary( nodeA, nodeB ) );
        }
        else {
          nextNodes.push( nodeA );
        }
      }
      nodes = nextNodes;
    }
    return nodes[ 0 ];
  }
}

class BoundsTreeLeaf extends BoundsTreeNode {
  public constructor( public edge: IntegerEdge ) {
    super( edge.bounds, edge.y0 === edge.y1, edge.x0 === edge.x1 );
  }

  public override selfIntersect(): void {
    // NOTHING
  }

  public override crossIntersect( other: BoundsTreeNode ): void {
    if ( IntegerEdge.hasBoundsIntersection( this.bounds, other.bounds, this.hasVertical, this.hasHorizontal ) ) {
      if ( other instanceof BoundsTreeLeaf ) {
        LineIntersector.processIntegerEdgeIntersection( this.edge, other.edge );
      }
      else if ( other instanceof BoundsTreeBinary ) {
        this.crossIntersect( other.left );
        this.crossIntersect( other.right );
      }
    }
  }
}

class BoundsTreeBinary extends BoundsTreeNode {
  public constructor(
    public readonly left: BoundsTreeNode,
    public readonly right: BoundsTreeNode
  ) {
    super( left.bounds.union( right.bounds ), left.hasHorizontal || right.hasHorizontal, left.hasVertical || right.hasVertical );
  }

  public override selfIntersect(): void {
    this.left.selfIntersect();
    this.right.selfIntersect();
    this.left.crossIntersect( this.right );
  }

  public override crossIntersect( other: BoundsTreeNode ): void {
    if ( IntegerEdge.hasBoundsIntersection( this.bounds, other.bounds, this.hasVertical, this.hasHorizontal ) ) {
      if ( other instanceof BoundsTreeLeaf ) {
        this.left.crossIntersect( other );
        this.right.crossIntersect( other );
      }
      else if ( other instanceof BoundsTreeBinary ) {
        this.left.crossIntersect( other.left );
        this.left.crossIntersect( other.right );
        this.right.crossIntersect( other.left );
        this.right.crossIntersect( other.right );
      }
    }
  }
}

scenery.register( 'LineIntersector', LineIntersector );
