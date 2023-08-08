// Copyright 2023, University of Colorado Boulder

/**
 * Acceleration of pairwise intersection tests for anything bounds-related
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */
import { scenery } from '../../imports.js';
import Bounds2 from '../../../../dot/js/Bounds2.js';
import Orientation from '../../../../phet-core/js/Orientation.js';
import OrientationPair from '../../../../phet-core/js/OrientationPair.js';
import { SegmentTree } from '../../../../kite/js/imports.js';

type Item = {
  bounds: Bounds2;
};

class ItemSegmentTree extends SegmentTree<Item> {
  public getMinX( item: Item, epsilon: number ): number {
    return item.bounds.left - epsilon;
  }

  public getMaxX( item: Item, epsilon: number ): number {
    return item.bounds.right + epsilon;
  }
}

export default class BoundsIntersectionFilter {

  public static quadraticIntersect( maximumBounds: Bounds2, items: Item[], callback: ( itemA: Item, itemB: Item ) => void ): void {
    for ( let i = 0; i < items.length; i++ ) {
      const itemA = items[ i ];
      const boundsA = itemA.bounds;
      for ( let j = i + 1; j < items.length; j++ ) {
        const itemB = items[ j ];
        if ( boundsA.intersectsBounds( itemB.bounds ) ) {
          callback( itemA, itemB );
        }
      }
    }
  }

  public static sweepLineIntersect( maximumBounds: Bounds2, items: Item[], callback: ( itemA: Item, itemB: Item ) => void ): void {

    // We'll expand bounds by this amount, so that "adjacent" bounds (with a potentially overlapping vertical or
    // horizontal line) will have a non-zero amount of area overlapping.
    const epsilon = 1e-10;

    // Our queue will store entries of { start: boolean, edge: Edge }, representing a sweep line similar to the
    // Bentley-Ottmann approach. We'll track which edges are passing through the sweep line.
    // ts-ignore-line
    const queue = new FlatQueue();

    // Tracks which edges are through the sweep line, but in a graph structure like a segment/interval tree, so that we
    // can have fast lookup (what edges are in a certain range) and also fast inserts/removals.
    const segmentTree = new ItemSegmentTree( epsilon );

    // Adds an edge to the queue
    const addToQueue = ( item: Item ) => {
      const bounds = item.bounds;

      // TODO: see if object allocations are slow here
      queue.push( { start: true, item: item }, bounds.minY - epsilon );
      queue.push( { start: false, item: item }, bounds.maxY + epsilon );
    };

    for ( let i = 0; i < items.length; i++ ) {
      addToQueue( items[ i ] );
    }

    while ( queue.length ) {
      const entry: { start: boolean; item: Item } = queue.pop();
      const item = entry.item;

      if ( entry.start ) {
        segmentTree.query( item, otherItem => {
          callback( item, otherItem );
          return false;
        } );

        segmentTree.addItem( item );
      }
      else {
        // Removal can't trigger an intersection, so we can safely remove it
        segmentTree.removeItem( item );
      }
    }
  }

  public static filterIntersect( maximumBounds: Bounds2, items: Item[], callback: ( itemA: Item, itemB: Item ) => void ): void {
    BoundsIntersectionFilter.recurse(
      Orientation.HORIZONTAL,
      maximumBounds,
      items,
      [],
      false,
      new OrientationPair( false, false ),
      new OrientationPair( false, false ),
      new OrientationPair( false, false ),
      callback
    );
  }

  private static intersect(
    internalItems: Item[],
    externalItems: Item[],
    external: boolean,
    callback: ( itemA: Item, itemB: Item ) => void
  ): void {
    for ( let i = 0; i < internalItems.length; i++ ) {
      const item = internalItems[ i ];
      for ( let j = 0; j < externalItems.length; j++ ) {
        callback( item, externalItems[ j ] );
      }

      if ( !external ) {
        for ( let j = i + 1; j < internalItems.length; j++ ) {
          callback( item, internalItems[ j ] );
        }
      }
    }
  }

  private static recurse(
    orientation: Orientation,
    bounds: Bounds2,
    internalItems: Item[],
    externalItems: Item[],
    otherComplete: boolean,
    external: OrientationPair<boolean>,
    minSplitLast: OrientationPair<boolean>,
    clipped: OrientationPair<boolean>,
    callback: ( itemA: Item, itemB: Item ) => void
  ): void {

    const anyExternal = external.horizontal || external.vertical;

    // No intersections if we are fully external and one side has nothing
    if ( anyExternal && externalItems.length === 0 ) {
      return;
    }

    // If we have no internal items, nothing to do
    if ( internalItems.length === 0 ) {
      return;
    }

    // TODO: efficiency
    const bailOrientation = () => {
      if ( otherComplete ) {
        BoundsIntersectionFilter.intersect( internalItems, externalItems, anyExternal, callback );
      }
      else {
        BoundsIntersectionFilter.recurse(
          orientation.opposite,
          bounds,
          internalItems,
          externalItems,
          true,
          external,
          minSplitLast,
          clipped,
          callback
        );
      }
    };

    let split = 0;
    let minValue = Number.POSITIVE_INFINITY;
    let maxValue = Number.NEGATIVE_INFINITY;

    const isClipped = clipped.get( orientation );
    const minSplitLastValue = minSplitLast.get( orientation );
    const minSide = orientation.minSide;
    const maxSide = orientation.maxSide;

    const boundsMin = bounds[ minSide ];
    const boundsMax = bounds[ maxSide ];

    for ( let i = 0; i < internalItems.length; i++ ) {
      const itemBounds = internalItems[ i ].bounds;
      const itemMin = Math.max( itemBounds[ minSide ], boundsMin );
      const itemMax = Math.min( itemBounds[ maxSide ], boundsMax );

      minValue = Math.min( minValue, itemMin );
      maxValue = Math.max( maxValue, itemMax );

      if ( external && isClipped ) {
        split = minSplitLastValue ? itemMax : itemMin;
      }
      else {
        // the center of the item
        split += 0.5 * ( itemMin + itemMax );
      }
    }
    for ( let i = 0; i < externalItems.length; i++ ) {
      const itemBounds = externalItems[ i ].bounds;
      const itemMin = Math.max( itemBounds[ minSide ], boundsMin );
      const itemMax = Math.min( itemBounds[ maxSide ], boundsMax );

      minValue = Math.min( minValue, itemMin );
      maxValue = Math.max( maxValue, itemMax );

      // if it's clipped, we'll use the ending coordinate only
      if ( isClipped ) {
        split = minSplitLastValue ? itemMax : itemMin;
      }
      else {
        split += 0.5 * ( itemMin + itemMax );
      }
    }

    if ( maxValue - minValue < 1e-6 ) {
      bailOrientation();
      return;
    }

    // average x
    split /= internalItems.length + externalItems.length;

    // Disjoint
    const minActiveItems = internalItems.filter( item => item.bounds[ maxSide ] < split );
    const maxActiveItems = internalItems.filter( item => item.bounds[ minSide ] > split );

    if ( minActiveItems.length === 0 || maxActiveItems.length === 0 ) {
      bailOrientation();
      return;
    }

    const bothActiveItems = internalItems.filter( item => item.bounds[ minSide ] <= split && item.bounds[ maxSide ] >= split );

    const minBounds = orientation === Orientation.HORIZONTAL ? bounds.withMaxX( split ) : bounds.withMaxY( split );
    const maxBounds = orientation === Orientation.HORIZONTAL ? bounds.withMinX( split ) : bounds.withMinY( split );

    // Possible overlap
    const minInactiveItems = externalItems.filter( item => item.bounds[ maxSide ] < split );
    const maxInactiveItems = externalItems.filter( item => item.bounds[ minSide ] > split );
    const bothInactiveItems = externalItems.filter( item => item.bounds[ minSide ] <= split && item.bounds[ maxSide ] >= split );

    const newClipped = clipped.with( orientation, true );
    const newExternal = external.with( orientation, true );
    const minSplit = minSplitLast.with( orientation, true );
    const maxSplit = minSplitLast.with( orientation, false );

    // "minimum" case, internal for "min", moving "both" to external
    BoundsIntersectionFilter.recurse(
      otherComplete ? orientation : orientation.opposite,
      minBounds,
      minActiveItems,
      external.get( orientation ) ? minInactiveItems : minInactiveItems.concat( bothActiveItems ),
      otherComplete,
      external,
      minSplit,
      newClipped,
      callback
    );

    // "maximum" case, internal for "max", moving "both" to external
    BoundsIntersectionFilter.recurse(
      otherComplete ? orientation : orientation.opposite,
      maxBounds,
      maxActiveItems,
      external.get( orientation ) ? maxInactiveItems : maxInactiveItems.concat( bothActiveItems ),
      otherComplete,
      external,
      maxSplit,
      newClipped,
      callback
    );

    if ( otherComplete ) {
      // "both" case (we're both xComplete and yComplete, so finalize it)
      BoundsIntersectionFilter.intersect( bothActiveItems, bothInactiveItems, anyExternal, callback );
    }
    else {
      // "both" case, internal for "both", just including external "both" so we are x-complete
      BoundsIntersectionFilter.recurse(
        orientation.opposite,
        bounds,
        bothActiveItems,
        bothInactiveItems,
        true,
        external,
        minSplitLast,
        clipped, // we did NOT change clips!
        callback
      );
    }

    if ( !external.get( orientation ) ) {
      // min external "both" case
      BoundsIntersectionFilter.recurse(
        orientation,
        minBounds,
        bothActiveItems,
        minInactiveItems,
        otherComplete,
        newExternal,
        minSplit,
        newClipped,
        callback
      );

      // max external "both" case
      BoundsIntersectionFilter.recurse(
        orientation,
        maxBounds,
        bothActiveItems,
        maxInactiveItems,
        otherComplete,
        newExternal,
        maxSplit,
        newClipped,
        callback
      );
    }
  }
}

scenery.register( 'BoundsIntersectionFilter', BoundsIntersectionFilter );
