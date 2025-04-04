// Copyright 2013-2025, University of Colorado Boulder

/**
 * Contains information about what renderers (and a few other flags) are supported for an entire subtree.
 *
 * We effectively do this by tracking bitmask changes from scenery.js (used for rendering properties in general). In particular, we count
 * how many zeros in the bitmask we have in key places.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Renderer from '../display/Renderer.js';
import scenery from '../scenery.js';
import type Node from '../nodes/Node.js';

const summaryBits = [
  // renderer bits ("Is renderer X supported by the entire sub-tree?")
  Renderer.bitmaskCanvas,
  Renderer.bitmaskSVG,
  Renderer.bitmaskDOM,
  Renderer.bitmaskWebGL,

  // summary bits (added to the renderer bitmask to handle special flags for the summary)
  Renderer.bitmaskSingleCanvas,
  Renderer.bitmaskSingleSVG,
  Renderer.bitmaskNotPainted,
  Renderer.bitmaskBoundsValid,
  // NOTE: This could be separated out into its own implementation for this flag, since
  // there are cases where we actually have nothing fromt he PDOM DUE to things being pulled out by another pdom order.
  // This is generally NOT the case, so I've left this in here because it significantly simplifies the implementation.
  Renderer.bitmaskNoPDOM,

  // inverse renderer bits ("Do all painted nodes NOT support renderer X in this sub-tree?")
  Renderer.bitmaskLacksCanvas,
  Renderer.bitmaskLacksSVG,
  Renderer.bitmaskLacksDOM,
  Renderer.bitmaskLacksWebGL
];

const summaryBitIndices: Record<number, number> = {};
summaryBits.forEach( ( bit, index ) => {
  summaryBitIndices[ bit ] = index;
} );

const numSummaryBits = summaryBits.length;

// A bitmask with all of the bits set that we record
let bitmaskAll = 0;
for ( let l = 0; l < numSummaryBits; l++ ) {
  bitmaskAll |= summaryBits[ l ];
}

export default class RendererSummary {

  // maps bitmask indices (see summaryBitIndices, the index of the bitmask in summaryBits) to
  // a count of how many children (or self) have that property (e.g. can't renderer all of their contents with Canvas)
  private _counts: Int16Array = new Int16Array( numSummaryBits );

  // (scenery-internal)
  public bitmask: number = bitmaskAll;

  private selfBitmask: number;

  public constructor( private readonly node: Node ) {
    // NOTE: assumes that we are created in the Node constructor
    assert && assert( node._rendererBitmask === Renderer.bitmaskNodeDefault, 'Node must have a default bitmask when creating a RendererSummary' );
    assert && assert( node._children.length === 0, 'Node cannot have children when creating a RendererSummary' );

    this.selfBitmask = RendererSummary.summaryBitmaskForNodeSelf( node );

    this.summaryChange( this.bitmask, this.selfBitmask );

    // required listeners to update our summary based on painted/non-painted information
    const listener = this.selfChange.bind( this );
    this.node.filterChangeEmitter.addListener( listener );
    this.node.clipAreaProperty.lazyLink( listener );
    this.node.rendererSummaryRefreshEmitter.addListener( listener );
  }

  /**
   * Use a bitmask of all 1s to represent 'does not exist' since we count zeros
   */
  public summaryChange( oldBitmask: number, newBitmask: number ): void {
    assert && this.audit();

    const changeBitmask = oldBitmask ^ newBitmask; // bit set only if it changed

    let ancestorOldMask = 0;
    let ancestorNewMask = 0;
    for ( let i = 0; i < numSummaryBits; i++ ) {
      const bit = summaryBits[ i ];
      const bitIndex = summaryBitIndices[ bit ];

      // If the bit for the renderer has changed
      if ( bit & changeBitmask ) {

        // If it is now set (wasn't before), gained support for the renderer
        if ( bit & newBitmask ) {
          this._counts[ bitIndex ]--; // reduce count, since we count the number of 0s (unsupported)
          if ( this._counts[ bitIndex ] === 0 ) {
            ancestorNewMask |= bit; // add our bit to the "new" mask we will send to ancestors
          }
        }
        // It was set before (now isn't), lost support for the renderer
        else {
          this._counts[ bitIndex ]++; // increment the count, since we count the number of 0s (unsupported)
          if ( this._counts[ bitIndex ] === 1 ) {
            ancestorOldMask |= bit; // add our bit to the "old" mask we will send to ancestors
          }
        }
      }
    }

    if ( ancestorOldMask || ancestorNewMask ) {

      const oldSubtreeBitmask = this.bitmask;
      assert && assert( oldSubtreeBitmask !== undefined );

      for ( let j = 0; j < numSummaryBits; j++ ) {
        const ancestorBit = summaryBits[ j ];
        // Check for added bits
        if ( ancestorNewMask & ancestorBit ) {
          this.bitmask |= ancestorBit;
        }

        // Check for removed bits
        if ( ancestorOldMask & ancestorBit ) {
          this.bitmask ^= ancestorBit;
          assert && assert( !( this.bitmask & ancestorBit ),
            'Should be cleared, doing cheaper XOR assuming it already was set' );
        }
      }

      this.node.instanceRefreshEmitter.emit();
      this.node.onSummaryChange( oldSubtreeBitmask, this.bitmask );

      const len = this.node._parents.length;
      for ( let k = 0; k < len; k++ ) {
        this.node._parents[ k ]._rendererSummary.summaryChange( ancestorOldMask, ancestorNewMask );
      }

      assert && assert( this.bitmask === this.computeBitmask(), 'Sanity check' );
    }

    assert && this.audit();
  }

  public selfChange(): void {
    const oldBitmask = this.selfBitmask;
    const newBitmask = RendererSummary.summaryBitmaskForNodeSelf( this.node );
    if ( oldBitmask !== newBitmask ) {
      this.summaryChange( oldBitmask, newBitmask );
      this.selfBitmask = newBitmask;
    }
  }

  private computeBitmask(): number {
    let bitmask = 0;
    for ( let i = 0; i < numSummaryBits; i++ ) {
      if ( this._counts[ i ] === 0 ) {
        bitmask |= summaryBits[ i ];
      }
    }
    return bitmask;
  }

  /**
   * Is the renderer compatible with every single painted node under this subtree?
   * (Can this entire sub-tree be rendered with just this renderer)
   *
   * @param renderer - Single bit preferred. If multiple bits set, requires ALL painted nodes are compatible
   *                   with ALL of the bits.
   */
  public isSubtreeFullyCompatible( renderer: number ): boolean {
    return !!( renderer & this.bitmask );
  }

  /**
   * Is the renderer compatible with at least one painted node under this subtree?
   *
   * @param renderer - Single bit preferred. If multiple bits set, will return if a single painted node is
   *                   compatible with at least one of the bits.
   */
  public isSubtreeContainingCompatible( renderer: number ): boolean {
    return !( ( renderer << Renderer.bitmaskLacksShift ) & this.bitmask );
  }

  public isSingleCanvasSupported(): boolean {
    return !!( Renderer.bitmaskSingleCanvas & this.bitmask );
  }

  public isSingleSVGSupported(): boolean {
    return !!( Renderer.bitmaskSingleSVG & this.bitmask );
  }

  public isNotPainted(): boolean {
    return !!( Renderer.bitmaskNotPainted & this.bitmask );
  }

  public hasNoPDOM(): boolean {
    return !!( Renderer.bitmaskNoPDOM & this.bitmask );
  }

  public areBoundsValid(): boolean {
    return !!( Renderer.bitmaskBoundsValid & this.bitmask );
  }

  /**
   * Given a bitmask representing a list of ordered preferred renderers, we check to see if all of our nodes can be
   * displayed in a single SVG block, AND that given the preferred renderers, that it will actually happen in our
   * rendering process.
   */
  public isSubtreeRenderedExclusivelySVG( preferredRenderers: number ): boolean {
    // Check if we have anything that would PREVENT us from having a single SVG block
    if ( !this.isSingleSVGSupported() ) {
      return false;
    }

    // Check for any renderer preferences that would CAUSE us to choose not to display with a single SVG block
    for ( let i = 0; i < Renderer.numActiveRenderers; i++ ) {
      // Grab the next-most preferred renderer
      const renderer = Renderer.bitmaskOrder( preferredRenderers, i );

      // If it's SVG, congrats! Everything will render in SVG (since SVG is supported, as noted above)
      if ( Renderer.bitmaskSVG & renderer ) {
        return true;
      }

      // Since it's not SVG, if there's a single painted node that supports this renderer (which is preferred over SVG),
      // then it will be rendered with this renderer, NOT SVG.
      if ( this.isSubtreeContainingCompatible( renderer ) ) {
        return false;
      }
    }

    return false; // sanity check
  }

  /**
   * Given a bitmask representing a list of ordered preferred renderers, we check to see if all of our nodes can be
   * displayed in a single Canvas block, AND that given the preferred renderers, that it will actually happen in our
   * rendering process.
   */
  public isSubtreeRenderedExclusivelyCanvas( preferredRenderers: number ): boolean {
    // Check if we have anything that would PREVENT us from having a single Canvas block
    if ( !this.isSingleCanvasSupported() ) {
      return false;
    }

    // Check for any renderer preferences that would CAUSE us to choose not to display with a single Canvas block
    for ( let i = 0; i < Renderer.numActiveRenderers; i++ ) {
      // Grab the next-most preferred renderer
      const renderer = Renderer.bitmaskOrder( preferredRenderers, i );

      // If it's Canvas, congrats! Everything will render in Canvas (since Canvas is supported, as noted above)
      if ( Renderer.bitmaskCanvas & renderer ) {
        return true;
      }

      // Since it's not Canvas, if there's a single painted node that supports this renderer (which is preferred over Canvas),
      // then it will be rendered with this renderer, NOT Canvas.
      if ( this.isSubtreeContainingCompatible( renderer ) ) {
        return false;
      }
    }

    return false; // sanity check
  }

  /**
   * For debugging purposes
   */
  public audit(): void {
    if ( assert ) {
      for ( let i = 0; i < numSummaryBits; i++ ) {
        const bit = summaryBits[ i ];
        const countIsZero = this._counts[ i ] === 0;
        const bitmaskContainsBit = !!( this.bitmask & bit );
        assert( countIsZero === bitmaskContainsBit, 'Bits should be set if count is zero' );
      }
    }
  }

  /**
   * Returns a string form of this object
   */
  public toString(): string {
    let result = RendererSummary.bitmaskToString( this.bitmask );
    for ( let i = 0; i < numSummaryBits; i++ ) {
      const bit = summaryBits[ i ];
      const countForBit = this._counts[ i ];
      if ( countForBit !== 0 ) {
        result += ` ${RendererSummary.bitToString( bit )}:${countForBit}`;
      }
    }
    return result;
  }

  /**
   * Determines which of the summary bits can be set for a specific Node (ignoring children/ancestors).
   * For instance, for bitmaskSingleSVG, we only don't include the flag if THIS node prevents its usage
   * (even though child nodes may prevent it in the renderer summary itself).
   */
  public static summaryBitmaskForNodeSelf( node: Node ): number {
    let bitmask = node._rendererBitmask;

    if ( node.isPainted() ) {
      bitmask |= ( ( node._rendererBitmask & Renderer.bitmaskCurrentRendererArea ) ^ Renderer.bitmaskCurrentRendererArea ) << Renderer.bitmaskLacksShift;
    }
    else {
      bitmask |= Renderer.bitmaskCurrentRendererArea << Renderer.bitmaskLacksShift;
    }

    // NOTE: If changing, see Instance.updateRenderingState
    const requiresSplit = node._cssTransform || node._layerSplit;
    const rendererHint = node._renderer;

    // Whether this subtree will be able to support a single SVG element
    // NOTE: If changing, see Instance.updateRenderingState
    if ( !requiresSplit && // Can't have a single SVG element if we are split
         Renderer.isSVG( node._rendererBitmask ) && // If our node doesn't support SVG, can't do it
         ( !rendererHint || Renderer.isSVG( rendererHint ) ) ) { // Can't if a renderer hint is set to something else
      bitmask |= Renderer.bitmaskSingleSVG;
    }

    // Whether this subtree will be able to support a single Canvas element
    // NOTE: If changing, see Instance.updateRenderingState
    if ( !requiresSplit && // Can't have a single SVG element if we are split
         Renderer.isCanvas( node._rendererBitmask ) && // If our node doesn't support Canvas, can't do it
         ( !rendererHint || Renderer.isCanvas( rendererHint ) ) ) { // Can't if a renderer hint is set to something else
      bitmask |= Renderer.bitmaskSingleCanvas;
    }

    if ( !node.isPainted() ) {
      bitmask |= Renderer.bitmaskNotPainted;
    }
    if ( node.areSelfBoundsValid() ) {
      bitmask |= Renderer.bitmaskBoundsValid;
    }
    if ( !node.hasPDOMContent && !node.hasPDOMOrder() ) {
      bitmask |= Renderer.bitmaskNoPDOM;
    }

    return bitmask;
  }

  /**
   * For debugging purposes
   */
  public static bitToString( bit: number ): string {
    if ( bit === Renderer.bitmaskCanvas ) { return 'Canvas'; }
    if ( bit === Renderer.bitmaskSVG ) { return 'SVG'; }
    if ( bit === Renderer.bitmaskDOM ) { return 'DOM'; }
    if ( bit === Renderer.bitmaskWebGL ) { return 'WebGL'; }
    if ( bit === Renderer.bitmaskLacksCanvas ) { return '(-Canvas)'; }
    if ( bit === Renderer.bitmaskLacksSVG ) { return '(-SVG)'; }
    if ( bit === Renderer.bitmaskLacksDOM ) { return '(-DOM)'; }
    if ( bit === Renderer.bitmaskLacksWebGL ) { return '(-WebGL)'; }
    if ( bit === Renderer.bitmaskSingleCanvas ) { return 'SingleCanvas'; }
    if ( bit === Renderer.bitmaskSingleSVG ) { return 'SingleSVG'; }
    if ( bit === Renderer.bitmaskNotPainted ) { return 'NotPainted'; }
    if ( bit === Renderer.bitmaskBoundsValid ) { return 'BoundsValid'; }
    if ( bit === Renderer.bitmaskNoPDOM ) { return 'NotAccessible'; }
    return '?';
  }

  /**
   * For debugging purposes
   */
  public static bitmaskToString( bitmask: number ): string {
    let result = '';
    for ( let i = 0; i < numSummaryBits; i++ ) {
      const bit = summaryBits[ i ];
      if ( bitmask & bit ) {
        result += `${RendererSummary.bitToString( bit )} `;
      }
    }
    return result;
  }

  public static bitmaskAll = bitmaskAll;
}

scenery.register( 'RendererSummary', RendererSummary );