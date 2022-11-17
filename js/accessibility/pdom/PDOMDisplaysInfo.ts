// Copyright 2018-2022, University of Colorado Boulder

/**
 * Per-node information required to track what PDOM Displays our Node is visible under. A PDOM display is a Display that
 * is marked true with the `accessibility` option, and thus creates and manages a ParallelDOM (see ParallelDOM and
 * general scenery accessibility doc for more details). Acts like a multimap
 * (duplicates allowed) to indicate how many times we appear in an pdom display.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { Display, Node, Renderer, scenery } from '../../imports.js';

export default class PDOMDisplaysInfo {

  private readonly node: Node;

  // (duplicates allowed) - There is one copy of each pdom
  // Display for each trail (from its root node to this node) that is fully visible (assuming this subtree has
  // pdom content).
  // Thus, the value of this is:
  // - If this node is invisible OR the subtree has no pdomContent/pdomOrder: []
  // - Otherwise, it is the concatenation of our parents' pdomDisplays (AND any pdom displays rooted
  //   at this node).
  // This value is synchronously updated, and supports pdomInstances by letting them know when certain
  // nodes are visible on the display.
  public readonly pdomDisplays: Display[];

  /**
   * Tracks pdom display information for our given node.
   * (scenery-internal)
   */
  public constructor( node: Node ) {
    this.node = node;
    this.pdomDisplays = [];
  }

  /**
   * Called when the node is added as a child to this node AND the node's subtree contains pdom content. (scenery-internal)
   */
  public onAddChild( node: Node ): void {
    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.PDOMDisplaysInfo( `onAddChild n#${node.id} (parent:n#${this.node.id})` );
    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.push();

    if ( node._pdomDisplaysInfo.canHavePDOMDisplays() ) {
      node._pdomDisplaysInfo.addPDOMDisplays( this.pdomDisplays );
    }

    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.pop();
  }

  /**
   * Called when the node is removed as a child from this node AND the node's subtree contains pdom content. (scenery-internal)
   */
  public onRemoveChild( node: Node ): void {
    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.PDOMDisplaysInfo( `onRemoveChild n#${node.id} (parent:n#${this.node.id})` );
    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.push();

    if ( node._pdomDisplaysInfo.canHavePDOMDisplays() ) {
      node._pdomDisplaysInfo.removePDOMDisplays( this.pdomDisplays );
    }

    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.pop();
  }

  /**
   * Called when our summary bitmask changes (scenery-internal)
   */
  public onSummaryChange( oldBitmask: number, newBitmask: number ): void {
    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.PDOMDisplaysInfo( `onSummaryChange n#${this.node.id} wasPDOM:${!( Renderer.bitmaskNoPDOM & oldBitmask )}, isPDOM:${!( Renderer.bitmaskNoPDOM & newBitmask )}` );
    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.push();

    // If we are invisible, our pdomDisplays would not have changed ([] => [])
    if ( this.node.visible && this.node.pdomVisible ) {
      const hadPDOM = !( Renderer.bitmaskNoPDOM & oldBitmask );
      const hasPDOM = !( Renderer.bitmaskNoPDOM & newBitmask );

      // If we changed to have pdom content, we need to recursively add pdom displays.
      if ( hasPDOM && !hadPDOM ) {
        this.addAllPDOMDisplays();
      }

      // If we changed to NOT have pdom content, we need to recursively remove pdom displays.
      if ( !hasPDOM && hadPDOM ) {
        this.removeAllPDOMDisplays();
      }
    }

    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.pop();
  }

  /**
   * Called when our visibility changes. (scenery-internal)
   */
  public onVisibilityChange( visible: boolean ): void {
    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.PDOMDisplaysInfo( `onVisibilityChange n#${this.node.id} visible:${visible}` );
    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.push();

    // If we don't have pdom (or pdomVisible), our pdomDisplays would not have changed ([] => [])
    if ( this.node.pdomVisible && !this.node._rendererSummary.hasNoPDOM() ) {
      if ( visible ) {
        this.addAllPDOMDisplays();
      }
      else {
        this.removeAllPDOMDisplays();
      }
    }

    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.pop();
  }

  /**
   * Called when our pdomVisibility changes. (scenery-internal)
   */
  public onPDOMVisibilityChange( visible: boolean ): void {
    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.PDOMDisplaysInfo( `onPDOMVisibilityChange n#${this.node.id} pdomVisible:${visible}` );
    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.push();

    // If we don't have pdom, our pdomDisplays would not have changed ([] => [])
    if ( this.node.visible && !this.node._rendererSummary.hasNoPDOM() ) {
      if ( visible ) {
        this.addAllPDOMDisplays();
      }
      else {
        this.removeAllPDOMDisplays();
      }
    }

    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.pop();
  }

  /**
   * Called when we have a rooted display added to this node. (scenery-internal)
   */
  public onAddedRootedDisplay( display: Display ): void {
    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.PDOMDisplaysInfo( `onAddedRootedDisplay n#${this.node.id}` );
    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.push();

    if ( display._accessible && this.canHavePDOMDisplays() ) {
      this.addPDOMDisplays( [ display ] );
    }

    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.pop();
  }

  /**
   * Called when we have a rooted display removed from this node. (scenery-internal)
   */
  public onRemovedRootedDisplay( display: Display ): void {
    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.PDOMDisplaysInfo( `onRemovedRootedDisplay n#${this.node.id}` );
    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.push();

    if ( display._accessible && this.canHavePDOMDisplays() ) {
      this.removePDOMDisplays( [ display ] );
    }

    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.pop();
  }

  /**
   * Returns whether we can have pdomDisplays specified in our array. (scenery-internal)
   */
  public canHavePDOMDisplays(): boolean {
    return this.node.visible && this.node.pdomVisible && !this.node._rendererSummary.hasNoPDOM();
  }

  /**
   * Adds all of our pdom displays to our array (and propagates).
   */
  private addAllPDOMDisplays(): void {
    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.PDOMDisplaysInfo( `addAllPDOMDisplays n#${this.node.id}` );
    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.push();

    assert && assert( this.pdomDisplays.length === 0, 'Should be empty before adding everything' );
    assert && assert( this.canHavePDOMDisplays(), 'Should happen when we can store pdomDisplays' );

    let i;
    const displays: Display[] = [];

    // Concatenation of our parents' pdomDisplays
    for ( i = 0; i < this.node._parents.length; i++ ) {
      Array.prototype.push.apply( displays, this.node._parents[ i ]._pdomDisplaysInfo.pdomDisplays );
    }

    // AND any acessible displays rooted at this node
    for ( i = 0; i < this.node._rootedDisplays.length; i++ ) {
      const display = this.node._rootedDisplays[ i ];
      if ( display._accessible ) {
        displays.push( display );
      }
    }

    this.addPDOMDisplays( displays );

    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.pop();
  }

  /**
   * Removes all of our pdom displays from our array (and propagates).
   */
  private removeAllPDOMDisplays(): void {
    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.PDOMDisplaysInfo( `removeAllPDOMDisplays n#${this.node.id}` );
    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.push();

    assert && assert( !this.canHavePDOMDisplays(), 'Should happen when we cannot store pdomDisplays' );

    // TODO: is there a way to avoid a copy?
    this.removePDOMDisplays( this.pdomDisplays.slice() );

    assert && assert( this.pdomDisplays.length === 0, 'Should be empty after removing everything' );

    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.pop();
  }

  /**
   * Adds a list of pdom displays to our internal list. See pdomDisplays documentation.
   */
  private addPDOMDisplays( displays: Display[] ): void {
    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.PDOMDisplaysInfo( `addPDOMDisplays n#${this.node.id} numDisplays:${displays.length}` );
    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.push();

    assert && assert( Array.isArray( displays ) );

    // Simplifies things if we can stop no-ops here.
    if ( displays.length !== 0 ) {
      Array.prototype.push.apply( this.pdomDisplays, displays );

      // Propagate the change to our children
      for ( let i = 0; i < this.node._children.length; i++ ) {
        const child = this.node._children[ i ];
        if ( child._pdomDisplaysInfo.canHavePDOMDisplays() ) {
          this.node._children[ i ]._pdomDisplaysInfo.addPDOMDisplays( displays );
        }
      }

      this.node.pdomDisplaysEmitter.emit();
    }

    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.pop();
  }

  /**
   * Removes a list of pdom displays from our internal list. See pdomDisplays documentation.
   */
  private removePDOMDisplays( displays: Display[] ): void {
    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.PDOMDisplaysInfo( `removePDOMDisplays n#${this.node.id} numDisplays:${displays.length}` );
    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.push();

    assert && assert( Array.isArray( displays ) );
    assert && assert( this.pdomDisplays.length >= displays.length, 'there should be at least as many PDOMDisplays as Displays' );

    // Simplifies things if we can stop no-ops here.
    if ( displays.length !== 0 ) {
      let i;

      for ( i = displays.length - 1; i >= 0; i-- ) {
        const index = this.pdomDisplays.lastIndexOf( displays[ i ] );
        assert && assert( index >= 0 );
        this.pdomDisplays.splice( i, 1 );
      }

      // Propagate the change to our children
      for ( i = 0; i < this.node._children.length; i++ ) {
        const child = this.node._children[ i ];
        // NOTE: Since this gets called many times from the RendererSummary (which happens before the actual child
        // modification happens), we DO NOT want to traverse to the child node getting removed. Ideally a better
        // solution than this flag should be found.
        if ( child._pdomDisplaysInfo.canHavePDOMDisplays() && !child._isGettingRemovedFromParent ) {
          child._pdomDisplaysInfo.removePDOMDisplays( displays );
        }
      }

      this.node.pdomDisplaysEmitter.emit();
    }

    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.pop();
  }
}

scenery.register( 'PDOMDisplaysInfo', PDOMDisplaysInfo );
