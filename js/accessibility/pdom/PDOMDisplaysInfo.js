// Copyright 2018-2020, University of Colorado Boulder

/**
 * Per-node information required to track what accessible Displays our node is visible under. Acts like a multimap
 * (duplicates allowed) to indicate how many times we appear in an accessible display.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Renderer from '../../display/Renderer.js';
import scenery from '../../scenery.js';

class PDOMDisplaysInfo {
  /**
   * Tracks accessible display information for our given node.
   * @public (scenery-internal)
   *
   * @param {Node} node
   */
  constructor( node ) {

    // @public {Node}
    this.node = node;

    // @public (scenery-internal) {Array.<Display>} - (duplicates allowed) - There is one copy of each accessible
    // Display for each trail (from its root node to this node) that is fully visible (assuming this subtree is
    // accessible).
    // Thus, the value of this is:
    // - If this node is invisible OR the subtree has no pdomContent/pdomOrder: []
    // - Otherwise, it is the concatenation of our parents' pdomDisplays (AND any accessible displays rooted
    //   at this node).
    // This value is synchronously updated, and supports pdomInstances by letting them know when certain
    // nodes are visible on the display.
    this.pdomDisplays = [];
  }

  /**
   * Called when the node is added as a child to this node AND the node's subtree contains accessible content.
   * @public (scenery-internal)
   *
   * @param {Node} node
   */
  onAddChild( node ) {
    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.PDOMDisplaysInfo( 'onAddChild n#' + node.id + ' (parent:n#' + this.node.id + ')' );
    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.push();

    if ( node._pdomDisplaysInfo.canHavePDOMDisplays() ) {
      node._pdomDisplaysInfo.addAccessibleDisplays( this.pdomDisplays );
    }

    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.pop();
  }

  /**
   * Called when the node is removed as a child from this node AND the node's subtree contains accessible content.
   * @public (scenery-internal)
   *
   * @param {Node} node
   */
  onRemoveChild( node ) {
    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.PDOMDisplaysInfo( 'onRemoveChild n#' + node.id + ' (parent:n#' + this.node.id + ')' );
    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.push();

    if ( node._pdomDisplaysInfo.canHavePDOMDisplays() ) {
      node._pdomDisplaysInfo.removeAccessibleDisplays( this.pdomDisplays );
    }

    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.pop();
  }

  /**
   * Called when our summary bitmask changes
   * @public (scenery-internal)
   *
   * @param {number} oldBitmask
   * @param {number} newBitmask
   */
  onSummaryChange( oldBitmask, newBitmask ) {
    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.PDOMDisplaysInfo( 'onSummaryChange n#' + this.node.id + ' wasA11y:' + !( Renderer.bitmaskNoPDOM & oldBitmask ) + ', isA11y:' + !( Renderer.bitmaskNoPDOM & newBitmask ) );
    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.push();

    // If we are invisible, our pdomDisplays would not have changed ([] => [])
    if ( this.node.visible && this.node.pdomVisible ) {
      const wasAccessible = !( Renderer.bitmaskNoPDOM & oldBitmask );
      const isAccessible = !( Renderer.bitmaskNoPDOM & newBitmask );

      // If we changed to be accessible, we need to recursively add accessible displays.
      if ( isAccessible && !wasAccessible ) {
        this.addAllAccessibleDisplays();
      }

      // If we changed to NOT be accessible, we need to recursively remove accessible displays.
      if ( !isAccessible && wasAccessible ) {
        this.removeAllAccessibleDisplays();
      }
    }

    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.pop();
  }

  /**
   * Called when our visibility changes.
   * @public (scenery-internal)
   *
   * @param {boolean} visible
   */
  onVisibilityChange( visible ) {
    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.PDOMDisplaysInfo( 'onVisibilityChange n#' + this.node.id + ' visible:' + visible );
    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.push();

    // If we are not accessible (or pdomVisible), our pdomDisplays would not have changed ([] => [])
    if ( this.node.pdomVisible && !this.node._rendererSummary.hasNoPDOM() ) {
      if ( visible ) {
        this.addAllAccessibleDisplays();
      }
      else {
        this.removeAllAccessibleDisplays();
      }
    }

    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.pop();
  }

  /**
   * Called when our accessibleVisibility changes.
   * @public (scenery-internal)
   *
   * @param {boolean} visible
   */
  onAccessibleVisibilityChange( visible ) {
    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.PDOMDisplaysInfo( 'onAccessibleVisibilityChange n#' + this.node.id + ' pdomVisible:' + visible );
    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.push();

    // If we are not accessible, our pdomDisplays would not have changed ([] => [])
    if ( this.node.visible && !this.node._rendererSummary.hasNoPDOM() ) {
      if ( visible ) {
        this.addAllAccessibleDisplays();
      }
      else {
        this.removeAllAccessibleDisplays();
      }
    }

    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.pop();
  }

  /**
   * Called when we have a rooted display added to this node.
   * @public (scenery-internal)
   *
   * @param {Display} display
   */
  onAddedRootedDisplay( display ) {
    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.PDOMDisplaysInfo( 'onAddedRootedDisplay n#' + this.node.id );
    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.push();

    if ( display._accessible && this.canHavePDOMDisplays() ) {
      this.addAccessibleDisplays( [ display ] );
    }

    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.pop();
  }

  /**
   * Called when we have a rooted display removed from this node.
   * @public (scenery-internal)
   *
   * @param {Display} display
   */
  onRemovedRootedDisplay( display ) {
    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.PDOMDisplaysInfo( 'onRemovedRootedDisplay n#' + this.node.id );
    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.push();

    if ( display._accessible && this.canHavePDOMDisplays() ) {
      this.removeAccessibleDisplays( [ display ] );
    }

    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.pop();
  }

  /**
   * Returns whether we can have pdomDisplays specified in our array.
   * @public (scenery-internal)
   *
   * @returns {boolean}
   */
  canHavePDOMDisplays() {
    return this.node.visible && this.node.pdomVisible && !this.node._rendererSummary.hasNoPDOM();
  }

  /**
   * Adds all of our accessible displays to our array (and propagates).
   * @private
   */
  addAllAccessibleDisplays() {
    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.PDOMDisplaysInfo( 'addAllAccessibleDisplays n#' + this.node.id );
    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.push();

    assert && assert( this.pdomDisplays.length === 0, 'Should be empty before adding everything' );
    assert && assert( this.canHavePDOMDisplays(), 'Should happen when we can store pdomDisplays' );

    let i;
    const displays = [];

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

    this.addAccessibleDisplays( displays );

    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.pop();
  }

  /**
   * Removes all of our accessible displays from our array (and propagates).
   * @private
   */
  removeAllAccessibleDisplays() {
    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.PDOMDisplaysInfo( 'removeAllAccessibleDisplays n#' + this.node.id );
    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.push();

    assert && assert( !this.canHavePDOMDisplays(), 'Should happen when we cannot store pdomDisplays' );

    // TODO: is there a way to avoid a copy?
    this.removeAccessibleDisplays( this.pdomDisplays.slice() );

    assert && assert( this.pdomDisplays.length === 0, 'Should be empty after removing everything' );

    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.pop();
  }

  /**
   * Adds a list of accessible displays to our internal list. See pdomDisplays documentation.
   * @private
   *
   * @param {Array.<Display>} displays
   */
  addAccessibleDisplays( displays ) {
    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.PDOMDisplaysInfo( 'addAccessibleDisplays n#' + this.node.id + ' numDisplays:' + displays.length );
    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.push();

    assert && assert( Array.isArray( displays ) );

    // Simplifies things if we can stop no-ops here.
    if ( displays.length !== 0 ) {
      Array.prototype.push.apply( this.pdomDisplays, displays );

      // Propagate the change to our children
      for ( let i = 0; i < this.node._children.length; i++ ) {
        const child = this.node._children[ i ];
        if ( child._pdomDisplaysInfo.canHavePDOMDisplays() ) {
          this.node._children[ i ]._pdomDisplaysInfo.addAccessibleDisplays( displays );
        }
      }

      this.node.pdomDisplaysEmitter.emit();
    }

    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.pop();
  }

  /**
   * Removes a list of accessible displays from our internal list. See pdomDisplays documentation.
   * @private
   *
   * @param {Array.<Display>} displays
   */
  removeAccessibleDisplays( displays ) {
    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.PDOMDisplaysInfo( 'removeAccessibleDisplays n#' + this.node.id + ' numDisplays:' + displays.length );
    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.push();

    assert && assert( Array.isArray( displays ) );
    assert && assert( this.pdomDisplays.length >= displays.length );

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
          child._pdomDisplaysInfo.removeAccessibleDisplays( displays );
        }
      }

      this.node.pdomDisplaysEmitter.emit();
    }

    sceneryLog && sceneryLog.PDOMDisplaysInfo && sceneryLog.pop();
  }
}

scenery.register( 'PDOMDisplaysInfo', PDOMDisplaysInfo );
export default PDOMDisplaysInfo;