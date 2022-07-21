// Copyright 2018-2022, University of Colorado Boulder

/**
 * Runs PDOM-tree-related scenery operations randomly (with assertions) to try to find any bugs.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Permutation from '../../../../dot/js/Permutation.js';
import Random from '../../../../dot/js/Random.js';
import arrayDifference from '../../../../phet-core/js/arrayDifference.js';
import { Display, Node, PDOMTree, scenery } from '../../imports.js';

class PDOMFuzzer {
  /**
   * @param {number} nodeCount
   * @param {boolean} logToConsole
   * @param {number} [seed]
   */
  constructor( nodeCount, logToConsole, seed ) {
    assert && assert( nodeCount >= 2 );

    seed = seed || null;

    // @private {number}
    this.nodeCount = nodeCount;

    // @private {boolean}
    this.logToConsole = logToConsole;

    // @private {Array.<Node>}
    this.nodes = _.range( 0, nodeCount ).map( () => new Node() );

    // @private {Display}
    this.display = new Display( this.nodes[ 0 ] );

    // @private {Random}
    this.random = new Random( { seed: seed } );

    // @private {Array.<Action>}
    this.actionsTaken = [];
  }

  /**
   * Runs one action randomly (printing out the action and result).
   * @public
   */
  step() {
    const action = this.random.sample( this.enumerateActions() );
    this.logToConsole && console.log( action.text );
    this.actionsTaken.push( action );
    action.execute();
    this.display._rootPDOMInstance.auditRoot();
    PDOMTree.auditPDOMDisplays( this.display.rootNode );
    if ( this.logToConsole ) {
      for ( let i = 0; i < this.nodes.length; i++ ) {
        const node = this.nodes[ i ];
        console.log( `${i}#${node.id} ${node.tagName} ch:${PDOMTree.debugOrder( node.children )} or:${PDOMTree.debugOrder( node.pdomOrder )} vis:${node.visible} avis:${node.pdomVisible}` );
      }
    }
  }

  /**
   * Find all of the possible actions that are legal.
   * @private
   *
   * @returns {Array.<Object>} - like { text: {string}, execute: {function} }
   */
  enumerateActions() {
    const actions = [];

    this.nodes.forEach( a => {
      actions.push( {
        text: `#${a.id}.visible = ${!a.visible}`,
        execute: () => {
          a.visible = !a.visible;
        }
      } );
      actions.push( {
        text: `#${a.id}.pdomVisible = ${!a.pdomVisible}`,
        execute: () => {
          a.pdomVisible = !a.pdomVisible;
        }
      } );
      [ 'span', 'div', null ].forEach( tagName => {
        if ( a.tagName !== tagName ) {
          actions.push( {
            text: `#${a.id}.tagName = ${tagName}`,
            execute: () => {
              a.tagName = tagName;
            }
          } );
        }
      } );

      this.powerSet( arrayDifference( this.nodes, [ a ] ).concat( [ null ] ) ).forEach( subset => {
        Permutation.forEachPermutation( subset, order => {
          // TODO: Make sure it's not the CURRENT order?
          if ( this.isPDOMOrderChangeLegal( a, order ) ) {
            actions.push( {
              text: `#${a.id}.pdomOrder = ${PDOMTree.debugOrder( order )}`,
              execute: () => {
                a.pdomOrder = order;
              }
            } );
          }
        } );
      } );

      this.nodes.forEach( b => {
        if ( this.isAddChildLegal( a, b ) ) {
          _.range( 0, a.children.length + 1 ).forEach( i => {
            actions.push( {
              text: `#${a.id}.insertChild(${i},#${b.id})`,
              execute: () => {
                a.insertChild( i, b );
              }
            } );
          } );
        }
        if ( a.hasChild( b ) ) {
          actions.push( {
            text: `#${a.id}.removeChild(#${b.id})`,
            execute: () => {
              a.removeChild( b );
            }
          } );
        }
      } );
    } );

    return actions;
  }

  /**
   * Checks whether the child can be added (as a child) to the parent.
   * @private
   *
   * @param {Node} parent
   * @param {Node} child
   * @returns {boolean}
   */
  isAddChildLegal( parent, child ) {
    return !parent.hasChild( child ) && this.isAcyclic( parent, child );
  }

  /**
   * Returns the power set of a set (all subsets).
   * @private
   *
   * @param {Array.<*>} list
   * @returns {Array.<Array.<*>>}
   */
  powerSet( list ) {
    if ( list.length === 0 ) {
      return [ [] ];
    }
    else {
      const lists = this.powerSet( list.slice( 1 ) );
      return lists.concat( lists.map( subList => [ list[ 0 ] ].concat( subList ) ) );
    }
  }

  /**
   * Returns whether an accessible order change is legal.
   * @private
   *
   * @param {Node} node
   * @param {Array.<Node|null>|null} order
   */
  isPDOMOrderChangeLegal( node, order ) {
    // remap for equivalence, so it's an array of nodes
    if ( order === null ) { order = []; }
    order = order.filter( n => n !== null );

    if ( _.includes( order, node ) ||
         _.uniq( order ).length < order.length ) {
      return false;
    }

    // Can't include nodes that are included in other accessible orders
    for ( let i = 0; i < order.length; i++ ) {
      if ( order[ i ]._pdomParent && order[ i ]._pdomParent !== node ) {
        return false;
      }
    }

    const hasConnection = ( a, b ) => {
      if ( a === node ) {
        return a.hasChild( b ) || _.includes( order, b );
      }
      else {
        return a.hasChild( b ) || ( !!a.pdomOrder && _.includes( a.pdomOrder, b ) );
      }
    };

    const effectiveChildren = node.children.concat( order );
    return _.every( effectiveChildren, child => this.isAcyclic( node, child, hasConnection ) );
  }

  /**
   * Checks whether a connection (parent-child or accessible order) is legal (doesn't cause a cycle).
   * @private
   *
   * @param {Node} parent
   * @param {Node} child
   * @param {function} hasConnection - determines whether there is a parent-child-style relationship between params
   * @returns {boolean}
   */
  isAcyclic( parent, child, hasConnection ) {
    if ( parent === child ) {
      return false;
    }

    const nodes = child.children.concat( child.pdomOrder ).filter( n => n !== null ); // super defensive

    while ( nodes.length ) {
      const node = nodes.pop();
      if ( node === parent ) {
        return false;
      }

      if ( hasConnection ) {
        this.nodes.forEach( potentialChild => {
          if ( hasConnection( node, potentialChild ) ) {
            nodes.push( potentialChild );
          }
        } );
      }
      else {
        // Add in children and accessible children (don't worry about duplicates since perf isn't critical)
        Array.prototype.push.apply( nodes, node.children );
        if ( node.pdomOrder ) {
          Array.prototype.push.apply( nodes, node.pdomOrder.filter( n => n !== null ) );
        }
      }
    }

    return true;
  }

  /**
   * Releases references
   * @public
   */
  dispose() {
    this.display.dispose();
  }
}

scenery.register( 'PDOMFuzzer', PDOMFuzzer );
export default PDOMFuzzer;