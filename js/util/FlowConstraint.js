// Copyright 2021, University of Colorado Boulder

/**
 * TODO: doc
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Orientation from '../../../phet-core/js/Orientation.js';
import arrayRemove from '../../../phet-core/js/arrayRemove.js';
import mutate from '../../../phet-core/js/mutate.js';
import Node from '../nodes/Node.js';
import scenery from '../scenery.js';
import Constraint from './Constraint.js';
import FlowCell from './FlowCell.js';
import FlowConfigurable from './FlowConfigurable.js';

const FLOW_CONSTRAINT_KEYS = [
  'orientation'
];

class FlowConstraint extends FlowConfigurable( Constraint ) {
  /**
   * @param {Node} rootNode
   * @param {Object} [options]
   */
  constructor( rootNode, options ) {
    assert && assert( rootNode instanceof Node );

    super( rootNode );

    // @private {Array.<FlowCell>}
    this.cells = [];

    // @private {Orientation}
    this._orientation = Orientation.HORIZONTAL;

    // @private {boolean}
    this._excludeInvisibleChildrenFromBounds = true;

    this.setConfigToBaseDefault();
    this.mutateConfigurable( options );
    mutate( this, FLOW_CONSTRAINT_KEYS, options );
  }

  /**
   * @protected
   * @override
   */
  layout() {
    super.layout();

    // const oppositeOrientation = this.orientation.opposite;

    const cells = this.cells.filter( cell => {
      return cell.node.bounds.isValid() && ( !this._excludeInvisibleChildrenFromBounds || cell.node.visible );
    } );

    if ( !cells.length ) {
      return;
    }

    // TODO: wrapping
    const lines = [ cells ];

    lines.forEach( () => {

    } );
  }

  /**
   * @public
   *
   * @returns {Orientation}
   */
  get orientation() {
    return this._orientation;
  }

  /**
   * @public
   *
   * @param {Orientation|string} value
   */
  set orientation( value ) {
    if ( value === 'horizontal' ) {
      value = Orientation.HORIZONTAL;
    }
    if ( value === 'vertical' ) {
      value = Orientation.VERTICAL;
    }

    assert && assert( Orientation.includes( value ) );

    if ( this._orientation !== value ) {
      this._orientation = value;

      this.updateLayoutAutomatically();
    }
  }

  /**
   * @public
   *
   * @returns {string}
   */
  get excludeInvisibleChildrenFromBounds() {
    return this._excludeInvisibleChildrenFromBounds;
  }

  /**
   * @public
   *
   * @param {Orientation|string} value
   */
  set excludeInvisibleChildrenFromBounds( value ) {
    assert && assert( typeof value === 'boolean' );

    if ( this._excludeInvisibleChildrenFromBounds !== value ) {
      this._excludeInvisibleChildrenFromBounds = value;

      this.updateLayoutAutomatically();
    }
  }

  /**
   * @public
   *
   * @param {number} index
   * @param {FlowCell} cell
   */
  insertCell( index, cell ) {
    assert && assert( typeof index === 'number' );
    assert && assert( index >= 0 );
    assert && assert( index <= this.cells.length );
    assert && assert( cell instanceof FlowCell );
    assert && assert( !_.includes( this.cells, cell ) );

    this.cells.splice( index, 0, cell );

    this.updateLayoutAutomatically();
  }

  /**
   * @public
   *
   * @param {FlowCell} cell
   */
  removeCell( cell ) {
    assert && assert( cell instanceof FlowCell );
    assert && assert( _.includes( this.cells, cell ) );

    arrayRemove( this.cells, cell );

    this.updateLayoutAutomatically();
  }
}

scenery.register( 'FlowConstraint', FlowConstraint );

export default FlowConstraint;