// Copyright 2021, University of Colorado Boulder

/**
 * TODO: doc
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Orientation from '../../../phet-core/js/Orientation.js';
import scenery from '../scenery.js';
import FlowConfigurable from './FlowConfigurable.js';

class FlowCell extends FlowConfigurable( Object ) {
  /**
   * @param {Node} node
   * @param {Object} [options]
   */
  constructor( node, options ) {
    super();

    // @private {Node}
    this._node = node;

    this.setConfigToInherit();
    this.mutateConfigurable( options );
  }

  /**
   * @public
   *
   * @returns {Node}
   */
  get node() {
    return this._node;
  }

  // TODO: Can swap out the Node?
  // TODO: Can pool these?
  // set node( value ) {
  //   if ( this._node !== value ) {
  //     this._node = value;
  //   }
  // }

  /**
   * @public
   *
   * @param {Orientation} orientation
   * @returns {number}
   */
  getMinimumSize( orientation ) {
    // TODO: better handling for preferred size
    if ( orientation === Orientation.HORIZONTAL ) {
      return this.leftMargin + this.node.width + this.rightMargin;
    }
    else {
      return this.topMargin + this.node.height + this.bottomMargin;
    }
  }
}

scenery.register( 'FlowCell', FlowCell );

export default FlowCell;