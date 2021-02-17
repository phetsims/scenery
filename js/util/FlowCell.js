// Copyright 2021, University of Colorado Boulder

/**
 * TODO: doc
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Utils from '../../../dot/js/Utils.js';
import Orientation from '../../../phet-core/js/Orientation.js';
import scenery from '../scenery.js';
import FlowConfigurable from './FlowConfigurable.js';
import Sizable from './Sizable.js';

class FlowCell extends FlowConfigurable( Object ) {
  /**
   * @param {Node} node
   * @param {Object} [options]
   */
  constructor( node, options ) {
    super();

    // @private {Node}
    this._node = node;

    // @public (scenery-internal) {number}
    this._pendingSize = 0;

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
   * @param {FlowConfigurable} defaultConfig
   * @returns {number}
   */
  getMinimumSize( orientation, defaultConfig ) {
    const isSizable = !!this.node.sizable;

    if ( orientation === Orientation.HORIZONTAL ) {
      return this.withDefault( 'leftMargin', defaultConfig ) +
             Math.max(
               isSizable ? this.node.minimumWidth : this.node.width,
               this.withDefault( 'minCellWidth', defaultConfig ) || 0
             ) +
             this.withDefault( 'rightMargin', defaultConfig );
    }
    else {
      return this.withDefault( 'topMargin', defaultConfig ) +
             Math.max(
               isSizable ? this.node.minimumHeight : this.node.height,
               this.withDefault( 'minCellHeight', defaultConfig ) || 0
             ) +
             this.withDefault( 'bottomMargin', defaultConfig );
    }
  }

  /**
   * @public
   *
   * @param {Orientation} orientation
   * @param {FlowConfigurable} defaultConfig
   * @returns {number}
   */
  getMaximumSize( orientation, defaultConfig ) {
    const isSizable = this.node instanceof Sizable;

    if ( orientation === Orientation.HORIZONTAL ) {
      return this.withDefault( 'leftMargin', defaultConfig ) +
             Math.min(
               isSizable ? this.node.maximumWidth : this.node.width,
               this.withDefault( 'maxCellWidth', defaultConfig ) || 0
             ) +
             this.withDefault( 'rightMargin', defaultConfig );
    }
    else {
      return this.withDefault( 'topMargin', defaultConfig ) +
             Math.min(
               isSizable ? this.node.maximumHeight : this.node.height,
               this.withDefault( 'maxCellHeight', defaultConfig ) || 0
             ) +
             this.withDefault( 'bottomMargin', defaultConfig );
    }
  }

  /**
   * @public
   *
   * @param {Orientation} orientation
   * @param {FlowConfigurable} defaultConfig
   * @param {number} value
   */
  attemptedPreferredSize( orientation, defaultConfig, value ) {
    if ( this.node.sizable ) {
      const minimumSize = this.getMinimumSize( orientation, defaultConfig );
      const maximumSize = this.getMaximumSize( orientation, defaultConfig );

      assert && assert( isFinite( minimumSize ) );
      assert && assert( isFinite( maximumSize ) );
      assert && assert( maximumSize >= minimumSize );

      value = Utils.clamp( value, minimumSize, maximumSize );

      if ( orientation === Orientation.HORIZONTAL ) {
        this.node.preferredWidth = value - this.withDefault( 'leftMargin', defaultConfig ) - this.withDefault( 'rightMargin', defaultConfig );
      }
      else {
        this.node.preferredHeight = value - this.withDefault( 'topMargin', defaultConfig ) - this.withDefault( 'bottomMargin', defaultConfig );
      }
      // TODO: warnings if those preferred sizes weren't reached?
    }
  }

  /**
   * @public
   *
   * @param {Orientation} orientation
   * @param {FlowConfigurable} defaultConfig
   * @param {number} value
   */
  positionStart( orientation, defaultConfig, value ) {
    if ( orientation === Orientation.HORIZONTAL ) {
      this.node.left = this.withDefault( 'leftMargin', defaultConfig ) + value;
    }
    else {
      this.node.top = this.withDefault( 'topMargin', defaultConfig ) + value;
    }
  }
}

scenery.register( 'FlowCell', FlowCell );

export default FlowCell;