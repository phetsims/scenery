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

    this.setOptions( options );

    // @private {function}
    this.layoutOptionsListener = this.onLayoutOptionsChange.bind( this );

    this.node.layoutOptionsChangedEmitter.addListener( this.layoutOptionsListener );
  }

  /**
   * @private
   */
  onLayoutOptionsChange() {
    this.setOptions( this.node.layoutOptions || undefined );
  }

  /**
   * @private
   *
   * @param {Object} [options]
   */
  setOptions( options ) {
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

  /**
   * @public
   *
   * @param {Orientation} orientation
   * @param {FlowConfigurable} defaultConfig
   * @returns {number}
   */
  getMinimumSize( orientation, defaultConfig ) {
    const isSizable = !!( orientation === Orientation.HORIZONTAL ? this.node.widthSizable : this.node.heightSizable );

    if ( orientation === Orientation.HORIZONTAL ) {
      return this.withDefault( 'leftMargin', defaultConfig ) +
             Math.max(
               isSizable ? this.node.minimumWidth : this.node.width,
               this.withDefault( 'minContentWidth', defaultConfig ) || 0
             ) +
             this.withDefault( 'rightMargin', defaultConfig );
    }
    else {
      return this.withDefault( 'topMargin', defaultConfig ) +
             Math.max(
               isSizable ? this.node.minimumHeight : this.node.height,
               this.withDefault( 'minContentHeight', defaultConfig ) || 0
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
    const isSizable = !!( orientation === Orientation.HORIZONTAL ? this.node.widthSizable : this.node.heightSizable );

    if ( orientation === Orientation.HORIZONTAL ) {
      return this.withDefault( 'leftMargin', defaultConfig ) +
             Math.min(
               isSizable ? Number.POSITIVE_INFINITY : this.node.width,
               this.withDefault( 'maxContentWidth', defaultConfig ) || Number.POSITIVE_INFINITY
             ) +
             this.withDefault( 'rightMargin', defaultConfig );
    }
    else {
      return this.withDefault( 'topMargin', defaultConfig ) +
             Math.min(
               isSizable ? Number.POSITIVE_INFINITY : this.node.height,
               this.withDefault( 'maxContentHeight', defaultConfig ) || Number.POSITIVE_INFINITY
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
  attemptPreferredSize( orientation, defaultConfig, value ) {
    if ( orientation === Orientation.HORIZONTAL ? this.node.widthSizable : this.node.heightSizable ) {
      const minimumSize = this.getMinimumSize( orientation, defaultConfig );
      const maximumSize = this.getMaximumSize( orientation, defaultConfig );

      assert && assert( isFinite( minimumSize ) );
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
    // TODO: coordinate transform handling, to our ancestorNode!!!!!
    if ( orientation === Orientation.HORIZONTAL ) {
      const left = this.withDefault( 'leftMargin', defaultConfig ) + value;

      if ( Math.abs( this.node.left - left ) > 1e-9 ) {
        this.node.left = left;
      }
    }
    else {
      const top = this.withDefault( 'topMargin', defaultConfig ) + value;

      if ( Math.abs( this.node.top - top ) > 1e-9 ) {
        this.node.top = top;
      }
    }
  }

  /**
   * @public
   *
   * @param {Orientation} orientation
   * @param {FlowConfigurable} defaultConfig
   * @param {number} value
   */
  positionOrigin( orientation, defaultConfig, value ) {
    if ( orientation === Orientation.HORIZONTAL ) {
      if ( Math.abs( this.node.x - value ) > 1e-9 ) {
        this.node.x = value;
      }
    }
    else {
      if ( Math.abs( this.node.y - value ) > 1e-9 ) {
        this.node.y = value;
      }
    }
  }

  /**
   * @public
   *
   * @param {FlowConfigurable} defaultConfig
   * @returns {Bounds2}
   */
  getCellBounds( defaultConfig ) {
    const leftMargin = this.withDefault( 'leftMargin', defaultConfig );
    const rightMargin = this.withDefault( 'rightMargin', defaultConfig );
    const topMargin = this.withDefault( 'topMargin', defaultConfig );
    const bottomMargin = this.withDefault( 'bottomMargin', defaultConfig );

    return this.node.bounds.withOffsets( leftMargin, topMargin, rightMargin, bottomMargin );
  }

  /**
   * Releases references
   * @public
   */
  dispose() {
    this.node.layoutOptionsChangedEmitter.removeListener( this.layoutOptionsListener );
  }
}

scenery.register( 'FlowCell', FlowCell );
export default FlowCell;