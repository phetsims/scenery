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
   * @param {FlowConstraint} constraint
   * @param {Node} node
   * @param {Object} [options]
   */
  constructor( constraint, node, options ) {
    super();

    // @private {FlowConstraint}
    this._constraint = constraint;

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
   * @public
   *
   * @returns {FlowConfigurable.Align}
   */
  get effectiveAlign() {
    return this._align !== null ? this._align : this._constraint._align;
  }

  /**
   * @public
   *
   * @returns {number}
   */
  get effectiveLeftMargin() {
    return this._leftMargin !== null ? this._leftMargin : this._constraint._leftMargin;
  }

  /**
   * @public
   *
   * @returns {number}
   */
  get effectiveRightMargin() {
    return this._rightMargin !== null ? this._rightMargin : this._constraint._rightMargin;
  }

  /**
   * @public
   *
   * @returns {number}
   */
  get effectiveTopMargin() {
    return this._topMargin !== null ? this._topMargin : this._constraint._topMargin;
  }

  /**
   * @public
   *
   * @returns {number}
   */
  get effectiveBottomMargin() {
    return this._bottomMargin !== null ? this._bottomMargin : this._constraint._bottomMargin;
  }

  /**
   * @public
   *
   * @returns {number}
   */
  get effectiveGrow() {
    return this._grow !== null ? this._grow : this._constraint._grow;
  }

  /**
   * @public
   *
   * @returns {number}
   */
  get effectiveMinContentWidth() {
    return this._minContentWidth !== null ? this._minContentWidth : this._constraint._minContentWidth;
  }

  /**
   * @public
   *
   * @returns {number}
   */
  get effectiveMinContentHeight() {
    return this._minContentHeight !== null ? this._minContentHeight : this._constraint._minContentHeight;
  }

  /**
   * @public
   *
   * @returns {number}
   */
  get effectiveMaxContentWidth() {
    return this._maxContentWidth !== null ? this._maxContentWidth : this._constraint._maxContentWidth;
  }

  /**
   * @public
   *
   * @returns {number}
   */
  get effectiveMaxContentHeight() {
    return this._maxContentHeight !== null ? this._maxContentHeight : this._constraint._maxContentHeight;
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
      return this.effectiveLeftMargin +
             Math.max( isSizable ? this.node.minimumWidth : this.node.width, this.effectiveMinContentWidth || 0 ) +
             this.effectiveRightMargin;
    }
    else {
      return this.effectiveTopMargin +
             Math.max( isSizable ? this.node.minimumHeight : this.node.height, this.effectiveMinContentHeight || 0 ) +
             this.effectiveBottomMargin;
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
      return this.effectiveLeftMargin +
             Math.min( isSizable ? Number.POSITIVE_INFINITY : this.node.width, this.effectiveMaxContentWidth || Number.POSITIVE_INFINITY ) +
             this.effectiveRightMargin;
    }
    else {
      return this.effectiveTopMargin +
             Math.min( isSizable ? Number.POSITIVE_INFINITY : this.node.height, this.effectiveMaxContentHeight || Number.POSITIVE_INFINITY ) +
             this.effectiveBottomMargin;
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
        this.node.preferredWidth = value - this.effectiveLeftMargin - this.effectiveRightMargin;
      }
      else {
        this.node.preferredHeight = value - this.effectiveTopMargin - this.effectiveBottomMargin;
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
      const left = this.effectiveLeftMargin + value;

      if ( Math.abs( this.node.left - left ) > 1e-9 ) {
        this.node.left = left;
      }
    }
    else {
      const top = this.effectiveTopMargin + value;

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
    return this.node.bounds.withOffsets(
      this.effectiveLeftMargin,
      this.effectiveTopMargin,
      this.effectiveRightMargin,
      this.effectiveBottomMargin
     );
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