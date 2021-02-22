// Copyright 2021, University of Colorado Boulder

/**
 * TODO: doc
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Utils from '../../../dot/js/Utils.js';
import merge from '../../../phet-core/js/merge.js';
import scenery from '../scenery.js';
import GridConfigurable from './GridConfigurable.js';

class GridCell extends GridConfigurable( Object ) {
  /**
   * @param {Node} node
   * @param {Object} [options]
   */
  constructor( node, options ) {

    options = merge( {
      x: 0,
      y: 0,
      width: 1,
      height: 1
    }, options );

    assert && assert( typeof options.x === 'number' && Number.isInteger( options.x ) && isFinite( options.x ) && options.x >= 0 );
    assert && assert( typeof options.y === 'number' && Number.isInteger( options.y ) && isFinite( options.y ) && options.y >= 0 );
    assert && assert( typeof options.width === 'number' && Number.isInteger( options.width ) && isFinite( options.width ) && options.width >= 1 );
    assert && assert( typeof options.height === 'number' && Number.isInteger( options.height ) && isFinite( options.height ) && options.height >= 1 );

    super();

    // @public {number} - These are only set initially, and ignored for the future
    this.x = options.x;
    this.y = options.y;
    this.width = options.width;
    this.height = options.height;

    // @private {Node}
    this._node = node;

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
   * @param {GridConfigurable} defaultConfig
   * @returns {number}
   */
  getMinimumWidth( defaultConfig ) {
    return this.withDefault( 'leftMargin', defaultConfig ) +
           Math.max(
             this.node.hSizable ? this.node.minimumWidth : this.node.width,
             this.withDefault( 'minCellWidth', defaultConfig )
           ) +
           this.withDefault( 'rightMargin', defaultConfig );
  }

  /**
   * @public
   *
   * @param {GridConfigurable} defaultConfig
   * @returns {number}
   */
  getMinimumHeight( defaultConfig ) {
    return this.withDefault( 'topMargin', defaultConfig ) +
           Math.max(
             this.node.vSizable ? this.node.minimumHeight : this.node.height,
             this.withDefault( 'minCellHeight', defaultConfig )
           ) +
           this.withDefault( 'bottomMargin', defaultConfig );
  }

  /**
   * @public
   *
   * @param {GridConfigurable} defaultConfig
   * @returns {number}
   */
  getMaximumWidth( defaultConfig ) {
    return this.withDefault( 'leftMargin', defaultConfig ) +
           Math.min(
             this.withDefault( 'maxCellWidth', defaultConfig ) || Number.POSITIVE_INFINITY
           ) +
           this.withDefault( 'rightMargin', defaultConfig );
  }

  /**
   * @public
   *
   * @param {GridConfigurable} defaultConfig
   * @returns {number}
   */
  getMaximumHeight( defaultConfig ) {
    return this.withDefault( 'topMargin', defaultConfig ) +
           Math.min(
             this.withDefault( 'maxCellHeight', defaultConfig ) || Number.POSITIVE_INFINITY
           ) +
           this.withDefault( 'bottomMargin', defaultConfig );
  }

  /**
   * @public
   *
   * @param {GridConfigurable} defaultConfig
   * @param {number} value
   */
  attemptedPreferredWidth( defaultConfig, value ) {
    if ( this.node.hSizable ) {
      const minimumWidth = this.getMinimumWidth( defaultConfig );
      const maximumWidth = this.getMaximumWidth( defaultConfig );

      assert && assert( isFinite( minimumWidth ) );
      assert && assert( maximumWidth >= minimumWidth );

      value = Utils.clamp( value, minimumWidth, maximumWidth );

      this.node.preferredWidth = value - this.withDefault( 'leftMargin', defaultConfig ) - this.withDefault( 'rightMargin', defaultConfig );
    }
  }

  /**
   * @public
   *
   * @param {GridConfigurable} defaultConfig
   * @param {number} value
   */
  attemptedPreferredHeight( defaultConfig, value ) {
    if ( this.node.vSizable ) {
      const minimumHeight = this.getMinimumHeight( defaultConfig );
      const maximumHeight = this.getMaximumHeight( defaultConfig );

      assert && assert( isFinite( minimumHeight ) );
      assert && assert( maximumHeight >= minimumHeight );

      value = Utils.clamp( value, minimumHeight, maximumHeight );

      this.node.preferredHeight = value - this.withDefault( 'topMargin', defaultConfig ) - this.withDefault( 'bottomMargin', defaultConfig );
    }
  }

  /**
   * @public
   *
   * @param {GridConfigurable} defaultConfig
   * @param {number} value
   */
  xStart( defaultConfig, value ) {
    // TODO: coordinate transform handling, to our rootNode!!!!!
    const left = this.withDefault( 'leftMargin', defaultConfig ) + value;

    if ( Math.abs( this.node.left - left ) > 1e-9 ) {
      this.node.left = left;
    }
  }

  /**
   * @public
   *
   * @param {GridConfigurable} defaultConfig
   * @param {number} value
   */
  yStart( defaultConfig, value ) {
    // TODO: coordinate transform handling, to our rootNode!!!!!
    const top = this.withDefault( 'topMargin', defaultConfig ) + value;

    if ( Math.abs( this.node.top - top ) > 1e-9 ) {
      this.node.top = top;
    }
  }

  /**
   * @public
   *
   * @param {GridConfigurable} defaultConfig
   * @param {number} value
   */
  xOrigin( defaultConfig, value ) {
    if ( Math.abs( this.node.x - value ) > 1e-9 ) {
      this.node.x = value;
    }
  }

  /**
   * @public
   *
   * @param {GridConfigurable} defaultConfig
   * @param {number} value
   */
  yOrigin( defaultConfig, value ) {
    if ( Math.abs( this.node.y - value ) > 1e-9 ) {
      this.node.y = value;
    }
  }

  /**
   * @public
   *
   * @param {GridConfigurable} defaultConfig
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
   * @public
   *
   * @param {number} row
   * @returns {boolean}
   */
  containsRow( row ) {
    return row >= this.y && row < this.y + this.height;
  }

  /**
   * @public
   *
   * @param {number} column
   * @returns {boolean}
   */
  containsColumn( column ) {
    return column >= this.x && column < this.x + this.width;
  }

  /**
   * @public
   *
   * @returns {number}
   */
  getRowIndices() {
    return _.range( this.y, this.y + this.height );
  }

  /**
   * @public
   *
   * @returns {number}
   */
  getColumnIndices() {
    return _.range( this.x, this.x + this.width );
  }

  /**
   * Releases references
   * @public
   */
  dispose() {
    this.node.layoutOptionsChangedEmitter.removeListener( this.layoutOptionsListener );
  }
}

scenery.register( 'GridCell', GridCell );
export default GridCell;