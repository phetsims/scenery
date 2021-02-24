// Copyright 2021, University of Colorado Boulder

/**
 * TODO: doc
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Utils from '../../../dot/js/Utils.js';
import Orientation from '../../../phet-core/js/Orientation.js';
import OrientationPair from '../../../phet-core/js/OrientationPair.js';
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

    // @public {OrientationPair.<number>} - These are only set initially, and ignored for the future
    this.position = new OrientationPair( options.x, options.y );
    this.size = new OrientationPair( options.width, options.height );

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

  /**
   * @public
   *
   * @param {Orientation} orientation
   * @param {GridConfigurable} defaultConfig
   * @returns {number}
   */
  getMinimumSize( orientation, defaultConfig ) {
    return orientation === Orientation.HORIZONTAL ? this.getMinimumWidth( defaultConfig ) : this.getMinimumHeight( defaultConfig );
  }

  /**
   * @public
   *
   * @param {GridConfigurable} defaultConfig
   * @returns {number}
   */
  getMinimumWidth( defaultConfig ) {
    return this.withDefault( 'leftMargin', defaultConfig ) +
           Math.max(
             this.node.widthSizable ? this.node.minimumWidth : this.node.width,
             this.withDefault( 'minContentWidth', defaultConfig )
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
             this.node.heightSizable ? this.node.minimumHeight : this.node.height,
             this.withDefault( 'minContentHeight', defaultConfig )
           ) +
           this.withDefault( 'bottomMargin', defaultConfig );
  }

  /**
   * @public
   *
   * @param {Orientation} orientation
   * @param {GridConfigurable} defaultConfig
   * @returns {number}
   */
  getMaximumSize( orientation, defaultConfig ) {
    return orientation === Orientation.HORIZONTAL ? this.getMaximumWidth( defaultConfig ) : this.getMaximumHeight( defaultConfig );
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
             this.withDefault( 'maxContentWidth', defaultConfig ) || Number.POSITIVE_INFINITY
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
             this.withDefault( 'maxContentHeight', defaultConfig ) || Number.POSITIVE_INFINITY
           ) +
           this.withDefault( 'bottomMargin', defaultConfig );
  }

  /**
   * @public
   *
   * @param {Orientation} orientation
   * @param {GridConfigurable} defaultConfig
   * @param {number} value
   */
  attemptedPreferredSize( orientation, defaultConfig, value ) {
    orientation === Orientation.HORIZONTAL ? this.attemptedPreferredWidth( defaultConfig, value ) : this.attemptedPreferredHeight( defaultConfig, value );
  }

  /**
   * @public
   *
   * @param {GridConfigurable} defaultConfig
   * @param {number} value
   */
  attemptedPreferredWidth( defaultConfig, value ) {
    if ( this.node.widthSizable ) {
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
    if ( this.node.heightSizable ) {
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
   * @param {Orientation} orientation
   * @param {GridConfigurable} defaultConfig
   * @param {number} value
   */
  positionStart( orientation, defaultConfig, value ) {
    orientation === Orientation.HORIZONTAL ? this.xStart( defaultConfig, value ) : this.yStart( defaultConfig, value );
  }

  /**
   * @public
   *
   * @param {GridConfigurable} defaultConfig
   * @param {number} value
   */
  xStart( defaultConfig, value ) {
    // TODO: coordinate transform handling, to our ancestorNode!!!!!
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
    // TODO: coordinate transform handling, to our ancestorNode!!!!!
    const top = this.withDefault( 'topMargin', defaultConfig ) + value;

    if ( Math.abs( this.node.top - top ) > 1e-9 ) {
      this.node.top = top;
    }
  }

  /**
   * @public
   *
   * @param {Orientation} orientation
   * @param {GridConfigurable} defaultConfig
   * @param {number} value
   */
  positionOrigin( orientation, defaultConfig, value ) {
    orientation === Orientation.HORIZONTAL ? this.xOrigin( defaultConfig, value ) : this.yOrigin( defaultConfig, value );
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
   * @param {Orientation} orientation
   * @param {number} index
   * @returns {boolean}
   */
  containsIndex( orientation, index ) {
    const position = this.position.get( orientation );
    const size = this.size.get( orientation );
    return index >= position && index < position + size;
  }

  /**
   * @public
   *
   * @param {number} row
   * @returns {boolean}
   */
  containsRow( row ) {
    return this.containsIndex( Orientation.VERTICAL, row );
  }

  /**
   * @public
   *
   * @param {number} column
   * @returns {boolean}
   */
  containsColumn( column ) {
    return this.containsIndex( Orientation.HORIZONTAL, column );
  }

  /**
   * @public
   *
   * @param {Orientation} orientation
   * @returns {number}
   */
  getIndices( orientation ) {
    const position = this.position.get( orientation );
    const size = this.size.get( orientation );
    return _.range( position, position + size );
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