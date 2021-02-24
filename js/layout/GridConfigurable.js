// Copyright 2021, University of Colorado Boulder

/**
 * TODO: doc
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import TinyEmitter from '../../../axon/js/TinyEmitter.js';
import Enumeration from '../../../phet-core/js/Enumeration.js';
import memoize from '../../../phet-core/js/memoize.js';
import mutate from '../../../phet-core/js/mutate.js';
import scenery from '../scenery.js';

const GRID_CONFIGURABLE_OPTION_KEYS = [
  'xAlign',
  'yAlign',
  'grow',
  'xGrow',
  'yGrow',
  'margin',
  'xMargin',
  'yMargin',
  'leftMargin',
  'rightMargin',
  'topMargin',
  'bottomMargin',
  'minContentWidth',
  'minContentHeight',
  'maxContentWidth',
  'maxContentHeight'
];

const GridConfigurable = memoize( type => {
  return class extends type {
    constructor( ...args ) {
      super( ...args );

      // @private {GridConfigurable.Align|null} - Null value inherits from a base config
      this._xAlign = null;
      this._yAlign = null;

      // @private {number|null} - Null value inherits from a base config
      this._leftMargin = null;
      this._rightMargin = null;
      this._topMargin = null;
      this._bottomMargin = null;
      this._xGrow = null;
      this._yGrow = null;
      this._minContentWidth = null;
      this._minContentHeight = null;
      this._maxContentWidth = null;
      this._maxContentHeight = null;

      // @public {TinyEmitter}
      this.changedEmitter = new TinyEmitter();
    }

    /**
     * @public
     *
     * @param {Object} [options]
     */
    mutateConfigurable( options ) {
      mutate( this, GRID_CONFIGURABLE_OPTION_KEYS, options );
    }

    /**
     * @public
     */
    setConfigToBaseDefault() {
      this._xAlign = GridConfigurable.Align.CENTER;
      this._yAlign = GridConfigurable.Align.CENTER;
      this._leftMargin = 0;
      this._rightMargin = 0;
      this._topMargin = 0;
      this._bottomMargin = 0;
      this._xGrow = 0;
      this._yGrow = 0;
      this._minContentWidth = null;
      this._minContentHeight = null;
      this._maxContentWidth = null;
      this._maxContentHeight = null;

      this.changedEmitter.emit();
    }

    /**
     * Resets values to their original state
     * @public
     */
    setConfigToInherit() {
      this._xAlign = null;
      this._yAlign = null;
      this._leftMargin = null;
      this._rightMargin = null;
      this._topMargin = null;
      this._bottomMargin = null;
      this._xGrow = null;
      this._yGrow = null;
      this._minContentWidth = null;
      this._minContentHeight = null;
      this._maxContentWidth = null;
      this._maxContentHeight = null;

      this.changedEmitter.emit();
    }

    /**
     * @public
     *
     * @param {string} propertyName
     * @param {GridConfigurable} defaultConfig
     * @returns {*}
     */
    withDefault( propertyName, defaultConfig ) {
      return this[ propertyName ] !== null ? this[ propertyName ] : defaultConfig[ propertyName ];
    }

    /**
     * @public
     *
     * @returns {GridConfigurable.Align|null}
     */
    get xAlign() {
      return this._xAlign;
    }

    /**
     * @public
     *
     * @param {GridConfigurable.Align|string|null} value
     */
    set xAlign( value ) {
      // remapping xAlign values to an independent set, so they aren't orientation-dependent
      // TODO: consider if this is wise
      if ( typeof value === 'string' ) {
        value = xAlignMapping[ value ];
      }

      assert && assert( value === null || GridConfigurable.Align.includes( value ) );

      if ( this._xAlign !== value ) {
        this._xAlign = value;

        this.changedEmitter.emit();
      }
    }

    /**
     * @public
     *
     * @returns {GridConfigurable.Align|null}
     */
    get yAlign() {
      return this._yAlign;
    }

    /**
     * @public
     *
     * @param {GridConfigurable.Align|string|null} value
     */
    set yAlign( value ) {
      // remapping yAlign values to an independent set, so they aren't orientation-dependent
      // TODO: consider if this is wise
      if ( typeof value === 'string' ) {
        value = yAlignMapping[ value ];
      }

      assert && assert( value === null || GridConfigurable.Align.includes( value ) );

      if ( this._yAlign !== value ) {
        this._yAlign = value;

        this.changedEmitter.emit();
      }
    }

    /**
     * @public
     *
     * @returns {number|null}
     */
    get leftMargin() {
      return this._leftMargin;
    }

    /**
     * @public
     *
     * @param {number|null} value
     */
    set leftMargin( value ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) ) );

      if ( this._leftMargin !== value ) {
        this._leftMargin = value;

        this.changedEmitter.emit();
      }
    }

    /**
     * @public
     *
     * @returns {number|null}
     */
    get rightMargin() {
      return this._rightMargin;
    }

    /**
     * @public
     *
     * @param {number|null} value
     */
    set rightMargin( value ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) ) );

      if ( this._rightMargin !== value ) {
        this._rightMargin = value;

        this.changedEmitter.emit();
      }
    }

    /**
     * @public
     *
     * @returns {number|null}
     */
    get topMargin() {
      return this._topMargin;
    }

    /**
     * @public
     *
     * @param {number|null} value
     */
    set topMargin( value ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) ) );

      if ( this._topMargin !== value ) {
        this._topMargin = value;

        this.changedEmitter.emit();
      }
    }

    /**
     * @public
     *
     * @returns {number|null}
     */
    get bottomMargin() {
      return this._bottomMargin;
    }

    /**
     * @public
     *
     * @param {number|null} value
     */
    set bottomMargin( value ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) ) );

      if ( this._bottomMargin !== value ) {
        this._bottomMargin = value;

        this.changedEmitter.emit();
      }
    }

    /**
     * @public
     *
     * @returns {number|null}
     */
    get grow() {
      assert && assert( this._xGrow === this._yGrow );

      return this._xGrow;
    }

    /**
     * @public
     *
     * @param {number|null} value
     */
    set grow( value ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) && value >= 0 ) );

      if ( this._xGrow !== value || this._yGrow !== value ) {
        this._xGrow = value;
        this._yGrow = value;

        this.changedEmitter.emit();
      }
    }

    /**
     * @public
     *
     * @returns {number|null}
     */
    get xGrow() {
      return this._xGrow;
    }

    /**
     * @public
     *
     * @param {number|null} value
     */
    set xGrow( value ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) && value >= 0 ) );

      if ( this._xGrow !== value ) {
        this._xGrow = value;

        this.changedEmitter.emit();
      }
    }

    /**
     * @public
     *
     * @returns {number|null}
     */
    get yGrow() {
      return this._yGrow;
    }

    /**
     * @public
     *
     * @param {number|null} value
     */
    set yGrow( value ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) && value >= 0 ) );

      if ( this._yGrow !== value ) {
        this._yGrow = value;

        this.changedEmitter.emit();
      }
    }

    /**
     * @public
     *
     * @returns {number|null}
     */
    get xMargin() {
      assert && assert( this._leftMargin === this._rightMargin );

      return this._leftMargin;
    }

    /**
     * @public
     *
     * @param {number|null} value
     */
    set xMargin( value ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) ) );

      if ( this._leftMargin !== value || this._rightMargin !== value ) {
        this._leftMargin = value;
        this._rightMargin = value;

        this.changedEmitter.emit();
      }
    }

    /**
     * @public
     *
     * @returns {number|null}
     */
    get yMargin() {
      assert && assert( this._topMargin === this._bottomMargin );

      return this._topMargin;
    }

    /**
     * @public
     *
     * @param {number|null} value
     */
    set yMargin( value ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) ) );

      if ( this._topMargin !== value || this._bottomMargin !== value ) {
        this._topMargin = value;
        this._bottomMargin = value;

        this.changedEmitter.emit();
      }
    }

    /**
     * @public
     *
     * @returns {number|null}
     */
    get margin() {
      assert && assert(
        this._leftMargin === this._rightMargin &&
        this._leftMargin === this._topMargin &&
        this._leftMargin === this._bottomMargin
      );

      return this._topMargin;
    }

    /**
     * @public
     *
     * @param {number|null} value
     */
    set margin( value ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) ) );

      if ( this._leftMargin !== value || this._rightMargin !== value || this._topMargin !== value || this._bottomMargin !== value ) {
        this._leftMargin = value;
        this._rightMargin = value;
        this._topMargin = value;
        this._bottomMargin = value;

        this.changedEmitter.emit();
      }
    }

    /**
     * @public
     *
     * @returns {number|null}
     */
    get minContentWidth() {
      return this._minContentWidth;
    }

    /**
     * @public
     *
     * @param {number|null} value
     */
    set minContentWidth( value ) {
      if ( this._minContentWidth !== value ) {
        this._minContentWidth = value;

        this.changedEmitter.emit();
      }
    }

    /**
     * @public
     *
     * @returns {number|null}
     */
    get minContentHeight() {
      return this._minContentHeight;
    }

    /**
     * @public
     *
     * @param {number|null} value
     */
    set minContentHeight( value ) {
      if ( this._minContentHeight !== value ) {
        this._minContentHeight = value;

        this.changedEmitter.emit();
      }
    }

    /**
     * @public
     *
     * @returns {number|null}
     */
    get maxContentWidth() {
      return this._maxContentWidth;
    }

    /**
     * @public
     *
     * @param {number|null} value
     */
    set maxContentWidth( value ) {
      if ( this._maxContentWidth !== value ) {
        this._maxContentWidth = value;

        this.changedEmitter.emit();
      }
    }

    /**
     * @public
     *
     * @returns {number|null}
     */
    get maxContentHeight() {
      return this._maxContentHeight;
    }

    /**
     * @public
     *
     * @param {number|null} value
     */
    set maxContentHeight( value ) {
      if ( this._maxContentHeight !== value ) {
        this._maxContentHeight = value;

        this.changedEmitter.emit();
      }
    }
  };
} );

// @public {Enumeration}
GridConfigurable.Align = Enumeration.byKeys( [
  'START',
  'END',
  'CENTER',
  'ORIGIN',
  'STRETCH'
] );

const xAlignMapping = {
  'left': GridConfigurable.Align.START,
  'right': GridConfigurable.Align.END,
  'center': GridConfigurable.Align.CENTER,
  'origin': GridConfigurable.Align.ORIGIN,
  'stretch': GridConfigurable.Align.STRETCH
};

const yAlignMapping = {
  'top': GridConfigurable.Align.START,
  'bottom': GridConfigurable.Align.END,
  'center': GridConfigurable.Align.CENTER,
  'origin': GridConfigurable.Align.ORIGIN,
  'stretch': GridConfigurable.Align.STRETCH
};

// @public {Object}
GridConfigurable.GRID_CONFIGURABLE_OPTION_KEYS = GRID_CONFIGURABLE_OPTION_KEYS;

scenery.register( 'GridConfigurable', GridConfigurable );
export default GridConfigurable;