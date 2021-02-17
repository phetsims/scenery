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

const FLOW_CONFIGURABLE_KEYS = [
  'align',
  'grow',
  'margin',
  'xMargin',
  'yMargin',
  'leftMargin',
  'rightMargin',
  'topMargin',
  'bottomMargin',
  'minCellWidth',
  'minCellHeight',
  'maxCellWidth',
  'maxCellHeight'
];

const FlowConfigurable = memoize( type => {
  return class extends type {
    constructor( ...args ) {
      super( ...args );

      // @private {FlowConfigurable.Align|null} - Null value inherits from a base config
      this._align = null;

      // @private {number|null} - Null value inherits from a base config
      this._leftMargin = null;
      this._rightMargin = null;
      this._topMargin = null;
      this._bottomMargin = null;
      this._grow = null;
      this._minCellWidth = null;
      this._minCellHeight = null;
      this._maxCellWidth = null;
      this._maxCellHeight = null;

      // @public {TinyEmitter}
      this.changedEmitter = new TinyEmitter();
    }

    /**
     * @public
     *
     * @param {Object} [options]
     */
    mutateConfigurable( options ) {
      mutate( this, FLOW_CONFIGURABLE_KEYS, options );
    }

    /**
     * @public
     */
    setConfigToBaseDefault() {
      this._align = FlowConfigurable.Align.CENTER;
      this._leftMargin = 0;
      this._rightMargin = 0;
      this._topMargin = 0;
      this._bottomMargin = 0;
      this._grow = 0;
      this._minCellWidth = null;
      this._minCellHeight = null;
      this._maxCellWidth = null;
      this._maxCellHeight = null;

      this.changedEmitter.emit();
    }

    /**
     * Resets values to their original state
     * @public
     */
    setConfigToInherit() {
      this._align = null;
      this._leftMargin = null;
      this._rightMargin = null;
      this._topMargin = null;
      this._bottomMargin = null;
      this._grow = null;
      this._minCellWidth = null;
      this._minCellHeight = null;
      this._maxCellWidth = null;
      this._maxCellHeight = null;

      this.changedEmitter.emit();
    }

    /**
     * @public
     *
     * @param {string} propertyName
     * @param {FlowConfigurable} defaultConfig
     * @returns {*}
     */
    withDefault( propertyName, defaultConfig ) {
      return this[ propertyName ] !== null ? this[ propertyName ] : defaultConfig[ propertyName ];
    }

    /**
     * @public
     *
     * @returns {FlowConfigurable.Align|null}
     */
    get align() {
      return this._align;
    }

    /**
     * @public
     *
     * @param {FlowConfigurable.Align|string|null} value
     */
    set align( value ) {
      // remapping align values to an independent set, so they aren't orientation-dependent
      // TODO: consider if this is wise
      if ( typeof value === 'string' ) {
        value = alignMapping[ value ];
      }

      assert && assert( value === null || FlowConfigurable.Align.includes( value ) );

      if ( this._align !== value ) {
        this._align = value;

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
    get minCellWidth() {
      return this._minCellWidth;
    }

    /**
     * @public
     *
     * @param {number|null} value
     */
    set minCellWidth( value ) {
      if ( this._minCellWidth !== value ) {
        this._minCellWidth = value;

        this.changedEmitter.emit();
      }
    }

    /**
     * @public
     *
     * @returns {number|null}
     */
    get minCellHeight() {
      return this._minCellHeight;
    }

    /**
     * @public
     *
     * @param {number|null} value
     */
    set minCellHeight( value ) {
      if ( this._minCellHeight !== value ) {
        this._minCellHeight = value;

        this.changedEmitter.emit();
      }
    }

    /**
     * @public
     *
     * @returns {number|null}
     */
    get maxCellWidth() {
      return this._maxCellWidth;
    }

    /**
     * @public
     *
     * @param {number|null} value
     */
    set maxCellWidth( value ) {
      if ( this._maxCellWidth !== value ) {
        this._maxCellWidth = value;

        this.changedEmitter.emit();
      }
    }

    /**
     * @public
     *
     * @returns {number|null}
     */
    get maxCellHeight() {
      return this._maxCellHeight;
    }

    /**
     * @public
     *
     * @param {number|null} value
     */
    set maxCellHeight( value ) {
      if ( this._maxCellHeight !== value ) {
        this._maxCellHeight = value;

        this.changedEmitter.emit();
      }
    }
  };
} );

FlowConfigurable.Align = Enumeration.byKeys( [
  'START',
  'END',
  'CENTER',
  'ORIGIN',
  'STRETCH'
] );

const alignMapping = {
  'left': FlowConfigurable.Align.START,
  'top': FlowConfigurable.Align.START,
  'right': FlowConfigurable.Align.END,
  'bottom': FlowConfigurable.Align.END,
  'center': FlowConfigurable.Align.CENTER,
  'origin': FlowConfigurable.Align.ORIGIN,
  'stretch': FlowConfigurable.Align.STRETCH
};

scenery.register( 'FlowConfigurable', FlowConfigurable );
export default FlowConfigurable;