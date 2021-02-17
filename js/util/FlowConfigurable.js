// Copyright 2021, University of Colorado Boulder

/**
 * TODO: doc
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import TinyEmitter from '../../../axon/js/TinyEmitter.js';
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
  'bottomMargin'
];

const FlowConfigurable = memoize( type => {
  return class extends type {
    /**
     * @param {Object} [options]
     */
    constructor( ...args ) {
      super( ...args );

      // @private {string|null} - Null value inherits from a base config
      this._align = null;

      // @private {number|null} - Null value inherits from a base config
      this._leftMargin = null;
      this._rightMargin = null;
      this._topMargin = null;
      this._bottomMargin = null;
      this._grow = null;

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
      this._align = 'center';
      this._leftMargin = 0;
      this._rightMargin = 0;
      this._topMargin = 0;
      this._bottomMargin = 0;
      this._grow = 0;

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

      this.changedEmitter.emit();
    }

    /**
     * @public
     *
     * @returns {string|null}
     */
    get align() {
      return this._align;
    }

    /**
     * @public
     *
     * @param {string|null} value
     */
    set align( value ) {
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
      if ( this._leftMargin !== value || this._rightMargin !== value || this._topMargin !== value || this._bottomMargin !== value ) {
        this._leftMargin = value;
        this._rightMargin = value;
        this._topMargin = value;
        this._bottomMargin = value;

        this.changedEmitter.emit();
      }
    }
  };
} );

scenery.register( 'FlowConfigurable', FlowConfigurable );

export default FlowConfigurable;