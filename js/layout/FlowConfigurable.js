// Copyright 2021, University of Colorado Boulder

/**
 * TODO: doc
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import TinyEmitter from '../../../axon/js/TinyEmitter.js';
import Enumeration from '../../../phet-core/js/Enumeration.js';
import Orientation from '../../../phet-core/js/Orientation.js';
import memoize from '../../../phet-core/js/memoize.js';
import mutate from '../../../phet-core/js/mutate.js';
import scenery from '../scenery.js';

const FLOW_CONFIGURABLE_OPTION_KEYS = [
  'orientation',
  'align',
  'grow',
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

const FlowConfigurable = memoize( type => {
  return class extends type {
    constructor( ...args ) {
      super( ...args );

      // @private {Orientation}
      this._orientation = Orientation.HORIZONTAL;

      // @private {FlowConfigurable.Align|null} - Null value inherits from a base config
      this._align = null;

      // @private {number|null} - Null value inherits from a base config
      this._leftMargin = null;
      this._rightMargin = null;
      this._topMargin = null;
      this._bottomMargin = null;
      this._grow = null;
      this._minContentWidth = null;
      this._minContentHeight = null;
      this._maxContentWidth = null;
      this._maxContentHeight = null;

      // @public {TinyEmitter}
      this.changedEmitter = new TinyEmitter();
      this.orientationChangedEmitter = new TinyEmitter();
    }

    /**
     * @public
     *
     * @param {Object} [options]
     */
    mutateConfigurable( options ) {
      mutate( this, FLOW_CONFIGURABLE_OPTION_KEYS, options );
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
      this._align = null;
      this._leftMargin = null;
      this._rightMargin = null;
      this._topMargin = null;
      this._bottomMargin = null;
      this._grow = null;
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
     * @param {FlowConfigurable} defaultConfig
     * @returns {*}
     */
    withDefault( propertyName, defaultConfig ) {
      return this[ propertyName ] !== null ? this[ propertyName ] : defaultConfig[ propertyName ];
    }

    /**
     * @public
     *
     * @param {FlowConfigurable} defaultConfig
     * @returns {FlowConfigurable}
     */
    withDefaults( defaultConfig ) {
      const configurable = new FlowConfigurableObject();

      configurable._align = this._align !== null ? this._align : defaultConfig._align;
      configurable._leftMargin = this._leftMargin !== null ? this._leftMargin : defaultConfig._leftMargin;
      configurable._rightMargin = this._rightMargin !== null ? this._rightMargin : defaultConfig._rightMargin;
      configurable._topMargin = this._topMargin !== null ? this._topMargin : defaultConfig._topMargin;
      configurable._bottomMargin = this._bottomMargin !== null ? this._bottomMargin : defaultConfig._bottomMargin;
      configurable._grow = this._grow !== null ? this._grow : defaultConfig._grow;
      configurable._minContentWidth = this._minContentWidth !== null ? this._minContentWidth : defaultConfig._minContentWidth;
      configurable._minContentHeight = this._minContentHeight !== null ? this._minContentHeight : defaultConfig._minContentHeight;
      configurable._maxContentWidth = this._maxContentWidth !== null ? this._maxContentWidth : defaultConfig._maxContentWidth;
      configurable._maxContentHeight = this._maxContentHeight !== null ? this._maxContentHeight : defaultConfig._maxContentHeight;

      return configurable;
    }

    /**
     * @public
     *
     * @returns {Orientation}
     */
    get orientation() {
      // TODO: Consider the returning 'horizontal/vertical'? Bleh
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

        this.orientationChangedEmitter.emit();
        this.changedEmitter.emit();
      }
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
    get grow() {
      return this._grow;
    }

    /**
     * @public
     *
     * @param {number|null} value
     */
    set grow( value ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) && value >= 0 ) );

      if ( this._grow !== value ) {
        this._grow = value;

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
FlowConfigurable.Align = Enumeration.byKeys( [
  'START',
  'END',
  'CENTER',
  'ORIGIN',
  'STRETCH'
] );

const alignMapping = {
  left: FlowConfigurable.Align.START,
  top: FlowConfigurable.Align.START,
  right: FlowConfigurable.Align.END,
  bottom: FlowConfigurable.Align.END,
  center: FlowConfigurable.Align.CENTER,
  origin: FlowConfigurable.Align.ORIGIN,
  stretch: FlowConfigurable.Align.STRETCH
};

// @public {Object}
FlowConfigurable.FLOW_CONFIGURABLE_OPTION_KEYS = FLOW_CONFIGURABLE_OPTION_KEYS;

const FlowConfigurableObject = FlowConfigurable( Object );

scenery.register( 'FlowConfigurable', FlowConfigurable );
export default FlowConfigurable;