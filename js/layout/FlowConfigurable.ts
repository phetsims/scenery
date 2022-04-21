// Copyright 2022, University of Colorado Boulder

/**
 * Mixin for storing options that can affect each cell.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import TinyEmitter from '../../../axon/js/TinyEmitter.js';
import Orientation from '../../../phet-core/js/Orientation.js';
import memoize from '../../../phet-core/js/memoize.js';
import mutate from '../../../phet-core/js/mutate.js';
import { scenery } from '../imports.js';
import Constructor from '../../../phet-core/js/types/Constructor.js';
import EnumerationValue from '../../../phet-core/js/EnumerationValue.js';
import Enumeration from '../../../phet-core/js/Enumeration.js';

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

const flowHorizontalAligns = [ 'top', 'bottom', 'center', 'origin', 'stretch' ] as const;
const flowVerticalAligns = [ 'left', 'right', 'center', 'origin', 'stretch' ] as const;
const flowOrientations = [ 'horizontal', 'vertical' ] as const;

export type FlowHorizontalAlign = typeof flowHorizontalAligns[number];
export type FlowVerticalAlign = typeof flowVerticalAligns[number];
export type FlowOrientation = typeof flowOrientations[number];

export type FlowConfigurableOptions = {
  orientation?: FlowOrientation | null;
  align?: FlowHorizontalAlign | FlowVerticalAlign | null;
  grow?: number | null;
  margin?: number | null;
  xMargin?: number | null;
  yMargin?: number | null;
  leftMargin?: number | null;
  rightMargin?: number | null;
  topMargin?: number | null;
  bottomMargin?: number | null;

  // For these, the nullable portion is actually part of the possible "value"
  minContentWidth?: number | null;
  minContentHeight?: number | null;
  maxContentWidth?: number | null;
  maxContentHeight?: number | null;
};

const getAllowedAligns = ( orientation: Orientation ): readonly ( string | null )[] => {
  return [ ...( orientation === Orientation.HORIZONTAL ? flowHorizontalAligns : flowVerticalAligns ), null ];
};

export class FlowConfigurableAlign extends EnumerationValue {
  static START = new FlowConfigurableAlign( 'top', 'left', 0 );
  static END = new FlowConfigurableAlign( 'bottom', 'right', 1 );
  static CENTER = new FlowConfigurableAlign( 'center', 'center', 0.5 );
  static ORIGIN = new FlowConfigurableAlign( 'origin', 'origin' );
  static STRETCH = new FlowConfigurableAlign( 'stretch', 'stretch' );

  horizontal: FlowHorizontalAlign;
  vertical: FlowVerticalAlign;
  padRatio: number;

  constructor( horizontal: FlowHorizontalAlign, vertical: FlowVerticalAlign, padRatio: number = Number.POSITIVE_INFINITY ) {
    super();

    this.horizontal = horizontal;
    this.vertical = vertical;
    this.padRatio = padRatio;
  }

  static enumeration = new Enumeration( FlowConfigurableAlign, {
    phetioDocumentation: 'Align for FlowConfigurable'
  } );
}

const horizontalAlignMap = {
  top: FlowConfigurableAlign.START,
  bottom: FlowConfigurableAlign.END,
  center: FlowConfigurableAlign.CENTER,
  origin: FlowConfigurableAlign.ORIGIN,
  stretch: FlowConfigurableAlign.STRETCH
};
const verticalAlignMap = {
  left: FlowConfigurableAlign.START,
  right: FlowConfigurableAlign.END,
  center: FlowConfigurableAlign.CENTER,
  origin: FlowConfigurableAlign.ORIGIN,
  stretch: FlowConfigurableAlign.STRETCH
};
const alignToInternal = ( orientation: Orientation, key: FlowHorizontalAlign | FlowVerticalAlign | null ): FlowConfigurableAlign | null => {
  if ( key === null ) {
    return null;
  }
  else if ( orientation === Orientation.HORIZONTAL ) {
    assert && assert( horizontalAlignMap[ key as 'top' | 'bottom' | 'center' | 'origin' | 'stretch' ] );

    return horizontalAlignMap[ key as 'top' | 'bottom' | 'center' | 'origin' | 'stretch' ];
  }
  else {
    assert && assert( verticalAlignMap[ key as 'left' | 'right' | 'center' | 'origin' | 'stretch' ] );

    return verticalAlignMap[ key as 'left' | 'right' | 'center' | 'origin' | 'stretch' ];
  }
};
const internalToAlign = ( orientation: Orientation, align: FlowConfigurableAlign | null ): FlowHorizontalAlign | FlowVerticalAlign | null => {
  if ( align === null ) {
    return null;
  }
  else if ( orientation === Orientation.HORIZONTAL ) {
    return align.horizontal;
  }
  else {
    return align.vertical;
  }
};

const FlowConfigurable = memoize( <SuperType extends Constructor>( type: SuperType ) => {
  return class extends type {

    _orientation: Orientation;

    // Null value inherits from a base config
    _align: FlowConfigurableAlign | null;

    // Null value inherits from a base config
    _leftMargin: number | null;
    _rightMargin: number | null;
    _topMargin: number | null;
    _bottomMargin: number | null;
    _grow: number | null;
    _minContentWidth: number | null;
    _minContentHeight: number | null;
    _maxContentWidth: number | null;
    _maxContentHeight: number | null;

    // {TinyEmitter}
    changedEmitter: TinyEmitter;
    orientationChangedEmitter: TinyEmitter;

    constructor( ...args: any[] ) {
      super( ...args );

      // @protected {Orientation}
      this._orientation = Orientation.HORIZONTAL;

      // @protected {FlowConfigurableAlign|null} - Null value inherits from a base config
      this._align = null;

      // @protected {number|null} - Null value inherits from a base config
      this._leftMargin = null;
      this._rightMargin = null;
      this._topMargin = null;
      this._bottomMargin = null;
      this._grow = null;
      this._minContentWidth = null;
      this._minContentHeight = null;
      this._maxContentWidth = null;
      this._maxContentHeight = null;

      // {TinyEmitter}
      this.changedEmitter = new TinyEmitter();
      this.orientationChangedEmitter = new TinyEmitter();
    }

    mutateConfigurable( options?: FlowConfigurableOptions ) {
      mutate( this, FLOW_CONFIGURABLE_OPTION_KEYS, options );
    }

    setConfigToBaseDefault() {
      this._align = FlowConfigurableAlign.CENTER;
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

    get orientation(): FlowOrientation {
      return this._orientation === Orientation.HORIZONTAL ? 'horizontal' : 'vertical';
    }

    set orientation( value: FlowOrientation ) {
      assert && assert( value === 'horizontal' || value === 'vertical' );

      const enumOrientation = value === 'horizontal' ? Orientation.HORIZONTAL : Orientation.VERTICAL;

      if ( this._orientation !== enumOrientation ) {
        this._orientation = enumOrientation;

        this.orientationChangedEmitter.emit();
        this.changedEmitter.emit();
      }
    }

    get align(): FlowHorizontalAlign | FlowVerticalAlign | null {
      const result = internalToAlign( this._orientation, this._align );

      assert && assert( result === null || typeof result === 'string' );

      return result;
    }

    set align( value: FlowHorizontalAlign | FlowVerticalAlign | null ) {
      assert && assert( getAllowedAligns( this._orientation ).includes( value ),
        `align ${value} not supported, with the orientation ${this._orientation}, the valid values are ${getAllowedAligns( this._orientation )}` );

      // remapping align values to an independent set, so they aren't orientation-dependent
      const mappedValue = alignToInternal( this._orientation, value );

      assert && assert( mappedValue === null || mappedValue instanceof FlowConfigurableAlign );

      if ( this._align !== mappedValue ) {
        this._align = mappedValue;

        this.changedEmitter.emit();
      }
    }

    get leftMargin(): number | null {
      return this._leftMargin;
    }

    set leftMargin( value: number | null ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) ) );

      if ( this._leftMargin !== value ) {
        this._leftMargin = value;

        this.changedEmitter.emit();
      }
    }

    get rightMargin(): number | null {
      return this._rightMargin;
    }

    set rightMargin( value: number | null ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) ) );

      if ( this._rightMargin !== value ) {
        this._rightMargin = value;

        this.changedEmitter.emit();
      }
    }

    get topMargin(): number | null {
      return this._topMargin;
    }

    set topMargin( value: number | null ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) ) );

      if ( this._topMargin !== value ) {
        this._topMargin = value;

        this.changedEmitter.emit();
      }
    }

    get bottomMargin(): number | null {
      return this._bottomMargin;
    }

    set bottomMargin( value: number | null ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) ) );

      if ( this._bottomMargin !== value ) {
        this._bottomMargin = value;

        this.changedEmitter.emit();
      }
    }

    get grow(): number | null {
      return this._grow;
    }

    set grow( value: number | null ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) && value >= 0 ) );

      if ( this._grow !== value ) {
        this._grow = value;

        this.changedEmitter.emit();
      }
    }

    get xMargin(): number | null {
      assert && assert( this._leftMargin === this._rightMargin );

      return this._leftMargin;
    }

    set xMargin( value: number | null ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) ) );

      if ( this._leftMargin !== value || this._rightMargin !== value ) {
        this._leftMargin = value;
        this._rightMargin = value;

        this.changedEmitter.emit();
      }
    }

    get yMargin(): number | null {
      assert && assert( this._topMargin === this._bottomMargin );

      return this._topMargin;
    }

    set yMargin( value: number | null ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) ) );

      if ( this._topMargin !== value || this._bottomMargin !== value ) {
        this._topMargin = value;
        this._bottomMargin = value;

        this.changedEmitter.emit();
      }
    }

    get margin(): number | null {
      assert && assert(
      this._leftMargin === this._rightMargin &&
      this._leftMargin === this._topMargin &&
      this._leftMargin === this._bottomMargin
      );

      return this._topMargin;
    }

    set margin( value: number | null ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) ) );

      if ( this._leftMargin !== value || this._rightMargin !== value || this._topMargin !== value || this._bottomMargin !== value ) {
        this._leftMargin = value;
        this._rightMargin = value;
        this._topMargin = value;
        this._bottomMargin = value;

        this.changedEmitter.emit();
      }
    }

    get minContentWidth(): number | null {
      return this._minContentWidth;
    }

    set minContentWidth( value: number | null ) {
      if ( this._minContentWidth !== value ) {
        this._minContentWidth = value;

        this.changedEmitter.emit();
      }
    }

    get minContentHeight(): number | null {
      return this._minContentHeight;
    }

    set minContentHeight( value: number | null ) {
      if ( this._minContentHeight !== value ) {
        this._minContentHeight = value;

        this.changedEmitter.emit();
      }
    }

    get maxContentWidth(): number | null {
      return this._maxContentWidth;
    }

    set maxContentWidth( value: number | null ) {
      if ( this._maxContentWidth !== value ) {
        this._maxContentWidth = value;

        this.changedEmitter.emit();
      }
    }

    get maxContentHeight(): number | null {
      return this._maxContentHeight;
    }

    set maxContentHeight( value: number | null ) {
      if ( this._maxContentHeight !== value ) {
        this._maxContentHeight = value;

        this.changedEmitter.emit();
      }
    }
  };
} );

scenery.register( 'FlowConfigurable', FlowConfigurable );
export default FlowConfigurable;
export { FLOW_CONFIGURABLE_OPTION_KEYS };
