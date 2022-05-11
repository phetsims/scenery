// Copyright 2021-2022, University of Colorado Boulder

/**
 * Mixin for storing options that can affect each cell.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import TinyEmitter from '../../../axon/js/TinyEmitter.js';
import Constructor from '../../../phet-core/js/types/Constructor.js';
import Enumeration from '../../../phet-core/js/Enumeration.js';
import EnumerationValue from '../../../phet-core/js/EnumerationValue.js';
import memoize from '../../../phet-core/js/memoize.js';
import mutate from '../../../phet-core/js/mutate.js';
import { scenery } from '../imports.js';

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

const gridHorizontalAligns = [ 'left', 'right', 'center', 'origin', 'stretch' ] as const;
const gridVerticalAligns = [ 'top', 'bottom', 'center', 'origin', 'stretch' ] as const;

export type GridHorizontalAlign = typeof gridHorizontalAligns[number];
export type GridVerticalAlign = typeof gridVerticalAligns[number];

export class GridConfigurableAlign extends EnumerationValue {
  static readonly START = new GridConfigurableAlign( 'left', 'top', 0 );
  static readonly END = new GridConfigurableAlign( 'right', 'bottom', 1 );
  static readonly CENTER = new GridConfigurableAlign( 'center', 'center', 0.5 );
  static readonly ORIGIN = new GridConfigurableAlign( 'origin', 'origin' );
  static readonly STRETCH = new GridConfigurableAlign( 'stretch', 'stretch', 0 );

  readonly horizontal: GridHorizontalAlign;
  readonly vertical: GridVerticalAlign;
  readonly padRatio: number;

  constructor( horizontal: GridHorizontalAlign, vertical: GridVerticalAlign, padRatio: number = Number.POSITIVE_INFINITY ) {
    super();

    this.horizontal = horizontal;
    this.vertical = vertical;
    this.padRatio = padRatio;
  }

  static readonly enumeration = new Enumeration( GridConfigurableAlign, {
    phetioDocumentation: 'Align for GridConfigurable'
  } );
}

export type GridConfigurableOptions = {
  // NOTE: 'origin' aligns will only apply to cells that are 1 grid line in that orientation (width/height)
  xAlign?: GridHorizontalAlign | null;
  yAlign?: GridVerticalAlign | null;
  grow?: number | null;
  xGrow?: number | null;
  yGrow?: number | null;
  margin?: number | null;
  xMargin?: number | null;
  yMargin?: number | null;
  leftMargin?: number | null;
  rightMargin?: number | null;
  topMargin?: number | null;
  bottomMargin?: number | null;
  minContentWidth?: number | null;
  minContentHeight?: number | null;
  maxContentWidth?: number | null;
  maxContentHeight?: number | null;
};

const horizontalAlignMap = {
  left: GridConfigurableAlign.START,
  right: GridConfigurableAlign.END,
  center: GridConfigurableAlign.CENTER,
  origin: GridConfigurableAlign.ORIGIN,
  stretch: GridConfigurableAlign.STRETCH
};
const verticalAlignMap = {
  top: GridConfigurableAlign.START,
  bottom: GridConfigurableAlign.END,
  center: GridConfigurableAlign.CENTER,
  origin: GridConfigurableAlign.ORIGIN,
  stretch: GridConfigurableAlign.STRETCH
};
const horizontalAlignToInternal = ( key: GridHorizontalAlign | null ): GridConfigurableAlign | null => {
  if ( key === null ) {
    return null;
  }

  assert && assert( horizontalAlignMap[ key as 'left' | 'right' | 'center' | 'origin' | 'stretch' ] );

  return horizontalAlignMap[ key as 'left' | 'right' | 'center' | 'origin' | 'stretch' ];
};
const verticalAlignToInternal = ( key: GridVerticalAlign | null ): GridConfigurableAlign | null => {
  if ( key === null ) {
    return null;
  }

  assert && assert( verticalAlignMap[ key as 'top' | 'bottom' | 'center' | 'origin' | 'stretch' ] );

  return verticalAlignMap[ key as 'top' | 'bottom' | 'center' | 'origin' | 'stretch' ];
};

const GridConfigurable = memoize( <SuperType extends Constructor>( type: SuperType ) => {
  return class extends type {

    _xAlign: GridConfigurableAlign | null;
    _yAlign: GridConfigurableAlign | null;
    _leftMargin: number | null;
    _rightMargin: number | null;
    _topMargin: number | null;
    _bottomMargin: number | null;
    _xGrow: number | null;
    _yGrow: number | null;
    _minContentWidth: number | null;
    _minContentHeight: number | null;
    _maxContentWidth: number | null;
    _maxContentHeight: number | null;

    readonly changedEmitter: TinyEmitter;

    constructor( ...args: any[] ) {
      super( ...args );

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

      this.changedEmitter = new TinyEmitter();
    }

    mutateConfigurable( options?: GridConfigurableOptions ): void {
      mutate( this, GRID_CONFIGURABLE_OPTION_KEYS, options );
    }

    setConfigToBaseDefault(): void {
      this._xAlign = GridConfigurableAlign.CENTER;
      this._yAlign = GridConfigurableAlign.CENTER;
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
     */
    setConfigToInherit(): void {
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

    get xAlign(): GridHorizontalAlign | null {
      const result = this._xAlign === null ? null : this._xAlign.horizontal;

      assert && assert( result === null || typeof result === 'string' );

      return result;
    }

    set xAlign( value: GridHorizontalAlign | null ) {
      assert && assert( value === null || gridHorizontalAligns.includes( value ),
        `align ${value} not supported, the valid values are ${gridHorizontalAligns} or null` );

      // remapping align values to an independent set, so they aren't orientation-dependent
      const mappedValue = horizontalAlignToInternal( value );

      assert && assert( mappedValue === null || mappedValue instanceof GridConfigurableAlign );

      if ( this._xAlign !== mappedValue ) {
        this._xAlign = mappedValue;

        this.changedEmitter.emit();
      }
    }

    get yAlign(): GridVerticalAlign | null {
      const result = this._yAlign === null ? null : this._yAlign.vertical;

      assert && assert( result === null || typeof result === 'string' );

      return result;
    }

    set yAlign( value: GridVerticalAlign | null ) {
      assert && assert( value === null || gridVerticalAligns.includes( value ),
        `align ${value} not supported, the valid values are ${gridVerticalAligns} or null` );

      // remapping align values to an independent set, so they aren't orientation-dependent
      const mappedValue = verticalAlignToInternal( value );

      assert && assert( mappedValue === null || mappedValue instanceof GridConfigurableAlign );

      if ( this._yAlign !== mappedValue ) {
        this._yAlign = mappedValue;

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
      assert && assert( this._xGrow === this._yGrow );

      return this._xGrow;
    }

    set grow( value: number | null ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) && value >= 0 ) );

      if ( this._xGrow !== value || this._yGrow !== value ) {
        this._xGrow = value;
        this._yGrow = value;

        this.changedEmitter.emit();
      }
    }

    get xGrow(): number | null {
      return this._xGrow;
    }

    set xGrow( value: number | null ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) && value >= 0 ) );

      if ( this._xGrow !== value ) {
        this._xGrow = value;

        this.changedEmitter.emit();
      }
    }

    get yGrow(): number | null {
      return this._yGrow;
    }

    set yGrow( value: number | null ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) && value >= 0 ) );

      if ( this._yGrow !== value ) {
        this._yGrow = value;

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

scenery.register( 'GridConfigurable', GridConfigurable );
export default GridConfigurable;
export { GRID_CONFIGURABLE_OPTION_KEYS };
